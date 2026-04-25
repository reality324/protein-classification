#!/usr/bin/env python3
"""
Multitask 训练脚本

功能：训练多任务神经网络，同时预测EC主类、细胞定位、分子功能
"""
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np
import pandas as pd
import json
import time
import re
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 默认配置
DEFAULT_DATA_DIR = "data/datasets/train_subset.parquet"
DEFAULT_ESM2_DIR = "data/processed/esm2_aligned"
DEFAULT_OUTPUT_DIR = "models/multitask"
RANDOM_SEED = 42


class MultitaskModel(nn.Module):
    def __init__(self, input_dim, task_dims, hidden_dims=[256, 128], dropout=0.3):
        super().__init__()
        self.task_dims = task_dims
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        self.shared = nn.Sequential(*layers)
        
        self.heads = nn.ModuleDict({
            task: nn.Linear(prev_dim, dim) 
            for task, dim in task_dims.items()
        })
    
    def forward(self, x):
        features = self.shared(x)
        outputs = {task: head(features) for task, head in self.heads.items()}
        return outputs


def load_data(data_dir, esm2_dir):
    """加载多任务数据
    
    注意: 数据划分必须与 generate_esm2_features.py 保持一致 (60/20/20)
    """
    df = pd.read_parquet(data_dir)
    
    # EC标签 - 使用 one-hot 编码列
    ec_cols = [c for c in df.columns if c.startswith('ec_') and c.startswith('ec_') and c not in ['ec_number', 'ec_main_class']]
    ec_cols = [c for c in df.columns if re.match(r'^ec_\d+$', c)]  # ec_1, ec_2, ..., ec_7
    y_ec = np.argmax(df[ec_cols].values, axis=1)  # 0-6 的类别 (EC 1-7)
    
    # Localization - 使用 one-hot 编码列
    loc_cols = [c for c in df.columns if c.startswith('loc_') and c != 'loc_normalized']
    y_loc = np.argmax(df[loc_cols].values, axis=1)
    
    # Function - 使用 one-hot 编码列
    func_cols = [c for c in df.columns if c.startswith('func_') and c != 'func_normalized']
    y_func = np.argmax(df[func_cols].values, axis=1)
    
    # 随机划分 - 必须与 generate_esm2_features.py 完全一致！
    indices = np.arange(len(df))
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(indices)
    
    n = len(df)
    train_end = int(n * 0.6)    # 60%
    val_end = int(n * 0.8)       # 20%
    
    train_idx = indices[:train_end]   # 0-2999 (3000)
    val_idx = indices[train_end:val_end]  # 3000-3999 (1000)
    test_idx = indices[val_end:]      # 4000-4999 (1000)
    
    # ESM2特征 - 从预生成的特征文件加载
    X_train = np.load(esm2_dir / "train_features.npy")
    X_val = np.load(esm2_dir / "val_features.npy")
    X_test = np.load(esm2_dir / "test_features.npy")
    
    # 验证数据维度一致性
    assert len(X_train) == len(train_idx), f"Train features mismatch: {len(X_train)} vs {len(train_idx)}"
    assert len(X_val) == len(val_idx), f"Val features mismatch: {len(X_val)} vs {len(val_idx)}"
    assert len(X_test) == len(test_idx), f"Test features mismatch: {len(X_test)} vs {len(test_idx)}"
    
    # 标签 - 按划分索引对应
    y_train = {
        'ec': y_ec[train_idx],
        'localization': y_loc[train_idx],
        'function': y_func[train_idx],
    }
    y_val = {
        'ec': y_ec[val_idx],
        'localization': y_loc[val_idx],
        'function': y_func[val_idx],
    }
    y_test = {
        'ec': y_ec[test_idx],
        'localization': y_loc[test_idx],
        'function': y_func[test_idx],
    }
    
    # 标签名称
    class_names = {
        'ec': {i: f"EC{i+1}" for i in range(7)},
        'localization': {i: c.replace('loc_', '') for i, c in enumerate(loc_cols)},
        'function': {i: c.replace('func_', '') for i, c in enumerate(func_cols)},
    }
    
    # 任务维度
    task_dims = {
        'ec': len(ec_cols),
        'localization': len(loc_cols),
        'function': len(func_cols),
    }
    
    return X_train, X_val, X_test, y_train, y_val, y_test, class_names, task_dims


def compute_task_weights(y_train_task, scale_factor=10.0):
    """计算类别权重，处理数据不平衡问题
    
    使用增强的 balanced 策略，对少数类给予更高权重
    scale_factor: 权重放大倍数，越大对少数类越友好
    """
    classes = np.unique(y_train_task)
    weights = compute_class_weight('balanced', classes=classes, y=y_train_task)
    # 进一步放大权重差异
    weights = weights ** 0.5 * scale_factor  # 平方根平滑，再乘以放大因子
    weights = np.clip(weights, 1.0, 100.0)  # 限制最大权重
    return torch.FloatTensor(weights)


class FocalLoss(nn.Module):
    """Focal Loss 用于处理类别不平衡问题
    
    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
    
    - p_t: 模型预测为正确类别的概率
    - gamma: 聚焦参数，越大越关注难分类样本（通常2.0）
    - alpha: 类别权重
    
    核心思想：当模型对某个样本很有把握时（p_t 接近1），loss会很小；
    只有当模型难以分类时（p_t 接近0.5），loss才会比较大。
    """
    
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def train_multitask(X_train, X_val, X_test, y_train, y_val, y_test, task_dims, 
                    epochs=100, batch_size=64, lr=0.001):
    """训练多任务模型
    
    Args:
        X_train, X_val, X_test: 训练集、验证集、测试集特征
        y_train, y_val, y_test: 对应的标签字典
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_t = torch.FloatTensor(X_train_scaled).to(device)
    X_val_t = torch.FloatTensor(X_val_scaled).to(device)
    X_test_t = torch.FloatTensor(X_test_scaled).to(device)
    
    model = MultitaskModel(X_train.shape[1], task_dims).to(device)
    
    # 损失函数 - 使用增强的类别权重 CrossEntropy
    criteria = {}
    for task, num_classes in task_dims.items():
        weights = compute_task_weights(y_train[task], scale_factor=10.0).to(device)
        criteria[task] = nn.CrossEntropyLoss(weight=weights)
    
    # 使用更小的学习率让模型更好收敛
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # 使用 Macro F1 作为早停指标（而不是准确率），更能反映少数类表现
    best_val_f1 = 0
    best_model_state = None
    patience = 20
    no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(X_train_t))
        total_loss = 0
        
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train_t[batch_idx]
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            
            loss = 0
            for task in task_dims:
                y_task = torch.LongTensor(y_train[task][batch_idx.cpu().numpy()]).to(device)
                loss += criteria[task](outputs[task], y_task)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # 验证 - 使用 Macro F1 作为主要指标
        model.eval()
        with torch.no_grad():
            outputs = model(X_val_t)
            
            # 计算每个任务的 Macro F1（对所有类别一视同仁）
            f1_per_task = {}
            for task in task_dims:
                _, preds = torch.max(outputs[task], 1)
                preds_np = preds.cpu().numpy()
                # 使用 zero_division=0 避免某个类不存在时的问题
                f1 = f1_score(y_val[task], preds_np, average='macro', zero_division=0)
                f1_per_task[task] = f1
            
            # 使用 Macro F1 平均值作为早停依据（各任务权重相同）
            val_f1 = np.mean(list(f1_per_task.values()))
            
            # 早停
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
        
        if (epoch + 1) % 20 == 0:
            f1_str = ", ".join([f"{t}={f:.4f}" for t, f in f1_per_task.items()])
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Val Macro-F1: {val_f1:.4f} ({f1_str})")
        
        if no_improve >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_model_state)
    return model, scaler


def main():
    parser = argparse.ArgumentParser(description="训练多任务模型")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_DIR, help="数据路径")
    parser.add_argument("--esm2-dir", type=str, default=DEFAULT_ESM2_DIR, help="ESM2特征目录")
    parser.add_argument("--output", "-o", type=str, default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=64, help="批次大小")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("训练多任务模型")
    print("=" * 60)
    
    # 任务配置 - 从数据中动态获取
    task_dims = {
        'ec': 7,           # EC主类
        'localization': 9,  # 细胞定位
        'function': 7,      # 分子功能
    }
    
    # 加载数据
    print("\n[1/4] 加载数据...")
    X_train, X_val, X_test, y_train, y_val, y_test, class_names, task_dims = load_data(Path(args.data), Path(args.esm2_dir))
    print(f"    训练样本: {len(X_train)}, 验证样本: {len(X_val)}, 测试样本: {len(X_test)}")
    print(f"    任务维度: EC={task_dims['ec']}, 定位={task_dims['localization']}, 功能={task_dims['function']}")
    print(f"    任务: {list(task_dims.keys())}")
    
    # 训练
    print("\n[2/4] 训练模型...")
    start_time = time.time()
    model, scaler = train_multitask(X_train, X_val, X_test, y_train, y_val, y_test, task_dims,
                                   epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    train_time = time.time() - start_time
    print(f"    训练完成! 耗时: {train_time:.2f}s")
    
    # 评估
    print("\n[3/4] 评估模型...")
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_test_scaled = scaler.transform(X_test)
        X_test_t = torch.FloatTensor(X_test_scaled).to(device)
        outputs = model(X_test_t)
    
    results = {}
    print("\n    各任务详细评估结果:")
    for task in task_dims:
        _, preds = torch.max(outputs[task], 1)
        preds = preds.cpu().numpy()
        
        acc = accuracy_score(y_test[task], preds)
        f1_macro = f1_score(y_test[task], preds, average='macro', zero_division=0)
        f1_weighted = f1_score(y_test[task], preds, average='weighted', zero_division=0)
        
        results[task] = {
            'accuracy': acc, 
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted
        }
        print(f"      [{task}] Accuracy={acc:.4f}, F1-macro={f1_macro:.4f}, F1-weighted={f1_weighted:.4f}")
    
    # 总体 Macro F1
    avg_f1_macro = np.mean([results[t]['f1_macro'] for t in task_dims])
    print(f"\n    总体 Macro F1: {avg_f1_macro:.4f}")
    
    # 保存
    print("\n[4/4] 保存模型...")
    checkpoint = {
        "model_state": model.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "input_dim": X_train.shape[1],
        "task_dims": task_dims,
        "hidden_dims": [256, 128],
        "class_names": class_names,  # 保存类别名称
    }
    model_path = output_dir / "multitask_model.pt"
    torch.save(checkpoint, model_path)
    
    config = {
        "algorithm": "MultitaskMLP",
        "encoding": "esm2",
        "input_dim": int(X_train.shape[1]),
        "task_dims": task_dims,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "train_samples": len(y_train['ec']),
        "val_samples": len(y_val['ec']),
        "test_samples": len(y_test['ec']),
        "test_results": results,
        "train_time": float(train_time),
    }
    config_path = output_dir / "multitask_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n模型已保存: {model_path}")
    print(f"配置已保存: {config_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
