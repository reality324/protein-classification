#!/usr/bin/env python3
"""
多指标分类器训练脚本

功能：使用同一数据集训练多个算法的多指标分类器
- EC主类预测
- 细胞定位预测
- 分子功能预测
"""
import sys
import argparse
import pickle
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def load_data(data_dir, esm2_dir):
    """加载数据"""
    df = pd.read_parquet(data_dir)
    print(f"    数据集大小: {len(df)} 条")
    
    # 加载ESM2特征
    esm2_train = np.load(esm2_dir / "train_features.npy")
    esm2_test = np.load(esm2_dir / "test_features.npy")
    esm2_full = np.vstack([esm2_train, esm2_test])
    
    n = len(esm2_full)  # 使用ESM2特征的数量作为基准
    print(f"    ESM2特征: {n} 条")
    
    # 截取对应的数据集
    df = df.iloc[:n]
    
    # 随机划分
    indices = np.arange(n)
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(indices)
    
    train_end = int(n * 0.8)
    train_idx = indices[:train_end]
    test_idx = indices[train_end:]
    
    # 提取标签
    ec_cols = [c for c in df.columns if c.startswith('ec_')]
    y_ec_all = np.argmax(df[ec_cols].values, axis=1)
    
    # EC主类
    def get_main_class(idx_list):
        return np.array([
            int(ec_cols[y_ec_all[i]].split('_')[1].split('.')[0]) - 1
            for i in idx_list
        ])
    
    # Localization
    loc_cols = [c for c in df.columns if c.startswith('loc_')]
    y_loc_all = np.argmax(df[loc_cols].values, axis=1)
    
    # Function
    func_cols = [c for c in df.columns if c.startswith('func_')]
    y_func_all = np.argmax(df[func_cols].values, axis=1)
    
    # 标签名称
    class_names = {
        'ec': {i: f"EC{i+1}" for i in range(7)},
        'localization': {i: c.replace('loc_', '') for i, c in enumerate(loc_cols)},
        'function': {i: c.replace('func_', '') for i, c in enumerate(func_cols)},
    }
    
    X_train = esm2_full[train_idx]
    X_test = esm2_full[test_idx]
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': {
            'ec': get_main_class(train_idx),
            'localization': y_loc_all[train_idx],
            'function': y_func_all[train_idx],
        },
        'y_test': {
            'ec': get_main_class(test_idx),
            'localization': y_loc_all[test_idx],
            'function': y_func_all[test_idx],
        },
        'class_names': class_names,
        'task_dims': {
            'ec': 7,
            'localization': len(loc_cols),
            'function': len(func_cols),
        }
    }


class MLP(nn.Module):
    """MLP分类器"""
    def __init__(self, input_dim, num_classes, hidden_dims=[256, 128], dropout=0.3):
        super().__init__()
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
        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


def train_models(data, output_dir, epochs=100, batch_size=64):
    """训练所有模型"""
    results = {}
    
    for task_name in ['ec', 'localization', 'function']:
        print(f"\n{'='*50}")
        print(f"训练任务: {task_name}")
        print('='*50)
        
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train'][task_name]
        y_test = data['y_test'][task_name]
        num_classes = data['task_dims'][task_name]
        
        task_results = {}
        
        # 1. RandomForest
        print(f"\n  [1/2] RandomForest...")
        rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=RANDOM_SEED, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_pred)
        rf_f1 = f1_score(y_test, rf_pred, average='macro')
        print(f"        Acc: {rf_acc:.4f}, F1: {rf_f1:.4f}")
        task_results['rf'] = {'accuracy': rf_acc, 'f1': rf_f1, 'model': rf}
        
        # 2. MLP
        print(f"  [2/2] MLP...")
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mlp = MLP(X_train.shape[1], num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(mlp.parameters(), lr=0.001)
        
        X_train_t = torch.FloatTensor(X_train_sc).to(device)
        y_train_t = torch.LongTensor(y_train).to(device)
        X_test_t = torch.FloatTensor(X_test_sc).to(device)
        
        mlp.train()
        indices = torch.randperm(len(X_train_t))
        for epoch in range(epochs):
            for i in range(0, len(indices), batch_size):
                batch_idx = indices[i:i+batch_size]
                optimizer.zero_grad()
                outputs = mlp(X_train_t[batch_idx])
                loss = criterion(outputs, y_train_t[batch_idx])
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 25 == 0:
                print(f"        Epoch {epoch+1}/{epochs}")
        
        mlp.eval()
        with torch.no_grad():
            mlp_pred = mlp(X_test_t).argmax(dim=1).cpu().numpy()
        mlp_acc = accuracy_score(y_test, mlp_pred)
        mlp_f1 = f1_score(y_test, mlp_pred, average='macro')
        print(f"        Acc: {mlp_acc:.4f}, F1: {mlp_f1:.4f}")
        task_results['mlp'] = {
            'accuracy': mlp_acc, 'f1': mlp_f1,
            'model': mlp.state_dict(),
            'scaler': scaler
        }
        
        results[task_name] = task_results
    
    return results


def save_models(results, data, output_dir):
    """保存所有模型"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存每个任务的模型
    for task_name in ['ec', 'localization', 'function']:
        task_dir = output_dir / task_name
        task_dir.mkdir(exist_ok=True)
        
        # RF
        with open(task_dir / "rf_model.pkl", 'wb') as f:
            pickle.dump(results[task_name]['rf']['model'], f)
        
        # MLP
        torch.save({
            'model_state': results[task_name]['mlp']['model'],
            'scaler_mean': results[task_name]['mlp']['scaler'].mean_,
            'scaler_scale': results[task_name]['mlp']['scaler'].scale_,
            'input_dim': data['X_train'].shape[1],
            'num_classes': data['task_dims'][task_name],
        }, task_dir / "mlp_model.pt")
        
        # 配置文件
        config = {
            'task': task_name,
            'class_names': data['class_names'][task_name],
            'task_dims': data['task_dims'],
            'results': {
                'rf': {'accuracy': results[task_name]['rf']['accuracy'], 'f1': results[task_name]['rf']['f1']},
                'mlp': {'accuracy': results[task_name]['mlp']['accuracy'], 'f1': results[task_name]['mlp']['f1']},
            }
        }
        with open(task_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"\n  {task_name} 模型已保存到 {task_dir}/")
    
    # 保存汇总结果
    summary = {
        'class_names': data['class_names'],
        'task_dims': data['task_dims'],
        'results': {}
    }
    for task_name in ['ec', 'localization', 'function']:
        summary['results'][task_name] = {
            'rf': results[task_name]['rf'],
            'mlp': results[task_name]['mlp'],
        }
        del summary['results'][task_name]['rf']['model']
        del summary['results'][task_name]['mlp']['model']
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="训练多指标分类器")
    parser.add_argument("--data", type=str, default="data/datasets/train_subset.parquet")
    parser.add_argument("--esm2-dir", type=str, default="data/processed/esm2_aligned")
    parser.add_argument("--output", "-o", type=str, default="models/multitask")
    parser.add_argument("--epochs", type=int, default=100, help="MLP训练轮数")
    parser.add_argument("--batch-size", type=int, default=64, help="批次大小")
    args = parser.parse_args()
    
    print("=" * 60)
    print("多指标分类器训练")
    print("  - EC主类预测")
    print("  - 细胞定位预测")
    print("  - 分子功能预测")
    print("=" * 60)
    
    # 加载数据
    print("\n[1/3] 加载数据...")
    data = load_data(Path(args.data), Path(args.esm2_dir))
    print(f"    训练样本: {len(data['X_train'])}, 测试样本: {len(data['X_test'])}")
    print(f"    特征维度: {data['X_train'].shape[1]}")
    
    # 训练模型
    print("\n[2/3] 训练模型...")
    start_time = time.time()
    results = train_models(data, args.output, epochs=args.epochs, batch_size=args.batch_size)
    train_time = time.time() - start_time
    print(f"\n    训练完成! 耗时: {train_time:.2f}s")
    
    # 保存模型
    print("\n[3/3] 保存模型...")
    save_models(results, data, args.output)
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("训练结果汇总")
    print("=" * 60)
    print(f"{'任务':<15} {'RF Acc':<10} {'MLP Acc':<10}")
    print("-" * 35)
    for task in ['ec', 'localization', 'function']:
        rf_acc = results[task]['rf']['accuracy']
        mlp_acc = results[task]['mlp']['accuracy']
        print(f"{task:<15} {rf_acc:.4f}     {mlp_acc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
