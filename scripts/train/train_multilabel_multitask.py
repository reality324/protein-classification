#!/usr/bin/env python3
"""
多标签功能分类器训练脚本
- 编码: ESM2 (蛋白质语言模型)
- 任务: 多标签功能预测 (一个蛋白质可以有多个功能)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MultiLabelBinarizer
from pathlib import Path
import json

from src.encodings.esm2 import ESM2Encoder


class MultiTaskMultiLabelMLP(nn.Module):
    """多标签 MLP - 每个任务输出多个标签"""

    def __init__(self, input_dim, ec_classes, loc_classes, func_count, hidden=[512, 256], dropout=0.3):
        super().__init__()

        layers = []
        prev = input_dim
        for h in hidden:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h

        self.encoder = nn.Sequential(*layers)

        self.ec_head = nn.Linear(prev, ec_classes)
        self.loc_head = nn.Linear(prev, loc_classes)
        self.func_head = nn.Linear(prev, func_count)

    def forward(self, x):
        h = self.encoder(x)
        return self.ec_head(h), self.loc_head(h), self.func_head(h)


def load_data(data_dir):
    data_dir = Path(data_dir)
    train_df = pd.read_parquet(data_dir / 'train.parquet')
    val_df = pd.read_parquet(data_dir / 'val.parquet')
    test_df = pd.read_parquet(data_dir / 'test.parquet')
    return train_df, val_df, test_df


def prepare_multilabel(train_df, val_df, test_df):
    """准备多标签数据"""
    # EC - 仍然是单标签
    ec_classes = sorted(train_df['ec_main_class'].unique())
    ec_map = {c: i for i, c in enumerate(ec_classes)}

    # Loc - 仍然是单标签
    loc_classes = sorted(train_df['location_normalized'].unique())
    loc_map = {c: i for i, c in enumerate(loc_classes)}

    # Func - 多标签
    mlb = MultiLabelBinarizer()
    train_funcs = [list(fs) for fs in train_df['functions']]
    mlb.fit(train_funcs)

    # 处理测试集中可能出现的未知标签
    all_funcs = set()
    for fs in train_df['functions']:
        all_funcs.update(fs)
    for fs in val_df['functions']:
        all_funcs.update(fs)
    for fs in test_df['functions']:
        all_funcs.update(fs)

    # 重新 fit 确保覆盖所有标签
    mlb = MultiLabelBinarizer(classes=sorted(all_funcs))
    mlb.fit(train_funcs)

    # 标签
    y_ec_train = np.array([ec_map[c] for c in train_df['ec_main_class']])
    y_ec_val = np.array([ec_map[c] for c in val_df['ec_main_class']])
    y_ec_test = np.array([ec_map[c] for c in test_df['ec_main_class']])

    y_loc_train = np.array([loc_map[c] for c in train_df['location_normalized']])
    y_loc_val = np.array([loc_map[c] for c in val_df['location_normalized']])
    y_loc_test = np.array([loc_map[c] for c in test_df['location_normalized']])

    y_func_train = mlb.transform(train_funcs)
    y_func_val = mlb.transform([list(fs) for fs in val_df['functions']])
    y_func_test = mlb.transform([list(fs) for fs in test_df['functions']])

    return {
        'ec': (ec_classes, ec_map),
        'loc': (loc_classes, loc_map),
        'func': (mlb,),
    }, {
        'ec': (y_ec_train, y_ec_val, y_ec_test),
        'loc': (y_loc_train, y_loc_val, y_loc_test),
        'func': (y_func_train, y_func_val, y_func_test),
    }


def compute_multilabel_metrics(y_true, y_pred, threshold=0.5):
    """计算多标签评估指标"""
    y_pred_binary = (y_pred >= threshold).astype(int)

    # 精确匹配率 (Exact Match Ratio)
    exact_match = np.mean(np.all(y_pred_binary == y_true, axis=1))

    # Hamming Loss
    hamming_loss = np.mean(y_pred_binary != y_true)

    # 子集准确率
    subset_correct = 0
    for p, t in zip(y_pred_binary, y_true):
        pred_pos = set(np.where(p == 1)[0])
        true_pos = set(np.where(t == 1)[0])
        if pred_pos == true_pos:
            subset_correct += 1
    subset_acc = subset_correct / len(y_true) if len(y_true) > 0 else 0

    # 每个标签的精确率和召回率
    n_classes = y_true.shape[1]
    precisions, recalls = [], []

    for i in range(n_classes):
        tp = ((y_pred_binary[:, i] == 1) & (y_true[:, i] == 1)).sum()
        fp = ((y_pred_binary[:, i] == 1) & (y_true[:, i] == 0)).sum()
        fn = ((y_pred_binary[:, i] == 0) & (y_true[:, i] == 1)).sum()

        if tp + fp > 0:
            precisions.append(tp / (tp + fp))
        else:
            precisions.append(0)

        if tp + fn > 0:
            recalls.append(tp / (tp + fn))
        else:
            recalls.append(0)

    return {
        'exact_match_ratio': float(exact_match),
        'hamming_loss': float(hamming_loss),
        'subset_accuracy': float(subset_acc),
        'mean_precision': float(np.mean(precisions)),
        'mean_recall': float(np.mean(recalls)),
        'mean_f1': float(2 * np.mean(precisions) * np.mean(recalls) /
                        (np.mean(precisions) + np.mean(recalls) + 1e-8)),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='多标签功能分类器训练')
    parser.add_argument('--data', type=str, default='data/datasets/balanced_with_go')
    parser.add_argument('--output', type=str, default='models/multilabel_esm2_multitask')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=15)
    args = parser.parse_args()

    device = torch.device('cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/5] 加载数据...")
    train_df, val_df, test_df = load_data(args.data)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    print("\n[2/5] 准备标签...")
    label_info, labels = prepare_multilabel(train_df, val_df, test_df)

    ec_classes = label_info['ec'][0]
    loc_classes = label_info['loc'][0]
    mlb = label_info['func'][0]
    func_classes = list(mlb.classes_)

    y_ec_train, y_ec_val, y_ec_test = labels['ec']
    y_loc_train, y_loc_val, y_loc_test = labels['loc']
    y_func_train, y_func_val, y_func_test = labels['func']

    print(f"EC: {len(ec_classes)} 类, Loc: {len(loc_classes)} 类, Func: {len(func_classes)} 标签")

    print("\n[3/5] 获取特征...")
    feat_cache = Path(args.data) / 'esm2_features.npy'
    if feat_cache.exists():
        print("加载缓存...")
        all_features = np.load(feat_cache)
        n_train, n_val = len(train_df), len(val_df)
        X_train = all_features[:n_train]
        X_val = all_features[n_train:n_train+n_val]
        X_test = all_features[n_train+n_val:]
    else:
        encoder = ESM2Encoder(pooling='mean', device=device)
        X_train = encoder.encode_batch(train_df['sequence'].tolist())
        X_val = encoder.encode_batch(val_df['sequence'].tolist())
        X_test = encoder.encode_batch(test_df['sequence'].tolist())

    print(f"特征维度: {X_train.shape[1]}")

    print("\n[4/5] 训练...")

    model = MultiTaskMultiLabelMLP(
        input_dim=X_train.shape[1],
        ec_classes=len(ec_classes),
        loc_classes=len(loc_classes),
        func_count=len(func_classes),
        hidden=[512, 256],
        dropout=0.3,
    ).to(device)

    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    X_train_t = torch.FloatTensor(X_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)

    train_ds = TensorDataset(
        X_train_t,
        torch.LongTensor(y_ec_train).to(device),
        torch.LongTensor(y_loc_train).to(device),
        torch.FloatTensor(y_func_train).to(device),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    bce_criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    best_state = None
    patience_cnt = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for X, y_ec, y_loc, y_func in train_loader:
            optimizer.zero_grad()
            ec_out, loc_out, func_out = model(X)

            loss = (criterion(ec_out, y_ec) +
                   criterion(loc_out, y_loc) +
                   bce_criterion(func_out, y_func))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            ec_out, loc_out, func_out = model(X_val_t)
            val_loss = (nn.CrossEntropyLoss()(ec_out, torch.LongTensor(y_ec_val).to(device)) +
                       nn.CrossEntropyLoss()(loc_out, torch.LongTensor(y_loc_val).to(device)) +
                       bce_criterion(func_out, torch.FloatTensor(y_func_val).to(device)))

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss={train_loss/len(train_loader):.4f}, val={val_loss.item():.4f}")

        if patience_cnt >= args.patience:
            print(f"早停于 epoch {epoch+1}")
            break

    model.load_state_dict(best_state)

    print("\n[5/5] 评估...")

    model.eval()
    X_test_t = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        ec_out, loc_out, func_out = model(X_test_t)
        ec_pred = ec_out.argmax(1).cpu().numpy()
        loc_pred = loc_out.argmax(1).cpu().numpy()
        func_pred_prob = torch.sigmoid(func_out).cpu().numpy()
        func_pred = (func_pred_prob >= args.threshold).astype(int)

    # EC 和 Loc 指标
    ec_acc = (ec_pred == y_ec_test).mean()
    loc_acc = (loc_pred == y_loc_test).mean()

    # Func 多标签指标
    func_metrics = compute_multilabel_metrics(y_func_test, func_pred_prob, args.threshold)

    print(f"\n测试结果:")
    print(f"  EC 准确率:   {ec_acc:.4f}")
    print(f"  定位准确率:  {loc_acc:.4f}")
    print(f"  功能多标签:")
    print(f"    - Exact Match Ratio: {func_metrics['exact_match_ratio']:.4f}")
    print(f"    - Hamming Loss:      {func_metrics['hamming_loss']:.4f}")
    print(f"    - Mean F1:           {func_metrics['mean_f1']:.4f}")

    # 保存
    torch.save({
        'model_state_dict': model.state_dict(),
        'ec_classes': list(ec_classes),
        'loc_classes': list(loc_classes),
        'func_classes': list(func_classes),
        'input_dim': X_train.shape[1],
        'threshold': args.threshold,
    }, output_dir / 'model.pt')

    results = {
        'ec_accuracy': float(ec_acc),
        'loc_accuracy': float(loc_acc),
        'func_multilabel': func_metrics,
    }
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n模型保存到: {output_dir}")


if __name__ == '__main__':
    main()
