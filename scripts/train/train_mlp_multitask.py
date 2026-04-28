#!/usr/bin/env python3
"""
MLP 多任务分类器训练脚本
- 编码: ESM2 (蛋白质语言模型)
- 算法: 多层感知机 (MLP)
- 任务: EC主类 + 细胞定位 + 分子功能
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
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import json

from src.encodings import EncoderRegistry


class MultiTaskMLP(nn.Module):
    """多任务 MLP"""

    def __init__(self, input_dim, ec_classes, loc_classes, func_classes, hidden=[512, 256], dropout=0.3):
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
        self.func_head = nn.Linear(prev, func_classes)

    def forward(self, x):
        h = self.encoder(x)
        return self.ec_head(h), self.loc_head(h), self.func_head(h)


def load_data(data_dir):
    """加载数据集"""
    data_dir = Path(data_dir)

    train_df = pd.read_parquet(data_dir / 'train.parquet')
    val_df = pd.read_parquet(data_dir / 'val.parquet')
    test_df = pd.read_parquet(data_dir / 'test.parquet')

    return train_df, val_df, test_df


def encode_labels(train_df, val_df, test_df):
    """编码标签"""
    ec_enc = LabelEncoder()
    loc_enc = LabelEncoder()
    func_enc = LabelEncoder()

    # 收集所有功能标签（处理 numpy 数组）
    all_funcs = set()
    for fs in train_df['functions']:
        for f in fs:
            all_funcs.add(f)
    all_funcs = sorted(all_funcs)

    ec_enc.fit(train_df['ec_main_class'])
    loc_enc.fit(train_df['location_normalized'])
    func_enc.fit(all_funcs)

    def get_single_label(funcs_list):
        """多标签转单标签（取第一个存在的）"""
        # 处理 numpy 数组
        if hasattr(funcs_list, 'tolist'):
            funcs_list = funcs_list.tolist()
        for f in funcs_list:
            if f in func_enc.classes_:
                return np.where(func_enc.classes_ == f)[0][0]
        return 0

    y_ec_train = ec_enc.transform(train_df['ec_main_class'])
    y_ec_val = ec_enc.transform(val_df['ec_main_class'])
    y_ec_test = ec_enc.transform(test_df['ec_main_class'])

    y_loc_train = loc_enc.transform(train_df['location_normalized'])
    y_loc_val = loc_enc.transform(val_df['location_normalized'])
    y_loc_test = loc_enc.transform(test_df['location_normalized'])

    y_func_train = np.array([get_single_label(fs) for fs in train_df['functions']])
    y_func_val = np.array([get_single_label(fs) for fs in val_df['functions']])
    y_func_test = np.array([get_single_label(fs) for fs in test_df['functions']])

    return {
        'ec': (ec_enc, y_ec_train, y_ec_val, y_ec_test),
        'loc': (loc_enc, y_loc_train, y_loc_val, y_loc_test),
        'func': (func_enc, y_func_train, y_func_val, y_func_test),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='MLP 多任务分类器训练')
    parser.add_argument('--data', type=str, default='data/datasets/balanced_with_go',
                        help='数据集目录')
    parser.add_argument('--encoding', type=str, default='esm2', choices=['onehot', 'ctd', 'esm2'],
                        help='特征编码方式: onehot (20维), ctd (147维), esm2 (1280维)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--hidden', type=str, default='512,256', help='隐藏层结构')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout 比率')
    parser.add_argument('--patience', type=int, default=15, help='早停耐心值')
    parser.add_argument('--device', type=str, default='auto', help='设备 (cuda/cpu)')
    args = parser.parse_args()

    if args.output is None:
        args.output = f'models/mlp_{args.encoding}_multitask'

    # 设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    hidden_dims = [int(x) for x in args.hidden.split(',')]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    train_df, val_df, test_df = load_data(args.data)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 2. 编码标签
    print("\n[2/5] 编码标签...")
    labels = encode_labels(train_df, val_df, test_df)
    ec_enc, y_ec_train, y_ec_val, y_ec_test = labels['ec']
    loc_enc, y_loc_train, y_loc_val, y_loc_test = labels['loc']
    func_enc, y_func_train, y_func_val, y_func_test = labels['func']

    print(f"EC: {len(ec_enc.classes_)} 类, Loc: {len(loc_enc.classes_)} 类, Func: {len(func_enc.classes_)} 类")

    # 3. 获取特征
    print("\n[3/5] 获取特征...")
    print(f"使用编码器: {args.encoding}")
    feat_cache = Path(args.data) / f'{args.encoding}_features.npy'

    if feat_cache.exists():
        print("加载缓存特征...")
        all_features = np.load(feat_cache)

        n_train = len(train_df)
        n_val = len(val_df)
        X_train = all_features[:n_train]
        X_val = all_features[n_train:n_train+n_val]
        X_test = all_features[n_train+n_val:]
    else:
        print(f"实时生成 {args.encoding} 特征...")
        encoder = EncoderRegistry.get(args.encoding)
        X_train = encoder.encode_batch(train_df['sequence'].tolist())
        X_val = encoder.encode_batch(val_df['sequence'].tolist())
        X_test = encoder.encode_batch(test_df['sequence'].tolist())

        np.save(feat_cache, np.vstack([X_train, X_val, X_test]))
        print(f"特征已缓存到: {feat_cache}")

    print(f"特征维度: {X_train.shape[1]}")

    # 4. 训练
    print("\n[4/5] 训练 MLP...")
    model = MultiTaskMLP(
        input_dim=X_train.shape[1],
        ec_classes=len(ec_enc.classes_),
        loc_classes=len(loc_enc.classes_),
        func_classes=len(func_enc.classes_),
        hidden=hidden_dims,
        dropout=args.dropout,
    ).to(device)

    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    X_train_t = torch.FloatTensor(X_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)

    train_ds = TensorDataset(
        X_train_t,
        torch.LongTensor(y_ec_train).to(device),
        torch.LongTensor(y_loc_train).to(device),
        torch.LongTensor(y_func_train).to(device),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_state = None
    patience_cnt = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for X, y_ec, y_loc, y_func in train_loader:
            optimizer.zero_grad()
            ec_out, loc_out, func_out = model(X)
            loss = criterion(ec_out, y_ec) + criterion(loc_out, y_loc) + criterion(func_out, y_func)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            ec_out, loc_out, func_out = model(X_val_t)
            val_loss = (criterion(ec_out, torch.LongTensor(y_ec_val).to(device)) +
                       criterion(loc_out, torch.LongTensor(y_loc_val).to(device)) +
                       criterion(func_out, torch.LongTensor(y_func_val).to(device)))

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

    # 5. 评估
    print("\n[5/5] 评估...")
    model.eval()
    X_test_t = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        ec_out, loc_out, func_out = model(X_test_t)

    ec_acc = (ec_out.argmax(1).cpu() == torch.LongTensor(y_ec_test)).float().mean().item()
    loc_acc = (loc_out.argmax(1).cpu() == torch.LongTensor(y_loc_test)).float().mean().item()
    func_acc = (func_out.argmax(1).cpu() == torch.LongTensor(y_func_test)).float().mean().item()

    print(f"\n测试结果:")
    print(f"  EC 准确率:   {ec_acc:.4f}")
    print(f"  定位准确率:  {loc_acc:.4f}")
    print(f"  功能准确率:  {func_acc:.4f}")

    # 保存
    torch.save({
        'model_state_dict': model.state_dict(),
        'ec_classes': list(ec_enc.classes_),
        'loc_classes': list(loc_enc.classes_),
        'func_classes': list(func_enc.classes_),
        'input_dim': X_train.shape[1],
        'hidden_dims': hidden_dims,
        'dropout': args.dropout,
    }, output_dir / 'model.pt')

    results = {
        'ec_accuracy': ec_acc,
        'loc_accuracy': loc_acc,
        'func_accuracy': func_acc,
    }
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n模型保存到: {output_dir}")


if __name__ == '__main__':
    main()
