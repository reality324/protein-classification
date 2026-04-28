#!/usr/bin/env python3
"""
RandomForest 多任务分类器训练脚本
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import json
import pickle

from src.encodings import EncoderRegistry


def load_data(data_dir):
    data_dir = Path(data_dir)
    return (pd.read_parquet(data_dir / 'train.parquet'),
            pd.read_parquet(data_dir / 'val.parquet'),
            pd.read_parquet(data_dir / 'test.parquet'))


def main():
    import argparse
    parser = argparse.ArgumentParser(description='RandomForest 多任务训练')
    parser.add_argument('--data', type=str, default='data/datasets/balanced_with_go')
    parser.add_argument('--encoding', type=str, default='esm2', choices=['onehot', 'ctd', 'esm2'],
                        help='特征编码方式: onehot (20维), ctd (147维), esm2 (1280维)')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    # 自动设置输出目录
    if args.output is None:
        args.output = f'models/rf_{args.encoding}_multitask'

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] 加载数据...")
    train_df, val_df, test_df = load_data(args.data)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    print("\n[2/4] 编码标签...")
    
    # EC
    ec_classes = sorted(train_df['ec_main_class'].unique())
    ec_map = {c: i for i, c in enumerate(ec_classes)}
    y_ec_train = np.array([ec_map[c] for c in train_df['ec_main_class']])
    y_ec_test = np.array([ec_map[c] for c in test_df['ec_main_class']])

    # Loc
    loc_classes = sorted(train_df['location_normalized'].unique())
    loc_map = {c: i for i, c in enumerate(loc_classes)}
    y_loc_train = np.array([loc_map[c] for c in train_df['location_normalized']])
    y_loc_test = np.array([loc_map[c] for c in test_df['location_normalized']])

    # Func
    first_labels = sorted(set(fs[0] for fs in train_df['functions'] if len(fs) > 0))
    func_map = {c: i for i, c in enumerate(first_labels)}
    def get_first_func(fs):
        if len(fs) > 0:
            return func_map.get(fs[0], 0)
        return 0
    y_func_train = np.array([get_first_func(fs) for fs in train_df['functions']])
    y_func_test = np.array([get_first_func(fs) for fs in test_df['functions']])

    print(f"EC: {len(ec_classes)} 类, Loc: {len(loc_classes)} 类, Func: {len(first_labels)} 类")

    print("\n[3/4] 获取特征...")
    print(f"使用编码器: {args.encoding}")

    # 尝试加载缓存的特征，如果没有缓存则实时生成
    feat_cache = Path(args.data) / f'{args.encoding}_features.npy'

    if feat_cache.exists():
        print("加载缓存特征...")
        all_features = np.load(feat_cache)
        n_train, n_val = len(train_df), len(val_df)
        X_train = all_features[:n_train]
        X_test = all_features[n_train+n_val:]
    else:
        print(f"实时生成 {args.encoding} 特征...")
        encoder = EncoderRegistry.get(args.encoding)
        X_train = encoder.encode_batch(train_df['sequence'].tolist())
        X_val = encoder.encode_batch(val_df['sequence'].tolist())
        X_test = encoder.encode_batch(test_df['sequence'].tolist())

        # 保存缓存
        np.save(feat_cache, np.vstack([X_train, X_val, X_test]))
        print(f"特征已缓存到: {feat_cache}")

    print("\n[4/4] 训练 RandomForest...")

    ec_clf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    ec_clf.fit(X_train, y_ec_train)
    ec_acc = (ec_clf.predict(X_test) == y_ec_test).mean()
    print(f"  EC: {ec_acc:.4f}")

    loc_clf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    loc_clf.fit(X_train, y_loc_train)
    loc_acc = (loc_clf.predict(X_test) == y_loc_test).mean()
    print(f"  Loc: {loc_acc:.4f}")

    func_clf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    func_clf.fit(X_train, y_func_train)
    func_acc = (func_clf.predict(X_test) == y_func_test).mean()
    print(f"  Func: {func_acc:.4f}")

    with open(output_dir / 'model.pkl', 'wb') as f:
        pickle.dump({
            'ec_clf': ec_clf, 'loc_clf': loc_clf, 'func_clf': func_clf,
            'ec_classes': ec_classes,
            'loc_classes': loc_classes,
            'func_classes': first_labels,
        }, f)

    results = {'ec_accuracy': float(ec_acc), 'loc_accuracy': float(loc_acc), 'func_accuracy': float(func_acc)}
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n保存到: {output_dir}")


if __name__ == '__main__':
    main()
