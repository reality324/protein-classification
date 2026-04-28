#!/usr/bin/env python3
"""
XGBoost 多任务分类器训练脚本
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from pathlib import Path
import json
import pickle

from src.encodings.esm2 import ESM2Encoder


def load_data(data_dir):
    data_dir = Path(data_dir)
    train_df = pd.read_parquet(data_dir / 'train.parquet')
    val_df = pd.read_parquet(data_dir / 'val.parquet')
    test_df = pd.read_parquet(data_dir / 'test.parquet')
    return train_df, val_df, test_df


def main():
    import argparse
    parser = argparse.ArgumentParser(description='XGBoost 多任务训练')
    parser.add_argument('--data', type=str, default='data/datasets/balanced_with_go')
    parser.add_argument('--output', type=str, default='models/xgb_esm2_multitask')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] 加载数据...")
    train_df, val_df, test_df = load_data(args.data)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    print("\n[2/4] 编码标签...")
    
    # EC - 用连续索引
    ec_classes = sorted(train_df['ec_main_class'].unique())
    ec_map = {c: i for i, c in enumerate(ec_classes)}
    y_ec_train = np.array([ec_map[c] for c in train_df['ec_main_class']])
    y_ec_test = np.array([ec_map[c] for c in test_df['ec_main_class']])
    print(f"EC: {len(ec_classes)} 类")

    # Loc - 用连续索引
    loc_classes = sorted(train_df['location_normalized'].unique())
    loc_map = {c: i for i, c in enumerate(loc_classes)}
    y_loc_train = np.array([loc_map[c] for c in train_df['location_normalized']])
    y_loc_test = np.array([loc_map[c] for c in test_df['location_normalized']])
    print(f"Loc: {len(loc_classes)} 类")

    # Func - 只用第一标签建立映射
    first_labels = sorted(set(fs[0] for fs in train_df['functions'] if len(fs) > 0))
    func_map = {c: i for i, c in enumerate(first_labels)}
    
    def get_first_func_idx(fs):
        for f in fs:
            if f in func_map:
                return func_map[f]
        return 0
    
    y_func_train = np.array([get_first_func_idx(fs) for fs in train_df['functions']])
    y_func_test = np.array([get_first_func_idx(fs) for fs in test_df['functions']])
    print(f"Func: {len(first_labels)} 类")

    print("\n[3/4] 获取特征...")
    feat_cache = Path(args.data) / 'esm2_features.npy'
    if feat_cache.exists():
        print("加载缓存...")
        all_features = np.load(feat_cache)
        n_train, n_val = len(train_df), len(val_df)
        X_train = all_features[:n_train]
        X_test = all_features[n_train+n_val:]
    else:
        encoder = ESM2Encoder(pooling='mean', device='cpu')
        X_train = encoder.encode_batch(train_df['sequence'].tolist())
        X_test = encoder.encode_batch(test_df['sequence'].tolist())

    print(f"特征维度: {X_train.shape[1]}")

    print("\n[4/4] 训练 XGBoost...")

    ec_clf = XGBClassifier(n_estimators=50, max_depth=6, learning_rate=0.1,
                           objective='multi:softprob',
                           tree_method='hist', random_state=42, n_jobs=2,
                           eval_metric='mlogloss')
    ec_clf.fit(X_train, y_ec_train)
    ec_acc = (ec_clf.predict(X_test) == y_ec_test).mean()
    print(f"  EC: {ec_acc:.4f}")

    loc_clf = XGBClassifier(n_estimators=50, max_depth=6, learning_rate=0.1,
                             objective='multi:softprob',
                             tree_method='hist', random_state=42, n_jobs=2,
                             eval_metric='mlogloss')
    loc_clf.fit(X_train, y_loc_train)
    loc_acc = (loc_clf.predict(X_test) == y_loc_test).mean()
    print(f"  Loc: {loc_acc:.4f}")

    func_clf = XGBClassifier(n_estimators=50, max_depth=6, learning_rate=0.1,
                              objective='multi:softprob',
                              tree_method='hist', random_state=42, n_jobs=2,
                              eval_metric='mlogloss')
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
