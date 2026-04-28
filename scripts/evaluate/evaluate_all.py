#!/usr/bin/env python3
"""
完整评估脚本 - 计算所有模型的详细指标
包括: Accuracy, Precision, Recall, F1, Confusion Matrix, Classification Report
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import torch
import pickle
import json
from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, classification_report, confusion_matrix)

from src.encodings.esm2 import ESM2Encoder


def load_test_data(data_dir):
    """加载测试数据"""
    data_dir = Path(data_dir)
    test_df = pd.read_parquet(data_dir / 'test.parquet')
    return test_df


def get_predictions_bnn(model_path, test_df, device='cpu'):
    """BNN 模型预测"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    from scripts.train.train_bnn_multitask import MultiTaskBNN

    model = MultiTaskBNN(
        input_dim=checkpoint['input_dim'],
        ec_classes=len(checkpoint['ec_classes']),
        loc_classes=len(checkpoint['loc_classes']),
        func_classes=len(checkpoint['func_classes']),
        hidden=checkpoint.get('hidden_dims', [512, 256]),
        dropout=checkpoint.get('dropout', 0.3),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    encoder = ESM2Encoder(pooling='mean', device=device)
    features = encoder.encode_batch(test_df['sequence'].tolist())

    with torch.no_grad():
        X = torch.FloatTensor(features).to(device)
        ec_out, loc_out, func_out = model(X)
        ec_pred = ec_out.argmax(1).cpu().numpy()
        loc_pred = loc_out.argmax(1).cpu().numpy()
        func_pred = func_out.argmax(1).cpu().numpy()

    return ec_pred, loc_pred, func_pred, checkpoint


def get_predictions_mlp(model_path, test_df, device='cpu'):
    """MLP 模型预测"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    from scripts.train.train_mlp_multitask import MultiTaskMLP

    model = MultiTaskMLP(
        input_dim=checkpoint['input_dim'],
        ec_classes=len(checkpoint['ec_classes']),
        loc_classes=len(checkpoint['loc_classes']),
        func_classes=len(checkpoint['func_classes']),
        hidden=checkpoint.get('hidden_dims', [512, 256]),
        dropout=checkpoint.get('dropout', 0.3),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    encoder = ESM2Encoder(pooling='mean', device=device)
    features = encoder.encode_batch(test_df['sequence'].tolist())

    with torch.no_grad():
        X = torch.FloatTensor(features).to(device)
        ec_out, loc_out, func_out = model(X)
        ec_pred = ec_out.argmax(1).cpu().numpy()
        loc_pred = loc_out.argmax(1).cpu().numpy()
        func_pred = func_out.argmax(1).cpu().numpy()

    return ec_pred, loc_pred, func_pred, checkpoint


def get_predictions_rf(model_path, test_df):
    """RandomForest 模型预测"""
    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    encoder = ESM2Encoder(pooling='mean', device='cpu')
    features = encoder.encode_batch(test_df['sequence'].tolist())

    ec_pred = data['ec_clf'].predict(features)
    loc_pred = data['loc_clf'].predict(features)
    func_pred = data['func_clf'].predict(features)

    return ec_pred, loc_pred, func_pred, data


def get_predictions_xgb(model_path, test_df):
    """XGBoost 模型预测"""
    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    encoder = ESM2Encoder(pooling='mean', device='cpu')
    features = encoder.encode_batch(test_df['sequence'].tolist())

    ec_pred = data['ec_clf'].predict(features)
    loc_pred = data['loc_clf'].predict(features)
    func_pred = data['func_clf'].predict(features)

    return ec_pred, loc_pred, func_pred, data


def encode_labels(test_df, model_data):
    """编码真实标签"""
    # EC
    ec_classes = model_data['ec_classes']
    ec_map = {c: i for i, c in enumerate(ec_classes)}
    y_ec = np.array([ec_map[c] for c in test_df['ec_main_class']])

    # Loc
    loc_classes = model_data['loc_classes']
    loc_map = {c: i for i, c in enumerate(loc_classes)}
    y_loc = np.array([loc_map[c] for c in test_df['location_normalized']])

    # Func
    func_classes = model_data['func_classes']
    func_map = {c: i for i, c in enumerate(func_classes)}

    def get_first_func(fs):
        if len(fs) > 0:
            return func_map.get(fs[0], 0)
        return 0

    y_func = np.array([get_first_func(fs) for fs in test_df['functions']])

    return y_ec, y_loc, y_func


def compute_metrics(y_true, y_pred, classes, task_name):
    """计算完整指标"""
    acc = accuracy_score(y_true, y_pred)

    # 加权平均 (处理类别不平衡)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Macro 平均
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 使用实际出现的类别名
    unique_labels = sorted(set(y_true) | set(y_pred))
    actual_classes = [classes[i] for i in unique_labels if i < len(classes)]

    # 详细报告
    report = classification_report(y_true, y_pred, labels=unique_labels,
                                  target_names=actual_classes,
                                  zero_division=0, output_dict=True)

    return {
        'accuracy': acc,
        'precision_weighted': prec,
        'recall_weighted': rec,
        'f1_weighted': f1,
        'precision_macro': prec_macro,
        'recall_macro': rec_macro,
        'f1_macro': f1_macro,
        'confusion_matrix': cm.tolist(),
        'per_class': report
    }


def evaluate_model(model_name, model_path, test_df, get_pred_fn, output_dir):
    """评估单个模型"""
    print(f"\n{'='*60}")
    print(f"评估模型: {model_name}")
    print('='*60)

    # 预测
    ec_pred, loc_pred, func_pred, model_data = get_pred_fn(model_path, test_df)

    # 真实标签
    y_ec, y_loc, y_func = encode_labels(test_df, model_data)

    # 计算指标
    results = {}

    print("\n[1] EC 主类预测")
    ec_metrics = compute_metrics(y_ec, ec_pred, model_data['ec_classes'], 'EC')
    results['ec'] = ec_metrics
    print(f"    Accuracy:  {ec_metrics['accuracy']:.4f}")
    print(f"    F1 (weighted): {ec_metrics['f1_weighted']:.4f}")
    print(f"    F1 (macro):    {ec_metrics['f1_macro']:.4f}")

    print("\n[2] 细胞定位预测")
    loc_metrics = compute_metrics(y_loc, loc_pred, model_data['loc_classes'], 'Loc')
    results['loc'] = loc_metrics
    print(f"    Accuracy:  {loc_metrics['accuracy']:.4f}")
    print(f"    F1 (weighted): {loc_metrics['f1_weighted']:.4f}")
    print(f"    F1 (macro):    {loc_metrics['f1_macro']:.4f}")

    print("\n[3] 分子功能预测")
    func_metrics = compute_metrics(y_func, func_pred, model_data['func_classes'], 'Func')
    results['func'] = func_metrics
    print(f"    Accuracy:  {func_metrics['accuracy']:.4f}")
    print(f"    F1 (weighted): {func_metrics['f1_weighted']:.4f}")
    print(f"    F1 (macro):    {func_metrics['f1_macro']:.4f}")

    # 保存结果
    output_path = output_dir / f"{model_name}_evaluation.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存: {output_path}")

    return results, model_data


def main():
    import argparse
    parser = argparse.ArgumentParser(description='完整模型评估')
    parser.add_argument('--data', type=str, default='data/datasets/balanced_with_go',
                        help='数据集目录')
    parser.add_argument('--output', type=str, default='results/evaluation',
                        help='输出目录')
    parser.add_argument('--models', type=str, default='all',
                        help='模型列表: all, bnn, mlp, rf, xgb')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载测试数据
    print("加载测试数据...")
    test_df = load_test_data(args.data)
    print(f"测试集: {len(test_df)} 样本")

    model_configs = {
        'bnn': ('models/bnn_esm2_multitask/model.pt', get_predictions_bnn),
        'mlp': ('models/mlp_esm2_multitask/model.pt', get_predictions_mlp),
        'rf': ('models/rf_esm2_multitask/model.pkl', get_predictions_rf),
        'xgb': ('models/xgb_esm2_multitask/model.pkl', get_predictions_xgb),
    }

    if args.models == 'all':
        models_to_eval = list(model_configs.keys())
    else:
        models_to_eval = args.models.split(',')

    all_results = {}
    model_data_cache = {}

    for model_name in models_to_eval:
        if model_name not in model_configs:
            print(f"未知模型: {model_name}")
            continue

        model_path, pred_fn = model_configs[model_name]
        if not Path(model_path).exists():
            print(f"模型不存在: {model_path}")
            continue

        results, model_data = evaluate_model(model_name, model_path, test_df, pred_fn, output_dir)
        all_results[model_name] = results
        model_data_cache[model_name] = model_data

    # 汇总对比
    print("\n" + "="*60)
    print("模型对比汇总")
    print("="*60)

    summary_data = []
    for model_name in all_results:
        r = all_results[model_name]
        summary_data.append({
            'Model': model_name.upper(),
            'EC Acc': f"{r['ec']['accuracy']:.4f}",
            'EC F1': f"{r['ec']['f1_weighted']:.4f}",
            'Loc Acc': f"{r['loc']['accuracy']:.4f}",
            'Loc F1': f"{r['loc']['f1_weighted']:.4f}",
            'Func Acc': f"{r['func']['accuracy']:.4f}",
            'Func F1': f"{r['func']['f1_weighted']:.4f}",
            'Avg F1': f"{(r['ec']['f1_weighted'] + r['loc']['f1_weighted'] + r['func']['f1_weighted'])/3:.4f}",
        })

    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))

    # 保存汇总
    summary_path = output_dir / 'summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\n汇总已保存: {summary_path}")

    # 保存完整结果
    with open(output_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
