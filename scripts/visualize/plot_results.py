#!/usr/bin/env python3
"""
可视化脚本 - 生成模型对比图和混淆矩阵热力图
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_results(eval_dir='results/evaluation'):
    """加载评估结果"""
    eval_dir = Path(eval_dir)

    results = {}
    for model_name in ['bnn', 'mlp', 'rf', 'xgb']:
        result_file = eval_dir / f'{model_name}_evaluation.json'
        if result_file.exists():
            with open(result_file) as f:
                results[model_name] = json.load(f)

    return results


def plot_model_comparison(results, output_dir):
    """绘制模型对比图"""
    models = list(results.keys())
    if not models:
        print("没有结果可绘图")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    tasks = ['ec', 'loc', 'func']
    task_names = ['EC Classification', 'Localization', 'Molecular Function']

    for idx, (task, title) in enumerate(zip(tasks, task_names)):
        ax = axes[idx]

        metrics = ['accuracy', 'f1_weighted', 'f1_macro']
        metric_names = ['Accuracy', 'F1 (Weighted)', 'F1 (Macro)']

        x = np.arange(len(metrics))
        width = 0.2

        for i, model in enumerate(models):
            if task in results[model]:
                values = [results[model][task].get(m, 0) for m in metrics]
                bars = ax.bar(x + i * width, values, width, label=model.upper())

        ax.set_xlabel('Metric')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(metric_names)
        ax.set_ylim(0, 1.0)
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'model_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"模型对比图已保存: {output_path}")
    plt.close()


def plot_confusion_matrix(results, model_name, task, classes, output_dir):
    """绘制单个任务的混淆矩阵"""
    if task not in results[model_name]:
        return

    cm = np.array(results[model_name][task]['confusion_matrix'])

    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 0.8), max(6, len(classes) * 0.6)))

    # 归一化
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    sns.heatmap(cm_norm, annot=False, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax,
                vmin=0, vmax=1)

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'{model_name.upper()} - {task.upper()} Confusion Matrix')

    plt.tight_layout()
    output_path = output_dir / f'{model_name}_{task}_confusion_matrix.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"混淆矩阵已保存: {output_path}")
    plt.close()


def plot_all_confusion_matrices(results, output_dir):
    """绘制所有模型的混淆矩阵"""
    tasks_info = {
        'ec': 'EC Classification',
        'loc': 'Cellular Localization',
        'func': 'Molecular Function'
    }

    for model_name in results:
        for task in tasks_info:
            if task in results[model_name]:
                cm = np.array(results[model_name][task]['confusion_matrix'])
                classes = [f'C{i}' for i in range(cm.shape[0])]
                plot_confusion_matrix(results, model_name, task, classes, output_dir)


def plot_overall_comparison(results, output_dir):
    """绘制总体对比"""
    models = list(results.keys())

    # 计算平均 F1
    avg_f1 = []
    for model in models:
        f1s = []
        for task in ['ec', 'loc', 'func']:
            if task in results[model]:
                f1s.append(results[model][task]['f1_weighted'])
        avg_f1.append(np.mean(f1s) if f1s else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：Average F1
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    bars = ax1.bar([m.upper() for m in models], avg_f1, color=colors)
    ax1.set_ylabel('Average F1 (Weighted)')
    ax1.set_title('Overall Model Comparison')
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, avg_f1):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    # 右图：各任务 F1 热力图
    tasks = ['ec', 'loc', 'func']
    task_names = ['EC', 'Localization', 'Function']
    f1_matrix = []

    for model in models:
        row = []
        for task in tasks:
            if task in results[model]:
                row.append(results[model][task]['f1_weighted'])
            else:
                row.append(0)
        f1_matrix.append(row)

    f1_matrix = np.array(f1_matrix)

    im = ax2.imshow(f1_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks(range(len(tasks)))
    ax2.set_xticklabels(task_names)
    ax2.set_yticks(range(len(models)))
    ax2.set_yticklabels([m.upper() for m in models])
    ax2.set_title('F1 Scores by Task')

    # 添加数值
    for i in range(len(models)):
        for j in range(len(tasks)):
            text = ax2.text(j, i, f'{f1_matrix[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=10)

    plt.colorbar(im, ax=ax2, label='F1 Score')
    plt.tight_layout()

    output_path = output_dir / 'overall_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"总体对比图已保存: {output_path}")
    plt.close()


def plot_per_class_f1(results, model_name, task, output_dir):
    """绘制各类别 F1 分数"""
    if task not in results[model_name]:
        return

    per_class = results[model_name][task]['per_class']

    # 过滤非类别条目
    classes = [k for k in per_class.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]

    if not classes:
        return

    f1_scores = [per_class[c]['f1-score'] for c in classes]

    fig, ax = plt.subplots(figsize=(max(10, len(classes) * 0.5), 6))

    y_pos = np.arange(len(classes))
    colors = plt.cm.RdYlGn(np.array(f1_scores))

    bars = ax.barh(y_pos, f1_scores, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel('F1 Score')
    ax.set_title(f'{model_name.upper()} - {task.upper()} Per-Class F1 Scores')
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', alpha=0.3)

    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
               f'{score:.2f}', va='center', fontsize=8)

    plt.tight_layout()
    output_path = output_dir / f'{model_name}_{task}_per_class_f1.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Per-class F1 图已保存: {output_path}")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='可视化评估结果')
    parser.add_argument('--eval_dir', type=str, default='results/evaluation')
    parser.add_argument('--output', type=str, default='results/visualization')
    parser.add_argument('--models', type=str, default=None,
                       help='模型列表: all, bnn, mlp, rf, xgb')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("加载评估结果...")
    results = load_results(args.eval_dir)

    if not results:
        print("没有找到评估结果，请先运行 evaluate_all.py")
        return

    print(f"已加载 {len(results)} 个模型的结果")

    models = args.models.split(',') if args.models else list(results.keys())

    print("\n生成可视化...")

    # 1. 模型对比图
    plot_model_comparison(results, output_dir)

    # 2. 总体对比
    plot_overall_comparison(results, output_dir)

    # 3. 各类别 F1
    for model in models:
        if model in results:
            for task in ['ec', 'loc', 'func']:
                plot_per_class_f1(results, model, task, output_dir)

    print(f"\n可视化结果已保存到: {output_dir}")


if __name__ == '__main__':
    main()
