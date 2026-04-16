#!/usr/bin/env python3
"""
综合训练脚本 - 同时训练BNN和MLP并绘制对比图
"""
import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import f1_score, accuracy_score

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.append(str(Path(__file__).parent.parent))
from src.models.bnn import BayesianMultiTaskMLP
from src.models.mlp import MultiTaskMLP

# ============ 配置 ============
GPU_ID = 0
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "processed" / "esm2_balanced"
DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "models"
OUTPUT_DIR = Path(__file__).parent.parent / "experiments"
OUTPUT_DIR.mkdir(exist_ok=True)

# 类别配置
EC_CLASSES = 8
LOC_CLASSES = 11
FUNC_CLASSES = 17

EC_NAMES = ['No EC', 'EC1-Oxidoreductases', 'EC2-Transferases', 'EC3-Hydrolases',
            'EC4-Lyases', 'EC5-Isomerases', 'EC6-Ligases', 'EC7-Translocases']

LOC_NAMES = ['Cytoplasm', 'Membrane', 'Plasma', 'Nucleus', 'Mitochondria',
            'ER', 'Golgi', 'Secreted', 'Cell_Wall', 'Lysosome', 'Peroxisome']

FUNC_NAMES = ['Binding', 'Transporter', 'Hydrolase', 'Transferase', 'Oxidoreductase',
              'Lyase', 'Ligase', 'Isomerase', 'Translocase', 'Signaling',
              'Structural', 'Kinase', 'Protease', 'Transcription', 'Antioxidant',
              'Enzyme', 'Motor']


class Trainer:
    """训练器类"""
    
    def __init__(self, model_type: str, device: torch.device, data_dir: Path):
        self.model_type = model_type
        self.device = device
        self.data_dir = data_dir
        
        self.model = None
        self.history = {
            'train_loss': [], 'val_loss': [],
            'ec_acc': [], 'loc_f1': [], 'func_f1': []
        }
    
    def load_data(self):
        """加载数据 - 标签是单标签格式 (0-7)"""
        print("加载数据...")
        
        train_features = np.load(self.data_dir / "train_features.npy")
        train_labels = np.load(self.data_dir / "train_labels.npy")  # shape: (N,), 值0-7
        val_features = np.load(self.data_dir / "val_features.npy")
        val_labels = np.load(self.data_dir / "val_labels.npy")
        test_features = np.load(self.data_dir / "test_features.npy")
        test_labels = np.load(self.data_dir / "test_labels.npy")
        
        # 标签是单标签格式，转换为 LongTensor
        train_ec = torch.LongTensor(train_labels)
        val_ec = torch.LongTensor(val_labels)
        test_ec = torch.LongTensor(test_labels)
        
        self.train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(train_features), train_ec),
            batch_size=128, shuffle=True
        )
        self.val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(val_features), val_ec),
            batch_size=128
        )
        self.test_loader = DataLoader(
            TensorDataset(torch.FloatTensor(test_features), test_ec),
            batch_size=128
        )
        
        print(f"  训练集: {len(self.train_loader.dataset)} 样本")
        print(f"  验证集: {len(self.val_loader.dataset)} 样本")
        print(f"  测试集: {len(self.test_loader.dataset)} 样本")
        
        return train_features.shape[1]
    
    def build_model(self, input_dim: int):
        """构建模型"""
        # 只做EC分类
        output_dims = {'ec': EC_CLASSES}
        
        if self.model_type == 'bnn':
            self.model = BayesianMultiTaskMLP(
                input_dim=input_dim,
                output_dims=output_dims,
                hidden_dims=[512, 256],
                dropout=0.3
            ).to(self.device)
        else:
            self.model = MultiTaskMLP(
                input_dim=input_dim,
                output_dims=output_dims,
                hidden_dims=[512, 256],
                dropout=0.3
            ).to(self.device)
        
        print(f"  模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train(self, epochs: int = 100, patience: int = 15) -> Dict:
        """训练"""
        optimizer = Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        criterion = nn.CrossEntropyLoss()
        
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        
        print(f"\n开始训练 {self.model_type.upper()}...")
        
        for epoch in range(epochs):
            # 训练
            self.model.train()
            train_loss = 0
            
            for batch in self.train_loader:
                X, ec_y = batch[0], batch[1]
                X = X.to(self.device)
                ec_y = ec_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X)
                
                loss = criterion(outputs['ec'], ec_y)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(self.train_loader)
            
            # 验证
            self.model.eval()
            val_loss = 0
            all_ec_preds, all_ec_labels = [], []
            
            with torch.no_grad():
                for batch in self.val_loader:
                    X, ec_y = batch[0], batch[1]
                    X = X.to(self.device)
                    ec_y = ec_y.to(self.device)
                    outputs = self.model(X)
                    val_loss += criterion(outputs['ec'], ec_y).item()
                    
                    all_ec_preds.append(outputs['ec'].argmax(dim=1).cpu())
                    all_ec_labels.append(ec_y.cpu())
            
            val_loss /= len(self.val_loader)
            scheduler.step(val_loss)
            
            # 指标
            ec_acc = accuracy_score(torch.cat(all_ec_labels), torch.cat(all_ec_preds))
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['ec_acc'].append(ec_acc)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | EC Acc: {ec_acc:.4f}")
            
            # 保存最佳
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停于 Epoch {epoch+1}")
                    break
        
        self.model.load_state_dict(best_state)
        return {'best_val_loss': best_val_loss}
    
    def evaluate(self) -> Dict:
        """评估"""
        self.model.eval()
        all_ec_preds, all_ec_labels = [], []
        
        with torch.no_grad():
            for batch in self.test_loader:
                X, ec_y = batch[0], batch[1]
                X = X.to(self.device)
                outputs = self.model(X)
                
                all_ec_preds.append(outputs['ec'].argmax(dim=1).cpu())
                all_ec_labels.append(ec_y.cpu())
        
        all_ec_preds = torch.cat(all_ec_preds)
        all_ec_labels = torch.cat(all_ec_labels)
        
        ec_acc = accuracy_score(all_ec_labels, all_ec_preds)
        ec_f1_per_class = f1_score(all_ec_labels, all_ec_preds, average=None)
        ec_f1_macro = f1_score(all_ec_labels, all_ec_preds, average='macro')
        ec_f1_micro = f1_score(all_ec_labels, all_ec_preds, average='micro')
        
        return {
            'ec_accuracy': ec_acc,
            'ec_f1_macro': ec_f1_macro,
            'ec_f1_micro': ec_f1_micro,
            'ec_f1_per_class': ec_f1_per_class.tolist()
        }
    
    def save_model(self, output_path: Path, metrics: Dict):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'config': {
                'input_dim': 480,
                'output_dims': {'ec': EC_CLASSES},
                'hidden_dims': [512, 256],
                'dropout': 0.3
            },
            'class_names': {'ec': EC_NAMES},
            'metrics': metrics,
            'history': self.history
        }, output_path)
        print(f"模型已保存: {output_path}")


def plot_loss_curves(results: Dict, output_path: Path):
    """绘制Loss曲线对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for model_type, history in results.items():
        epochs = range(1, len(history['train_loss']) + 1)
        
        axes[0].plot(epochs, history['train_loss'], label=f'{model_type.upper()} Train', 
                linewidth=2, marker='o', markevery=5, markersize=4)
        axes[0].plot(epochs, history['val_loss'], label=f'{model_type.upper()} Val', 
                linewidth=2, linestyle='--', marker='s', markevery=5, markersize=4)
        
        axes[1].plot(epochs, history['val_loss'], label=f'{model_type.upper()}', 
                linewidth=2, marker='s', markevery=5, markersize=4)
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training & Validation Loss', fontsize=14)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Validation Loss', fontsize=12)
    axes[1].set_title('Validation Loss Comparison', fontsize=14)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Loss曲线已保存: {output_path}")
    plt.close()


def plot_metrics_comparison(results: Dict, metrics: Dict, output_path: Path):
    """绘制模型性能比较图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    model_types = list(metrics.keys())
    colors = {'bnn': '#2E86AB', 'mlp': '#A23B72'}
    
    # EC准确率
    ax1 = axes[0]
    ec_accs = [metrics[m]['ec_accuracy'] for m in model_types]
    bars1 = ax1.bar(model_types, ec_accs, color=[colors.get(m, 'gray') for m in model_types], 
                    edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('EC Classification Accuracy', fontsize=14)
    ax1.set_ylim(0, 1)
    for bar, val in zip(bars1, ec_accs):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # F1 Macro
    ax2 = axes[1]
    f1_macros = [metrics[m]['ec_f1_macro'] for m in model_types]
    bars2 = ax2.bar(model_types, f1_macros, color=[colors.get(m, 'gray') for m in model_types],
                    edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.set_title('EC F1 Score (Macro)', fontsize=14)
    ax2.set_ylim(0, 1)
    for bar, val in zip(bars2, f1_macros):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # F1 Micro
    ax3 = axes[2]
    f1_micros = [metrics[m]['ec_f1_micro'] for m in model_types]
    bars3 = ax3.bar(model_types, f1_micros, color=[colors.get(m, 'gray') for m in model_types],
                    edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('F1 Score', fontsize=12)
    ax3.set_title('EC F1 Score (Micro)', fontsize=14)
    ax3.set_ylim(0, 1)
    for bar, val in zip(bars3, f1_micros):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"性能比较图已保存: {output_path}")
    plt.close()


def plot_ec_per_class(results: Dict, metrics: Dict, output_path: Path):
    """绘制EC各类别F1对比"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    model_types = list(metrics.keys())
    x = np.arange(len(EC_NAMES))
    width = 0.35
    
    colors = {'bnn': '#2E86AB', 'mlp': '#A23B72'}
    
    for i, model_type in enumerate(model_types):
        ec_f1 = metrics[model_type]['ec_f1_per_class']
        offset = width * (i - 0.5)
        bars = ax.bar(x + offset, ec_f1, width, label=model_type.upper(), 
                     color=colors.get(model_type, 'gray'), edgecolor='black')
    
    ax.set_xlabel('EC Class', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('EC Classification F1 per Class', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(EC_NAMES, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"EC各类别F1图已保存: {output_path}")
    plt.close()


def plot_accuracy_curve(results: Dict, output_path: Path):
    """绘制准确率变化曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model_type, history in results.items():
        epochs = range(1, len(history['ec_acc']) + 1)
        ax.plot(epochs, history['ec_acc'], label=f'{model_type.upper()}', 
                linewidth=2, marker='o', markevery=5, markersize=4)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Accuracy', fontsize=12)
    ax.set_title('EC Classification Accuracy During Training', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"准确率曲线已保存: {output_path}")
    plt.close()


def main():
    print("="*60)
    print("BNN vs MLP 多任务模型训练与对比")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")
    
    results = {}
    all_metrics = {}
    
    for model_type in ['bnn', 'mlp']:
        print("\n" + "="*60)
        print(f"训练 {model_type.upper()} 模型")
        print("="*60)
        
        trainer = Trainer(model_type, device, DEFAULT_DATA_DIR)
        input_dim = trainer.load_data()
        trainer.build_model(input_dim)
        trainer.train(epochs=100, patience=15)
        
        print(f"\n{model_type.upper()} 测试集评估:")
        eval_metrics = trainer.evaluate()
        print(f"  Accuracy: {eval_metrics['ec_accuracy']:.4f}")
        print(f"  F1 (Macro): {eval_metrics['ec_f1_macro']:.4f}")
        print(f"  F1 (Micro): {eval_metrics['ec_f1_micro']:.4f}")
        print(f"  各类别 F1:")
        for i, name in enumerate(EC_NAMES):
            print(f"    {name:15s}: {eval_metrics['ec_f1_per_class'][i]:.4f}")
        
        # 保存模型
        model_path = DEFAULT_MODEL_DIR / f"{model_type}_multitask.pt"
        trainer.save_model(model_path, eval_metrics)
        
        results[model_type] = trainer.history
        all_metrics[model_type] = eval_metrics
    
    # 绘制对比图
    print("\n" + "="*60)
    print("绘制对比图")
    print("="*60)
    
    plot_loss_curves(results, OUTPUT_DIR / "loss_comparison.png")
    plot_metrics_comparison(results, all_metrics, OUTPUT_DIR / "metrics_comparison.png")
    plot_ec_per_class(results, all_metrics, OUTPUT_DIR / "ec_per_class_comparison.png")
    plot_accuracy_curve(results, OUTPUT_DIR / "accuracy_curve.png")
    
    # 保存结果
    all_results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'models': all_metrics
    }
    with open(OUTPUT_DIR / "comparison_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"结果JSON已保存: {OUTPUT_DIR / 'comparison_results.json'}")
    
    # 打印总结
    print("\n" + "="*60)
    print("训练完成 - 结果总结")
    print("="*60)
    print(f"\n{'模型':<10} {'Accuracy':<12} {'F1(Macro)':<12} {'F1(Micro)':<12}")
    print("-"*48)
    for model_type in ['bnn', 'mlp']:
        m = all_metrics[model_type]
        print(f"{model_type.upper():<10} {m['ec_accuracy']:<12.4f} {m['ec_f1_macro']:<12.4f} {m['ec_f1_micro']:<12.4f}")
    
    print("\n输出文件:")
    print(f"  - models/bnn_multitask.pt")
    print(f"  - models/mlp_multitask.pt")
    print(f"  - experiments/loss_comparison.png")
    print(f"  - experiments/metrics_comparison.png")
    print(f"  - experiments/ec_per_class_comparison.png")
    print(f"  - experiments/accuracy_curve.png")
    print("="*60)


if __name__ == "__main__":
    main()