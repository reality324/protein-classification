#!/usr/bin/env python3
"""
统一的多任务训练脚本
支持: BNN (贝叶斯神经网络) / MLP / 传统机器学习
同时预测: EC分类 + 细胞定位 + 蛋白质功能
"""
import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).parent.parent))
from src.models.bnn import BayesianMultiTaskMLP
from src.models.mlp import MultiTaskMLP

# ============ 配置 ============
GPU_ID = 0
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "processed" / "esm2_balanced"
DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "models"

# 类别数
EC_CLASSES = 8
LOC_CLASSES = 11
FUNC_CLASSES = 17

# 类别名称
EC_NAMES = ['无EC', 'EC1-氧化还原酶', 'EC2-转移酶', 'EC3-水解酶',
            'EC4-裂解酶', 'EC5-异构酶', 'EC6-连接酶', 'EC7-转位酶']
LOC_NAMES = ['Cytoplasm', 'Membrane', 'Plasma', 'Nucleus', 'Mitochondria',
            'ER', 'Golgi', 'Secreted', 'Cell_Wall', 'Lysosome', 'Peroxisome']
FUNC_NAMES = ['Binding', 'Transporter', 'Hydrolase', 'Transferase', 'Oxidoreductase',
              'Lyase', 'Ligase', 'Isomerase', 'Translocase', 'Signaling',
              'Structural', 'Kinase', 'Protease', 'Transcription', 'Antioxidant',
              'Enzyme', 'Motor']


class UnifiedTrainer:
    """统一训练器"""
    
    def __init__(
        self,
        model_type: str = 'bnn',
        data_dir: Path = DEFAULT_DATA_DIR,
        output_dir: Path = DEFAULT_MODEL_DIR,
        gpu_id: int = GPU_ID
    ):
        self.model_type = model_type
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        
        self.model = None
        self.traditional_model = None
        self.scaler = None
        
        print(f"初始化训练器 | 模型类型: {model_type} | 设备: {self.device}")
    
    def load_data(self) -> Dict:
        """加载数据"""
        print("\n加载数据...")
        
        train_features = np.load(self.data_dir / "train_features.npy")
        train_labels = np.load(self.data_dir / "train_labels.npy")
        val_features = np.load(self.data_dir / "val_features.npy")
        val_labels = np.load(self.data_dir / "val_labels.npy")
        test_features = np.load(self.data_dir / "test_features.npy")
        test_labels = np.load(self.data_dir / "test_labels.npy")
        
        print(f"  训练集: {train_features.shape[0]} 样本")
        print(f"  验证集: {val_features.shape[0]} 样本")
        print(f"  测试集: {test_features.shape[0]} 样本")
        
        # 解析标签
        train_ec = torch.LongTensor(train_labels[:, :EC_CLASSES].argmax(axis=1))
        train_loc = torch.FloatTensor(train_labels[:, EC_CLASSES:EC_CLASSES+LOC_CLASSES])
        train_func = torch.FloatTensor(train_labels[:, EC_CLASSES+LOC_CLASSES:])
        
        val_ec = torch.LongTensor(val_labels[:, :EC_CLASSES].argmax(axis=1))
        val_loc = torch.FloatTensor(val_labels[:, EC_CLASSES:EC_CLASSES+LOC_CLASSES])
        val_func = torch.FloatTensor(val_labels[:, EC_CLASSES+LOC_CLASSES:])
        
        test_ec = torch.LongTensor(test_labels[:, :EC_CLASSES].argmax(axis=1))
        test_loc = torch.FloatTensor(test_labels[:, EC_CLASSES:EC_CLASSES+LOC_CLASSES])
        test_func = torch.FloatTensor(test_labels[:, EC_CLASSES+LOC_CLASSES:])
        
        # 创建 DataLoader
        self.train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(train_features), train_ec, train_loc, train_func),
            batch_size=128, shuffle=True
        )
        self.val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(val_features), val_ec, val_loc, val_func),
            batch_size=128
        )
        self.test_loader = DataLoader(
            TensorDataset(torch.FloatTensor(test_features), test_ec, test_loc, test_func),
            batch_size=128
        )
        
        # 保存原始数据用于传统ML
        self.X_train = train_features
        self.y_train_ec = train_labels[:, :EC_CLASSES]
        self.y_train_loc = train_labels[:, EC_CLASSES:EC_CLASSES+LOC_CLASSES]
        self.y_train_func = train_labels[:, EC_CLASSES+LOC_CLASSES:]
        
        self.X_val = val_features
        self.X_test = test_features
        self.y_test_ec = test_labels[:, :EC_CLASSES].argmax(axis=1)
        self.y_test_loc = test_labels[:, EC_CLASSES:EC_CLASSES+LOC_CLASSES]
        self.y_test_func = test_labels[:, EC_CLASSES+LOC_CLASSES:]
        
        return {'input_dim': train_features.shape[1]}
    
    def build_neural_model(self, input_dim: int):
        """构建神经网络模型"""
        output_dims = {'ec': EC_CLASSES, 'loc': LOC_CLASSES, 'func': FUNC_CLASSES}
        
        if self.model_type == 'bnn':
            self.model = BayesianMultiTaskMLP(
                input_dim=input_dim,
                output_dims=output_dims,
                hidden_dims=[512, 256],
                dropout=0.3,
                prior_std=0.1
            ).to(self.device)
        else:  # mlp
            self.model = MultiTaskMLP(
                input_dim=input_dim,
                output_dims=output_dims,
                hidden_dims=[512, 256],
                dropout=0.3
            ).to(self.device)
        
        print(f"  模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def build_traditional_model(self, model_name: str):
        """构建传统机器学习模型"""
        if model_name == 'rf':
            self.traditional_model = MultiOutputClassifier(
                RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
            )
        elif model_name == 'xgb':
            try:
                import xgboost as xgb
                self.traditional_model = MultiOutputClassifier(
                    xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42)
                )
            except ImportError:
                print("XGBoost 未安装，使用 GradientBoosting")
                self.traditional_model = MultiOutputClassifier(
                    GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
                )
        elif model_name == 'svm':
            self.scaler = StandardScaler()
            self.traditional_model = MultiOutputClassifier(
                SVC(C=1.0, kernel='rbf', probability=True, random_state=42)
            )
        elif model_name == 'lr':
            self.scaler = StandardScaler()
            self.traditional_model = MultiOutputClassifier(
                LogisticRegression(max_iter=1000, random_state=42)
            )
    
    def train_neural(self, epochs: int = 100, patience: int = 15) -> Dict:
        """训练神经网络"""
        optimizer = Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        ec_criterion = nn.CrossEntropyLoss()
        bce_criterion = nn.BCEWithLogitsLoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # 训练
            self.model.train()
            train_loss = 0
            
            for X, ec_y, loc_y, func_y in self.train_loader:
                X = X.to(self.device)
                ec_y = ec_y.to(self.device)
                loc_y = loc_y.to(self.device)
                func_y = func_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X)
                
                loss = (ec_criterion(outputs['ec'], ec_y) +
                        bce_criterion(outputs['loc'], loc_y) +
                        bce_criterion(outputs['func'], func_y))
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(self.train_loader)
            
            # 验证
            self.model.eval()
            val_loss = 0
            all_ec_preds, all_ec_labels = [], []
            all_loc_preds, all_loc_labels = [], []
            
            with torch.no_grad():
                for X, ec_y, loc_y, func_y in self.val_loader:
                    X = X.to(self.device)
                    outputs = self.model(X)
                    val_loss += (ec_criterion(outputs['ec'], ec_y) +
                                bce_criterion(outputs['loc'], loc_y) +
                                bce_criterion(outputs['func'], func_y)).item()
                    
                    all_ec_preds.append(outputs['ec'].argmax(dim=1).cpu())
                    all_ec_labels.append(ec_y.cpu())
                    all_loc_preds.append((torch.sigmoid(outputs['loc']) > 0.5).float().cpu())
                    all_loc_labels.append(loc_y.cpu())
            
            val_loss /= len(self.val_loader)
            scheduler.step(val_loss)
            
            # 指标
            ec_acc = accuracy_score(torch.cat(all_ec_labels), torch.cat(all_ec_preds))
            loc_f1 = f1_score(torch.cat(all_loc_labels), torch.cat(all_loc_preds), average='micro')
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f}/{val_loss:.4f} | "
                      f"EC Acc: {ec_acc:.4f} | Loc F1: {loc_f1:.4f}")
            
            # 保存最佳
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停于 Epoch {epoch+1}")
                    break
        
        # 加载最佳模型
        self.model.load_state_dict(best_model_state)
        return {'best_val_loss': best_val_loss}
    
    def train_traditional(self) -> Dict:
        """训练传统机器学习模型"""
        # 合并标签为多标签格式
        # 训练 EC 单任务
        print(f"\n训练传统ML模型: {self.model_type}")
        
        # 简单起见，训练 EC 分类
        y_train = self.y_train_ec.argmax(axis=1)
        
        if self.scaler:
            X_train_scaled = self.scaler.fit_transform(self.X_train)
            X_test_scaled = self.scaler.transform(self.X_test)
        else:
            X_train_scaled = self.X_train
            X_test_scaled = self.X_test
        
        print(f"  训练样本: {len(X_train_scaled)}")
        
        import time
        start = time.time()
        self.traditional_model.fit(X_train_scaled, y_train)
        train_time = time.time() - start
        
        print(f"  训练时间: {train_time:.2f}s")
        
        return {'train_time': train_time}
    
    def evaluate(self) -> Dict:
        """评估模型"""
        print("\n" + "="*60)
        print("测试集评估")
        print("="*60)
        
        results = {}
        
        if self.model_type in ['bnn', 'mlp']:
            # 神经网络评估
            self.model.eval()
            all_ec_preds, all_ec_labels = [], []
            all_loc_preds, all_loc_labels = [], []
            all_func_preds, all_func_labels = [], []
            
            with torch.no_grad():
                for X, ec_y, loc_y, func_y in self.test_loader:
                    X = X.to(self.device)
                    outputs = self.model(X)
                    
                    all_ec_preds.append(outputs['ec'].argmax(dim=1).cpu())
                    all_ec_labels.append(ec_y.cpu())
                    all_loc_preds.append((torch.sigmoid(outputs['loc']) > 0.5).float().cpu())
                    all_loc_labels.append(loc_y.cpu())
                    all_func_preds.append((torch.sigmoid(outputs['func']) > 0.5).float().cpu())
                    all_func_labels.append(func_y.cpu())
            
            all_ec_preds = torch.cat(all_ec_preds)
            all_ec_labels = torch.cat(all_ec_labels)
            all_loc_preds = torch.cat(all_loc_preds)
            all_loc_labels = torch.cat(all_loc_labels)
            all_func_preds = torch.cat(all_func_preds)
            all_func_labels = torch.cat(all_func_labels)
            
            ec_acc = accuracy_score(all_ec_labels, all_ec_preds)
            loc_f1 = f1_score(all_loc_labels, all_loc_preds, average='micro')
            func_f1 = f1_score(all_func_labels, all_func_preds, average='micro')
            ec_f1_per_class = f1_score(all_ec_labels, all_ec_preds, average=None)
            
            print(f"\n【EC分类】准确率: {ec_acc:.4f}")
            print("各类别 F1:")
            for i, name in enumerate(EC_NAMES):
                print(f"  {name:15s}: {ec_f1_per_class[i]:.4f}")
            
            print(f"\n【细胞定位】F1 (micro): {loc_f1:.4f}")
            print(f"【蛋白质功能】F1 (micro): {func_f1:.4f}")
            
            results = {
                'ec_accuracy': ec_acc,
                'loc_f1': loc_f1,
                'func_f1': func_f1,
                'ec_f1_per_class': ec_f1_per_class.tolist()
            }
        
        else:
            # 传统ML评估
            y_pred = self.traditional_model.predict(self.X_test)
            ec_acc = accuracy_score(self.y_test_ec, y_pred)
            print(f"\n【EC分类】准确率: {ec_acc:.4f}")
            
            results = {'ec_accuracy': ec_acc}
        
        return results
    
    def save_model(self, model_name: str, metrics: Dict):
        """保存模型"""
        model_path = self.output_dir / f"{model_name}_multitask.pt"
        
        if self.model_type in ['bnn', 'mlp']:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_type': self.model_type,
                'config': {
                    'input_dim': self.X_train.shape[1],
                    'output_dims': {'ec': EC_CLASSES, 'loc': LOC_CLASSES, 'func': FUNC_CLASSES},
                    'hidden_dims': [512, 256],
                    'dropout': 0.3
                },
                'class_names': {
                    'ec': EC_NAMES,
                    'loc': LOC_NAMES,
                    'func': FUNC_NAMES
                },
                'metrics': metrics
            }, model_path)
        else:
            import joblib
            joblib.dump(self.traditional_model, model_path)
        
        print(f"\n模型已保存: {model_path}")
        return model_path


def parse_args():
    parser = argparse.ArgumentParser(description='统一多任务训练')
    parser.add_argument('--model', '-m', type=str, default='bnn',
                        choices=['bnn', 'mlp', 'rf', 'xgb', 'svm', 'lr'],
                        help='模型类型')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='训练轮数')
    parser.add_argument('--patience', '-p', type=int, default=15, help='早停轮数')
    parser.add_argument('--data_dir', '-d', type=str, default=None, help='数据目录')
    parser.add_argument('--output_dir', '-o', type=str, default=None, help='输出目录')
    parser.add_argument('--gpu', '-g', type=int, default=GPU_ID, help='GPU ID')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*60)
    print("多任务蛋白质分类器训练")
    print(f"模型类型: {args.model.upper()}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # 初始化
    trainer = UnifiedTrainer(
        model_type=args.model,
        data_dir=Path(args.data_dir) if args.data_dir else DEFAULT_DATA_DIR,
        output_dir=Path(args.output_dir) if args.output_dir else DEFAULT_MODEL_DIR,
        gpu_id=args.gpu
    )
    
    # 加载数据
    info = trainer.load_data()
    
    # 构建模型
    if args.model in ['bnn', 'mlp']:
        trainer.build_neural_model(info['input_dim'])
        
        # 训练
        train_info = trainer.train_neural(epochs=args.epochs, patience=args.patience)
    else:
        trainer.build_traditional_model(args.model)
        train_info = trainer.train_traditional()
    
    # 评估
    results = trainer.evaluate()
    
    # 保存
    model_name = f"{args.model}_multitask"
    if args.model in ['bnn', 'mlp']:
        model_path = trainer.save_model(model_name, results)
    else:
        model_path = trainer.output_dir / f"{model_name}.pkl"
        import joblib
        joblib.dump(trainer.traditional_model, model_path)
        print(f"\n模型已保存: {model_path}")
    
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)


if __name__ == "__main__":
    main()