#!/usr/bin/env python3
"""
预测脚本 - 使用训练好的模型进行预测
支持: BNN, MLP, 传统ML模型
同时预测: EC分类 + 细胞定位 + 蛋白质功能
"""
import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).parent.parent))
from src.data.featurization import get_feature_extractor
from src.models.bnn import BayesianMultiTaskMLP
from src.models.mlp import MultiTaskMLP


class MultiTaskPredictor:
    """多任务预测器"""
    
    def __init__(
        self,
        model_path: Path,
        embedding_method: str = 'esm2_35M',
        device: str = None,
        mc_samples: int = 10
    ):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.embedding_method = embedding_method
        self.mc_samples = mc_samples
        
        self._load_model(model_path)
    
    def _load_model(self, model_path: Path):
        """加载模型"""
        print(f"加载模型: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # 获取配置
        config = checkpoint.get('config', {})
        class_names = checkpoint.get('class_names', {})
        model_type = checkpoint.get('model_type', 'mlp')
        
        input_dim = config.get('input_dim', 480)
        output_dims = config.get('output_dims', {'ec': 8, 'loc': 11, 'func': 17})
        
        print(f"模型类型: {model_type}")
        print(f"输入维度: {input_dim}")
        print(f"输出: EC({output_dims.get('ec')}) + 定位({output_dims.get('loc')}) + 功能({output_dims.get('func')})")
        
        # 创建模型
        if model_type == 'bnn':
            self.model = BayesianMultiTaskMLP(
                input_dim=input_dim,
                output_dims=output_dims,
                hidden_dims=config.get('hidden_dims', [512, 256]),
                dropout=config.get('dropout', 0.3)
            )
        else:
            self.model = MultiTaskMLP(
                input_dim=input_dim,
                output_dims=output_dims,
                hidden_dims=config.get('hidden_dims', [512, 256]),
                dropout=config.get('dropout', 0.3)
            )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 保存类别名称
        self.ec_names = class_names.get('ec', [f'EC{i}' for i in range(output_dims.get('ec', 8))])
        self.loc_names = class_names.get('loc', [f'Loc{i}' for i in range(output_dims.get('loc', 11))])
        self.func_names = class_names.get('func', [f'Func{i}' for i in range(output_dims.get('func', 17))])
        
        self.is_bnn = model_type == 'bnn'
        
        print(f"模型已加载，使用设备: {self.device}")
        if self.is_bnn:
            print(f"MC Dropout 采样次数: {self.mc_samples}")
    
    @torch.no_grad()
    def predict(self, sequence: str, threshold: float = 0.5) -> Dict:
        """预测单条序列"""
        extractor = get_feature_extractor(self.embedding_method)
        features = extractor.extract([sequence])
        features = torch.FloatTensor(features).to(self.device)
        
        if self.is_bnn:
            # BNN: MC Dropout 采样
            mc_results = self.model.mc_forward(features, n_samples=self.mc_samples)
            
            ec_probs = torch.softmax(mc_results['mean']['ec'], dim=1).cpu().numpy()[0]
            loc_probs = torch.sigmoid(mc_results['mean']['loc']).cpu().numpy()[0]
            func_probs = torch.sigmoid(mc_results['mean']['func']).cpu().numpy()[0]
            
            # 不确定性
            ec_uncertainty = mc_results['std']['ec'].cpu().numpy()[0]
            loc_uncertainty = mc_results['std']['loc'].cpu().numpy()[0]
            func_uncertainty = mc_results['std']['func'].cpu().numpy()[0]
        else:
            # MLP: 直接预测
            outputs = self.model(features)
            
            ec_probs = torch.softmax(outputs['ec'], dim=1).cpu().numpy()[0]
            loc_probs = torch.sigmoid(outputs['loc']).cpu().numpy()[0]
            func_probs = torch.sigmoid(outputs['func']).cpu().numpy()[0]
            
            ec_uncertainty = None
            loc_uncertainty = None
            func_uncertainty = None
        
        # EC 预测
        ec_pred_idx = np.argmax(ec_probs)
        ec_predictions = [(self.ec_names[i], ec_probs[i]) for i in range(len(ec_probs))]
        ec_predictions.sort(key=lambda x: x[1], reverse=True)
        
        # 定位预测
        loc_mask = loc_probs >= threshold
        loc_predictions = [(self.loc_names[i], loc_probs[i]) for i in range(len(loc_probs)) if loc_mask[i]]
        loc_predictions.sort(key=lambda x: x[1], reverse=True)
        
        # 功能预测
        func_mask = func_probs >= threshold
        func_predictions = [(self.func_names[i], func_probs[i]) for i in range(len(func_probs)) if func_mask[i]]
        func_predictions.sort(key=lambda x: x[1], reverse=True)
        
        result = {
            'sequence': sequence,
            'sequence_length': len(sequence),
            'ec_prediction': self.ec_names[ec_pred_idx],
            'ec_confidence': ec_probs[ec_pred_idx],
            'ec_top3': ec_predictions[:3],
            'location_predictions': loc_predictions[:5],
            'function_predictions': func_predictions[:10],
        }
        
        if self.is_bnn:
            result['uncertainty'] = {
                'ec': ec_uncertainty[ec_pred_idx],
                'location_avg': loc_uncertainty.mean(),
                'function_avg': func_uncertainty.mean()
            }
        
        return result
    
    @torch.no_grad()
    def predict_batch(self, sequences: List[str], batch_size: int = 32, threshold: float = 0.5) -> pd.DataFrame:
        """批量预测"""
        extractor = get_feature_extractor(self.embedding_method)
        all_features = extractor._batch_encode(sequences, batch_size)
        
        results = []
        n_samples = len(sequences)
        
        for i in range(0, n_samples, batch_size):
            batch_features = torch.FloatTensor(all_features[i:i+batch_size]).to(self.device)
            batch_seqs = sequences[i:i+batch_size]
            
            if self.is_bnn:
                mc_results = self.model.mc_forward(batch_features, n_samples=self.mc_samples)
                ec_probs = torch.softmax(mc_results['mean']['ec'], dim=1).cpu().numpy()
                loc_probs = torch.sigmoid(mc_results['mean']['loc']).cpu().numpy()
                func_probs = torch.sigmoid(mc_results['mean']['func']).cpu().numpy()
            else:
                outputs = self.model(batch_features)
                ec_probs = torch.softmax(outputs['ec'], dim=1).cpu().numpy()
                loc_probs = torch.sigmoid(outputs['loc']).cpu().numpy()
                func_probs = torch.sigmoid(outputs['func']).cpu().numpy()
            
            for j, (ec_p, loc_p, func_p) in enumerate(zip(ec_probs, loc_probs, func_probs)):
                ec_pred_idx = np.argmax(ec_p)
                
                loc_mask = loc_p >= threshold
                loc_pred = [self.loc_names[k] for k in range(len(loc_p)) if loc_mask[k]]
                
                func_mask = func_p >= threshold
                func_pred = [self.func_names[k] for k in range(len(func_p)) if func_mask[k]]
                
                results.append({
                    'sequence': batch_seqs[j],
                    'ec_prediction': self.ec_names[ec_pred_idx],
                    'ec_confidence': ec_p[ec_pred_idx],
                    'location_predictions': ','.join(loc_pred[:5]),
                    'function_predictions': ','.join(func_pred[:10]),
                })
        
        return pd.DataFrame(results)


def parse_args():
    parser = argparse.ArgumentParser(description='蛋白质多任务预测')
    
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='输入序列或FASTA文件')
    parser.add_argument('--fasta', action='store_true',
                        help='输入是FASTA文件')
    parser.add_argument('--output', '-o', type=str, default='predictions.tsv',
                        help='输出文件路径')
    parser.add_argument('--model', '-m', type=str,
                        default='models/bnn_multitask.pt',
                        help='模型文件路径')
    parser.add_argument('--embedding', '-e', type=str,
                        default='esm2_35M',
                        help='嵌入方法')
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                        help='预测阈值')
    parser.add_argument('--batch_size', '-b', type=int, default=32,
                        help='批处理大小')
    parser.add_argument('--mc_samples', type=int, default=10,
                        help='BNN的MC Dropout采样次数')
    parser.add_argument('--print', action='store_true',
                        help='打印详细结果')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    predictor = MultiTaskPredictor(
        model_path=Path(args.model),
        embedding_method=args.embedding,
        mc_samples=args.mc_samples
    )
    
    if args.fasta:
        from Bio import SeqIO
        records = list(SeqIO.parse(args.input, "fasta"))
        sequences = [str(rec.seq) for rec in records]
        ids = [rec.id for rec in records]
        
        print(f"从FASTA加载 {len(records)} 条序列")
        results = predictor.predict_batch(sequences, batch_size=args.batch_size, threshold=args.threshold)
        results.to_csv(args.output, index=False, sep='\t')
        print(f"结果已保存: {args.output}")
    
    else:
        sequence = args.input.strip()
        result = predictor.predict(sequence, threshold=args.threshold)
        
        print("\n" + "="*60)
        print("预测结果")
        print("="*60)
        print(f"序列长度: {result['sequence_length']}")
        
        print(f"\n【EC分类预测】")
        print(f"  预测类别: {result['ec_prediction']}")
        print(f"  置信度: {result['ec_confidence']:.4f}")
        print(f"  Top 3: {', '.join([f'{n}({p:.3f})' for n, p in result['ec_top3']])}")
        
        print(f"\n【细胞定位预测】")
        for name, prob in result['location_predictions'][:3]:
            print(f"  {name}: {prob:.4f}")
        
        print(f"\n【蛋白质功能预测】")
        for name, prob in result['function_predictions'][:5]:
            print(f"  {name}: {prob:.4f}")
        
        if 'uncertainty' in result:
            print(f"\n【不确定性估计】(BNN)")
            print(f"  EC: {result['uncertainty']['ec']:.4f}")
            print(f"  定位平均: {result['uncertainty']['location_avg']:.4f}")
            print(f"  功能平均: {result['uncertainty']['function_avg']:.4f}")


if __name__ == "__main__":
    main()