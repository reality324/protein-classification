#!/usr/bin/env python3
"""
Multitask 推理脚本

功能：使用训练好的多任务模型同时预测EC主类、细胞定位、分子功能
"""
import sys
import argparse
import torch
import torch.nn as nn
import json
from pathlib import Path
from typing import Dict
import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.encodings import EncoderRegistry

# 标签映射
LOCALIZATION_CLASSES = {
    0: "Cytoplasm", 1: "ER", 2: "Golgi", 3: "Membrane",
    4: "Mitochondria", 5: "Nucleus", 6: "Secreted"
}


class MultiTaskModel(nn.Module):
    """多任务神经网络 - 与训练脚本中的MultitaskModel结构一致"""

    def __init__(
        self,
        input_dim: int,
        task_dims: Dict[str, int],
        hidden_dims: list = [256, 128],  # 与训练脚本一致
        dropout: float = 0.3,
    ):
        super().__init__()
        self.task_dims = task_dims

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
        self.shared = nn.Sequential(*layers)

        self.heads = nn.ModuleDict()
        for task_name, num_classes in task_dims.items():
            self.heads[task_name] = nn.Linear(prev_dim, num_classes)

    def forward(self, x):
        features = self.shared(x)
        outputs = {name: head(features) for name, head in self.heads.items()}
        return outputs


class MultitaskModel(MultiTaskModel):
    """兼容性别名"""
    pass


def load_model(model_path):
    """加载模型"""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # 从checkpoint获取配置
    task_dims = checkpoint.get('task_dims', {'ec': 7, 'localization': 7, 'function': 7})
    
    # 从state_dict推断hidden_dims（只从Linear层推断）
    state_dict = checkpoint['model_state']
    hidden_dims = []
    for key in sorted(state_dict.keys()):
        # Linear层: shared.0.weight, shared.4.weight 等
        # BatchNorm层: shared.1.weight 等
        # 只取Linear层的权重
        if key.startswith('shared.') and 'weight' in key and '.weight' == key[-7:]:
            hidden_dims.append(state_dict[key].shape[0])
    
    if len(hidden_dims) != 2:
        hidden_dims = checkpoint.get('hidden_dims', [256, 128])
    
    model = MultiTaskModel(
        checkpoint['input_dim'],
        task_dims,
        hidden_dims
    )
    
    # 严格加载
    model.load_state_dict(checkpoint['model_state'])
    
    model.eval()
    
    # 处理没有scaler的情况
    scaler = StandardScaler()
    if 'scaler_mean' in checkpoint:
        scaler.mean_ = checkpoint['scaler_mean']
        scaler.scale_ = checkpoint['scaler_scale']
    else:
        # 没有scaler数据，创建单位scaler
        scaler.mean_ = np.zeros(checkpoint['input_dim'])
        scaler.scale_ = np.ones(checkpoint['input_dim'])
    
    return model, scaler, task_dims, checkpoint.get('class_names', {})


def predict(sequence, model, scaler, task_dims, class_names_map=None, top_k=3):
    """单条预测
    
    Args:
        sequence: 蛋白质序列
        model: 训练好的模型
        scaler: 数据标准化器
        task_dims: 任务维度字典
        class_names_map: 类别名称映射
        top_k: 显示前k个最可能的预测
    """
    encoder = EncoderRegistry.get('esm2')
    features = encoder.encode(sequence).reshape(1, -1)
    features_scaled = scaler.transform(features)
    
    # 类别名称 - 从模型加载或使用默认
    EC_CLASSES = {0: "EC1", 1: "EC2", 2: "EC3", 3: "EC4", 4: "EC5", 5: "EC6", 6: "EC7"}
    LOCALIZATION_CLASSES = {
        0: "Cytoplasm", 1: "Endoplasmic reticulum", 2: "Membrane",
        3: "Mitochondria", 4: "Nucleus", 5: "Other", 6: "Secreted", 7: "Unknown"
    }
    FUNCTION_CLASSES = {
        0: "Catalytic", 1: "Kinase", 2: "Other", 3: "Receptor", 4: "Transferase", 5: "Transporter"
    }
    
    class_names = {
        'ec': class_names_map.get('ec', EC_CLASSES) if class_names_map else EC_CLASSES,
        'localization': class_names_map.get('localization', LOCALIZATION_CLASSES) if class_names_map else LOCALIZATION_CLASSES,
        'function': class_names_map.get('function', FUNCTION_CLASSES) if class_names_map else FUNCTION_CLASSES,
    }
    
    with torch.no_grad():
        outputs = model(torch.FloatTensor(features_scaled))
        
        result = {"sequence": sequence[:50] + "..." if len(sequence) > 50 else sequence}
        
        for task, num_classes in task_dims.items():
            probs = torch.softmax(outputs[task], dim=1)[0].numpy()
            pred_class = int(np.argmax(probs))
            
            names = class_names.get(task, {})
            
            # 获取 Top-K 预测
            top_indices = np.argsort(probs)[::-1][:top_k]
            top_predictions = [
                {
                    "class": int(idx),
                    "label": names.get(idx, f"Class {idx}"),
                    "probability": float(probs[idx])
                }
                for idx in top_indices
            ]
            
            result[task] = {
                "predicted_class": pred_class,
                "predicted_label": names.get(pred_class, f"Class {pred_class}"),
                "confidence": float(probs[pred_class]),
                "top_predictions": top_predictions,
            }
    
    return result


def read_fasta(fasta_path):
    """读取FASTA文件"""
    headers, sequences = [], []
    with open(fasta_path, 'r') as f:
        current_seq, current_header = [], None
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header is not None:
                    headers.append(current_header)
                    sequences.append(''.join(current_seq))
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        if current_header is not None:
            headers.append(current_header)
            sequences.append(''.join(current_seq))
    return headers, sequences


def main():
    parser = argparse.ArgumentParser(description="多任务模型推理")
    parser.add_argument("--model", "-m", type=str, required=True, help="模型路径 (.pt)")
    parser.add_argument("--sequence", "-s", type=str, help="单条蛋白质序列")
    parser.add_argument("--fasta", "-f", type=str, help="FASTA文件路径")
    parser.add_argument("--output", "-o", type=str, help="输出JSON文件路径")
    parser.add_argument("--top-k", "-k", type=int, default=3, help="显示前k个预测结果 (默认3)")
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"错误: 模型文件不存在: {model_path}")
        return 1
    
    # 加载模型
    print(f"加载模型: {model_path}")
    model, scaler, task_dims, class_names = load_model(model_path)
    print(f"  任务: {list(task_dims.keys())}")
    print(f"  类别数: EC={task_dims.get('ec', '?')}, 定位={task_dims.get('localization', '?')}, 功能={task_dims.get('function', '?')}")
    
    print("\n" + "=" * 60)
    print("蛋白质多任务预测")
    print("  - EC主类分类")
    print("  - 细胞定位分类")
    print("  - 分子功能分类")
    print("=" * 60)
    
    results = []
    
    if args.sequence:
        result = predict(args.sequence, model, scaler, task_dims, class_names, top_k=args.top_k)
        result["sequence_length"] = len(args.sequence)
        results.append(result)
        
        print(f"\n序列: {result['sequence']}")
        print(f"长度: {result['sequence_length']} 氨基酸")
        print(f"\n预测结果:")
        for task in ['ec', 'localization', 'function']:
            if task in result:
                task_info = result[task]
                print(f"\n  [{task.upper()}]")
                print(f"    最佳预测: {task_info['predicted_label']} ({task_info['confidence']:.2%})")
                print(f"    Top-{args.top_k} 预测:")
                for i, pred in enumerate(task_info['top_predictions']):
                    marker = ">>>" if i == 0 else "   "
                    print(f"      {marker} {i+1}. {pred['label']}: {pred['probability']:.2%}")
    
    elif args.fasta:
        headers, sequences = read_fasta(args.fasta)
        print(f"\n读取 {len(sequences)} 条蛋白质序列")
        
        for i, (header, seq) in enumerate(zip(headers, sequences)):
            result = predict(seq, model, scaler, task_dims, class_names)
            result["header"] = header
            result["sequence_length"] = len(seq)
            results.append(result)
            
            print(f"\n[{i+1}/{len(sequences)}] {header[:50]}")
            ec_label = result.get('ec', {}).get('predicted_label', 'N/A')
            print(f"  EC: {ec_label}")
    else:
        print("错误: 请提供 --sequence 或 --fasta 参数")
        return 1
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {output_path}")
    
    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
