#!/usr/bin/env python3
"""
蛋白质多任务分类器推理脚本
- 输入: 蛋白质序列
- 输出: EC主类 + 细胞定位 + 分子功能预测
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from src.encodings.esm2 import ESM2Encoder


class MultiTaskBNN(nn.Module):
    """多任务 BNN"""

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


def predict(sequence, model, class_labels, esm2_encoder, device, n_samples=30):
    """带 MC Dropout 的预测"""
    # 编码
    features = esm2_encoder.encode(sequence).reshape(1, -1)
    X = torch.FloatTensor(features).to(device)

    # 启用 dropout
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    # MC 采样
    ec_preds, loc_preds, func_preds = [], [], []

    with torch.no_grad():
        for _ in range(n_samples):
            ec_out, loc_out, func_out = model(X)
            ec_preds.append(torch.softmax(ec_out, dim=1))
            loc_preds.append(torch.softmax(loc_out, dim=1))
            func_preds.append(torch.softmax(func_out, dim=1))

    ec_mean = torch.stack(ec_preds).mean(0)[0].cpu().numpy()
    loc_mean = torch.stack(loc_preds).mean(0)[0].cpu().numpy()
    func_mean = torch.stack(func_preds).mean(0)[0].cpu().numpy()

    # 计算不确定性 (熵)
    ec_entropy = -np.sum(ec_mean * np.log(ec_mean + 1e-10))
    loc_entropy = -np.sum(loc_mean * np.log(loc_mean + 1e-10))
    func_entropy = -np.sum(func_mean * np.log(func_mean + 1e-10))

    ec_classes = class_labels['ec_classes']
    loc_classes = class_labels['loc_classes']
    func_classes = class_labels['func_classes']

    return {
        'ec': {
            'class': ec_classes[np.argmax(ec_mean)],
            'confidence': float(np.max(ec_mean)),
            'uncertainty': float(ec_entropy),
            'top3': [(ec_classes[i], float(ec_mean[i])) for i in np.argsort(ec_mean)[-3:][::-1]],
        },
        'localization': {
            'class': loc_classes[np.argmax(loc_mean)],
            'confidence': float(np.max(loc_mean)),
            'uncertainty': float(loc_entropy),
            'top3': [(loc_classes[i], float(loc_mean[i])) for i in np.argsort(loc_mean)[-3:][::-1]],
        },
        'function': {
            'class': func_classes[np.argmax(func_mean)],
            'confidence': float(np.max(func_mean)),
            'uncertainty': float(func_entropy),
            'top3': [(func_classes[i], float(func_mean[i])) for i in np.argsort(func_mean)[-3:][::-1]],
        },
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='蛋白质多任务分类推理')
    parser.add_argument('--model', type=str, default='models/bnn_esm2_multitask/model.pt',
                        help='模型路径')
    parser.add_argument('--sequence', type=str, default=None,
                        help='蛋白质序列')
    parser.add_argument('--fasta', type=str, default=None,
                        help='FASTA 文件路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出 JSON 文件')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 加载模型
    print(f"\n加载模型: {args.model}")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)

    ec_classes = checkpoint['ec_classes']
    loc_classes = checkpoint['loc_classes']
    func_classes = checkpoint['func_classes']
    input_dim = checkpoint['input_dim']

    model = MultiTaskBNN(
        input_dim=input_dim,
        ec_classes=len(ec_classes),
        loc_classes=len(loc_classes),
        func_classes=len(func_classes),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    class_labels = {
        'ec_classes': ec_classes,
        'loc_classes': loc_classes,
        'func_classes': func_classes,
    }

    # ESM2 编码器
    esm2 = ESM2Encoder(pooling='mean', device=device)

    results = {}

    if args.fasta:
        # 批量推理
        print(f"\n从 FASTA 文件加载: {args.fasta}")
        with open(args.fasta) as f:
            content = f.read()

        sequences = []
        names = []
        current_name = None
        current_seq = []

        for line in content.strip().split('\n'):
            if line.startswith('>'):
                if current_name:
                    sequences.append(''.join(current_seq))
                    names.append(current_name)
                current_name = line[1:].strip()
                current_seq = []
            else:
                current_seq.append(line.strip())

        if current_name:
            sequences.append(''.join(current_seq))
            names.append(current_name)

        print(f"共 {len(sequences)} 条序列")

        for name, seq in zip(names, sequences):
            print(f"\n序列: {name}")
            result = predict(seq, model, class_labels, esm2, device)
            results[name] = result

            print(f"  EC: {result['ec']['class']} (置信度: {result['ec']['confidence']:.3f})")
            print(f"  定位: {result['localization']['class']} (置信度: {result['localization']['confidence']:.3f})")
            print(f"  功能: {result['function']['class']} (置信度: {result['function']['confidence']:.3f})")

    elif args.sequence:
        # 单条推理
        print(f"\n序列: {args.sequence[:50]}...")

        result = predict(args.sequence, model, class_labels, esm2, device)

        print(f"\n预测结果:")
        print(f"  EC 主类: {result['ec']['class']}")
        print(f"    置信度: {result['ec']['confidence']:.3f}")
        print(f"    不确定性: {result['ec']['uncertainty']:.3f}")
        print(f"    Top-3: {result['ec']['top3']}")

        print(f"\n  细胞定位: {result['localization']['class']}")
        print(f"    置信度: {result['localization']['confidence']:.3f}")
        print(f"    不确定性: {result['localization']['uncertainty']:.3f}")
        print(f"    Top-3: {result['localization']['top3']}")

        print(f"\n  分子功能: {result['function']['class']}")
        print(f"    置信度: {result['function']['confidence']:.3f}")
        print(f"    不确定性: {result['function']['uncertainty']:.3f}")
        print(f"    Top-3: {result['function']['top3']}")

        results['single'] = result

    else:
        print("请提供 --sequence 或 --fasta 参数")
        return

    # 保存结果
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果保存到: {args.output}")


if __name__ == '__main__':
    main()
