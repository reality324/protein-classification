#!/usr/bin/env python3
"""
MLP 多任务分类器推理脚本
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json

from src.encodings import EncoderRegistry


class MultiTaskMLP(nn.Module):
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


def predict(sequence, model, class_labels, encoder, device):
    features = encoder.encode(sequence).reshape(1, -1)
    X = torch.FloatTensor(features).to(device)
    model.eval()
    with torch.no_grad():
        ec_out, loc_out, func_out = model(X)
        ec_prob = torch.softmax(ec_out, dim=1)[0].cpu().numpy()
        loc_prob = torch.softmax(loc_out, dim=1)[0].cpu().numpy()
        func_prob = torch.softmax(func_out, dim=1)[0].cpu().numpy()

    ec_classes = class_labels['ec_classes']
    loc_classes = class_labels['loc_classes']
    func_classes = class_labels['func_classes']

    return {
        'ec': {'class': ec_classes[np.argmax(ec_prob)], 'confidence': float(np.max(ec_prob)),
               'top3': [(ec_classes[i], float(ec_prob[i])) for i in np.argsort(ec_prob)[-3:][::-1]]},
        'localization': {'class': loc_classes[np.argmax(loc_prob)], 'confidence': float(np.max(loc_prob)),
                         'top3': [(loc_classes[i], float(loc_prob[i])) for i in np.argsort(loc_prob)[-3:][::-1]]},
        'function': {'class': func_classes[np.argmax(func_prob)], 'confidence': float(np.max(func_prob)),
                     'top3': [(func_classes[i], float(func_prob[i])) for i in np.argsort(func_prob)[-3:][::-1]]},
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='MLP 多任务推理')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--encoding', type=str, default='esm2', choices=['onehot', 'ctd', 'esm2'],
                        help='特征编码方式')
    parser.add_argument('--sequence', type=str, default=None)
    parser.add_argument('--fasta', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    if args.model is None:
        args.model = f'models/mlp_{args.encoding}_multitask/model.pt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    ec_classes = checkpoint['ec_classes']
    loc_classes = checkpoint['loc_classes']
    func_classes = checkpoint['func_classes']
    input_dim = checkpoint['input_dim']
    hidden_dims = checkpoint.get('hidden_dims', [512, 256])
    dropout = checkpoint.get('dropout', 0.3)

    model = MultiTaskMLP(input_dim, len(ec_classes), len(loc_classes), len(func_classes),
                         hidden_dims, dropout).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    class_labels = {'ec_classes': ec_classes, 'loc_classes': loc_classes, 'func_classes': func_classes}
    print(f"使用编码器: {args.encoding}")
    encoder = EncoderRegistry.get(args.encoding)

    if args.fasta:
        with open(args.fasta) as f:
            content = f.read()
        sequences, names = [], []
        current_name, current_seq = None, []
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

        for name, seq in zip(names, sequences):
            result = predict(seq, model, class_labels, encoder, device)
            print(f"\n{name}:")
            print(f"  EC: {result['ec']['class']} ({result['ec']['confidence']:.3f})")
            print(f"  Loc: {result['localization']['class']} ({result['localization']['confidence']:.3f})")
            print(f"  Func: {result['function']['class']} ({result['function']['confidence']:.3f})")

    elif args.sequence:
        result = predict(args.sequence, model, class_labels, encoder, device)
        print(f"\n预测结果:")
        print(f"  EC: {result['ec']['class']} ({result['ec']['confidence']:.3f})")
        print(f"  定位: {result['localization']['class']} ({result['localization']['confidence']:.3f})")
        print(f"  功能: {result['function']['class']} ({result['function']['confidence']:.3f})")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result if args.sequence else {}, f, indent=2)
        print(f"\n保存到: {args.output}")


if __name__ == '__main__':
    main()
