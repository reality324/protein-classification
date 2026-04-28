#!/usr/bin/env python3
"""
XGBoost 多任务分类器推理脚本
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pickle
import json
from pathlib import Path

from src.encodings.esm2 import ESM2Encoder


def predict(sequence, model_data, esm2_encoder):
    features = esm2_encoder.encode(sequence).reshape(1, -1)

    ec_prob = model_data['ec_clf'].predict_proba(features)[0]
    loc_prob = model_data['loc_clf'].predict_proba(features)[0]
    func_prob = model_data['func_clf'].predict_proba(features)[0]

    ec_pred = ec_prob.argmax()
    loc_pred = loc_prob.argmax()
    func_pred = func_prob.argmax()

    return {
        'ec': {
            'class': model_data['ec_classes'][ec_pred],
            'confidence': float(ec_prob[ec_pred]),
            'top3': [(model_data['ec_classes'][i], float(ec_prob[i])) for i in np.argsort(ec_prob)[-3:][::-1]]
        },
        'localization': {
            'class': model_data['loc_classes'][loc_pred],
            'confidence': float(loc_prob[loc_pred]),
            'top3': [(model_data['loc_classes'][i], float(loc_prob[i])) for i in np.argsort(loc_prob)[-3:][::-1]]
        },
        'function': {
            'class': model_data['func_classes'][func_pred],
            'confidence': float(func_prob[func_pred]),
            'top3': [(model_data['func_classes'][i], float(func_prob[i])) for i in np.argsort(func_prob)[-3:][::-1]]
        },
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='XGBoost 多任务推理')
    parser.add_argument('--model', type=str, default='models/xgb_esm2_multitask/model.pkl')
    parser.add_argument('--sequence', type=str, default=None)
    parser.add_argument('--fasta', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    print(f"加载模型: {args.model}")
    with open(args.model, 'rb') as f:
        model_data = pickle.load(f)

    esm2 = ESM2Encoder(pooling='mean', device='cpu')

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
            result = predict(seq, model_data, esm2)
            print(f"\n{name}:")
            print(f"  EC: {result['ec']['class']} ({result['ec']['confidence']:.3f})")
            print(f"  Loc: {result['localization']['class']} ({result['localization']['confidence']:.3f})")
            print(f"  Func: {result['function']['class']} ({result['function']['confidence']:.3f})")

    elif args.sequence:
        result = predict(args.sequence, model_data, esm2)
        print(f"\n预测结果:")
        print(f"  EC: {result['ec']['class']} ({result['ec']['confidence']:.3f})")
        print(f"  定位: {result['localization']['class']} ({result['localization']['confidence']:.3f})")
        print(f"  功能: {result['function']['class']} ({result['function']['confidence']:.3f})")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result if args.sequence else {}, f, indent=2, ensure_ascii=False)
        print(f"\n保存到: {args.output}")


if __name__ == '__main__':
    main()
