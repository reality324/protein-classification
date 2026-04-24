#!/usr/bin/env python3
"""train.py - 单次训练脚本: 指定编码方式 + 指定算法

使用方法:
    python scripts/train.py --encoding ctd --algorithm rf
    python scripts/train.py --encoding esm2 --algorithm mlp --epochs 50
    python scripts/train.py --encoding ctd --algorithm bnn --save-model
"""
import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.encodings import EncoderRegistry
from src.algorithms import ClassifierRegistry
from src.pipeline import ProteinDataset, Trainer, Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description="蛋白质分类器: 单次训练")
    parser.add_argument("--encoding", "-e", type=str, default="ctd",
                        choices=["onehot", "ctd", "esm2"],
                        help="编码方式")
    parser.add_argument("--algorithm", "-a", type=str, default="rf",
                        choices=["rf", "xgb", "svm", "lr", "mlp", "bnn"],
                        help="分类算法")
    parser.add_argument("--data", "-d", type=str, default=None,
                        help="数据目录")
    parser.add_argument("--output", "-o", type=str, default="results/single_train",
                        help="输出目录")
    parser.add_argument("--save-model", action="store_true",
                        help="保存模型")
    parser.add_argument("--verbose", "-v", action="store_true", default=True)
    return parser.parse_args()


def load_data(encoding: str, data_path: str = None):
    """加载数据"""
    if encoding == "esm2":
        # 加载预计算的 ESM2 特征
        print("加载预计算 ESM2 特征...")
        dataset = ProteinDataset(encoding="esm2", task="ec")
        dataset.load_from_esm2_features(
            features_dir="data/processed/esm2_aligned",
            labels_parquet="data/datasets/train_subset.parquet",
        )
        return dataset
    elif data_path and Path(data_path).exists():
        print(f"从文件加载数据: {data_path}")
        dataset = ProteinDataset()
        dataset.load_from_files(data_path)
    else:
        # 生成模拟数据
        print("使用模拟数据进行演示...")
        np.random.seed(42)
        n_train, n_val, n_test = 1000, 200, 200
        n_classes = 8
        dim = EncoderRegistry.get(encoding).dim

        X_train = np.random.randn(n_train, dim)
        X_val = np.random.randn(n_val, dim)
        X_test = np.random.randn(n_test, dim)
        y_train = np.random.randint(0, n_classes, n_train)
        y_val = np.random.randint(0, n_classes, n_val)
        y_test = np.random.randint(0, n_classes, n_test)

        dataset = ProteinDataset()
        dataset.load_from_arrays(X_train, y_train, X_val, y_val, X_test, y_test)

    return dataset


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print(f"蛋白质分类器 - 单次训练")
    print(f"编码: {args.encoding}, 算法: {args.algorithm}")
    print("=" * 60 + "\n")

    # 加载数据
    dataset = load_data(args.encoding, args.data)
    print(dataset)

    # 获取分类器
    clf = ClassifierRegistry.get(args.algorithm)
    print(f"\n分类器: {clf.name}")

    # 训练
    trainer = Trainer(
        model=clf,
        save_dir=Path(args.output) / "models" if args.save_model else None,
        verbose=args.verbose,
    )

    X_train, y_train = dataset.get_train()
    X_val, y_val = dataset.get_val()
    X_test, y_test = dataset.get_test()

    result = trainer.train(X_train, y_train, X_val, y_val)

    # 评估
    if X_test is not None and y_test is not None:
        print("\n测试集评估:")
        evaluator = Evaluator()
        metrics = evaluator.evaluate_model(clf, X_test, y_test)
        evaluator.print_summary(metrics)

        # 保存结果
        if args.save_model:
            output_path = Path(args.output) / "evaluation_results.json"
            evaluator.save_results(metrics, str(output_path))

    print("\n训练完成!")


if __name__ == "__main__":
    main()
