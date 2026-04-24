"""数据集模块 - 支持 ESM2 预计算特征 + 多任务标签"""
import numpy as np
import pandas as pd
from pathlib import Path

from configs.config import ALL_TASKS


def extract_ec_main_class(df: pd.DataFrame) -> np.ndarray:
    """从 one-hot EC 标签提取主类 (EC1-7)"""
    ec_cols = [c for c in df.columns if c.startswith('ec_') and c[3:].replace('.', '').isdigit()]
    ec_values = df[ec_cols].values
    ec_indices = np.argmax(ec_values, axis=1)
    main_classes = []
    for idx in ec_indices:
        col_name = ec_cols[idx]
        main_class = int(col_name.split('_')[1].split('.')[0])
        main_classes.append(main_class - 1)  # 0-indexed
    return np.array(main_classes)


def extract_localization(df: pd.DataFrame) -> np.ndarray:
    """从 one-hot Localization 标签提取类别索引"""
    loc_cols = [c for c in df.columns if c.startswith('loc_')]
    loc_values = df[loc_cols].values
    return np.argmax(loc_values, axis=1)


def extract_function(df: pd.DataFrame) -> np.ndarray:
    """从 one-hot Function 标签提取类别索引"""
    func_cols = [c for c in df.columns if c.startswith('func_')]
    func_values = df[func_cols].values
    return np.argmax(func_values, axis=1)


class ProteinDataset:
    """统一蛋白质数据集"""

    def __init__(self, encoding: str = "esm2", task: str = "multi-task"):
        self.encoding = encoding
        self.task = task
        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None
        self.y_train_dict = self.y_val_dict = self.y_test_dict = None

    def load_from_esm2_features(
        self,
        features_dir: str = "data/processed/esm2_aligned",
        labels_parquet: str = "data/datasets/train_subset.parquet",
    ):
        """加载 ESM2 特征和标签

        特征和标签必须来自同一个 parquet 文件，使用相同的随机种子划分
        """
        features_dir = Path(features_dir)
        labels_parquet = Path(labels_parquet)

        print(f"[ProteinDataset] 加载 ESM2 特征: {features_dir}")
        X_train = np.load(features_dir / "train_features.npy")
        X_val = np.load(features_dir / "val_features.npy")
        X_test = np.load(features_dir / "test_features.npy")

        print(f"[ProteinDataset] 加载标签: {labels_parquet}")
        df = pd.read_parquet(labels_parquet)

        # 使用与特征生成相同的随机种子和划分
        indices = np.arange(len(df))
        np.random.seed(42)
        np.random.shuffle(indices)

        n = len(df)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        # 对齐特征和标签
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test

        # 提取标签
        if self.task == "multi-task":
            self.y_train_dict = {
                "ec": extract_ec_main_class(df.iloc[train_idx]),
                "localization": extract_localization(df.iloc[train_idx]),
                "function": extract_function(df.iloc[train_idx]),
            }
            self.y_val_dict = {
                "ec": extract_ec_main_class(df.iloc[val_idx]),
                "localization": extract_localization(df.iloc[val_idx]),
                "function": extract_function(df.iloc[val_idx]),
            }
            self.y_test_dict = {
                "ec": extract_ec_main_class(df.iloc[test_idx]),
                "localization": extract_localization(df.iloc[test_idx]),
                "function": extract_function(df.iloc[test_idx]),
            }
            self.y_train = self.y_train_dict["ec"]

            print(f"[ProteinDataset] EC: {len(np.unique(self.y_train_dict['ec']))}类, "
                  f"Loc: {len(np.unique(self.y_train_dict['localization']))}类, "
                  f"Func: {len(np.unique(self.y_train_dict['function']))}类")
        else:
            if self.task == "ec":
                self.y_train = extract_ec_main_class(df.iloc[train_idx])
                self.y_val = extract_ec_main_class(df.iloc[val_idx])
                self.y_test = extract_ec_main_class(df.iloc[test_idx])
            elif self.task == "localization":
                self.y_train = extract_localization(df.iloc[train_idx])
                self.y_val = extract_localization(df.iloc[val_idx])
                self.y_test = extract_localization(df.iloc[test_idx])
            else:
                self.y_train = extract_function(df.iloc[train_idx])
                self.y_val = extract_function(df.iloc[val_idx])
                self.y_test = extract_function(df.iloc[test_idx])

        print(f"[ProteinDataset] 特征: train={len(self.X_train)}, val={len(self.X_val)}, test={len(self.X_test)}")
        print(f"[ProteinDataset] 特征维度: {self.X_train.shape[1]}")
        print(f"[ProteinDataset] 加载完成!")
        return self

    def load_from_parquet(self, parquet_path: str, encoding: str = "ctd"):
        """从 Parquet 文件加载并编码"""
        from ..encodings import EncoderRegistry

        print(f"[ProteinDataset] 从 {parquet_path} 加载数据...")
        df = pd.read_parquet(parquet_path)
        print(f"[ProteinDataset] 原始数据: {len(df)} 条")

        indices = np.arange(len(df))
        np.random.seed(42)
        np.random.shuffle(indices)

        n = len(df)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        # 标签
        if self.task == "multi-task":
            self.y_train_dict = {
                "ec": extract_ec_main_class(df.iloc[train_idx]),
                "localization": extract_localization(df.iloc[train_idx]),
                "function": extract_function(df.iloc[train_idx]),
            }
            self.y_val_dict = {
                "ec": extract_ec_main_class(df.iloc[val_idx]),
                "localization": extract_localization(df.iloc[val_idx]),
                "function": extract_function(df.iloc[val_idx]),
            }
            self.y_test_dict = {
                "ec": extract_ec_main_class(df.iloc[test_idx]),
                "localization": extract_localization(df.iloc[test_idx]),
                "function": extract_function(df.iloc[test_idx]),
            }
            self.y_train = self.y_train_dict["ec"]
        else:
            if self.task == "ec":
                labels = extract_ec_main_class(df)
            elif self.task == "localization":
                labels = extract_localization(df)
            else:
                labels = extract_function(df)
            self.y_train = labels[train_idx]
            self.y_val = labels[val_idx]
            self.y_test = labels[test_idx]

        # 编码
        print(f"[ProteinDataset] 编码序列 (使用 {encoding})...")
        encoder = EncoderRegistry.get(encoding)
        self.X_train = encoder.encode_batch(df['sequence'].iloc[train_idx].tolist())
        self.X_val = encoder.encode_batch(df['sequence'].iloc[val_idx].tolist())
        self.X_test = encoder.encode_batch(df['sequence'].iloc[test_idx].tolist())

        print(f"[ProteinDataset] 特征维度: {self.X_train.shape[1]}")
        print(f"[ProteinDataset] 划分: train={len(self.X_train)}, val={len(self.X_val)}, test={len(self.X_test)}")
        return self

    @property
    def input_dim(self):
        return self.X_train.shape[1] if self.X_train is not None else 0

    def load_from_arrays(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None):
        """从 numpy 数组加载数据"""
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val if X_val is not None else None
        self.y_val = y_val if y_val is not None else None
        self.X_test = X_test if X_test is not None else None
        self.y_test = y_test if y_test is not None else None
        return self

    def get_train(self):
        return self.X_train, self.y_train

    def get_val(self):
        return self.X_val, self.y_val

    def get_test(self):
        return self.X_test, self.y_test

    def __repr__(self):
        return f"ProteinDataset(encoding={self.encoding}, task={self.task}, dim={self.input_dim})"
