"""
数据预处理模块
包括: 标签编码、数据清洗、数据集划分
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder as SklearnLabelEncoder, MultiLabelBinarizer
import joblib

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, DATASETS_DIR


class BaseLabelEncoder:
    """标签编码器基类"""
    
    def __init__(self):
        self.classes_ = None
        self.class_to_idx = None
        self.idx_to_class = None
    
    def fit(self, labels):
        """拟合编码器"""
        raise NotImplementedError
    
    def transform(self, labels):
        """转换标签"""
        raise NotImplementedError
    
    def inverse_transform(self, encoded):
        """逆转换"""
        raise NotImplementedError
    
    def save(self, path: Path):
        """保存编码器"""
        joblib.dump(self, path)
    
    @classmethod
    def load(cls, path: Path):
        """加载编码器"""
        return joblib.load(path)


class ECNumberEncoder(BaseLabelEncoder):
    """EC 编号编码器 (支持层级结构)"""
    
    def __init__(self, min_depth: int = 2):
        super().__init__()
        self.min_depth = min_depth
        self.all_ecs = []
    
    def fit(self, ec_strings: pd.Series):
        """拟合编码器
        
        EC 号格式: "1.2.3.4" 四个层级
        - 层级1: 酶的类型 (氧化还原酶, 转移酶, 水解酶, 裂解酶, 异构酶, 连接酶)
        - 层级2: 底物类型
        - 层级3: 反应类型
        - 层级4: 具体底物
        """
        # 收集所有 EC 号
        all_ecs = set()
        for ec_str in ec_strings:
            if pd.isna(ec_str) or ec_str == '':
                continue
            for ec in ec_str.split(','):
                ec = ec.strip()
                if not ec:
                    continue
                # 根据层级截取
                parts = ec.split('.')
                if len(parts) >= self.min_depth:
                    # 保留到指定层级
                    ec_truncated = '.'.join(parts[:self.min_depth])
                    all_ecs.add(ec_truncated)
        
        # 排序并编码
        self.all_ecs = sorted(list(all_ecs))
        self.classes_ = self.all_ecs
        self.class_to_idx = {ec: idx for idx, ec in enumerate(self.all_ecs)}
        self.idx_to_class = {idx: ec for ec, idx in self.class_to_idx.items()}
        
        return self
    
    def transform(self, ec_strings: pd.Series) -> np.ndarray:
        """转换 EC 字符串为多标签编码"""
        n_samples = len(ec_strings)
        n_classes = len(self.classes_)
        
        # 多标签二进制编码
        encoded = np.zeros((n_samples, n_classes), dtype=np.float32)
        
        for i, ec_str in enumerate(ec_strings):
            if pd.isna(ec_str) or ec_str == '':
                continue
            for ec in ec_str.split(','):
                ec = ec.strip()
                if not ec:
                    continue
                parts = ec.split('.')
                if len(parts) >= self.min_depth:
                    ec_truncated = '.'.join(parts[:self.min_depth])
                    if ec_truncated in self.class_to_idx:
                        encoded[i, self.class_to_idx[ec_truncated]] = 1
        
        return encoded
    
    def get_hierarchy_info(self) -> Dict[str, List[str]]:
        """获取 EC 层级信息"""
        hierarchy = {
            'class1': [],  # 氧化还原酶
            'class2': [],  # 转移酶
            'class3': [],  # 水解酶
            'class4': [],  # 裂解酶
            'class5': [],  # 异构酶
            'class6': [],  # 连接酶
        }
        
        for ec in self.classes_:
            main_class = ec.split('.')[0]
            class_key = f'class{main_class}'
            if class_key in hierarchy:
                hierarchy[class_key].append(ec)
        
        return hierarchy


class LocalizationEncoder(BaseLabelEncoder):
    """细胞定位编码器"""
    
    # 标准化的定位分类
    LOCATION_CATEGORIES = {
        # 细胞核
        'Nucleus': ['nucleus', 'nuclear', 'nucleolus', 'chromosome'],
        # 细胞质
        'Cytoplasm': ['cytoplasm', 'cytosol', 'cytoplasmatic'],
        # 线粒体
        'Mitochondria': ['mitochondrion', 'mitochondrial'],
        # 细胞膜
        'Membrane': ['membrane', 'cell membrane', 'plasma membrane', 'cell surface'],
        # 内质网
        'Endoplasmic reticulum': ['endoplasmic reticulum', 'er membrane'],
        # 高尔基体
        'Golgi apparatus': ['golgi', 'golgi apparatus'],
        # 溶酶体
        'Lysosome': ['lysosome', 'lysosomal'],
        # 过氧化物酶体
        'Peroxisome': ['peroxisome', 'peroxisomal'],
        # 分泌/细胞外
        'Secreted': ['secreted', 'secretory', 'extracellular', 'extracellular space', 'cell wall'],
        # 核糖体
        'Ribosome': ['ribosome', 'ribosomal'],
        # 溶血
        'Cytoskeleton': ['cytoskeleton', 'actin', 'microtubule', 'intermediate filament'],
        # 内体
        'Endosome': ['endosome'],
        # 质膜
        'Cell junction': ['cell junction', 'synapse', 'synaptic'],
    }
    
    def fit(self, location_strings: pd.Series):
        """拟合编码器"""
        # 标准化定位名称
        standardized = []
        for loc in location_strings:
            if pd.isna(loc) or loc == '':
                standardized.append('Unknown')
                continue
            
            loc_lower = loc.lower()
            matched = False
            
            for standard_name, keywords in self.LOCATION_CATEGORIES.items():
                for kw in keywords:
                    if kw in loc_lower:
                        standardized.append(standard_name)
                        matched = True
                        break
                if matched:
                    break
            
            if not matched:
                # 如果没有匹配，提取第一个主要词
                main_loc = loc.split(';')[0].strip()
                if main_loc:
                    standardized.append(main_loc)
                else:
                    standardized.append('Unknown')
        
        # 对标准化后的标签进行编码
        unique_locations = sorted(set(standardized))
        self.classes_ = unique_locations
        self.class_to_idx = {loc: idx for idx, loc in enumerate(unique_locations)}
        self.idx_to_class = {idx: loc for loc, idx in self.class_to_idx.items()}
        
        return self
    
    def transform(self, location_strings: pd.Series) -> np.ndarray:
        """转换为单标签编码"""
        n_samples = len(location_strings)
        n_classes = len(self.classes_)
        
        encoded = np.zeros(n_samples, dtype=np.int32)
        
        for i, loc in enumerate(location_strings):
            if pd.isna(loc) or loc == '':
                encoded[i] = self.class_to_idx.get('Unknown', 0)
                continue
            
            loc_lower = loc.lower()
            matched = False
            
            for standard_name, keywords in self.LOCATION_CATEGORIES.items():
                for kw in keywords:
                    if kw in loc_lower:
                        encoded[i] = self.class_to_idx[standard_name]
                        matched = True
                        break
                if matched:
                    break
            
            if not matched:
                main_loc = loc.split(';')[0].strip()
                encoded[i] = self.class_to_idx.get(main_loc, self.class_to_idx.get('Unknown', 0))
        
        return encoded


class FunctionEncoder(BaseLabelEncoder):
    """蛋白质功能编码器 (基于 Keywords)"""
    
    def __init__(self, top_k: int = 50):
        super().__init__()
        self.top_k = top_k
        self.mlb = MultiLabelBinarizer()
    
    def fit(self, keywords_strings: pd.Series):
        """拟合编码器 - 选择最常见的 top_k 功能"""
        # 统计功能频率
        func_counts = {}
        for kw_str in keywords_strings:
            if pd.isna(kw_str) or kw_str == '':
                continue
            for kw in kw_str.split(','):
                kw = kw.strip()
                if kw:
                    func_counts[kw] = func_counts.get(kw, 0) + 1
        
        # 选择 top_k
        top_funcs = sorted(func_counts.items(), key=lambda x: x[1], reverse=True)[:self.top_k]
        self.classes_ = [f[0] for f in top_funcs]
        self.class_to_idx = {f: idx for idx, f in enumerate(self.classes_)}
        self.idx_to_class = {idx: f for f, idx in self.class_to_idx.items()}
        
        return self
    
    def transform(self, keywords_strings: pd.Series) -> np.ndarray:
        """转换为多标签编码"""
        # 构建多标签列表
        labels = []
        for kw_str in keywords_strings:
            if pd.isna(kw_str) or kw_str == '':
                labels.append([])
                continue
            
            kws = [kw.strip() for kw in kw_str.split(',') if kw.strip() in self.class_to_idx]
            labels.append(kws)
        
        return self.mlb.fit_transform(labels)


class ProteinDataProcessor:
    """蛋白质数据处理器"""
    
    def __init__(self):
        self.ec_encoder = ECNumberEncoder(min_depth=3)  # EC号保留到第三层
        self.loc_encoder = LocalizationEncoder()
        self.func_encoder = FunctionEncoder(top_k=50)
        
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'ProteinDataProcessor':
        """拟合所有编码器"""
        print("拟合标签编码器...")
        
        # EC 编码器
        self.ec_encoder.fit(df['ec_number'])
        print(f"  EC 类别数: {len(self.ec_encoder.classes_)}")
        
        # 定位编码器
        self.loc_encoder.fit(df['location'])
        print(f"  定位类别数: {len(self.loc_encoder.classes_)}")
        
        # 功能编码器
        self.func_encoder.fit(df['keywords'])
        print(f"  功能类别数: {len(self.func_encoder.classes_)}")
        
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换数据"""
        if not self.is_fitted:
            raise ValueError("需要先调用 fit() 方法")
        
        # 编码标签
        df['ec_encoded'] = list(self.ec_encoder.transform(df['ec_number']))
        df['loc_encoded'] = self.loc_encoder.transform(df['location'])
        df['func_encoded'] = list(self.func_encoder.transform(df['keywords']))
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """拟合并转换"""
        return self.fit(df).transform(df)
    
    def save_encoders(self, output_dir: Path = None):
        """保存编码器"""
        output_dir = output_dir or DATASETS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.ec_encoder.save(output_dir / 'ec_encoder.joblib')
        self.loc_encoder.save(output_dir / 'loc_encoder.joblib')
        self.func_encoder.save(output_dir / 'func_encoder.joblib')
        
        # 保存类别信息
        np.save(output_dir / 'ec_classes.npy', self.ec_encoder.classes_)
        np.save(output_dir / 'loc_classes.npy', np.array(self.loc_encoder.classes_))
        np.save(output_dir / 'func_classes.npy', self.func_encoder.classes_)
        
        print(f"编码器已保存到: {output_dir}")
    
    @classmethod
    def load_encoders(cls, input_dir: Path = None) -> 'ProteinDataProcessor':
        """加载编码器"""
        input_dir = input_dir or DATASETS_DIR
        
        processor = cls()
        processor.ec_encoder = joblib.load(input_dir / 'ec_encoder.joblib')
        processor.loc_encoder = joblib.load(input_dir / 'loc_encoder.joblib')
        processor.func_encoder = joblib.load(input_dir / 'func_encoder.joblib')
        processor.is_fitted = True
        
        return processor


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """划分训练/验证/测试集
    
    分层抽样确保各集合类别分布一致
    """
    # 先划分测试集
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    
    # 再划分验证集
    val_size_adjusted = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size_adjusted, random_state=random_state
    )
    
    print(f"数据集划分完成:")
    print(f"  训练集: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  验证集: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  测试集: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df


def preprocess_pipeline(input_file: Path = None) -> Dict[str, pd.DataFrame]:
    """完整的预处理流程
    
    Returns:
        dict with keys: 'train', 'val', 'test'
    """
    # 1. 加载数据
    if input_file is None:
        input_file = RAW_DATA_DIR / "protein_data_raw.parquet"
    
    print(f"加载数据: {input_file}")
    df = pd.read_parquet(input_file)
    print(f"原始数据: {len(df)} 条")
    
    # 2. 过滤有效数据
    # 必须有至少一个标签
    df = df[
        (df['ec_number'] != '') | 
        (df['location'] != '') | 
        (df['keywords'] != '')
    ]
    df = df[df['sequence'].str.len() > 10]  # 过滤短序列
    df = df.drop_duplicates(subset=['id'])
    
    print(f"过滤后数据: {len(df)} 条")
    
    # 3. 处理编码
    processor = ProteinDataProcessor()
    df = processor.fit_transform(df)
    
    # 4. 划分数据集
    train_df, val_df, test_df = split_dataset(df)
    
    # 5. 保存编码器
    processor.save_encoders()
    
    # 6. 保存数据集
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(DATASETS_DIR / 'train.parquet', index=False)
    val_df.to_parquet(DATASETS_DIR / 'val.parquet', index=False)
    test_df.to_parquet(DATASETS_DIR / 'test.parquet', index=False)
    
    print("预处理完成!")
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df,
        'processor': processor
    }


if __name__ == "__main__":
    preprocess_pipeline()
