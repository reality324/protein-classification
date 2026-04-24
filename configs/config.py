"""统一配置中心 - 所有路径、参数、可用编码和算法都集中管理"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ============== 基础路径 ==============
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "datasets"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATASETS_DIR = DATA_DIR / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# ============== 多任务分类配置 ==============
# 三个任务: EC主类预测 + 细胞定位预测 + 分子功能预测

MULTITASK_CONFIG = {
    "task_type": "multi-task",  # multi-task | single-task
    "tasks": {
        "ec": {
            "name": "EC主类",
            "num_classes": 6,
            "class_names": [
                "EC1-氧化还原酶",    # 1.x.x.x
                "EC2-转移酶",       # 2.x.x.x
                "EC3-水解酶",       # 3.x.x.x
                "EC4-裂解酶",       # 4.x.x.x
                "EC5-异构酶",       # 5.x.x.x
                "EC6-连接酶",       # 6.x.x.x
            ],
            "description": "酶学分类 - 按催化反应类型"
        },
        "localization": {
            "name": "细胞定位",
            "num_classes": 11,
            "class_names": [
                "Cell_Wall", "Cytoplasm", "ER", "Golgi",
                "Lysosome", "Membrane", "Mitochondria", "Nucleus",
                "Peroxisome", "Plasma", "Secreted"
            ],
            "description": "细胞内定位 - 蛋白质在细胞中的位置"
        },
        "function": {
            "name": "分子功能",
            "num_classes": 17,
            "class_names": [
                "Antioxidant", "Binding", "Enzyme", "Hydrolase",
                "Isomerase", "Kinase", "Ligase", "Lyase", "Motor",
                "Oxidoreductase", "Protease", "Signaling", "Structural",
                "Transcription", "Transferase", "Translocase", "Transporter"
            ],
            "description": "分子功能 - 蛋白质执行的功能"
        }
    }
}

# 兼容旧配置 - 默认使用 EC 主类任务
DATASET_CONFIG = {
    "name": "protein_classification",
    "num_classes": 6,  # EC1-6 六大类
    "task": "multiclass",
    "class_names": MULTITASK_CONFIG["tasks"]["ec"]["class_names"],
    "train_ratio": 0.6,
    "val_ratio": 0.2,
    "test_ratio": 0.2,
    "random_seed": 42,
}

# 任务列表
ALL_TASKS = list(MULTITASK_CONFIG["tasks"].keys())

# ============== 编码方式配置 ==============
@dataclass
class EncodingConfig:
    name: str
    dim: int
    description: str
    requires_gpu: bool = False
    requires_model_download: bool = False

ENCODINGS = {
    "onehot": EncodingConfig(
        name="onehot",
        dim=20,
        description="氨基酸单热编码 (Amino Acid Composition)",
    ),
    "ctd": EncodingConfig(
        name="ctd",
        dim=147,
        description="组成-转变-分布编码 (Composition-Transition-Distribution)",
    ),
    "esm2": EncodingConfig(
        name="esm2",
        dim=480,
        description="ESM2 预训练语言模型嵌入 (facebook/esm2_t6_8M_UR50D)",
        requires_gpu=True,
        requires_model_download=True,
    ),
}

DEFAULT_ENCODING = "ctd"

# ============== 分类算法配置 ==============
@dataclass
class AlgorithmConfig:
    name: str
    type: str  # "sklearn" | "pytorch"
    description: str
    supports_uncertainty: bool = False

ALGORITHMS = {
    "rf": AlgorithmConfig(
        name="rf",
        type="sklearn",
        description="Random Forest - 随机森林",
    ),
    "xgb": AlgorithmConfig(
        name="xgb",
        type="sklearn",
        description="XGBoost - 梯度提升树",
    ),
    "svm": AlgorithmConfig(
        name="svm",
        type="sklearn",
        description="SVM - 支持向量机 (RBF核)",
    ),
    "lr": AlgorithmConfig(
        name="lr",
        type="sklearn",
        description="Logistic Regression - 逻辑回归",
    ),
    "mlp": AlgorithmConfig(
        name="mlp",
        type="pytorch",
        description="MLP - 多层感知机",
    ),
    "bnn": AlgorithmConfig(
        name="bnn",
        type="pytorch",
        description="BNN - 贝叶斯神经网络 (MC Dropout)",
        supports_uncertainty=True,
    ),
}

DEFAULT_ALGORITHM = "rf"

# ============== 训练超参数 ==============
@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 15
    lr_scheduler: str = "plateau"  # plateau | cosine | none
    early_stopping: bool = True

    # 神经网络专用
    hidden_dims: list = field(default_factory=lambda: [256, 128])
    dropout: float = 0.3
    activation: str = "relu"

    # BNN专用
    mc_samples: int = 30  # Monte Carlo采样次数
    kl_weight: float = 0.1  # KL散度权重

    # 传统ML专用
    n_estimators: int = 200
    max_depth: Optional[int] = None
    cv_folds: int = 5

TRAIN_DEFAULTS = TrainConfig()

# ============== ESM2 模型配置 ==============
@dataclass
class ESM2Config:
    model_name: str = "facebook/esm2_t6_8M_UR50D"
    max_length: int = 1024
    pooling: str = "mean"  # mean | cls | max
    device: str = "cuda"  # cuda | cpu

ESM2_DEFAULTS = ESM2Config()

# ============== 实验结果保存 ==============
@dataclass
class ExperimentConfig:
    save_model: bool = True
    save_predictions: bool = True
    save_plots: bool = True
    save_metrics: bool = True
    verbose: bool = True

EXPERIMENT_DEFAULTS = ExperimentConfig()

# ============== 辅助函数 ==============
def get_encoding_dim(encoding_name: str) -> int:
    """获取指定编码方式的特征维度"""
    if encoding_name not in ENCODINGS:
        raise ValueError(f"Unknown encoding: {encoding_name}. Available: {list(ENCODINGS.keys())}")
    return ENCODINGS[encoding_name].dim

def get_algorithm_type(algorithm_name: str) -> str:
    """获取指定算法的框架类型"""
    if algorithm_name not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm_name}. Available: {list(ALGORITHMS.keys())}")
    return ALGORITHMS[algorithm_name].type

def ensure_dirs():
    """确保所有必要目录存在"""
    for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
