# ProteinClassifier - 多任务蛋白质分类器

一个基于深度学习的多任务蛋白质分类框架，同时预测:
- **EC Number** - 酶催化功能分类 (多标签)
- **蛋白质功能** - Gene Ontology 功能注释 (多标签)
- **细胞定位** - Subcellular Localization (多分类)

---

## 📁 项目结构

```
ProteinClassifier/
├── configs/
│   └── config.py              # 配置文件
├── data/
│   ├── raw/                   # 原始数据
│   ├── processed/             # 处理后数据
│   └── datasets/              # 训练/验证/测试集
├── src/
│   ├── data/
│   │   ├── download.py        # UniProt 数据下载
│   │   ├── preprocessing.py   # 标签编码
│   │   ├── featurization.py   # 特征提取
│   │   ├── dataset.py         # PyTorch Dataset
│   │   └── augmentation.py    # 数据增强
│   ├── models/
│   │   └── multi_task_model.py # 多任务分类器
│   └── utils/
│       ├── metrics.py         # 评估指标
│       └── visualization.py   # 可视化工具
├── scripts/
│   ├── prepare_data.py       # 数据准备脚本
│   ├── train.py              # 训练脚本
│   ├── predict.py           # 预测脚本
│   ├── validate_data.py      # 数据验证脚本
│   └── quick_predict.py      # 快速预测
├── notebooks/                # Jupyter notebooks
├── env.yaml                 # conda 环境配置
└── requirements.txt
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
cd /home/tianwangcong/ProteinClassifier

# 使用 conda 创建环境
conda env create -f env.yaml
conda activate protein_classifier

# 或使用 pip
pip install -r requirements.txt
```

### 2. 准备数据

```bash
# 完整数据准备流程
python scripts/prepare_data.py

# 只提取特征 (使用 ESM2)
python scripts/prepare_data.py --extract_features --embedding esm2_8M
```

### 3. 训练模型

```bash
# 使用 One-Hot 特征 (CPU 快速训练)
python scripts/train.py --embedding onehot --epochs 100

# 使用 ESM2 嵌入 (需要 GPU)
python scripts/train.py --embedding esm2_8M --epochs 100 --batch_size 32
```

### 4. 预测

```bash
# 从 FASTA 文件预测
python scripts/predict.py --input proteins.fasta --output results.tsv --fasta

# 单条序列预测
python scripts/predict.py --input "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"

# 快速预测
python scripts/quick_predict.py "YOUR_SEQUENCE"
```

---

## 📊 三个分类任务

| 任务 | 类型 | 数据来源 | 分类数 | 说明 |
|------|------|----------|--------|------|
| **EC Number** | 多标签 | UniProt | ~5000+ | 酶催化功能层级分类 |
| **蛋白质功能** | 多标签 | UniProt Keywords | ~50 | GO-like 功能注释 |
| **细胞定位** | 多分类 | UniProt Subcellular | ~30 | 亚细胞定位 |

### EC Number 层级结构

```
EC: 1.2.3.4
│ │ │ │
│ │ │ └── 第四层: 具体底物
│ │ └──── 第三层: 反应类型
│ └─────── 第二层: 底物类型
└───────── 第一层: 酶的类型 (1-6)
```

---

## 🔧 特征提取方法

| 方法 | 维度 | 速度 | 说明 |
|------|------|------|------|
| **One-Hot** | 20 | ⚡ 极快 | 基础方法，适合快速验证 |
| **ESM2-8M** | 320 | ⚡ 快 | 小型语言模型，推荐 |
| **ESM2-35M** | 480 | 🕐 中等 | 中型模型，效果更好 |
| **ESM2-150M** | 640 | 🐢 慢 | 大型模型，效果最好 |
| **ProtBERT** | 1024 | 🐢 慢 | BERT 风格嵌入 |

---

## 🧠 模型架构

```
Input (蛋白质序列, 长度 L)
    ↓
特征提取器 (ESM2/One-Hot)
    ↓
特征向量 (batch_size, dim)
    ↓
共享特征层 [dim → 512 → 256]
    ↓
┌─────────────┬─────────────┬─────────────┐
│   EC Head   │  Loc Head   │ Func Head   │
│  (多标签)   │  (多分类)   │  (多标签)   │
│  [256→500]  │  [256→30]   │  [256→50]   │
└─────────────┴─────────────┴─────────────┘
```

---

## 📈 评估指标

### EC Number / 蛋白质功能 (多标签)
- Precision (Macro)
- Recall (Macro)
- F1 Score (Macro)
- AUC-ROC (Macro)

### 细胞定位 (多分类)
- Accuracy
- Top-1 / Top-3 Accuracy
- F1 Score (Macro)
- Confusion Matrix

---

## 🔧 工具脚本

| 脚本 | 功能 |
|------|------|
| `prepare_data.py` | 完整数据准备流程 |
| `train.py` | 模型训练 |
| `predict.py` | 批量预测 |
| `quick_predict.py` | 命令行快速预测 |
| `validate_data.py` | 数据集验证和可视化 |

---

## 📖 使用示例

### Python API

```python
from src.models.multi_task_model import create_model
from src.data.dataset import ProteinDataset
from src.data.featurization import get_feature_extractor
from src.utils.metrics import Evaluator

# 创建模型
model = create_model(
    input_dim=320,
    ec_num_classes=500,
    loc_num_classes=30,
    func_num_classes=50,
)

# 提取特征
extractor = get_feature_extractor('esm2_8M')
features = extractor.extract(['MVLSPADKTNVKAAWG...'])

# 预测
model.eval()
with torch.no_grad():
    outputs = model(torch.FloatTensor(features))
```

### 数据增强

```python
from src.data.augmentation import ProteinAugmenter

augmenter = ProteinAugmenter(seed=42)

# 单序列增强
augmented = augmenter.augment(
    'MVLSPADKTNVKAAWG...',
    methods=['mutate', 'swap'],
    n_augments=5
)
```

---

## 🔬 数据来源

- **UniProt** - 蛋白质序列和功能注释
  - https://www.uniprot.org
- **Gene Ontology** - 功能分类标准
  - http://geneontology.org
- **CAFA** - 蛋白质功能预测基准
  - https://www.bioai.dk/ccg/cafa2/

---

## 📝 配置文件

```python
# configs/config.py
ROOT_DIR = "/path/to/ProteinClassifier"

# 嵌入方法配置
MODEL_CONFIGS = {
    'esm2_8M': {'embedding_dim': 320, 'max_length': 1024},
    'esm2_35M': {'embedding_dim': 480, 'max_length': 1024},
}

# 训练配置
TRAIN_CONFIG = {
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 1e-4,
}
```

---

## ⚠️ 注意事项

1. **GPU 要求**: 使用 ESM2/ProtBERT 需要 CUDA GPU
2. **内存**: 大规模训练需要足够的显存
3. **数据量**: 建议至少 10 万条蛋白质序列

---

## 📚 参考项目和引用

本项目参考了以下工作：

### EC Number 预测

EC Number 预测模块参考了 ECRECer 项目：

```tex
@article{shi2023enzyme,
  title={Enzyme Commission Number Prediction and Benchmarking with Hierarchical Dual-core Multitask Learning Framework},
  author={Shi, Zhenkun and Deng, Rui and Yuan, Qianqian and Mao, Zhitao and Wang, Ruoyu and Li, Haoran and Liao, Xiaoping and Ma, Hongwu},
  journal={Research},
  year={2023},
  publisher={AAAS}
}
```

### 细胞定位预测

细胞定位预测模块参考了 Amazon SageMaker 的蛋白质分类项目：

```tex
@misc{aws-sagemaker-protein-classification,
  title={Fine-tuning and deploying ProtBert Model for Protein Classification using Amazon SageMaker},
  author={AWS Samples},
  year={2021},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={\url{https://github.com/aws-samples/amazon-sagemaker-protein-classification}},
  url={https://github.com/aws-samples/amazon-sagemaker-protein-classification}
}
```

### 其他参考项目

- [ECRECer](https://github.com/kingstdio/ECRECer) - EC Number 预测
- [ESM](https://github.com/facebookresearch/esm) - 蛋白质语言模型
- [CAFA](https://www.bioai.dk/ccg/cafa2/) - 蛋白质功能预测

---

## 📄 License

MIT License
