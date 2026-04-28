# ProteinClassifier

蛋白质功能预测项目 - 支持多种机器学习算法

## 项目简介

ProteinClassifier 是一个用于预测蛋白质功能的机器学习工具包，支持多种编码方式和多种机器学习算法。项目特点：

- **多种编码方式**: OneHot、CTD、ESM2 蛋白质语言模型特征
- **多种算法**: RandomForest、XGBoost、MLP、贝叶斯神经网络
- **多任务学习**: 支持同时预测 EC 分类、细胞定位、分子功能
- **灵活的推理**: 支持单序列推理和批量 FASTA 文件推理

## 环境配置

### 依赖环境

```yaml
# 环境配置文件: env.yaml
python: >=3.8
pytorch: >=1.9
scikit-learn: >=1.0
xgboost: >=1.5
pandas: >=1.3
numpy: >=1.20
```

### 安装方式

```bash
# 1. 克隆项目
git clone https://github.com/reality324/protein-classifier.git
cd ProteinClassifier

# 2. 创建 conda 环境
conda env create -f env.yaml
conda activate protein-classifier

# 3. 安装 ESM2 (用于生成蛋白质嵌入特征)
pip install fair-esm
```

## 数据集来源

本项目使用的数据集来源于 **UniProt Swiss-Prot** 数据库。

### 数据源信息

| 属性 | 说明 |
|------|------|
| **数据库** | UniProt Swiss-Prot |
| **官网** | https://www.uniprot.org |
| **FTP 下载** | `https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/` |
| **原始文件** | `uniprot_sprot.dat.gz` (~570,000 条) |
| **处理后数据** | `data/datasets/train_subset.parquet` (**5,000 条**) |
| **数据质量** | 人工审查 (Reviewed)，最可靠的蛋白质注释 |

### 本地数据文件

```
ProteinClassifier/data/datasets/
├── protein_with_go.parquet      # 完整数据集 (~5万条)
├── train_subset.parquet         # 训练子集 (5000条)
├── uniprot_parsed.parquet      # 解析后的UniProt数据
└── balanced_with_go/            # 平衡后的数据集
    ├── train.parquet           # 训练集
    ├── val.parquet             # 验证集
    └── test.parquet            # 测试集
```

### 数据处理流程

原始 Swiss-Prot 数据经过以下处理得到训练数据集：
1. 筛选同时具有 EC 编号、细胞定位、功能注释的蛋白质
2. 过滤过短或过长的序列
3. 使用 ESM2 模型提取蛋白质特征
4. 按照 6:2:2 比例划分训练集、验证集、测试集
5. 对不平衡类别进行平衡处理

### 数据标签

数据集中的蛋白质同时拥有以下三种标注信息：

#### 1. EC 主类 (EC Number)

酶催化功能分类，依据酶学委员会编号体系划分为主类（首位数字）：

| 类别 | 说明 |
|------|------|
| EC1 | 氧化还原酶 (Oxidoreductases) |
| EC2 | 转移酶 (Transferases) |
| EC3 | 水解酶 (Hydrolases) |
| EC4 | 裂解酶 (Lyases) |
| EC5 | 异构酶 (Isomerases) |
| EC6 | 连接酶 (Ligases) |
| EC7 | 转运酶 (Translocases) |

#### 2. 细胞定位 (Subcellular Localization)

蛋白质在细胞内的主要定位区域：

| 类别 | 说明 |
|------|------|
| Cytoplasm | 细胞质 |
| Endoplasmic reticulum | 内质网 |
| Membrane | 细胞膜 |
| Mitochondria | 线粒体 |
| Nucleus | 细胞核 |
| Other | 其他定位 |
| Secreted | 分泌/胞外 |
| Unknown | 未知定位 |

#### 3. 分子功能 (Molecular Function)

蛋白质的分子功能注释：

| 类别 | 说明 |
|------|------|
| Catalytic | 催化活性 |
| Kinase | 激酶 |
| Other | 其他功能 |
| Receptor | 受体 |
| Transferase | 转移酶 |
| Transporter | 转运蛋白 |

> **说明**: Swiss-Prot 是 UniProt 的手工注释子集，数据质量高，广泛用于生物信息学研究和机器学习基准测试。

### 数据获取方式

```bash
# 方式1: FTP 下载完整 Swiss-Prot 数据
wget https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_sprot.dat.gz

# 方式2: 通过 REST API 查询特定蛋白质
curl "https://rest.uniprot.org/uniprotkb/stream?query=ec:[1 TO 6]+AND+comment%28SCL%29+AND+comment%28function%29&format=tsv&size=500"
```

详细的下载脚本和数据处理方法请参考 [docs/data_sources.md](docs/data_sources.md)。

## 项目结构

```
ProteinClassifier/
├── scripts/
│   ├── 01_process_data.py          # 数据处理脚本
│   ├── train/                      # 训练脚本
│   │   ├── train_rf_multitask.py       # RandomForest多任务训练
│   │   ├── train_xgb_multitask.py      # XGBoost多任务训练
│   │   ├── train_mlp_multitask.py      # MLP多任务训练
│   │   ├── train_bnn_multitask.py      # BNN多任务训练
│   │   └── train_multilabel_multitask.py # 多标签分类训练
│   │
│   ├── inference/                   # 推理脚本
│   │   ├── inference_rf_multitask.py      # RF多任务推理
│   │   ├── inference_xgb_multitask.py      # XGBoost多任务推理
│   │   ├── inference_mlp_multitask.py      # MLP多任务推理
│   │   ├── inference_bnn_multitask.py      # BNN多任务推理
│   │   └── inference_multilabel.py         # 多标签分类推理
│   │
│   ├── evaluate/                   # 评估脚本
│   │   └── evaluate_all.py         # 评估所有模型
│   │
│   └── visualize/                  # 可视化脚本
│       └── plot_results.py         # 绘制结果图表
│
├── models/                         # 模型文件 (训练后生成)
│   ├── rf_esm2_multitask/          # RandomForest模型
│   ├── xgb_esm2_multitask/         # XGBoost模型
│   ├── mlp_esm2_multitask/         # MLP模型
│   ├── bnn_esm2_multitask/         # 贝叶斯神经网络模型
│   └── multilabel_esm2_multitask/  # 多标签分类模型
│
├── src/
│   ├── encodings/                  # 特征编码
│   │   ├── onehot.py
│   │   ├── ctd.py
│   │   └── esm2.py
│   │
│   ├── algorithms/                 # 算法实现
│   │   ├── rf.py
│   │   ├── xgb.py
│   │   ├── mlp.py
│   │   └── bnn.py
│   │
│   └── pipeline/                   # 训练流程
│       ├── dataset.py
│       ├── trainer.py
│       └── evaluator.py
│
└── data/
    ├── datasets/                   # 数据集
    │   ├── protein_with_go.parquet    # 完整数据集
    │   ├── train_subset.parquet       # 训练子集
    │   └── balanced_with_go/          # 平衡后的数据集
    ├── processed/                  # 处理后的数据
    └── raw/                        # 原始数据
```

## 快速开始

### 训练模型

```bash
# RandomForest 多任务训练 (支持 onehot/ctd/esm2)
python scripts/train/train_rf_multitask.py --encoding esm2

# XGBoost 多任务训练
python scripts/train/train_xgb_multitask.py --encoding esm2

# MLP 多任务训练
python scripts/train/train_mlp_multitask.py --encoding esm2

# BNN 多任务训练
python scripts/train/train_bnn_multitask.py --encoding esm2

# 多标签分类训练
python scripts/train/train_multilabel_multitask.py --encoding esm2
```

> **提示**: 运行 `python scripts/train/train_rf_multitask.py -h` 可查看所有支持的编码方式 (`--encoding`)

### 模型推理

```bash
# RF模型推理
python scripts/inference/inference_rf_multitask.py \
    --model models/rf_esm2_multitask/ \
    --sequence "YOUR_PROTEIN_SEQUENCE"

# XGBoost模型推理
python scripts/inference/inference_xgb_multitask.py \
    --model models/xgb_esm2_multitask/ \
    --sequence "YOUR_PROTEIN_SEQUENCE"

# MLP模型推理
python scripts/inference/inference_mlp_multitask.py \
    --model models/mlp_esm2_multitask/ \
    --sequence "YOUR_PROTEIN_SEQUENCE"

# BNN模型推理 (带不确定性估计)
python scripts/inference/inference_bnn_multitask.py \
    --model models/bnn_esm2_multitask/ \
    --sequence "YOUR_PROTEIN_SEQUENCE"

# 多标签分类推理
python scripts/inference/inference_multilabel.py \
    --model models/multilabel_esm2_multitask/ \
    --sequence "YOUR_PROTEIN_SEQUENCE"
```

### 批量推理 (FASTA文件)

```bash
python scripts/inference/inference_rf_multitask.py \
    --model models/rf_esm2_multitask/ \
    --fasta proteins.fasta \
    --output results.json
```

### 模型评估

```bash
# 评估所有模型
python scripts/evaluate/evaluate_all.py

# 绘制结果可视化
python scripts/visualize/plot_results.py
```

## 支持的算法

| 算法 | 训练脚本 | 推理脚本 | 模型格式 | 特点 |
|------|---------|---------|----------|------|
| RandomForest | `train_rf_multitask.py` | `inference_rf_multitask.py` | 目录 | 传统机器学习，可解释性强，多任务预测 |
| XGBoost | `train_xgb_multitask.py` | `inference_xgb_multitask.py` | 目录 | 梯度提升，高精度，多任务预测 |
| MLP | `train_mlp_multitask.py` | `inference_mlp_multitask.py` | 目录 | 神经网络，全连接层，多任务预测 |
| BNN | `train_bnn_multitask.py` | `inference_bnn_multitask.py` | 目录 | 贝叶斯神经网络，可输出预测不确定性 |
| Multilabel | `train_multilabel_multitask.py` | `inference_multilabel.py` | 目录 | 多标签分类，支持部分标签缺失 |

## 支持的编码方式

| 编码 | 维度 | 说明 | 支持的算法 |
|------|------|------|-----------|
| `onehot` | 20维 | 20种氨基酸的组成比例编码 | RF, XGBoost, MLP, BNN, Multilabel |
| `ctd` | 147维 | 氨基酸物化性质编码 (Composition-Transition-Distribution) | RF, XGBoost, MLP, BNN, Multilabel |
| `esm2` | 1280维 | ESM2 蛋白质语言模型特征 | RF, XGBoost, MLP, BNN, Multilabel |

## 支持的分类任务

| 算法 | 预测任务 | 类别数 | 说明 |
|------|----------|--------|------|
| RandomForest | EC主类 + 细胞定位 + 分子功能 | 3任务 | 多任务学习，同时预测3个指标 |
| XGBoost | EC主类 + 细胞定位 + 分子功能 | 3任务 | 多任务学习，同时预测3个指标 |
| MLP | EC主类 + 细胞定位 + 分子功能 | 3任务 | 多任务学习，同时预测3个指标 |
| BNN | EC主类 + 细胞定位 + 分子功能 | 3任务 | 多任务学习，可输出预测不确定性 |
| Multilabel | EC主类 + 细胞定位 + 分子功能 | 3任务 | 多标签分类，支持部分标签缺失 |

> **说明**: 所有模型都支持同时预测 EC主类、细胞定位和分子功能三个分类指标。

## 模型性能 (Multitask)

### 测试集评估结果

| 任务 | Accuracy | F1-Macro | F1-Weighted | 说明 |
|------|----------|----------|-------------|------|
| **EC主类** | 97.94% | 96.66% | 97.94% | 7类分类，表现优异 |
| **细胞定位** | 95.48% | 47.62% | 96.12% | 8类分类，存在类别不平衡 |
| **分子功能** | 98.16% | 85.33% | 98.31% | 6类分类，表现良好 |
| **总体** | - | **76.54%** | - | Macro F1 平均值 |

> **重要说明**：
> - Accuracy 指标会被多数类主导。细胞定位任务中 "Unknown" 占 97%，因此 95% 的 Accuracy 主要反映的是对 "Unknown" 的预测能力。
> - **F1-Macro** 是更合理的指标，它对所有类别一视同仁，能更好地反映少数类的识别能力。
> - 由于训练数据中 `Unknown` 和 `Other` 类别占比过高，模型在定位和功能任务上更容易预测出这两个类别。

### 关于类别不平衡

训练数据集存在严重的类别不平衡问题：

| 任务 | 类别分布 |
|------|----------|
| **细胞定位** | Unknown: 97.0%, 其他7类合计: 3.0% |
| **分子功能** | Other: 95.6%, 其他5类合计: 4.4% |

本项目已采取以下措施缓解类别不平衡：
1. **增强类别权重**：使用 `balanced` 策略计算类别权重，少数类获得更高权重
2. **Macro F1 早停**：使用 Macro F1 而非准确率作为早停指标
3. **Top-K 预测**：推理时显示前3个最可能的预测结果

如需进一步提升少数类的识别能力，建议：
- 收集更多有明确标注的训练数据（排除 Unknown/Other）
- 使用过采样技术（如 SMOTE）增加少数类样本
- 使用数据增强技术扩充训练集

## 模型配置

每个模型目录下包含:
- `results.json`: 训练结果和评估指标
- 模型文件 (训练后生成)

```json
{
  "model_type": "RandomForest",
  "encoding": "esm2",
  "input_dim": 1280,
  "tasks": {
    "ec": {"n_classes": 7, "classes": ["EC1", "EC2", "EC3", "EC4", "EC5", "EC6", "EC7"]},
    "localization": {"n_classes": 8},
    "function": {"n_classes": 6}
  },
  "test_results": {
    "ec": {"accuracy": 0.95, "f1_macro": 0.90},
    "localization": {"accuracy": 0.95, "f1_macro": 0.45},
    "function": {"accuracy": 0.98, "f1_macro": 0.82}
  }
}
```

## 数据集划分

训练数据按照以下比例划分：

| 数据集 | 比例 | 说明 |
|--------|------|------|
| 训练集 | 60% | 用于模型训练 |
| 验证集 | 20% | 用于超参数调优 |
| 测试集 | 20% | 用于最终评估 |

划分使用随机种子 `42` 保证可重复性。

## 评估指标

模型训练和评估使用以下指标：

| 指标 | 说明 |
|------|------|
| Accuracy | 整体分类准确率 |
| F1-Macro | 各类别 F1 分数的宏平均 |
| F1-Weighted | 加权 F1 分数 |
| Precision | 精确率 |
| Recall | 召回率 |

## 编码算法详解

每种编码算法的具体原理和计算方法请参考：

- [技术文档_编码算法详解.md](技术文档_编码算法详解.md) - 包含 OneHot、CTD、ESM2 的详细说明

## 训练配置

各算法的默认超参数配置：

### RandomForest

```python
{
    "n_estimators": 100,
    "max_depth": 20,
    "min_samples_split": 5,
    "random_state": 42
}
```

### XGBoost

```python
{
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "objective": "multi:softmax"
}
```

### MLP

```python
{
    "hidden_dims": [256, 128],
    "dropout": 0.3,
    "epochs": 100,
    "batch_size": 32,
    "lr": 0.001
}
```

## 常见问题

### Q: 如何处理新的数据？

```bash
# 使用数据处理脚本
python scripts/01_process_data.py --input your_data.parquet --output processed/
```

### Q: 如何添加新的编码方式？

1. 在 `src/encodings/` 目录下创建新的编码器类
2. 继承 `EncoderBase` 基类
3. 实现 `encode()` 和 `encode_batch()` 方法
4. 在 `EncoderRegistry` 中注册新编码器

### Q: 支持 GPU 训练吗？

是的，MLP、BNN 模型支持 GPU 训练：

```bash
python scripts/train/train_mlp_multitask.py --device cuda
```

## 文档目录

| 文档 | 说明 |
|------|------|
| README.md | 项目总览 (本文档) |
| [技术文档_编码算法详解.md](技术文档_编码算法详解.md) | 编码算法原理详解 |
| [docs/data_sources.md](docs/data_sources.md) | 数据来源和下载方式 |
| [requirements.txt](requirements.txt) | Python 依赖列表 |

## 许可证

本项目基于 MIT 许可证开源。

## 致谢

- 数据来源: [UniProt Consortium](https://www.uniprot.org)
- ESM2 模型: [Facebook AI Research](https://github.com/facebookresearch/esm)
- 开源机器学习库: scikit-learn, XGBoost, PyTorch
