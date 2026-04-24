# 蛋白质数据标签来源

## 🎯 三标签数据集 (同时拥有 EC + 细胞定位 + 功能)

### 推荐数据库: UniProt (最佳选择)

**UniProt** 是目前最推荐的数据库，因为:
1. 同时包含 EC、细胞定位、功能注释
2. 数据质量高 (Swiss-Prot 人工审核)
3. 提供 REST API 方便下载
4. ~57 万条蛋白质，其中约 20-30 万条同时拥有这三种标签

**官网**: https://www.uniprot.org

#### 三标签查询示例

```bash
# 查询同时有 EC、细胞定位、功能注释的蛋白质
curl "https://rest.uniprot.org/uniprotkb/stream?query=ec:[1 TO 6]+AND+comment%28SCL%29+AND+comment%28function%29&format=tsv&size=500"

# 下载人类蛋白质
curl "https://rest.uniprot.org/uniprotkb/stream?query=organism:9606+AND+ec:[1+TO+6]+AND+comment(SCL)+AND+comment(function)&format=tsv"
```

#### 可下载字段

| 字段名 | UniProt 列名 | 说明 |
|--------|-------------|------|
| EC Number | `ec` | 酶催化功能 |
| 细胞定位 | `cc_scl_annotation` | 亚细胞定位 |
| 功能注释 | `cc_function` | 功能描述 |
| Keywords | `keywords` | 功能关键词 |
| 序列 | `sequence` | 氨基酸序列 |

---

## 📚 主要数据源

### 1. UniProt (最推荐 - 同时包含三种标签!)

**官网**: https://www.uniprot.org

UniProt 是全球最大的蛋白质数据库，提供高质量的蛋白质注释。

#### 获取方式

```bash
# 方式1: REST API
curl "https://rest.uniprot.org/uniprotkb/stream?query=reviewed:true&format=tsv&columns=accession,sequence,ec,comment(SUBCELLULAR LOCATION),keywords"

# 方式2: 下载完整数据库
# https://www.uniprot.org/downloads
```

#### Python API 示例

```python
import requests
import pandas as pd

# 获取带 EC 编号的蛋白质
url = "https://rest.uniprot.org/uniprotkb/stream"
params = {
    "query": "ec:[1 TO 6] AND reviewed:true",
    "format": "tsv",
    "columns": "accession,sequence,ec,gene_names,organism",
    "size": 500  # 每批数量
}

response = requests.get(url, params=params)
data = pd.read_csv(StringIO(response.text))
```

---

### 2. UniProt 提供的分类数据

#### 子细胞定位数据
- **文件**: `SUBCELLULAR_LOCATIONS.txt`
- **来源**: https://www.uniprot.org/docs/subcell
- **内容**: 约 300 种亚细胞定位分类

#### EC Number 数据
- **文件**: `enzyme.dat`
- **来源**: https://www.expasy.org/resources/uniprotkb/enzymes
- **内容**: 完整的 EC 分类系统

---

### 3. Gene Ontology (GO)

**官网**: http://geneontology.org

#### 获取方式

```bash
# 安装 GOA 工具
pip install goatools

# 下载 GO 术语
wget http://purl.obolibrary.org/obo/go/go-basic.obo

# 下载 GOA 注释
wget https://current.geneontology.org/annotations/goa_human.gaf.gz
```

#### GO 三个本体

| 本体 | 全称 | 示例 |
|------|------|------|
| MF | Molecular Function | catalytic activity, binding |
| BP | Biological Process | metabolic process, signaling |
| CC | Cellular Component | nucleus, mitochondrion |

---

### 4. CAFA (蛋白质功能预测挑战赛)

**官网**: https://www.bioai.dk/ccg/cafa2/

#### 数据集

| 数据集 | 描述 | 大小 |
|--------|------|------|
| CAFA1 | 2010 基准 | ~45K 蛋白质 |
| CAFA2 | 2013 基准 | ~100K 蛋白质 |
| CAFA3 | 2016 基准 | ~300K 蛋白质 |
| CAFA4 | 2020 基准 | ~500K 蛋白质 |

#### 下载

```bash
# CAFA4 数据
wget https://www.bioai.dk/ccg/cafa4/protein_function_predictions.tar.gz

# Ground Truth
wget https://www.bioai.dk/ccg/cafa4/ground_truth.tar.gz
```

---

### 5. 其他数据源

#### AlphaFold DB
- **官网**: https://alphafold.ebi.ac.uk
- **内容**: 预测蛋白质结构
- **用途**: 结合结构信息增强分类

#### PDB (蛋白质结构数据库)
- **官网**: https://www.rcsb.org
- **内容**: 实验验证的蛋白质结构
- **用途**: 结构域分析

#### STRING (蛋白质相互作用)
- **官网**: https://string-db.org
- **内容**: 蛋白质相互作用网络
- **用途**: 功能注释传递

---

## 📊 标签类型详解

### EC Number (酶分类)

```
EC 编号格式: X.X.X.X

层级结构:
├── EC 1 - 氧化还原酶 (Oxidoreductases)
├── EC 2 - 转移酶 (Transferases)  
├── EC 3 - 水解酶 (Hydrolases)
├── EC 4 - 裂解酶 (Lyases)
├── EC 5 - 异构酶 (Isomerases)
└── EC 6 - 连接酶 (Ligases)
```

### 细胞定位分类

| 类别 | 示例 | GO ID |
|------|------|-------|
| 细胞核 | Nucleus | GO:0005634 |
| 细胞质 | Cytoplasm | GO:0005737 |
| 线粒体 | Mitochondrion | GO:0005739 |
| 细胞膜 | Cell membrane | GO:0005886 |
| 内质网 | ER | GO:0005783 |
| 高尔基体 | Golgi | GO:0005794 |
| 分泌 | Secreted | GO:0005576 |
| 溶酶体 | Lysosome | GO:0005764 |

---

## 🔧 数据下载脚本

### 完整数据下载

```python
#!/usr/bin/env python3
"""从 UniProt 下载蛋白质数据"""

import requests
import pandas as pd
from typing import List, Dict
import time

class UniProtDownloader:
    """UniProt 数据下载器"""

    BASE_URL = "https://rest.uniprot.org/uniprotkb"

    def __init__(self):
        self.session = requests.Session()

    def search(
        self,
        query: str,
        fields: List[str],
        size: int = 500,
    ) -> pd.DataFrame:
        """搜索蛋白质"""
        all_results = []
        offset = 0

        while True:
            params = {
                "query": query,
                "format": "tsv",
                "fields": ",".join(fields),
                "size": size,
                "offset": offset,
            }

            response = self.session.get(self.BASE_URL, params=params)
            response.raise_for_status()

            df = pd.read_csv(StringIO(response.text), sep="\t")
            if len(df) == 0:
                break

            all_results.append(df)
            offset += size

            print(f"已获取 {offset} 条记录...")

            # 避免请求过快
            time.sleep(0.5)

        return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

    def download_with_ec(self, output_file: str):
        """下载带 EC 编号的蛋白质"""
        df = self.search(
            query="reviewed:true AND ec:[1 TO 6]",
            fields=["accession", "sequence", "ec", "gene_names", "organism"],
        )
        df.to_csv(output_file, sep="\t", index=False)
        print(f"已保存 {len(df)} 条记录到 {output_file}")

    def download_with_location(self, output_file: str):
        """下载带细胞定位的蛋白质"""
        df = self.search(
            query='reviewed:true AND "cytoplasm" OR "nucleus" OR "mitochondrion" OR "membrane"',
            fields=["accession", "sequence", "comment(SUBCELLULAR LOCATION)", "keywords"],
        )
        df.to_csv(output_file, sep="\t", index=False)
        print(f"已保存 {len(df)} 条记录到 {output_file}")

    def download_batch(self, accessions: List[str]) -> Dict:
        """批量下载指定蛋白质"""
        results = {}

        for i in range(0, len(accessions), 100):
            batch = accessions[i:i+100]
            ids = "+".join(batch)

            url = f"{self.BASE_URL}/accessions"
            params = {"accessions": ids}

            response = self.session.get(url, params=params)
            response.raise_for_status()

            # 处理响应
            data = response.json()
            for entry in data.get("results", []):
                accession = entry["primaryAccession"]
                results[accession] = entry

            time.sleep(0.2)

        return results


if __name__ == "__main__":
    downloader = UniProtDownloader()

    # 下载带 EC 编号的数据
    downloader.download_with_ec("proteins_with_ec.tsv")

    # 下载带定位信息的数据
    downloader.download_with_location("proteins_with_location.tsv")
```

---

## 📈 数据量估算

| 数据集 | 蛋白质数量 | 说明 |
|--------|------------|------|
| UniProt Reviewed | ~560,000 | 人工审查的高质量数据 |
| UniProt Swiss-Prot | ~220,000 | 最可靠的手动注释 |
| TrEMBL | ~238,000,000 | 自动注释 |
| CAFA4 | ~312,000 | 功能预测基准 |

---

## ⚠️ 数据质量注意事项

1. **EC 编号**: 并非所有蛋白质都有 EC 编号（只有酶类）
2. **细胞定位**: 同一蛋白质可能有多个定位
3. **GO 注释**: 有层次关系，需要考虑
4. **数据不平衡**: 某些类别样本很少

建议从 **UniProt Reviewed (Swiss-Prot)** 开始，它提供高质量的人工注释数据。
