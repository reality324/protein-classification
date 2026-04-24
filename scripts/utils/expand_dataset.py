#!/usr/bin/env python3
"""
扩大数据集脚本

从原始 uniprot_sprot.dat.gz 文件中提取更多符合条件的蛋白质数据，
筛选同时有 EC 编号、细胞定位、功能注释的蛋白质。

使用方法:
    python scripts/utils/expand_dataset.py --max-samples 50000
"""
import gzip
import re
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def parse_swissprot(filepath, max_samples=None):
    """解析 Swiss-Prot .dat.gz 文件，提取符合条件的蛋白质"""
    
    entries = []
    current = {}
    
    with gzip.open(filepath, 'rt') as f:
        for line in f:
            line = line.rstrip()
            
            if line.startswith('//'):
                if current:
                    # 检查是否同时有 EC、细胞定位、功能
                    if (current.get('has_ec') and 
                        current.get('has_scl') and 
                        current.get('has_func') and
                        current.get('sequence')):
                        entries.append(current)
                        
                        if max_samples and len(entries) >= max_samples:
                            break
                current = {}
            
            elif line.startswith('ID '):
                current['id'] = line.split()[1].rstrip(';')
            
            elif line.startswith('AC '):
                current['accession'] = line[5:].strip().rstrip(';')
            
            elif line.startswith('DE '):
                # EC 编号在 DE 行中
                if re.search(r'EC=(\d+\.\d+\.\d+\.\d+)', line):
                    current['has_ec'] = True
                if re.search(r'EC=(\d+\.\d+\.\d+\.\d+[-\d]*)', line):
                    ec_match = re.search(r'EC=(\d+\.\d+\.\d+\.\d+[-\d]*)', line)
                    if ec_match:
                        current['ec_number'] = ec_match.group(1)
            
            elif line.startswith('GN '):
                parts = line[5:].strip().split(';')
                for p in parts:
                    if p.startswith('Name='):
                        current['gene_name'] = p.replace('Name=', '').rstrip(';')
                        break
            
            elif 'SUBCELLULAR LOCATION' in line:
                current['has_scl'] = True
                # 提取定位信息
                loc_match = re.search(r'SUBCELLULAR LOCATION:[^{]*?\.\s*', line)
                if loc_match:
                    current['location'] = loc_match.group().replace('SUBCELLULAR LOCATION:', '').strip()
            
            elif '-!- FUNCTION:' in line:
                current['has_func'] = True
                func_text = line.replace('-!- FUNCTION:', '').strip()
                if func_text and len(func_text) > 10:
                    current['function'] = func_text[:200]
            
            elif line.startswith('SQ '):
                # 序列长度
                parts = line.split()
                for p in parts:
                    if p.isdigit():
                        current['seq_length'] = int(p)
                        break
            
            elif line.startswith('CC') and 'FUNCTION:' not in line:
                current['description'] = ''
            
            elif line[0:3] == '   ' or line.startswith('     '):
                # 序列行
                if 'sequence' not in current:
                    current['sequence'] = ''
                seq_parts = line.split()
                current['sequence'] += ''.join(seq_parts)
    
    return entries


def extract_main_ec_class(ec_str):
    """提取 EC 主类 (1-7)"""
    if not ec_str:
        return None
    match = re.match(r'(\d)', ec_str)
    if match:
        return int(match.group(1))
    return None


def main():
    parser = argparse.ArgumentParser(description="扩大数据集")
    parser.add_argument("--input", default="/home/tianwangcong/uniprot_sprot.dat.gz",
                       help="原始数据文件")
    parser.add_argument("--max-samples", type=int, default=50000,
                       help="最大样本数 (默认 50000)")
    parser.add_argument("--output", default="data/datasets/train_subset.parquet",
                       help="输出文件")
    args = parser.parse_args()
    
    print("=" * 60)
    print("扩大数据集")
    print("=" * 60)
    print(f"输入文件: {args.input}")
    print(f"目标样本数: {args.max_samples}")
    
    # 1. 解析数据
    print("\n[1/4] 解析 Swiss-Prot 数据...")
    entries = parse_swissprot(args.input, max_samples=args.max_samples)
    print(f"解析到 {len(entries)} 条符合条件的蛋白质")
    
    # 2. 转换为 DataFrame
    print("\n[2/4] 转换数据...")
    df = pd.DataFrame(entries)
    
    # 3. 过滤和清理
    print("\n[3/4] 数据清理...")
    
    # 确保有序列
    df = df[df['sequence'].notna() & (df['sequence'].str.len() > 0)]
    
    # 过滤序列长度 (50-2000 aa)
    df['seq_length'] = df['sequence'].str.len()
    df = df[(df['seq_length'] >= 50) & (df['seq_length'] <= 2000)]
    
    print(f"序列长度过滤后: {len(df)} 条")
    
    # 提取 EC 主类
    df['ec_main_class'] = df['ec_number'].apply(extract_main_ec_class)
    df = df[df['ec_main_class'].notna()]
    print(f"EC 主类提取后: {len(df)} 条")
    
    # 4. 创建标签列
    print("\n[4/4] 创建标签...")
    
    # EC 分类 (7 类)
    for i in range(1, 8):
        df[f'ec_{i}'] = (df['ec_main_class'] == i).astype(int)
    
    # 细胞定位分类 (标准化)
    def normalize_location(loc):
        if pd.isna(loc):
            return 'Unknown'
        loc_lower = loc.lower()
        if 'nucleus' in loc_lower or 'nuclear' in loc_lower:
            return 'Nucleus'
        elif 'cytoplasm' in loc_lower or 'cytosol' in loc_lower:
            return 'Cytoplasm'
        elif 'mitochondr' in loc_lower:
            return 'Mitochondria'
        elif 'membrane' in loc_lower or 'cell surface' in loc_lower:
            return 'Membrane'
        elif 'endoplasmic reticulum' in loc_lower or 'er ' in loc_lower:
            return 'Endoplasmic reticulum'
        elif 'secreted' in loc_lower or 'extracellular' in loc_lower:
            return 'Secreted'
        else:
            return 'Other'
    
    df['loc_normalized'] = df['location'].apply(normalize_location)
    unique_locs = sorted(df['loc_normalized'].unique())
    print(f"定位类别: {unique_locs}")
    
    for loc in unique_locs:
        df[f'loc_{loc}'] = (df['loc_normalized'] == loc).astype(int)
    
    # 分子功能分类 (基于 keywords/function 中的关键词)
    def extract_function(func_text):
        if pd.isna(func_text):
            return 'Other'
        func_lower = func_text.lower()
        if 'catalytic' in func_lower or 'hydrolase' in func_lower:
            return 'Catalytic'
        elif 'transferase' in func_lower:
            return 'Transferase'
        elif 'kinase' in func_lower:
            return 'Kinase'
        elif 'receptor' in func_lower or 'signaling' in func_lower:
            return 'Receptor'
        elif 'transporter' in func_lower or 'channel' in func_lower:
            return 'Transporter'
        elif 'dna binding' in func_lower or 'transcription' in func_lower:
            return 'DNA_binding'
        else:
            return 'Other'
    
    df['func_normalized'] = df['function'].apply(extract_function)
    unique_funcs = sorted(df['func_normalized'].unique())
    print(f"功能类别: {unique_funcs}")
    
    for func in unique_funcs:
        df[f'func_{func}'] = (df['func_normalized'] == func).astype(int)
    
    # 5. 选择需要的列
    label_cols = [c for c in df.columns if c.startswith(('ec_', 'loc_', 'func_'))]
    output_cols = ['AC', 'sequence'] + label_cols
    
    # 重命名 AC 列
    df['AC'] = df['accession']
    
    # 截断序列（如果太长）
    df['sequence'] = df['sequence'].str[:2000]
    
    # 选择列
    df_output = df[output_cols].copy()
    
    print(f"\n输出数据: {len(df_output)} 条")
    print(f"标签列数: {len(label_cols)}")
    
    # 6. 保存
    print(f"\n保存到: {args.output}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_parquet(output_path, index=False)
    
    print("\n" + "=" * 60)
    print("数据集扩大完成!")
    print("=" * 60)
    
    # 显示统计
    print("\n数据集统计:")
    print(f"  总样本数: {len(df_output)}")
    print(f"  序列长度: {df_output['sequence'].str.len().mean():.1f} ± {df_output['sequence'].str.len().std():.1f}")
    
    ec_cols = [c for c in label_cols if c.startswith('ec_')]
    loc_cols = [c for c in label_cols if c.startswith('loc_')]
    func_cols = [c for c in label_cols if c.startswith('func_')]
    
    print(f"\n  EC 类别数: {len(ec_cols)}")
    print(f"  定位类别数: {len(loc_cols)}")
    print(f"  功能类别数: {len(func_cols)}")


if __name__ == "__main__":
    main()
