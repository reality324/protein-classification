#!/usr/bin/env python3
"""
重新处理 Swiss-Prot 数据，提取完整的 GO 功能标注
"""
import gzip
import re
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path
import argparse

def parse_swissprot(filepath):
    """解析 Swiss-Prot .dat.gz 文件，提取更多信息"""
    entries = []
    current_entry = {}
    seq_lines = []
    in_function = False
    func_lines = []
    
    with gzip.open(filepath, 'rt') as f:
        for line in f:
            line = line.rstrip()
            
            if line.startswith('//'):
                # 保存累积的功能描述
                if func_lines:
                    current_entry['function_text'] = ' '.join(func_lines)
                
                if current_entry:
                    current_entry['sequence'] = ''.join(seq_lines)
                    if current_entry.get('sequence') and len(current_entry['sequence']) > 30:
                        entries.append(current_entry)
                current_entry = {}
                seq_lines = []
                func_lines = []
                in_function = False
                
            elif line.startswith('AC'):
                current_entry['accession'] = line[5:].strip().rstrip(';')
                
            elif line.startswith('DE'):
                ec_matches = re.findall(r'EC=(\d+\.\d+\.\d+\.\d+[-\d]*)', line)
                for ec in ec_matches:
                    if 'ec_numbers' not in current_entry:
                        current_entry['ec_numbers'] = []
                    if ec not in current_entry['ec_numbers']:
                        current_entry['ec_numbers'].append(ec)
                        
            elif line.startswith('KW'):
                current_entry['keywords'] = line[5:].strip().rstrip(';')
                
            elif line.startswith('CC'):
                content = line[5:].strip()
                # 细胞定位
                if content.startswith('-!- SUBCELLULAR LOCATION:'):
                    loc = content.replace('-!- SUBCELLULAR LOCATION:', '').strip()
                    loc = re.sub(r'\{.*?\}', '', loc)
                    loc = re.sub(r'\[.*?\]', '', loc)
                    loc = loc.split('.')[0].strip()
                    if loc:
                        current_entry['location'] = loc
                # 功能描述
                elif content.startswith('-!- FUNCTION:'):
                    func = content.replace('-!- FUNCTION:', '').strip()
                    func_lines.append(func)
                    in_function = True
                elif in_function and content.startswith('     '):
                    # 续行
                    func_lines.append(content.strip())
                elif not content.startswith('     '):
                    in_function = False
                    
            elif line.startswith('DR   GO;'):
                # 正确的 Swiss-Prot DR 行格式：
                # DR   GO; GO:0005524; F:ATP binding; IEA:UniProtKB-KW.
                # 用分号分割，而不是逗号
                content = line[5:].strip()
                parts = content.split(';')
                for p in parts:
                    p = p.strip()
                    if p.startswith('F:'):
                        # F: 开头的是 Molecular Function
                        func_name = p[2:].strip()
                        if func_name:
                            if 'go_mf' not in current_entry:
                                current_entry['go_mf'] = []
                            if func_name not in current_entry['go_mf']:
                                current_entry['go_mf'].append(func_name)
    
            elif line[0:3] == '   ' or line.startswith('     '):
                seq_parts = line.split()
                seq_lines.extend(seq_parts)
    
    return entries


def normalize_ec(ec_list):
    """获取 EC 主类"""
    if not isinstance(ec_list, list) or len(ec_list) == 0:
        return None
    ec = ec_list[0]
    if not ec:
        return None
    main_class = ec.split('.')[0]
    return f"EC{main_class}"


def normalize_location(loc_text):
    """标准化细胞定位"""
    if not loc_text or pd.isna(loc_text):
        return None
    
    loc = loc_text.lower().strip()
    
    location_mapping = {
        'nucleus': 'Nucleus', 'nuclear': 'Nucleus', 'nucleolus': 'Nucleus',
        'cytoplasm': 'Cytoplasm', 'cytosol': 'Cytoplasm',
        'mitochondri': 'Mitochondria',
        'membrane': 'Membrane', 'plasma membrane': 'Membrane',
        'endoplasmic reticulum': 'ER',
        'golgi': 'Golgi',
        'lysosome': 'Lysosome',
        'peroxisome': 'Peroxisome',
        'secreted': 'Secreted', 'extracellular': 'Secreted',
        'ribosome': 'Ribosome',
        'cell surface': 'Cell_Surface',
        'cell junction': 'Cell_Junction',
    }
    
    for key, value in location_mapping.items():
        if key in loc:
            return value
    
    return None


def normalize_function_from_go(go_mf_list, keywords):
    """从 GO Molecular Function 提取功能"""
    functions = []
    
    # GO Molecular Function 到类别的映射
    go_to_function = {
        # 氧化还原酶
        'oxidoreductase': 'Oxidoreductase',
        'oxidase': 'Oxidoreductase',
        'reductase': 'Oxidoreductase',
        'dehydrogenase': 'Oxidoreductase',
        'electron transfer': 'Oxidoreductase',
        
        # 转移酶
        'transferase': 'Transferase',
        'kinase': 'Kinase',
        'phosphotransferase': 'Kinase',
        
        # 水解酶
        'hydrolase': 'Hydrolase',
        'protease': 'Protease',
        'peptidase': 'Protease',
        'lipase': 'Lipase',
        'nuclease': 'Nuclease',
        'phosphatase': 'Phosphatase',
        'glycosidase': 'Glycosidase',
        
        # 裂解酶
        'lyase': 'Lyase',
        'decarboxylase': 'Lyase',
        'synthase': 'Lyase',
        
        # 异构酶
        'isomerase': 'Isomerase',
        'epimerase': 'Isomerase',
        'mutase': 'Isomerase',
        
        # 连接酶
        'ligase': 'Ligase',
        'synthetase': 'Ligase',
        
        # 其他功能
        'binding': 'Binding',
        'protein binding': 'Binding',
        'dna binding': 'DNA_Binding',
        'rna binding': 'RNA_Binding',
        
        'transcription': 'Transcription',
        'transcriptional': 'Transcription',
        
        'motor': 'Motor',
        'atpase': 'ATPase',
        'gtpase': 'GTPase',
        
        'receptor': 'Receptor',
        'signaling': 'Signaling',
        'signal transduction': 'Signaling',
        
        'transporter': 'Transporter',
        'channel': 'Channel',
        'ion channel': 'Ion_Channel',
        
        'structural molecule': 'Structural',
        'metal ion binding': 'Metal_Binding',
        'zinc': 'Metal_Binding',
        
        'catalytic': 'Catalytic',
        'enzyme': 'Enzyme',
    }
    
    # 从 GO MF 列表提取
    if isinstance(go_mf_list, list):
        for go in go_mf_list:
            go_lower = go.lower()
            for key, func in go_to_function.items():
                if key in go_lower and func not in functions:
                    functions.append(func)
    
    # 从 Keywords 提取
    if keywords and pd.notna(keywords):
        kw_lower = keywords.lower()
        for key, func in go_to_function.items():
            if key in kw_lower and func not in functions:
                functions.append(func)
    
    return functions


def main():
    print("=" * 60)
    print("重新处理 Swiss-Prot 数据 (含 GO 功能)")
    print("=" * 60)
    
    # 解析数据
    print("\n[1/4] 解析数据...")
    entries = parse_swissprot('/home/tianwangcong/uniprot_sprot.dat.gz')
    print(f"  解析到 {len(entries)} 条记录")
    
    df = pd.DataFrame(entries)
    
    # 标准化标签
    print("\n[2/4] 标准化标签...")
    df['ec_main_class'] = df['ec_numbers'].apply(normalize_ec)
    df['location_normalized'] = df['location'].apply(normalize_location)
    df['functions'] = df.apply(
        lambda x: normalize_function_from_go(x.get('go_mf'), x.get('keywords')), 
        axis=1
    )
    
    # 过滤
    print("\n[3/4] 过滤无效样本...")
    has_ec = df['ec_main_class'].notna()
    has_loc = df['location_normalized'].notna()
    has_func = df['functions'].apply(lambda x: len(x) > 0)
    
    print(f"  有 EC 编号: {has_ec.sum()}")
    print(f"  有细胞定位: {has_loc.sum()}")
    print(f"  有分子功能: {has_func.sum()}")
    
    # 至少有两种标签
    df['label_count'] = has_ec.astype(int) + has_loc.astype(int) + has_func.astype(int)
    df_filtered = df[df['label_count'] >= 2].copy()
    print(f"  有效样本 (>=2种标签): {len(df_filtered)}")
    
    # 统计
    print("\n[4/4] 类别分布...")
    
    print("\n  EC 主类:")
    for ec, cnt in df_filtered['ec_main_class'].value_counts().items():
        print(f"    {ec}: {cnt} ({cnt/len(df_filtered)*100:.1f}%)")
    
    print("\n  细胞定位:")
    for loc, cnt in df_filtered['location_normalized'].value_counts().items():
        print(f"    {loc}: {cnt} ({cnt/len(df_filtered)*100:.1f}%)")
    
    print("\n  分子功能:")
    all_funcs = []
    for funcs in df_filtered['functions']:
        all_funcs.extend(funcs)
    func_counts = Counter(all_funcs)
    for func, cnt in func_counts.most_common():
        print(f"    {func}: {cnt}")
    
    # 保存
    output_file = 'data/datasets/protein_with_go.parquet'
    df_filtered.to_parquet(output_file, index=False)
    print(f"\n保存到: {output_file}")


if __name__ == "__main__":
    main()
