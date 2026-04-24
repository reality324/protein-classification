#!/usr/bin/env python3
"""
解析 UniProt Swiss-Prot .dat.gz 文件，提取蛋白质序列和标签
"""
import gzip
import pandas as pd
from pathlib import Path
from collections import defaultdict


def parse_dat_file(filepath):
    """解析 Swiss-Prot .dat 文件"""
    entries = []
    current_entry = {}

    with gzip.open(filepath, 'rt') as f:
        for line in f:
            line = line.rstrip()

            if line.startswith('ID'):
                parts = line.split()
                current_entry['id'] = parts[1].rstrip(';')

            elif line.startswith('AC'):
                ac = line[5:].strip().rstrip(';')
                current_entry['accession'] = ac

            elif line.startswith('DE'):
                if 'description' not in current_entry:
                    current_entry['description'] = []
                current_entry['description'].append(line[5:].strip())

            elif line.startswith('GN'):
                if 'gene' not in current_entry:
                    current_entry['gene'] = []
                current_entry['gene'].append(line[5:].strip())

            elif line.startswith('KW'):
                if 'keywords' not in current_entry:
                    current_entry['keywords'] = []
                current_entry['keywords'].append(line[5:].strip())

            elif line.startswith('CC'):
                content = line[5:].strip()
                if content.startswith('-!- FUNCTION:'):
                    if 'function' not in current_entry:
                        current_entry['function'] = []
                    current_entry['function'].append(content.replace('-!- FUNCTION:', '').strip())

            elif line.startswith('SQ'):
                # 序列长度
                parts = line[5:].split()
                current_entry['seq_length'] = int(parts[0])

            elif line.startswith('//'):
                # 条目结束
                if current_entry:
                    # 提取序列 (在 SQ 行之后到 // 之前的所有行)
                    entries.append(current_entry)
                current_entry = {}

    return entries


def extract_sequence(lines_iter, lines):
    """从行迭代器中提取序列"""
    seq_parts = []
    for line in lines:
        if line.startswith('//'):
            break
        seq_parts.append(line.strip())
    return ''.join(seq_parts)


def parse_with_sequence(filepath):
    """解析 Swiss-Prot .dat 文件，提取序列"""
    entries = []
    current_entry = {}
    seq_lines = []

    with gzip.open(filepath, 'rt') as f:
        for line in f:
            line = line.rstrip()

            if line.startswith('//'):
                # 条目结束，保存
                if current_entry:
                    current_entry['sequence'] = ''.join(seq_lines)
                    entries.append(current_entry)
                current_entry = {}
                seq_lines = []

            elif line.startswith('ID'):
                parts = line.split()
                current_entry['id'] = parts[1].rstrip(';')

            elif line.startswith('AC'):
                current_entry['accession'] = line[5:].strip().rstrip(';')

            elif line.startswith('DE'):
                desc = line[5:].strip()
                if 'description' not in current_entry:
                    current_entry['description'] = desc
                else:
                    current_entry['description'] += ' ' + desc
                # 提取 EC 编号
                import re
                ec_matches = re.findall(r'EC=(\d+\.\d+\.\d+\.\d+[-\d]*)', line)
                for ec in ec_matches:
                    if 'ec_numbers' not in current_entry:
                        current_entry['ec_numbers'] = []
                    if ec not in current_entry['ec_numbers']:
                        current_entry['ec_numbers'].append(ec)

            elif line.startswith('GN'):
                parts = line[5:].strip().split(';')
                for p in parts:
                    if p.startswith('Name='):
                        current_entry['gene_name'] = p.replace('Name=', '').rstrip(';')
                        break

            elif line.startswith('KW'):
                kw = line[5:].strip().rstrip(';')
                current_entry['keywords'] = kw.replace(';', ',')

            elif line.startswith('EC'):
                ec = line[5:].strip().rstrip(';')
                if ec and ec not in ('-', ''):
                    if 'ec_numbers' not in current_entry:
                        current_entry['ec_numbers'] = []
                    current_entry['ec_numbers'].append(ec)

            elif line.startswith('DR'):
                # 数据库引用
                if 'dr' not in current_entry:
                    current_entry['dr'] = []
                current_entry['dr'].append(line[5:].strip())

            elif line.startswith('SQ '):
                # 序列长度行: "SQ   SEQUENCE   256 AA;  29735 MW;  B4840739BF7D4121 CRC64;"
                parts = line.split(';')
                if parts:
                    length_str = parts[0].replace('SQ', '').strip()
                    parts2 = length_str.split()
                    for p in parts2:
                        if p.isdigit():
                            current_entry['seq_length'] = int(p)
                            break

            elif line[0:3] == '   ' or line.startswith('     '):
                # 序列行
                seq_parts = line.split()
                seq_lines.extend(seq_parts)

    return entries


def main():
    import argparse
    parser = argparse.ArgumentParser(description="解析 UniProt Swiss-Prot .dat.gz 文件")
    parser.add_argument("input", nargs='?', default="data/raw/uniprot_sprot.dat.gz",
                        help="输入文件路径")
    parser.add_argument("-o", "--output", default="data/datasets/uniprot_parsed.parquet",
                        help="输出 parquet 文件路径")
    parser.add_argument("--limit", type=int, default=None,
                        help="限制条目数量（用于测试）")
    args = parser.parse_args()

    print(f"解析文件: {args.input}")
    entries = parse_with_sequence(args.input)

    if args.limit:
        entries = entries[:args.limit]

    print(f"解析到 {len(entries)} 条记录")

    # 转换为 DataFrame
    df = pd.DataFrame(entries)

    # 选择需要的列
    columns = ['id', 'accession', 'description', 'keywords', 'ec_numbers', 'ec_number',
               'gene_name', 'sequence', 'seq_length']
    for col in columns:
        if col not in df.columns:
            df[col] = None

    df = df[[c for c in columns if c in df.columns]]

    # 过滤有效记录（必须有序列）
    df = df[df['sequence'].notna() & (df['sequence'].str.len() > 0)]
    print(f"有效记录（带序列）: {len(df)} 条")

    # 显示统计
    print(f"\n统计信息:")
    print(f"  总记录数: {len(df)}")
    has_ec = 'ec_numbers' in df.columns
    if has_ec:
        df['ec_number'] = df['ec_numbers'].apply(lambda x: ','.join(x) if isinstance(x, list) and len(x) > 0 else None)
        print(f"  有 EC 编号: {df['ec_number'].notna().sum()}")
    print(f"  有 Keywords: {df['keywords'].notna().sum()}")
    print(f"  平均序列长度: {df['sequence'].str.len().mean():.1f}")

    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"\n已保存到: {output_path}")

    # 显示样本
    print("\n前3条记录:")
    for i, row in df.head(3).iterrows():
        print(f"  ID: {row['id']}, EC: {row.get('ec_number', 'N/A')}")
        print(f"    Keywords: {str(row.get('keywords', 'N/A'))[:80]}...")
        print(f"    Sequence: {row['sequence'][:50]}...")


if __name__ == "__main__":
    main()
