#!/usr/bin/env python3
"""Update comparison plots with ESM2 results"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

# Load results
results_dir = Path("results/test")
df = pd.read_csv(results_dir / "comparison_results.csv")

# Set style
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False

# Create comparison figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. Accuracy comparison
ax1 = axes[0]
colors = {'onehot': '#2ecc71', 'ctd': '#3498db', 'esm2': '#e74c3c'}
for enc in df['encoding'].unique():
    enc_data = df[df['encoding'] == enc].sort_values('test_accuracy', ascending=True)
    bars = ax1.barh(enc_data['algorithm'], enc_data['test_accuracy'], 
                    color=colors.get(enc, '#95a5a6'), alpha=0.8, label=enc)
    for bar, acc in zip(bars, enc_data['test_accuracy']):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{acc:.3f}', va='center', fontsize=9)

ax1.set_xlabel('Test Accuracy')
ax1.set_title('Accuracy by Encoding × Algorithm')
ax1.set_xlim(0, 1.1)
ax1.legend(title='Encoding')

# 2. F1 Micro comparison
ax2 = axes[1]
for enc in df['encoding'].unique():
    enc_data = df[df['encoding'] == enc].sort_values('test_f1_micro', ascending=True)
    bars = ax2.barh(enc_data['algorithm'], enc_data['test_f1_micro'],
                    color=colors.get(enc, '#95a5a6'), alpha=0.8, label=enc)
    for bar, f1 in zip(bars, enc_data['test_f1_micro']):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{f1:.3f}', va='center', fontsize=9)

ax2.set_xlabel('F1 (micro)')
ax2.set_title('F1 Score by Encoding × Algorithm')
ax2.set_xlim(0, 1.1)
ax2.legend(title='Encoding')

# 3. Heatmap of accuracy
ax3 = axes[2]
pivot = df.pivot_table(values='test_accuracy', index='algorithm', columns='encoding')
pivot = pivot[['onehot', 'ctd', 'esm2']]  # Order columns
pivot = pivot.reindex(['xgb', 'rf'])  # Order rows

sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax3, 
            vmin=0.5, vmax=1.0, cbar_kws={'label': 'Accuracy'})
ax3.set_title('Accuracy Heatmap')

plt.tight_layout()
plt.savefig(results_dir / "algorithm_comparison.png", dpi=150, bbox_inches='tight')
print(f"Saved: {results_dir / 'algorithm_comparison.png'}")

# Create summary comparison figure
fig2, ax = plt.subplots(figsize=(10, 6))

# Group by encoding
summary = df.groupby('encoding').agg({
    'test_accuracy': 'mean',
    'test_f1_micro': 'mean',
    'test_f1_macro': 'mean'
}).round(4)

x = range(len(summary))
width = 0.25

bars1 = ax.bar([i - width for i in x], summary['test_accuracy'], width, label='Accuracy', color='#3498db')
bars2 = ax.bar(x, summary['test_f1_micro'], width, label='F1 (micro)', color='#2ecc71')
bars3 = ax.bar([i + width for i in x], summary['test_f1_macro'], width, label='F1 (macro)', color='#e74c3c')

ax.set_ylabel('Score')
ax.set_title('Summary: Performance by Encoding Method')
ax.set_xticks(x)
ax.set_xticklabels(summary.index)
ax.legend()
ax.set_ylim(0, 1.1)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(results_dir / "comparison_summary.png", dpi=150, bbox_inches='tight')
print(f"Saved: {results_dir / 'comparison_summary.png'}")

print("\n" + "=" * 60)
print("Results Summary:")
print("=" * 60)
print(summary.to_string())
print("=" * 60)
