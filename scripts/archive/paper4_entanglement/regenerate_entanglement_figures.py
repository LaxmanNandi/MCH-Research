#!/usr/bin/env python3
"""Regenerate entanglement figures from pre-computed CSV data with VRI labels."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

# Load pre-computed data
df = pd.read_csv('c:/Users/barla/mch_experiments/results/tables/entanglement_position_data.csv')
corr_df = pd.read_csv('c:/Users/barla/mch_experiments/results/tables/entanglement_correlations.csv')
var_df = pd.read_csv('c:/Users/barla/mch_experiments/results/tables/entanglement_variance_summary.csv')

all_results = {name: group for name, group in df.groupby('model')}

fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: dRCI vs VRI (all models)
ax = fig.add_subplot(gs[0, 0])
for model_name, mdf in all_results.items():
    color = 'steelblue' if '(Phil)' in model_name else 'crimson'
    ax.scatter(mdf['mutual_info_proxy'], mdf['drci'], s=30, alpha=0.6, color=color,
               label=model_name if model_name in list(all_results.keys())[:2] else '')

all_mi = df['mutual_info_proxy']
all_drci = df['drci']
slope, intercept, r, p, _ = linregress(all_mi, all_drci)
x_fit = np.linspace(all_mi.min(), all_mi.max(), 100)
y_fit = slope * x_fit + intercept
ax.plot(x_fit, y_fit, 'k--', alpha=0.7, linewidth=2, label=f'Combined Fit (r={r:.3f}, p={p:.2e})')
ax.set_xlabel('VRI (1 - Var_Ratio)', fontsize=11)
ax.set_ylabel('dRCI', fontsize=11)
ax.set_title('Entanglement vs dRCI (All Positions)', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3)
ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)
ax.axvline(0, color='black', linewidth=0.5, alpha=0.3)

# Plot 2: Variance Ratio across positions (Philosophy)
ax = fig.add_subplot(gs[0, 1])
for model_name, mdf in all_results.items():
    if '(Phil)' in model_name:
        ax.plot(mdf['position'], mdf['var_ratio'], marker='o', markersize=4,
                alpha=0.7, linewidth=1.5, label=model_name.split(' (')[0])
ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Var_TRUE = Var_COLD')
ax.set_xlabel('Position', fontsize=11)
ax.set_ylabel('Var_Ratio (TRUE / COLD)', fontsize=11)
ax.set_title('Philosophy: Variance Ratio Across Positions', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 31)

# Plot 3: Variance Ratio across positions (Medical)
ax = fig.add_subplot(gs[0, 2])
for model_name, mdf in all_results.items():
    if '(Med)' in model_name:
        ax.plot(mdf['position'], mdf['var_ratio'], marker='o', markersize=4,
                alpha=0.7, linewidth=1.5, label=model_name.split(' (')[0])
ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Var_TRUE = Var_COLD')
ax.axvline(30, color='orange', linestyle=':', alpha=0.5, linewidth=2, label='P30 (Type 2)')
ax.set_xlabel('Position', fontsize=11)
ax.set_ylabel('Var_Ratio (TRUE / COLD)', fontsize=11)
ax.set_title('Medical: Variance Ratio Across Positions', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 31)

# Plot 4: dRCI vs Variance Ratio (by domain)
ax = fig.add_subplot(gs[1, 0])
phil_data = df[df['model'].str.contains('Phil')]
med_data = df[df['model'].str.contains('Med')]
ax.scatter(phil_data['var_ratio'], phil_data['drci'], s=30, alpha=0.6, color='steelblue', label='Philosophy')
ax.scatter(med_data['var_ratio'], med_data['drci'], s=30, alpha=0.6, color='crimson', label='Medical')
ax.axvline(1.0, color='red', linestyle='--', alpha=0.3, linewidth=1)
ax.set_xlabel('Variance Ratio (TRUE / COLD)', fontsize=11)
ax.set_ylabel('dRCI', fontsize=11)
ax.set_title('dRCI vs Variance Ratio (By Domain)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)

# Plot 5: Correlation heatmap
ax = fig.add_subplot(gs[1, 1])
corr_matrix = corr_df[['r_var_ratio', 'r_mi_proxy', 'r_mi_proxy_total']].T
model_labels = [m.split(' (')[0] for m in corr_df['model']]
im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
ax.set_xticks(range(len(model_labels)))
ax.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(3))
ax.set_yticklabels(['Var_Ratio', 'VRI', 'VRI_Total'], fontsize=9)
ax.set_title('Correlation with dRCI', fontsize=12, fontweight='bold')
for i in range(3):
    for j in range(len(model_labels)):
        ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha='center', va='center', color='black', fontsize=8)
plt.colorbar(im, ax=ax, label='Pearson r')

# Plot 6: Mean Variance Ratio by Model
ax = fig.add_subplot(gs[1, 2])
model_labels_bar = [m.split(' (')[0] for m in var_df['model']]
colors = ['steelblue' if '(Phil)' in m else 'crimson' for m in var_df['model']]
x = np.arange(len(model_labels_bar))
ax.bar(x, var_df['mean_var_ratio'], color=colors, alpha=0.7)
ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='No Entanglement')
ax.set_ylabel('Mean Var_Ratio (TRUE / COLD)', fontsize=11)
ax.set_title('Mean Variance Ratio by Model', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_labels_bar, rotation=45, ha='right', fontsize=9)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# Plot 7: dRCI Philosophy
ax = fig.add_subplot(gs[2, 0])
for model_name, mdf in all_results.items():
    if '(Phil)' in model_name:
        ax.plot(mdf['position'], mdf['drci'], marker='o', markersize=4,
                alpha=0.7, linewidth=1.5, label=model_name.split(' (')[0])
ax.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)
ax.set_xlabel('Position', fontsize=11)
ax.set_ylabel('dRCI', fontsize=11)
ax.set_title('Philosophy: dRCI Across Positions', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 31)

# Plot 8: dRCI Medical
ax = fig.add_subplot(gs[2, 1])
for model_name, mdf in all_results.items():
    if '(Med)' in model_name:
        ax.plot(mdf['position'], mdf['drci'], marker='o', markersize=4,
                alpha=0.7, linewidth=1.5, label=model_name.split(' (')[0])
ax.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)
ax.axvline(30, color='orange', linestyle=':', alpha=0.5, linewidth=2, label='P30 (Type 2)')
ax.set_xlabel('Position', fontsize=11)
ax.set_ylabel('dRCI', fontsize=11)
ax.set_title('Medical: dRCI Across Positions', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 31)

# Plot 9: P30 Spike
ax = fig.add_subplot(gs[2, 2])
med_models = [name for name in all_results.keys() if '(Med)' in name]
p30_drci = [all_results[m][all_results[m]['position'] == 30]['drci'].values[0] for m in med_models]
p30_var_ratio = [all_results[m][all_results[m]['position'] == 30]['var_ratio'].values[0] for m in med_models]
mlabels = [m.split(' (')[0] for m in med_models]
x = np.arange(len(mlabels))
ax2 = ax.twinx()
ax.bar(x - 0.2, p30_drci, 0.4, label='dRCI at P30', color='crimson', alpha=0.7)
ax2.bar(x + 0.2, p30_var_ratio, 0.4, label='Var_Ratio at P30', color='steelblue', alpha=0.7)
ax.set_ylabel('dRCI at P30', fontsize=11, color='crimson')
ax2.set_ylabel('Var_Ratio at P30', fontsize=11, color='steelblue')
ax.set_title('Medical P30: Spike + Entanglement', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(mlabels, rotation=45, ha='right', fontsize=9)
ax.tick_params(axis='y', labelcolor='crimson')
ax2.tick_params(axis='y', labelcolor='steelblue')
ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)
ax2.axhline(1.0, color='red', linestyle='--', alpha=0.3, linewidth=1)
ax.grid(True, alpha=0.3, axis='x')

# Save to all locations
os.makedirs('c:/Users/barla/mch_experiments/analysis', exist_ok=True)
for path in [
    'c:/Users/barla/mch_experiments/analysis/entanglement_theory_validation.png',
    'c:/Users/barla/mch_experiments/papers/paper4_entanglement/figures/entanglement_validation.png',
    'c:/Users/barla/mch_experiments/results/figures/entanglement_validation.png',
]:
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'[OK] Saved: {path}')

plt.close()

# === Now regenerate the 2-panel multipanel figure ===
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: Entanglement vs Information
ax = axes[0]
for model_name, mdf in all_results.items():
    marker = 'o' if '(Phil)' in model_name else 'x'
    color = 'steelblue' if '(Phil)' in model_name else 'darkorange'
    label_domain = 'philosophy' if '(Phil)' in model_name else 'medical'
    ax.scatter(mdf['mutual_info_proxy'], mdf['drci'], s=40, alpha=0.6,
               color=color, marker=marker)

# Add domain legend entries
ax.scatter([], [], color='steelblue', marker='o', label='philosophy')
ax.scatter([], [], color='darkorange', marker='x', label='medical')

slope, intercept, r, p, _ = linregress(df['mutual_info_proxy'], df['drci'])
x_fit = np.linspace(df['mutual_info_proxy'].min(), df['mutual_info_proxy'].max(), 100)
ax.plot(x_fit, slope * x_fit + intercept, 'k--', alpha=0.7, linewidth=2, label=f'Fit (r={r:.2f})')
ax.axvline(0, color='gray', linestyle=':', alpha=0.5, linewidth=1)
ax.text(-3, 0.35, f'r = {r:.2f}, p < 0.001', fontsize=10)
ax.text(-4.5, 0.15, 'Divergent Region\n(Var > 1)', color='red', fontsize=9, alpha=0.7)
ax.text(0.2, 0.15, 'Convergent Region\n(Var < 1)', color='green', fontsize=9, alpha=0.7)
ax.set_xlabel('VRI (Variance Reduction Index)', fontsize=11)
ax.set_ylabel('ΔRCI (Entanglement)', fontsize=11)
ax.set_title('(A) Entanglement vs. Information', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

# Panel B: Variance Ratio by Model
ax = axes[1]
model_order = var_df.sort_values('mean_var_ratio', ascending=False)
mlabels = [f"{m.split(' (')[0]} ({m.split('(')[1]}" for m in model_order['model']]
colors = ['steelblue' if '(Phil)' in m else 'darkorange' for m in model_order['model']]
x = np.arange(len(mlabels))
ax.bar(x, model_order['mean_var_ratio'].values, color=colors, alpha=0.8)
ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Var Ratio = 1')
ax.text(len(mlabels) - 0.5, 1.02, 'Reference (Var=1)', color='red', fontsize=8, alpha=0.7, ha='right')
ax.bar([], [], color='steelblue', label='Philosophy')
ax.bar([], [], color='darkorange', label='Medical')
ax.set_ylabel('Variance Ratio (σ² scrambled / σ² orig)', fontsize=11)
ax.set_title('(B) Variance Ratio by Model', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(mlabels, rotation=45, ha='right', fontsize=9)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2, axis='y')

plt.tight_layout()

# Save multipanel
for path in [
    'c:/Users/barla/mch_experiments/papers/paper4_entanglement/figures/fig4_entanglement_multipanel.png',
    'c:/Users/barla/mch_experiments/results/figures/fig4_entanglement_multipanel.png',
    'c:/Users/barla/mch_experiments/docs/figures/publication/fig4_entanglement_multipanel.png',
]:
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'[OK] Saved: {path}')

plt.close()
print('\nAll figures regenerated with VRI labels!')
