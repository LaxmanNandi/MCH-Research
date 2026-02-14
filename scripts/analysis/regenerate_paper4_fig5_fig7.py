#!/usr/bin/env python3
"""
Regenerate Paper 4 Figure 5 (Independence: RCI vs Variance Ratio)
and Figure 7 (Llama Safety Anomaly at P30).

Reads from pre-computed entanglement_position_data.csv.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load data
df = pd.read_csv('c:/Users/barla/mch_experiments/results/tables/entanglement_position_data.csv')

# ============================================================================
# FIGURE 5: Independence Test - RCI vs Variance Ratio
# ============================================================================
print("Generating Figure 5: Independence test (RCI vs Variance Ratio)...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: TRUE self-sim vs Var_Ratio
ax = axes[0]
phil = df[df['model'].str.contains('Phil')]
med = df[df['model'].str.contains('Med')]

ax.scatter(phil['true_self_sim'], phil['var_ratio'], s=25, alpha=0.5,
           color='steelblue', label='Philosophy')
ax.scatter(med['true_self_sim'], med['var_ratio'], s=25, alpha=0.5,
           color='crimson', label='Medical')

r_true, p_true = pearsonr(df['true_self_sim'], df['var_ratio'])
ax.set_xlabel('TRUE Self-Similarity (RCI_TRUE)', fontsize=11)
ax.set_ylabel('Variance Ratio (TRUE / COLD)', fontsize=11)
ax.set_title(f'(A) RCI_TRUE vs Var_Ratio (r={r_true:.3f}, p={p_true:.2e})', fontsize=12, fontweight='bold')
ax.axhline(1.0, color='red', linestyle='--', alpha=0.3)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)

# Panel B: COLD self-sim vs Var_Ratio
ax = axes[1]
ax.scatter(phil['cold_self_sim'], phil['var_ratio'], s=25, alpha=0.5,
           color='steelblue', label='Philosophy')
ax.scatter(med['cold_self_sim'], med['var_ratio'], s=25, alpha=0.5,
           color='crimson', label='Medical')

r_cold, p_cold = pearsonr(df['cold_self_sim'], df['var_ratio'])
ax.set_xlabel('COLD Self-Similarity (RCI_COLD)', fontsize=11)
ax.set_ylabel('Variance Ratio (TRUE / COLD)', fontsize=11)
ax.set_title(f'(B) RCI_COLD vs Var_Ratio (r={r_cold:.3f}, p={p_cold:.2e})', fontsize=12, fontweight='bold')
ax.axhline(1.0, color='red', linestyle='--', alpha=0.3)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)

plt.tight_layout()

for path in [
    'c:/Users/barla/mch_experiments/papers/paper4_entanglement/figures/fig5_independence_rci_var.png',
]:
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'  [OK] Saved: {path}')
plt.close()

# ============================================================================
# FIGURE 7: Llama Safety Anomaly at Medical P30
# ============================================================================
print("\nGenerating Figure 7: Llama safety anomaly (P30 medical)...")

# Get P30 data for all medical models
med_df = df[df['model'].str.contains('Med')]
p30_data = med_df[med_df['position'] == 30].copy()
p30_data['short_name'] = p30_data['model'].apply(lambda x: x.split(' (')[0])

# Sort by var_ratio for visualization
p30_data = p30_data.sort_values('var_ratio')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Var_Ratio at P30 by model
ax = axes[0]
colors = []
for _, row in p30_data.iterrows():
    if 'Llama' in row['short_name']:
        colors.append('red')
    elif row['var_ratio'] < 1.0:
        colors.append('green')
    else:
        colors.append('orange')

x = np.arange(len(p30_data))
bars = ax.bar(x, p30_data['var_ratio'].values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.axhline(1.0, color='black', linestyle='--', alpha=0.5, linewidth=1.5, label='Var_Ratio = 1')
ax.axhline(3.0, color='red', linestyle=':', alpha=0.5, linewidth=1.5, label='Safety threshold (3.0)')
ax.set_ylabel('Var_Ratio at P30', fontsize=11)
ax.set_title('(A) P30 Variance Ratio by Model', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(p30_data['short_name'].values, rotation=45, ha='right', fontsize=9)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2, axis='y')

# Panel B: DRCI vs Var_Ratio at P30 (scatter)
ax = axes[1]
for _, row in p30_data.iterrows():
    if 'Llama' in row['short_name']:
        color, marker, size = 'red', 'X', 120
    elif row['var_ratio'] < 1.0:
        color, marker, size = 'green', 'o', 80
    else:
        color, marker, size = 'orange', 's', 80
    ax.scatter(row['var_ratio'], row['drci'], s=size, color=color, marker=marker,
               edgecolors='black', linewidth=0.5, zorder=5)
    ax.annotate(row['short_name'], (row['var_ratio'], row['drci']),
                fontsize=8, ha='center', va='bottom', xytext=(0, 8),
                textcoords='offset points')

ax.axvline(1.0, color='black', linestyle='--', alpha=0.3)
ax.axhline(0.0, color='black', linestyle='-', alpha=0.3)
ax.axvline(3.0, color='red', linestyle=':', alpha=0.3, label='Safety threshold')

# Add regime labels
ax.text(0.3, 0.15, 'Convergent\n(Stable)', color='green', fontsize=10, alpha=0.7, ha='center')
ax.text(5.0, -0.15, 'Divergent\n(Unsafe)', color='red', fontsize=10, alpha=0.7, ha='center')

# Legend
ax.scatter([], [], color='green', marker='o', s=60, label='Convergent (VR < 1)')
ax.scatter([], [], color='orange', marker='s', s=60, label='Mild divergent (1 < VR < 3)')
ax.scatter([], [], color='red', marker='X', s=80, label='Extreme divergent (Llama)')

ax.set_xlabel('Var_Ratio at P30', fontsize=11)
ax.set_ylabel('ΔRCI at P30', fontsize=11)
ax.set_title('(B) P30 Entanglement Regime Map', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.2)

plt.tight_layout()

for path in [
    'c:/Users/barla/mch_experiments/papers/paper4_entanglement/figures/fig7_llama_safety_anomaly.png',
]:
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'  [OK] Saved: {path}')
plt.close()

print("\nAll figures regenerated!")
