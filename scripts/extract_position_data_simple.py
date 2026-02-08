#!/usr/bin/env python3
"""
Extract Position-Specific Data from existing position_drci_data.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import seaborn as sns

print("="*80)
print("EXTRACTING POSITION-SPECIFIC DATA")
print("="*80)

# Load the position-level data
df = pd.read_csv('c:/Users/barla/mch_experiments/analysis/position_drci_data.csv')

# Note: mean_drci_cold appears to be dRCI (TRUE - COLD) based on position_analysis scripts
# Let's verify by checking if values match known ranges

print(f"\nLoaded {len(df)} rows")
print(f"Models: {df['model'].unique()}")
print(f"Domains: {df['domain'].unique()}")

# ============================================================================
# PHILOSOPHY: Extract P10 and P30
# ============================================================================

print("\n" + "="*80)
print("PHILOSOPHY MODELS: P10 vs P30")
print("="*80)

philosophy_df = df[df['domain'] == 'philosophy'].copy()
philosophy_models = philosophy_df['model'].unique()

p10_p30_data = []

for model in philosophy_models:
    model_data = philosophy_df[philosophy_df['model'] == model]

    p10_data = model_data[model_data['position'] == 10]
    p30_data = model_data[model_data['position'] == 30]

    if len(p10_data) > 0 and len(p30_data) > 0:
        p10_drci = p10_data['mean_drci_cold'].values[0]
        p30_drci = p30_data['mean_drci_cold'].values[0]

        # Context for P10 (positions 8-12)
        context_data = model_data[(model_data['position'] >= 8) & (model_data['position'] <= 12)]
        p10_context_mean = context_data['mean_drci_cold'].mean()
        p10_context_std = context_data['mean_drci_cold'].std()

        p10_p30_data.append({
            'model': model,
            'p10_drci': p10_drci,
            'p30_drci': p30_drci,
            'diff': p30_drci - p10_drci,
            'p10_context_mean': p10_context_mean,
            'p10_context_std': p10_context_std
        })

        print(f"  {model:20s}: P10={p10_drci:.4f}, P30={p30_drci:.4f}, Diff={p30_drci - p10_drci:+.4f}")

p10_p30_df = pd.DataFrame(p10_p30_data)

if len(p10_p30_df) > 0:
    print(f"\nPhilosophy Summary:")
    print(f"  P10 mean: {p10_p30_df['p10_drci'].mean():.4f} +/- {p10_p30_df['p10_drci'].std():.4f}")
    print(f"  P30 mean: {p10_p30_df['p30_drci'].mean():.4f} +/- {p10_p30_df['p30_drci'].std():.4f}")
    print(f"  Mean difference (P30 - P10): {p10_p30_df['diff'].mean():+.4f}")

    if len(p10_p30_df) > 1:
        t_stat, p_val = ttest_rel(p10_p30_df['p30_drci'], p10_p30_df['p10_drci'])
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"  Paired t-test (P30 vs P10): t={t_stat:.3f}, p={p_val:.4f} {sig}")

# ============================================================================
# MEDICAL: Extract P29 and P30
# ============================================================================

print("\n" + "="*80)
print("MEDICAL MODELS: P29 vs P30 (Spike Analysis)")
print("="*80)

medical_df = df[df['domain'] == 'medical'].copy()
medical_models = medical_df['model'].unique()

p29_p30_data = []

for model in medical_models:
    model_data = medical_df[medical_df['model'] == model]

    p29_data = model_data[model_data['position'] == 29]
    p30_data = model_data[model_data['position'] == 30]

    if len(p29_data) > 0 and len(p30_data) > 0:
        p29_drci = p29_data['mean_drci_cold'].values[0]
        p30_drci = p30_data['mean_drci_cold'].values[0]
        spike = p30_drci - p29_drci

        # Z-score for P30
        pos_1_29 = model_data[(model_data['position'] >= 1) & (model_data['position'] <= 29)]
        mean_1_29 = pos_1_29['mean_drci_cold'].mean()
        std_1_29 = pos_1_29['mean_drci_cold'].std()
        z_score = (p30_drci - mean_1_29) / std_1_29 if std_1_29 > 0 else 0

        p29_p30_data.append({
            'model': model,
            'p29_drci': p29_drci,
            'p30_drci': p30_drci,
            'spike': spike,
            'z_score': z_score
        })

        print(f"  {model:20s}: P29={p29_drci:.4f}, P30={p30_drci:.4f}, Spike={spike:+.4f}, Z={z_score:+.2f}")

p29_p30_df = pd.DataFrame(p29_p30_data)

if len(p29_p30_df) > 0:
    print(f"\nMedical Summary:")
    print(f"  P29 mean: {p29_p30_df['p29_drci'].mean():.4f} +/- {p29_p30_df['p29_drci'].std():.4f}")
    print(f"  P30 mean: {p29_p30_df['p30_drci'].mean():.4f} +/- {p29_p30_df['p30_drci'].std():.4f}")
    print(f"  Mean spike (P30 - P29): {p29_p30_df['spike'].mean():+.4f}")
    print(f"  Mean Z-score: {p29_p30_df['z_score'].mean():+.2f} +/- {p29_p30_df['z_score'].std():.2f}")

    if len(p29_p30_df) > 1:
        t_stat, p_val = ttest_rel(p29_p30_df['p30_drci'], p29_p30_df['p29_drci'])
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"  Paired t-test (P30 vs P29): t={t_stat:.3f}, p={p_val:.4e} {sig}")

# ============================================================================
# PHILOSOPHY P10 CONTEXT ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("PHILOSOPHY P10 CONTEXT ANALYSIS")
print("="*80)

print("\nIs P10 different from surrounding positions (P8-P12)?")

for _, row in p10_p30_df.iterrows():
    model_name = row['model']
    p10_drci = row['p10_drci']
    context_mean = row['p10_context_mean']
    context_std = row['p10_context_std']

    deviation = abs(p10_drci - context_mean) / context_std if context_std > 0 else 0

    print(f"  {model_name:20s}: P10={p10_drci:.4f}, Context={context_mean:.4f}+/-{context_std:.4f}, Z={deviation:+.2f}")

# ============================================================================
# GENERATE PLOTS
# ============================================================================

print("\n" + "="*80)
print("GENERATING PLOTS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1A: Philosophy P10 vs P30 scatter
ax = axes[0, 0]
if len(p10_p30_df) > 0:
    ax.scatter(p10_p30_df['p10_drci'], p10_p30_df['p30_drci'], s=100, alpha=0.7, c='steelblue')

    for _, row in p10_p30_df.iterrows():
        ax.annotate(row['model'], (row['p10_drci'], row['p30_drci']),
                    fontsize=8, ha='right', va='bottom', alpha=0.7)

    lim_min = min(p10_p30_df['p10_drci'].min(), p10_p30_df['p30_drci'].min()) - 0.02
    lim_max = max(p10_p30_df['p10_drci'].max(), p10_p30_df['p30_drci'].max()) + 0.02
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.3, linewidth=1, label='P10 = P30')

    ax.set_xlabel('Position 10 dRCI', fontsize=11)
    ax.set_ylabel('Position 30 dRCI', fontsize=11)
    ax.set_title('Philosophy: P10 vs P30 (Type 2 at different positions)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Plot 1B: Philosophy P10-P30 differences
ax = axes[0, 1]
if len(p10_p30_df) > 0:
    diffs = p10_p30_df['diff']
    colors = ['red' if d > 0 else 'blue' for d in diffs]
    y_pos = range(len(p10_p30_df))
    ax.barh(y_pos, diffs, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(p10_p30_df['model'], fontsize=9)
    ax.axvline(0, color='black', linewidth=1, linestyle='-')
    ax.set_xlabel('dRCI Difference (P30 - P10)', fontsize=11)
    ax.set_title('Philosophy: Type 2 Effect at P30 vs P10', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

# Plot 2A: Medical P30 vs P29 spike
ax = axes[1, 0]
if len(p29_p30_df) > 0:
    x = np.arange(len(p29_p30_df))
    width = 0.35

    ax.bar(x - width/2, p29_p30_df['p29_drci'], width, label='Position 29', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, p29_p30_df['p30_drci'], width, label='Position 30', color='crimson', alpha=0.8)

    ax.set_ylabel('dRCI', fontsize=11)
    ax.set_title('Medical: Position 30 Spike vs Position 29', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(p29_p30_df['model'], rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

# Plot 2B: Medical full position curves
ax = axes[1, 1]
for model in medical_models:
    model_data = medical_df[medical_df['model'] == model]
    ax.plot(model_data['position'], model_data['mean_drci_cold'],
            marker='o', markersize=4, alpha=0.7, linewidth=1.5, label=model)

ax.axvline(30, color='red', linestyle='--', alpha=0.3, linewidth=2, label='P30 (summarization)')
ax.axvline(10, color='green', linestyle='--', alpha=0.3, linewidth=2, label='P10 (early summary)')

ax.set_xlabel('Position in Conversation', fontsize=11)
ax.set_ylabel('dRCI', fontsize=11)
ax.set_title('Medical: Full Position Curves (1-30)', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 31)

plt.tight_layout()
output_path = "c:/Users/barla/mch_experiments/analysis/position_specific_extracted.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"[OK] Plot saved: {output_path}")

# ============================================================================
# SAVE EXTRACTED DATA
# ============================================================================

print("\n" + "="*80)
print("SAVING DATA")
print("="*80)

p10_p30_df.to_csv('c:/Users/barla/mch_experiments/analysis/philosophy_p10_p30_extracted.csv', index=False)
print("[OK] Philosophy P10/P30 data: analysis/philosophy_p10_p30_extracted.csv")

p29_p30_df.to_csv('c:/Users/barla/mch_experiments/analysis/medical_p29_p30_spike.csv', index=False)
print("[OK] Medical P29/P30 spike data: analysis/medical_p29_p30_spike.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
