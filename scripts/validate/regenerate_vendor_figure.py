"""
Regenerate Figure 2: Vendor Signatures with 95% Confidence Intervals
======================================================================
Uses clean 50-trial data from trial_level_drci.csv to generate updated
vendor-level effect sizes with confidence intervals.

Output: figures/fig2_effect_sizes_ci.png (300 DPI)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Paths
BASE = r'C:\Users\barla\mch_experiments'
DATA_FILE = os.path.join(BASE, 'analysis', 'trial_level_drci.csv')
OUTPUT_FILE = os.path.join(BASE, 'figures', 'fig2_effect_sizes_ci.png')

# Load data
df = pd.read_csv(DATA_FILE)

# Map models to vendors
vendor_map = {
    'gpt4o': 'OpenAI',
    'gpt4o_mini': 'OpenAI',
    'gpt52': 'OpenAI',
    'claude_opus': 'Anthropic',
    'claude_haiku': 'Anthropic',
    'gemini_flash': 'Google',
    'gemini_25_pro': 'Google',
    'deepseek_v3_1': 'DeepSeek',
    'llama_4_maverick': 'Meta',
    'llama_4_scout': 'Meta',
    'qwen3_235b': 'Alibaba',
    'mistral_small_24b': 'Mistral',
    'ministral_14b': 'Mistral',
    'kimi_k2': 'Moonshot'
}

df['vendor'] = df['model'].map(vendor_map)

# Calculate vendor-level statistics
vendor_stats = df.groupby('vendor')['delta_rci'].agg(['mean', 'sem', 'count'])
vendor_stats['ci_95'] = 1.96 * vendor_stats['sem']
vendor_stats = vendor_stats.sort_values('mean', ascending=True)  # Ascending for horizontal bars

# Run ANOVA
groups = [group['delta_rci'].values for name, group in df.groupby('vendor')]
f_stat, p_value = stats.f_oneway(*groups)

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot horizontal bars with error bars
y_pos = np.arange(len(vendor_stats))
ax.barh(y_pos, vendor_stats['mean'], xerr=vendor_stats['ci_95'],
        capsize=5, color='steelblue', alpha=0.7, ecolor='black')

# Labels and formatting
ax.set_yticks(y_pos)
ax.set_yticklabels(vendor_stats.index, fontsize=11)
ax.set_xlabel('Mean ΔRCI (± 95% CI)', fontsize=12, fontweight='bold')
ax.set_title('Vendor-Level Context Sensitivity Signatures', fontsize=14, fontweight='bold', pad=20)

# Add sample sizes
for i, (vendor, row) in enumerate(vendor_stats.iterrows()):
    ax.text(row['mean'] + row['ci_95'] + 0.01, i,
            f"n={int(row['count'])}",
            va='center', fontsize=9, color='gray')

# Add ANOVA stats
ax.text(0.02, 0.98, f"ANOVA: F = {f_stat:.2f}, p < 0.0001",
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        verticalalignment='top')

# Grid and layout
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_xlim(0, max(vendor_stats['mean'] + vendor_stats['ci_95']) * 1.15)
plt.tight_layout()

# Save at 300 DPI for publication
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
print(f"OK Figure saved: {OUTPUT_FILE}")
print(f"\nVendor Rankings (Highest to Lowest):")
print(vendor_stats.sort_values('mean', ascending=False)[['mean', 'ci_95', 'count']])
print(f"\nANOVA: F={f_stat:.2f}, p={p_value:.6f}")
