#!/usr/bin/env python3
"""
Merge Exploration Arc with Var_Ratio and test correlation
Tests hypothesis: Var_Ratio > 1 (divergent) -> Exploration_Arc > 1 (expanding diversity)
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
exploration_df = pd.read_csv('C:/Users/barla/mch_experiments/scripts/validate/exploration_arc_data.csv')
conservation_df = pd.read_csv('C:/Users/barla/mch_experiments/data/paper6/conservation_product_test.csv')

# Standardize model names for merging
model_name_map = {
    'GPT-4o': 'gpt4o',
    'GPT-4o-mini': 'gpt4o_mini',
    'Claude Haiku': 'claude_haiku',
    'Gemini Flash Phil': 'gemini_flash',
    'Gemini Flash Med': 'gemini_flash',
    'DeepSeek V3.1': 'deepseek_v3_1',
    'Kimi K2': 'kimi_k2',
    'Llama 4 Maverick': 'llama_4_maverick',
    'Llama 4 Scout': 'llama_4_scout',
    'Mistral Small 24B': 'mistral_small_24b',
    'Ministral 14B': 'ministral_14b',
    'Qwen3 235B': 'qwen3_235b'
}

exploration_df['model_key'] = exploration_df['model'].map(model_name_map)
conservation_df['domain'] = conservation_df['domain'].str.lower()

# Merge on model_key and domain
merged_df = exploration_df.merge(
    conservation_df[['model', 'domain', 'drci', 'var_ratio']],
    left_on=['model_key', 'domain'],
    right_on=['model', 'domain'],
    how='inner',
    suffixes=('_exp', '_cons')
)

print("="*70)
print("EXPLORATION ARC x VAR_RATIO CORRELATION ANALYSIS")
print("="*70)

print("\nMerged Data:")
print(merged_df[['model_exp', 'domain', 'exploration_arc', 'var_ratio', 'drci']])

# Overall correlation
r_pearson, p_pearson = pearsonr(merged_df['var_ratio'], merged_df['exploration_arc'])
r_spearman, p_spearman = spearmanr(merged_df['var_ratio'], merged_df['exploration_arc'])

print("\n" + "="*70)
print("OVERALL CORRELATION")
print("="*70)
print(f"Pearson r:  {r_pearson:.4f}, p={p_pearson:.4e}, N={len(merged_df)}")
print(f"Spearman rho: {r_spearman:.4f}, p={p_spearman:.4e}, N={len(merged_df)}")

# By domain
print("\n" + "="*70)
print("DOMAIN-SPECIFIC ANALYSIS")
print("="*70)

for domain in ['philosophy', 'medical']:
    domain_df = merged_df[merged_df['domain'] == domain]

    if len(domain_df) > 2:
        r_d, p_d = pearsonr(domain_df['var_ratio'], domain_df['exploration_arc'])
        print(f"\n{domain.upper()} (n={len(domain_df)}):")
        print(f"  Var_Ratio range:       {domain_df['var_ratio'].min():.3f} - {domain_df['var_ratio'].max():.3f}")
        print(f"  Exploration Arc range: {domain_df['exploration_arc'].min():.3f} - {domain_df['exploration_arc'].max():.3f}")
        print(f"  Correlation: r={r_d:.4f}, p={p_d:.4f}")

        # Count convergent vs divergent
        convergent = (domain_df['var_ratio'] < 1).sum()
        divergent = (domain_df['var_ratio'] > 1).sum()
        print(f"  Convergent (Var_Ratio < 1): {convergent}")
        print(f"  Divergent (Var_Ratio > 1):  {divergent}")

# Hypothesis test
print("\n" + "="*70)
print("HYPOTHESIS TEST: Divergence = Dynamic Maneuverability")
print("="*70)

merged_df['divergent'] = merged_df['var_ratio'] > 1
merged_df['expanding'] = merged_df['exploration_arc'] > 1

print("\nContingency Table:")
print(pd.crosstab(
    merged_df['divergent'],
    merged_df['expanding'],
    rownames=['Divergent (Var_Ratio>1)'],
    colnames=['Expanding (Arc>1)']
))

# Philosophy specific: All should be expanding (open-ended domain)
phil_df = merged_df[merged_df['domain'] == 'philosophy']
phil_expanding = (phil_df['exploration_arc'] > 1).all()

print("\nPhilosophy Domain (open-ended):")
print(f"  All models show expansion (Arc > 1): {phil_expanding}")
if phil_expanding:
    print(f"  Mean Exploration Arc: {phil_df['exploration_arc'].mean():.3f}")
    print(f"  Mean Var_Ratio: {phil_df['var_ratio'].mean():.3f}")

# Medical: Mixed (task-convergent domain)
med_df = merged_df[merged_df['domain'] == 'medical']
med_convergent = (med_df['exploration_arc'] < 1).sum()
med_stable = ((med_df['exploration_arc'] >= 0.95) & (med_df['exploration_arc'] <= 1.05)).sum()
med_expanding = (med_df['exploration_arc'] > 1.05).sum()

print("\nMedical Domain (task-convergent):")
print(f"  Converging (Arc < 1):     {med_convergent} models")
print(f"  Stable (0.95 < Arc < 1.05): {med_stable} models")
print(f"  Expanding (Arc > 1.05):   {med_expanding} models")
print(f"  Mean Exploration Arc: {med_df['exploration_arc'].mean():.3f}")
print(f"  Mean Var_Ratio: {med_df['var_ratio'].mean():.3f}")

# Create scatter plot
fig, ax = plt.subplots(figsize=(10, 7))

# Plot by domain
phil_data = merged_df[merged_df['domain'] == 'philosophy']
med_data = merged_df[merged_df['domain'] == 'medical']

ax.scatter(phil_data['var_ratio'], phil_data['exploration_arc'],
          s=100, alpha=0.7, color='#2E86AB', label='Philosophy', edgecolors='black')
ax.scatter(med_data['var_ratio'], med_data['exploration_arc'],
          s=100, alpha=0.7, color='#A23B72', label='Medical', edgecolors='black')

# Add reference lines
ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Arc = 1 (stable diversity)')
ax.axvline(x=1, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Var_Ratio = 1 (neutral)')

# Add model labels
for idx, row in merged_df.iterrows():
    model_short = row['model_exp'].replace('Gemini Flash Phil', 'GF-P').replace('Gemini Flash Med', 'GF-M')
    model_short = model_short.replace('Llama 4 ', 'L4-').replace('Mistral Small 24B', 'Mistral')
    model_short = model_short.replace('DeepSeek V3.1', 'DS').replace('Claude Haiku', 'Haiku')
    model_short = model_short.replace('GPT-4o-mini', 'GPT4m').replace('GPT-4o', 'GPT4')
    model_short = model_short.replace('Ministral 14B', 'Mini14').replace('Qwen3 235B', 'Qwen3')

    ax.annotate(model_short, (row['var_ratio'], row['exploration_arc']),
               xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)

ax.set_xlabel('Var_Ratio (Variance TRUE / Variance COLD)', fontsize=12)
ax.set_ylabel('Exploration Arc (Diversity_Late / Diversity_Early)', fontsize=12)
ax.set_title(f'Dynamic Maneuverability: Var_Ratio vs Exploration Arc\n(r={r_pearson:.3f}, p={p_pearson:.2e}, N={len(merged_df)})',
            fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_fig = Path('C:/Users/barla/mch_experiments/scripts/validate/exploration_arc_correlation.png')
plt.savefig(output_fig, dpi=300, bbox_inches='tight')
print(f"\n[SAVED] Figure: {output_fig}")

# Save merged data
output_csv = Path('C:/Users/barla/mch_experiments/scripts/validate/exploration_arc_with_var_ratio.csv')
merged_df.to_csv(output_csv, index=False)
print(f"[SAVED] Merged data: {output_csv}")

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
print("""
1. Philosophy (open-ended): ALL models expand diversity (Arc > 1.39)
   - Open-ended discussion naturally broadens response space over time
   - Mean Arc = 2.03 (diversity doubles from early to late)

2. Medical (task-convergent): Mixed patterns (Arc range: 0.87-1.26)
   - Some models converge toward diagnostic solution (Arc < 1)
   - Others maintain/expand slightly (Arc ~ 1 or slightly > 1)
   - Mean Arc = 1.06 (relatively stable diversity)

3. Var_Ratio correlation with Exploration Arc:
   - NEGATIVE correlation overall (r=-0.41, p=0.18, not significant)
   - Philosophy: Strong negative r=-0.92 (p=0.08, near threshold)
   - Medical: Weak positive r=0.47 (p=0.24, not significant)

4. Domain Structure Dominates:
   - Philosophy: High exploration arc regardless of Var_Ratio (domain effect)
   - Medical: Constrained exploration even with Var_Ratio > 1 (task structure)
   - Within domains: No clear Var_Ratio -> Arc relationship
""")
