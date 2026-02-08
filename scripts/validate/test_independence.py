#!/usr/bin/env python3
"""
Independence Test: Base Capability vs Entanglement
Tests whether RCI_COLD (base capability) and dRCI (entanglement) are independent dimensions.

Hypothesis: AI cognition has two separable axes:
- Dimension 1: Base Capability (RCI_COLD) = zero-shot competence
- Dimension 2: Entanglement (dRCI, Var_Ratio) = context coupling

If truly independent: |r| < 0.2, regression beta not significant
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm

print("="*80)
print("INDEPENDENCE TEST: BASE CAPABILITY vs ENTANGLEMENT")
print("="*80)

# Models with full response data (8 models across 2 domains)
models = {
    # Philosophy
    'GPT-4o-mini (Phil)': 'c:/Users/barla/mch_experiments/data/closed_model_philosophy_rerun/mch_results_gpt4o_mini_philosophy_50trials.json',
    'GPT-4o (Phil)': 'c:/Users/barla/mch_experiments/data/closed_model_philosophy_rerun/mch_results_gpt4o_philosophy_50trials.json',
    'Claude Haiku (Phil)': 'c:/Users/barla/mch_experiments/data/closed_model_philosophy_rerun/mch_results_claude_haiku_philosophy_50trials.json',
    'Gemini Flash (Phil)': 'c:/Users/barla/mch_experiments/data/closed_model_philosophy_rerun/mch_results_gemini_flash_philosophy_50trials.json',
    # Medical
    'Gemini Flash (Med)': 'c:/Users/barla/mch_experiments/data/gemini_flash_medical_rerun/mch_results_gemini_flash_medical_50trials.json',
    'DeepSeek V3.1 (Med)': 'c:/Users/barla/mch_experiments/data/open_medical_rerun/mch_results_deepseek_v3_1_medical_50trials.json',
    'Llama 4 Maverick (Med)': 'c:/Users/barla/mch_experiments/data/open_medical_rerun/mch_results_llama_4_maverick_medical_50trials.json',
    'Llama 4 Scout (Med)': 'c:/Users/barla/mch_experiments/data/open_medical_rerun/mch_results_llama_4_scout_medical_50trials.json',
}

# ============================================================================
# EXTRACT DATA
# ============================================================================

all_data = []

print("\nExtracting data from 8 models...")

for model_name, file_path in models.items():
    print(f"  {model_name}...")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract domain
    domain = 'Philosophy' if '(Phil)' in model_name else 'Medical'

    # For each position (1-30)
    n_prompts = len(data['trials'][0]['prompts'])

    for pos in range(n_prompts):
        # Compute RCI_COLD for this position across all trials
        cold_aligns = []
        true_aligns = []
        var_ratios = []

        for trial in data['trials']:
            cold_aligns.append(trial['alignments']['cold'][pos])
            true_aligns.append(trial['alignments']['true'][pos])

        # Mean RCI values
        rci_cold = np.mean(cold_aligns)
        rci_true = np.mean(true_aligns)
        drci = rci_true - rci_cold

        all_data.append({
            'model': model_name,
            'domain': domain,
            'position': pos + 1,
            'rci_cold': rci_cold,
            'rci_true': rci_true,
            'drci': drci
        })

df = pd.DataFrame(all_data)

print(f"\nTotal data points: {len(df)}")
print(f"  Philosophy: {len(df[df['domain'] == 'Philosophy'])}")
print(f"  Medical: {len(df[df['domain'] == 'Medical'])}")

# ============================================================================
# TEST 1: POOLED CORRELATION
# ============================================================================

print("\n" + "="*80)
print("TEST 1: POOLED CORRELATION (N = 240)")
print("="*80)

r_pooled, p_pooled = pearsonr(df['rci_cold'], df['drci'])

print(f"\nPearson correlation:")
print(f"  r = {r_pooled:.4f}")
print(f"  p = {p_pooled:.4e}")
print(f"  N = {len(df)}")

if abs(r_pooled) < 0.2:
    print(f"\n  RESULT: Independence SUPPORTED (|r| = {abs(r_pooled):.3f} < 0.2)")
elif abs(r_pooled) < 0.3:
    print(f"\n  RESULT: Weak correlation (|r| = {abs(r_pooled):.3f})")
else:
    print(f"\n  RESULT: Independence QUESTIONABLE (|r| = {abs(r_pooled):.3f} >= 0.3)")

# ============================================================================
# TEST 2: PER-MODEL CORRELATION
# ============================================================================

print("\n" + "="*80)
print("TEST 2: PER-MODEL CORRELATION")
print("="*80)

model_correlations = []

for model_name in df['model'].unique():
    model_df = df[df['model'] == model_name]
    r, p = pearsonr(model_df['rci_cold'], model_df['drci'])
    model_correlations.append({
        'model': model_name,
        'r': r,
        'p_uncorrected': p,
        'n': len(model_df)
    })

# FDR correction (Benjamini-Hochberg)
p_values = [m['p_uncorrected'] for m in model_correlations]
reject, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

for i, m in enumerate(model_correlations):
    m['p_corrected'] = p_corrected[i]
    m['significant'] = reject[i]

model_corr_df = pd.DataFrame(model_correlations)

print("\nPer-model correlations (FDR-corrected):")
for _, row in model_corr_df.iterrows():
    sig = "*" if row['significant'] else ""
    print(f"  {row['model']:30s}: r={row['r']:+.3f}, p={row['p_corrected']:.4f} {sig}")

print(f"\nSummary statistics:")
print(f"  Median r = {model_corr_df['r'].median():.4f}")
print(f"  IQR = [{model_corr_df['r'].quantile(0.25):.4f}, {model_corr_df['r'].quantile(0.75):.4f}]")
print(f"  Significant correlations: {model_corr_df['significant'].sum()}/{len(model_corr_df)}")

# ============================================================================
# TEST 3: PER-DOMAIN CORRELATION
# ============================================================================

print("\n" + "="*80)
print("TEST 3: PER-DOMAIN CORRELATION")
print("="*80)

for domain in ['Philosophy', 'Medical']:
    domain_df = df[df['domain'] == domain]
    r, p = pearsonr(domain_df['rci_cold'], domain_df['drci'])
    print(f"\n{domain}:")
    print(f"  r = {r:+.4f}, p = {p:.4e}, N = {len(domain_df)}")

# ============================================================================
# TEST 4: REGRESSION ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("TEST 4: REGRESSION ANALYSIS")
print("="*80)

# Prepare regression data
reg_df = df.copy()
reg_df['domain_binary'] = (reg_df['domain'] == 'Medical').astype(int)

# Model: drci ~ rci_cold + domain + position
X = reg_df[['rci_cold', 'domain_binary', 'position']]
X = sm.add_constant(X)
y = reg_df['drci']

model = sm.OLS(y, X).fit()

print("\nRegression: dRCI ~ RCI_COLD + Domain + Position")
print(model.summary())

beta_rci_cold = model.params['rci_cold']
se_rci_cold = model.bse['rci_cold']
p_rci_cold = model.pvalues['rci_cold']

print(f"\nCoefficient for RCI_COLD:")
print(f"  beta = {beta_rci_cold:+.4f}")
print(f"  SE = {se_rci_cold:.4f}")
print(f"  p = {p_rci_cold:.4e}")

if p_rci_cold > 0.05:
    print(f"\n  RESULT: Independence SUPPORTED (p = {p_rci_cold:.4f} > 0.05)")
else:
    print(f"\n  RESULT: Independence QUESTIONABLE (p = {p_rci_cold:.4e} < 0.05)")

# ============================================================================
# TEST 5: PCA ORTHOGONALITY CHECK
# ============================================================================

print("\n" + "="*80)
print("TEST 5: PCA ORTHOGONALITY CHECK")
print("="*80)

# Feature matrix: [RCI_COLD, dRCI]
# Note: Var_Ratio requires response-level analysis, skip for now
features = df[['rci_cold', 'drci']].values

# Standardize
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# PCA
pca = PCA(n_components=2)
components = pca.fit_transform(features_scaled)

print("\nPCA Results:")
print(f"  Explained variance: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")
print(f"\nComponent loadings:")
print(f"             PC1      PC2")
print(f"  RCI_COLD:  {pca.components_[0,0]:+.3f}   {pca.components_[1,0]:+.3f}")
print(f"  dRCI:      {pca.components_[0,1]:+.3f}   {pca.components_[1,1]:+.3f}")

# Check if loadings are approximately orthogonal
loading_corr = abs(pca.components_[0,0] * pca.components_[0,1] +
                   pca.components_[1,0] * pca.components_[1,1])

print(f"\nLoading orthogonality: {loading_corr:.4f}")
if loading_corr < 0.3:
    print("  RESULT: Orthogonal loading (suggests independence)")
else:
    print("  RESULT: Non-orthogonal loading")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("GENERATING PLOTS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Pooled scatter
ax = axes[0, 0]
colors = {'Philosophy': '#2E86AB', 'Medical': '#A23B72'}
for domain in ['Philosophy', 'Medical']:
    domain_df = df[df['domain'] == domain]
    ax.scatter(domain_df['rci_cold'], domain_df['drci'],
              alpha=0.5, s=30, c=colors[domain], label=domain)

# Regression line with 95% CI
x_range = np.linspace(df['rci_cold'].min(), df['rci_cold'].max(), 100)
slope, intercept = np.polyfit(df['rci_cold'], df['drci'], 1)
y_pred = slope * x_range + intercept
ax.plot(x_range, y_pred, 'k--', alpha=0.8, linewidth=2, label=f'r={r_pooled:.3f}')

ax.set_xlabel('RCI_COLD (Base Capability)', fontsize=11)
ax.set_ylabel('ΔRCI (Entanglement)', fontsize=11)
ax.set_title(f'Pooled Correlation (N={len(df)})\nr = {r_pooled:+.3f}, p = {p_pooled:.4e}', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)
ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
ax.axvline(df['rci_cold'].mean(), color='gray', linestyle=':', linewidth=0.5)

# Plot 2: Per-model correlations
ax = axes[0, 1]
model_corr_sorted = model_corr_df.sort_values('r')
colors_sig = ['red' if sig else 'gray' for sig in model_corr_sorted['significant']]
ax.barh(range(len(model_corr_sorted)), model_corr_sorted['r'], color=colors_sig, alpha=0.7)
ax.set_yticks(range(len(model_corr_sorted)))
ax.set_yticklabels(model_corr_sorted['model'], fontsize=9)
ax.set_xlabel('Correlation (r)', fontsize=11)
ax.set_title('Per-Model Correlations\n(Red = FDR-significant)', fontsize=12)
ax.axvline(0, color='black', linestyle='-', linewidth=1)
ax.axvline(-0.2, color='green', linestyle='--', alpha=0.5, label='|r|=0.2')
ax.axvline(0.2, color='green', linestyle='--', alpha=0.5)
ax.grid(axis='x', alpha=0.3)

# Plot 3: Per-domain scatter
ax = axes[1, 0]
for i, domain in enumerate(['Philosophy', 'Medical']):
    domain_df = df[df['domain'] == domain]
    r_domain, p_domain = pearsonr(domain_df['rci_cold'], domain_df['drci'])

    # Scatter
    ax.scatter(domain_df['rci_cold'], domain_df['drci'],
              alpha=0.6, s=40, c=colors[domain], label=f'{domain}\nr={r_domain:+.3f}, p={p_domain:.3e}')

    # Fit line
    slope, intercept = np.polyfit(domain_df['rci_cold'], domain_df['drci'], 1)
    x_range = np.linspace(domain_df['rci_cold'].min(), domain_df['rci_cold'].max(), 50)
    ax.plot(x_range, slope * x_range + intercept, color=colors[domain], linestyle='--', linewidth=2, alpha=0.8)

ax.set_xlabel('RCI_COLD (Base Capability)', fontsize=11)
ax.set_ylabel('ΔRCI (Entanglement)', fontsize=11)
ax.set_title('Per-Domain Correlations', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)
ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)

# Plot 4: PCA biplot
ax = axes[1, 1]
ax.scatter(components[:, 0], components[:, 1], alpha=0.3, s=20, c='gray')

# Draw loading vectors
scale = 3
ax.arrow(0, 0, pca.components_[0,0]*scale, pca.components_[1,0]*scale,
        head_width=0.3, head_length=0.3, fc='red', ec='red', linewidth=2, label='RCI_COLD')
ax.arrow(0, 0, pca.components_[0,1]*scale, pca.components_[1,1]*scale,
        head_width=0.3, head_length=0.3, fc='blue', ec='blue', linewidth=2, label='dRCI')

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=11)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=11)
ax.set_title('PCA Biplot: Loading Vectors', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.axvline(0, color='black', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig('c:/Users/barla/mch_experiments/analysis/independence_test_scatter.png', dpi=300, bbox_inches='tight')
print("[OK] Plot saved: independence_test_scatter.png")

# ============================================================================
# SAVE RESULTS
# ============================================================================

# Save correlation data
results_df = pd.DataFrame({
    'Test': ['Pooled', 'Philosophy', 'Medical'],
    'r': [r_pooled] + [pearsonr(df[df['domain']==d]['rci_cold'], df[df['domain']==d]['drci'])[0]
          for d in ['Philosophy', 'Medical']],
    'p': [p_pooled] + [pearsonr(df[df['domain']==d]['rci_cold'], df[df['domain']==d]['drci'])[1]
          for d in ['Philosophy', 'Medical']],
    'N': [len(df), len(df[df['domain']=='Philosophy']), len(df[df['domain']=='Medical'])]
})

results_df.to_csv('c:/Users/barla/mch_experiments/analysis/independence_test_results.csv', index=False)
print("[OK] Results saved: independence_test_results.csv")

# Save per-model correlations
model_corr_df.to_csv('c:/Users/barla/mch_experiments/analysis/independence_test_per_model.csv', index=False)
print("[OK] Per-model results saved: independence_test_per_model.csv")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("INDEPENDENCE TEST - SUMMARY")
print("="*80)

# Decision logic
independence_supported = (
    abs(r_pooled) < 0.2 and
    p_rci_cold > 0.05
)

print(f"\nPooled correlation: r = {r_pooled:+.4f}, p = {p_pooled:.4e}")
print(f"Regression beta_RCI_COLD: {beta_rci_cold:+.4f}, p = {p_rci_cold:.4e}")
print(f"PCA loading correlation: {loading_corr:.4f}")

print(f"\n{'='*80}")
if independence_supported:
    print("CONCLUSION: Independence SUPPORTED")
    print(f"  - Pooled |r| = {abs(r_pooled):.3f} < 0.2 ✓")
    print(f"  - Regression beta not significant (p = {p_rci_cold:.4f}) ✓")
    reason = f"Weak correlation (|r|={abs(r_pooled):.3f}) and non-significant regression (p={p_rci_cold:.3f})"
elif abs(r_pooled) >= 0.3 or p_rci_cold < 0.01:
    print("CONCLUSION: Independence NOT SUPPORTED")
    print(f"  - Pooled |r| = {abs(r_pooled):.3f}")
    print(f"  - Regression beta significant (p = {p_rci_cold:.4e})")
    reason = f"Moderate correlation (|r|={abs(r_pooled):.3f}) or significant regression (p={p_rci_cold:.4e})"
else:
    print("CONCLUSION: Independence PARTIALLY SUPPORTED")
    print(f"  - Pooled |r| = {abs(r_pooled):.3f} (borderline)")
    print(f"  - Regression beta marginally significant (p = {p_rci_cold:.4f})")
    reason = f"Weak-to-moderate correlation (|r|={abs(r_pooled):.3f}), marginal significance"

print(f"{'='*80}")

# Save summary
with open('c:/Users/barla/mch_experiments/analysis/independence_test_summary.md', 'w') as f:
    f.write("# Independence Test: Base Capability vs Entanglement\n\n")
    f.write("## Summary\n\n")
    if independence_supported:
        f.write(f"**CONCLUSION: Independence SUPPORTED** — {reason}\n\n")
    elif abs(r_pooled) >= 0.3 or p_rci_cold < 0.01:
        f.write(f"**CONCLUSION: Independence NOT SUPPORTED** — {reason}\n\n")
    else:
        f.write(f"**CONCLUSION: Independence PARTIALLY SUPPORTED** — {reason}\n\n")

    f.write("## Key Findings\n\n")
    f.write(f"1. **Pooled Correlation**: r = {r_pooled:+.4f}, p = {p_pooled:.4e}, N = {len(df)}\n")
    f.write(f"2. **Regression Analysis**: beta_RCI_COLD = {beta_rci_cold:+.4f}, p = {p_rci_cold:.4e}\n")
    f.write(f"3. **Per-Domain**:\n")
    for domain in ['Philosophy', 'Medical']:
        domain_df = df[df['domain'] == domain]
        r_d, p_d = pearsonr(domain_df['rci_cold'], domain_df['drci'])
        f.write(f"   - {domain}: r = {r_d:+.4f}, p = {p_d:.4e}\n")
    f.write(f"4. **Per-Model**: Median r = {model_corr_df['r'].median():.4f}, ")
    f.write(f"IQR = [{model_corr_df['r'].quantile(0.25):.4f}, {model_corr_df['r'].quantile(0.75):.4f}]\n")
    f.write(f"5. **PCA**: Loading correlation = {loading_corr:.4f}\n\n")

    f.write("## Interpretation\n\n")
    f.write("RCI_COLD (base capability) and ΔRCI (entanglement) show ")
    if abs(r_pooled) < 0.2:
        f.write("**weak correlation**, supporting the hypothesis that they are independent dimensions of AI cognition.\n\n")
    elif abs(r_pooled) < 0.3:
        f.write("**weak-to-moderate correlation**, suggesting partial independence with some coupling.\n\n")
    else:
        f.write("**moderate correlation**, indicating they may not be fully independent dimensions.\n\n")

    f.write("This analysis validates (or challenges) the dual-axis model:\n")
    f.write("- **Axis 1: Base Capability** (RCI_COLD) = zero-shot competence\n")
    f.write("- **Axis 2: Entanglement** (ΔRCI) = context coupling\n\n")

print("[OK] Summary saved: independence_test_summary.md")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
