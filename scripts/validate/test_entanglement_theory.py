#!/usr/bin/env python3
"""
Test Entanglement Theory: Engagement Creates Entanglement of Possibility Spaces

Theory:
- Engagement (TRUE condition) narrows possibility space → lower variance
- No engagement (COLD condition) maintains wide possibilities → higher variance
- dRCI measures degree of entanglement

Tests:
1. Variance Ratio: Var_TRUE / Var_COLD (should be < 1 if entangled)
2. Mutual Information Proxy: I(X;Y) ~ variance reduction
3. Correlation: Does variance reduction correlate with dRCI?
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from scipy.stats import pearsonr, linregress
from sklearn.decomposition import PCA
import seaborn as sns

print("="*80)
print("ENTANGLEMENT THEORY VALIDATION")
print("="*80)

# Load embedding model
print("\nLoading embedding model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Models from both domains with 50 trials
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

def compute_entanglement_metrics(file_path, model_name, n_positions=30):
    """
    Compute entanglement metrics for each position

    Returns:
    - dRCI: empirical measure
    - variance_true: variance of TRUE embeddings
    - variance_cold: variance of COLD embeddings
    - variance_ratio: Var_TRUE / Var_COLD (< 1 means entangled)
    - mutual_info_proxy: 1 - variance_ratio (higher = more entanglement)
    """
    print(f"\n{model_name}:")
    print(f"  Loading...")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    n_trials = data['n_trials']
    results = []

    for pos in range(n_positions):
        # Extract responses
        true_responses = [trial['responses']['true'][pos] for trial in data['trials']]
        cold_responses = [trial['responses']['cold'][pos] for trial in data['trials']]

        # Encode
        true_emb = model.encode(true_responses, convert_to_numpy=True, show_progress_bar=False)
        cold_emb = model.encode(cold_responses, convert_to_numpy=True, show_progress_bar=False)

        # Compute dRCI (standard method)
        rci_true = []
        rci_cold = []

        for i in range(n_trials):
            other_idx = [j for j in range(n_trials) if j != i]
            true_sims = np.dot(true_emb[i], true_emb[other_idx].T)
            rci_true.append(np.mean(true_sims))
            cold_sims = np.dot(cold_emb[i], cold_emb[other_idx].T)
            rci_cold.append(np.mean(cold_sims))

        drci = np.mean(rci_true) - np.mean(rci_cold)

        # Variance metrics (entanglement signature)
        # Compute variance across all embedding dimensions
        var_true = np.var(true_emb, axis=0).mean()  # Mean variance across dimensions
        var_cold = np.var(cold_emb, axis=0).mean()

        # Alternative: Total variance (trace of covariance matrix)
        var_true_total = np.trace(np.cov(true_emb.T))
        var_cold_total = np.trace(np.cov(cold_emb.T))

        # Variance ratio: < 1 means TRUE has lower variance (entangled)
        var_ratio = var_true / var_cold if var_cold > 0 else 1.0
        var_ratio_total = var_true_total / var_cold_total if var_cold_total > 0 else 1.0

        # Mutual information proxy: higher means more entanglement
        mutual_info_proxy = 1.0 - var_ratio
        mutual_info_proxy_total = 1.0 - var_ratio_total

        # Pairwise similarity metrics
        true_self_sim = np.mean([np.dot(true_emb[i], true_emb[j])
                                  for i in range(n_trials) for j in range(i+1, n_trials)])
        cold_self_sim = np.mean([np.dot(cold_emb[i], cold_emb[j])
                                  for i in range(n_trials) for j in range(i+1, n_trials)])

        # Cross-similarity between TRUE and COLD
        cross_sim = np.mean([np.dot(true_emb[i], cold_emb[j])
                             for i in range(n_trials) for j in range(n_trials)])

        results.append({
            'position': pos + 1,
            'drci': drci,
            'var_true': var_true,
            'var_cold': var_cold,
            'var_ratio': var_ratio,
            'mutual_info_proxy': mutual_info_proxy,
            'var_true_total': var_true_total,
            'var_cold_total': var_cold_total,
            'var_ratio_total': var_ratio_total,
            'mutual_info_proxy_total': mutual_info_proxy_total,
            'true_self_sim': true_self_sim,
            'cold_self_sim': cold_self_sim,
            'cross_sim': cross_sim
        })

        if (pos + 1) % 10 == 0:
            print(f"  Position {pos+1}/30: dRCI = {drci:+.4f}, Var_ratio = {var_ratio:.4f}")

    return pd.DataFrame(results)

# ============================================================================
# COMPUTE FOR ALL MODELS
# ============================================================================

all_results = {}

for model_name, file_path in models.items():
    try:
        df = compute_entanglement_metrics(file_path, model_name)
        all_results[model_name] = df
    except Exception as e:
        print(f"  ERROR: {e}")

# ============================================================================
# TEST 1: CORRELATION BETWEEN dRCI AND MUTUAL INFO PROXY
# ============================================================================

print("\n" + "="*80)
print("TEST 1: CORRELATION BETWEEN dRCI AND ENTANGLEMENT METRICS")
print("="*80)

correlation_results = []

for model_name, df in all_results.items():
    # Correlation with variance ratio
    r_var, p_var = pearsonr(df['drci'], df['var_ratio'])

    # Correlation with mutual info proxy
    r_mi, p_mi = pearsonr(df['drci'], df['mutual_info_proxy'])

    # Correlation with mutual info proxy (total variance)
    r_mi_total, p_mi_total = pearsonr(df['drci'], df['mutual_info_proxy_total'])

    print(f"\n{model_name}:")
    print(f"  dRCI vs Var_Ratio:        r = {r_var:+.4f}, p = {p_var:.4e} {'***' if p_var < 0.001 else '**' if p_var < 0.01 else '*' if p_var < 0.05 else 'ns'}")
    print(f"  dRCI vs MI_Proxy:         r = {r_mi:+.4f}, p = {p_mi:.4e} {'***' if p_mi < 0.001 else '**' if p_mi < 0.01 else '*' if p_mi < 0.05 else 'ns'}")
    print(f"  dRCI vs MI_Proxy_Total:   r = {r_mi_total:+.4f}, p = {p_mi_total:.4e} {'***' if p_mi_total < 0.001 else '**' if p_mi_total < 0.01 else '*' if p_mi_total < 0.05 else 'ns'}")

    correlation_results.append({
        'model': model_name,
        'r_var_ratio': r_var,
        'p_var_ratio': p_var,
        'r_mi_proxy': r_mi,
        'p_mi_proxy': p_mi,
        'r_mi_proxy_total': r_mi_total,
        'p_mi_proxy_total': p_mi_total
    })

corr_df = pd.DataFrame(correlation_results)

# ============================================================================
# TEST 2: VARIANCE COMPARISON (TRUE vs COLD)
# ============================================================================

print("\n" + "="*80)
print("TEST 2: VARIANCE COMPARISON (Entanglement Signature)")
print("="*80)

print("\nTheory: If engagement creates entanglement:")
print("  - TRUE should have LOWER variance (narrowed possibilities)")
print("  - COLD should have HIGHER variance (wide possibilities)")
print("  - Variance_Ratio = Var_TRUE / Var_COLD < 1.0")

variance_summary = []

for model_name, df in all_results.items():
    mean_var_ratio = df['var_ratio'].mean()
    mean_var_ratio_total = df['var_ratio_total'].mean()

    # Count positions where TRUE has lower variance than COLD
    n_entangled = (df['var_ratio'] < 1.0).sum()
    pct_entangled = n_entangled / len(df) * 100

    print(f"\n{model_name}:")
    print(f"  Mean Var_Ratio:        {mean_var_ratio:.4f} ({'TRUE < COLD' if mean_var_ratio < 1.0 else 'TRUE > COLD'})")
    print(f"  Mean Var_Ratio_Total:  {mean_var_ratio_total:.4f} ({'TRUE < COLD' if mean_var_ratio_total < 1.0 else 'TRUE > COLD'})")
    print(f"  Positions with TRUE < COLD: {n_entangled}/30 ({pct_entangled:.1f}%)")

    variance_summary.append({
        'model': model_name,
        'mean_var_ratio': mean_var_ratio,
        'mean_var_ratio_total': mean_var_ratio_total,
        'n_entangled': n_entangled,
        'pct_entangled': pct_entangled
    })

var_summary_df = pd.DataFrame(variance_summary)

# ============================================================================
# TEST 3: POSITION-SPECIFIC ANALYSIS (P30 spike in medical)
# ============================================================================

print("\n" + "="*80)
print("TEST 3: POSITION 30 ANALYSIS (Medical Spike)")
print("="*80)

print("\nMedical P30 Entanglement Signature:")

for model_name, df in all_results.items():
    if '(Med)' in model_name:
        p30_data = df[df['position'] == 30].iloc[0]
        p29_data = df[df['position'] == 29].iloc[0]

        print(f"\n{model_name}:")
        print(f"  P30: dRCI = {p30_data['drci']:+.4f}, Var_Ratio = {p30_data['var_ratio']:.4f}, MI_Proxy = {p30_data['mutual_info_proxy']:+.4f}")
        print(f"  P29: dRCI = {p29_data['drci']:+.4f}, Var_Ratio = {p29_data['var_ratio']:.4f}, MI_Proxy = {p29_data['mutual_info_proxy']:+.4f}")
        print(f"  Spike: dRCI = {p30_data['drci'] - p29_data['drci']:+.4f}, Var_Ratio change = {p30_data['var_ratio'] - p29_data['var_ratio']:+.4f}")

# ============================================================================
# GENERATE PLOTS
# ============================================================================

print("\n" + "="*80)
print("GENERATING PLOTS")
print("="*80)

fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# ===================================================================
# Plot 1: dRCI vs Mutual Info Proxy (all models)
# ===================================================================

ax = fig.add_subplot(gs[0, 0])

for model_name, df in all_results.items():
    color = 'steelblue' if '(Phil)' in model_name else 'crimson'
    ax.scatter(df['mutual_info_proxy'], df['drci'], s=30, alpha=0.6,
               color=color, label=model_name if model_name in list(all_results.keys())[:2] else '')

# Fit line for all data combined
all_mi = pd.concat([df['mutual_info_proxy'] for df in all_results.values()])
all_drci = pd.concat([df['drci'] for df in all_results.values()])
slope, intercept, r, p, _ = linregress(all_mi, all_drci)

x_fit = np.linspace(all_mi.min(), all_mi.max(), 100)
y_fit = slope * x_fit + intercept
ax.plot(x_fit, y_fit, 'k--', alpha=0.7, linewidth=2,
        label=f'Combined Fit (r={r:.3f}, p={p:.2e})')

ax.set_xlabel('Mutual Info Proxy (1 - Var_Ratio)', fontsize=11)
ax.set_ylabel('dRCI', fontsize=11)
ax.set_title('Entanglement vs dRCI (All Positions)', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3)
ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)
ax.axvline(0, color='black', linewidth=0.5, alpha=0.3)

# ===================================================================
# Plot 2: Variance Ratio across positions (Philosophy)
# ===================================================================

ax = fig.add_subplot(gs[0, 1])

for model_name, df in all_results.items():
    if '(Phil)' in model_name:
        ax.plot(df['position'], df['var_ratio'], marker='o', markersize=4,
                alpha=0.7, linewidth=1.5, label=model_name.split(' (')[0])

ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Var_TRUE = Var_COLD')
ax.set_xlabel('Position', fontsize=11)
ax.set_ylabel('Var_Ratio (TRUE / COLD)', fontsize=11)
ax.set_title('Philosophy: Variance Ratio Across Positions', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 31)

# ===================================================================
# Plot 3: Variance Ratio across positions (Medical)
# ===================================================================

ax = fig.add_subplot(gs[0, 2])

for model_name, df in all_results.items():
    if '(Med)' in model_name:
        ax.plot(df['position'], df['var_ratio'], marker='o', markersize=4,
                alpha=0.7, linewidth=1.5, label=model_name.split(' (')[0])

ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Var_TRUE = Var_COLD')
ax.axvline(30, color='orange', linestyle=':', alpha=0.5, linewidth=2, label='P30 (Type 2)')
ax.set_xlabel('Position', fontsize=11)
ax.set_ylabel('Var_Ratio (TRUE / COLD)', fontsize=11)
ax.set_title('Medical: Variance Ratio Across Positions', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 31)

# ===================================================================
# Plot 4: dRCI vs Variance Ratio (scatterplot by domain)
# ===================================================================

ax = fig.add_subplot(gs[1, 0])

phil_data = pd.concat([df.assign(model=name) for name, df in all_results.items() if '(Phil)' in name])
med_data = pd.concat([df.assign(model=name) for name, df in all_results.items() if '(Med)' in name])

ax.scatter(phil_data['var_ratio'], phil_data['drci'], s=30, alpha=0.6,
           color='steelblue', label='Philosophy')
ax.scatter(med_data['var_ratio'], med_data['drci'], s=30, alpha=0.6,
           color='crimson', label='Medical')

ax.axvline(1.0, color='red', linestyle='--', alpha=0.3, linewidth=1)
ax.set_xlabel('Variance Ratio (TRUE / COLD)', fontsize=11)
ax.set_ylabel('dRCI', fontsize=11)
ax.set_title('dRCI vs Variance Ratio (By Domain)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)

# ===================================================================
# Plot 5: Correlation strengths (heatmap)
# ===================================================================

ax = fig.add_subplot(gs[1, 1])

corr_matrix = corr_df[['r_var_ratio', 'r_mi_proxy', 'r_mi_proxy_total']].T
model_labels = [m.split(' (')[0] for m in corr_df['model']]

im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
ax.set_xticks(range(len(model_labels)))
ax.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(3))
ax.set_yticklabels(['Var_Ratio', 'MI_Proxy', 'MI_Proxy_Total'], fontsize=9)
ax.set_title('Correlation with dRCI', fontsize=12, fontweight='bold')

for i in range(3):
    for j in range(len(model_labels)):
        text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=8)

plt.colorbar(im, ax=ax, label='Pearson r')

# ===================================================================
# Plot 6: Mean Variance Ratio by Model
# ===================================================================

ax = fig.add_subplot(gs[1, 2])

model_labels = [m.split(' (')[0] for m in var_summary_df['model']]
colors = ['steelblue' if '(Phil)' in m else 'crimson' for m in var_summary_df['model']]

x = np.arange(len(model_labels))
ax.bar(x, var_summary_df['mean_var_ratio'], color=colors, alpha=0.7)
ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='No Entanglement')

ax.set_ylabel('Mean Var_Ratio (TRUE / COLD)', fontsize=11)
ax.set_title('Mean Variance Ratio by Model', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=9)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# ===================================================================
# Plot 7: dRCI position curves (Philosophy)
# ===================================================================

ax = fig.add_subplot(gs[2, 0])

for model_name, df in all_results.items():
    if '(Phil)' in model_name:
        ax.plot(df['position'], df['drci'], marker='o', markersize=4,
                alpha=0.7, linewidth=1.5, label=model_name.split(' (')[0])

ax.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)
ax.set_xlabel('Position', fontsize=11)
ax.set_ylabel('dRCI', fontsize=11)
ax.set_title('Philosophy: dRCI Across Positions', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 31)

# ===================================================================
# Plot 8: dRCI position curves (Medical)
# ===================================================================

ax = fig.add_subplot(gs[2, 1])

for model_name, df in all_results.items():
    if '(Med)' in model_name:
        ax.plot(df['position'], df['drci'], marker='o', markersize=4,
                alpha=0.7, linewidth=1.5, label=model_name.split(' (')[0])

ax.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)
ax.axvline(30, color='orange', linestyle=':', alpha=0.5, linewidth=2, label='P30 (Type 2)')
ax.set_xlabel('Position', fontsize=11)
ax.set_ylabel('dRCI', fontsize=11)
ax.set_title('Medical: dRCI Across Positions', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 31)

# ===================================================================
# Plot 9: P30 Spike Analysis (Medical)
# ===================================================================

ax = fig.add_subplot(gs[2, 2])

med_models = [name for name in all_results.keys() if '(Med)' in name]
p30_drci = [all_results[m][all_results[m]['position'] == 30]['drci'].values[0] for m in med_models]
p30_var_ratio = [all_results[m][all_results[m]['position'] == 30]['var_ratio'].values[0] for m in med_models]

model_labels = [m.split(' (')[0] for m in med_models]
x = np.arange(len(model_labels))

ax2 = ax.twinx()

bars1 = ax.bar(x - 0.2, p30_drci, 0.4, label='dRCI at P30', color='crimson', alpha=0.7)
bars2 = ax2.bar(x + 0.2, p30_var_ratio, 0.4, label='Var_Ratio at P30', color='steelblue', alpha=0.7)

ax.set_ylabel('dRCI at P30', fontsize=11, color='crimson')
ax2.set_ylabel('Var_Ratio at P30', fontsize=11, color='steelblue')
ax.set_title('Medical P30: Spike + Entanglement', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=9)
ax.tick_params(axis='y', labelcolor='crimson')
ax2.tick_params(axis='y', labelcolor='steelblue')
ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)
ax2.axhline(1.0, color='red', linestyle='--', alpha=0.3, linewidth=1)
ax.grid(True, alpha=0.3, axis='x')

output_path = "c:/Users/barla/mch_experiments/analysis/entanglement_theory_validation.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"[OK] Plot saved: {output_path}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save correlation results
corr_df.to_csv('c:/Users/barla/mch_experiments/analysis/entanglement_correlations.csv', index=False)
print("[OK] Correlations: entanglement_correlations.csv")

# Save variance summary
var_summary_df.to_csv('c:/Users/barla/mch_experiments/analysis/entanglement_variance_summary.csv', index=False)
print("[OK] Variance summary: entanglement_variance_summary.csv")

# Save all position data
combined_df = pd.concat([df.assign(model=name) for name, df in all_results.items()])
combined_df.to_csv('c:/Users/barla/mch_experiments/analysis/entanglement_position_data.csv', index=False)
print("[OK] Position data: entanglement_position_data.csv")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("ENTANGLEMENT THEORY - FINDINGS")
print("="*80)

print("\n1. MUTUAL INFORMATION CORRELATION:")
print(f"   Combined correlation (all models): r = {r:.4f}, p = {p:.4e}")
if p < 0.001:
    print(f"   *** STRONG EVIDENCE: MI proxy correlates with dRCI!")
elif p < 0.05:
    print(f"   ** MODERATE EVIDENCE: MI proxy correlates with dRCI")
else:
    print(f"   No significant correlation")

print("\n2. VARIANCE SIGNATURE (TRUE vs COLD):")
overall_mean_var_ratio = var_summary_df['mean_var_ratio'].mean()
print(f"   Overall mean Var_Ratio = {overall_mean_var_ratio:.4f}")
if overall_mean_var_ratio < 1.0:
    print(f"   TRUE has LOWER variance on average (entanglement signature!)")
elif overall_mean_var_ratio > 1.0:
    print(f"   TRUE has HIGHER variance on average (unexpected!)")
else:
    print(f"   TRUE and COLD have similar variance")

n_models_entangled = (var_summary_df['mean_var_ratio'] < 1.0).sum()
print(f"   Models with Var_TRUE < Var_COLD: {n_models_entangled}/{len(var_summary_df)}")

print("\n3. DOMAIN DIFFERENCES:")
phil_mean_var = var_summary_df[var_summary_df['model'].str.contains('Phil')]['mean_var_ratio'].mean()
med_mean_var = var_summary_df[var_summary_df['model'].str.contains('Med')]['mean_var_ratio'].mean()
print(f"   Philosophy mean Var_Ratio: {phil_mean_var:.4f}")
print(f"   Medical mean Var_Ratio:    {med_mean_var:.4f}")

print("\n4. MEDICAL P30 SPIKE:")
for model_name in med_models:
    p30_drci_val = all_results[model_name][all_results[model_name]['position'] == 30]['drci'].values[0]
    p30_var_ratio_val = all_results[model_name][all_results[model_name]['position'] == 30]['var_ratio'].values[0]
    print(f"   {model_name.split('(')[0]:20s}: dRCI = {p30_drci_val:+.4f}, Var_Ratio = {p30_var_ratio_val:.4f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
