#!/usr/bin/env python3
"""
Simplified Test of DeepSeek's Theoretical Framework

Key tests:
1. dRCI vs log(position) trend
2. PCA eigenvalue distributions (TRUE vs COLD)
3. Correlation structure analysis
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
print("DEEPSEEK THEORY VALIDATION - SIMPLIFIED APPROACH")
print("="*80)

# Load embedding model
print("\nLoading embedding model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Philosophy AND Medical models with 50 trials
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

def compute_position_drci_simple(file_path, model_name, n_positions=30):
    """Compute position-level dRCI using standard method"""
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

        # Compute dRCI
        rci_true = []
        rci_cold = []

        for i in range(n_trials):
            other_idx = [j for j in range(n_trials) if j != i]
            true_sims = np.dot(true_emb[i], true_emb[other_idx].T)
            rci_true.append(np.mean(true_sims))
            cold_sims = np.dot(cold_emb[i], cold_emb[other_idx].T)
            rci_cold.append(np.mean(cold_sims))

        drci = np.mean(rci_true) - np.mean(rci_cold)

        # Correlation between TRUE and COLD embeddings
        # This gives us a measure of information overlap
        corr_matrix = np.corrcoef(true_emb.flatten(), cold_emb.flatten())[0, 1]

        # Average pairwise similarity within TRUE vs within COLD
        true_self_sim = np.mean([np.dot(true_emb[i], true_emb[j])
                                  for i in range(n_trials) for j in range(i+1, n_trials)])
        cold_self_sim = np.mean([np.dot(cold_emb[i], cold_emb[j])
                                  for i in range(n_trials) for j in range(i+1, n_trials)])

        results.append({
            'position': pos + 1,
            'drci': drci,
            'true_self_sim': true_self_sim,
            'cold_self_sim': cold_self_sim,
            'true_cold_corr': corr_matrix
        })

        if (pos + 1) % 5 == 0:
            print(f"  Position {pos+1}/30: dRCI = {drci:.4f}")

    return pd.DataFrame(results)

# ============================================================================
# COMPUTE FOR ALL MODELS
# ============================================================================

all_results = {}

for model_name, file_path in models.items():
    try:
        df = compute_position_drci_simple(file_path, model_name)
        all_results[model_name] = df
    except Exception as e:
        print(f"  ERROR: {e}")

# ============================================================================
# TEST 1: LOG-SCALE POSITION TREND
# ============================================================================

print("\n" + "="*80)
print("TEST 1: dRCI vs log(position) - DeepSeek's alpha * log(n) + beta")
print("="*80)

log_results = []

for model_name, df in all_results.items():
    # Exclude position 1 (log(0) undefined) and position 30 (potential outlier)
    df_fit = df[(df['position'] > 1) & (df['position'] <= 29)].copy()
    df_fit['log_n'] = np.log(df_fit['position'] - 1)

    # Fit: drci ~ alpha * log(n) + beta
    slope, intercept, r, p, stderr = linregress(df_fit['log_n'], df_fit['drci'])

    print(f"\n{model_name}:")
    print(f"  dRCI = {slope:.4f} * log(n) {intercept:+.4f}")
    print(f"  R² = {r**2:.4f}, p = {p:.4e}")

    log_results.append({
        'model': model_name,
        'alpha': slope,
        'beta': intercept,
        'R2': r**2,
        'p_value': p
    })

log_df = pd.DataFrame(log_results)

# ============================================================================
# TEST 2: PCA EIGENVALUE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("TEST 2: PCA - TRUE vs COLD Eigenvalue Distributions")
print("="*80)

pca_results = {}

for model_name, file_path in models.items():
    print(f"\n{model_name}:")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # All responses
    all_true = []
    all_cold = []
    for trial in data['trials']:
        all_true.extend(trial['responses']['true'])
        all_cold.extend(trial['responses']['cold'])

    print(f"  Encoding {len(all_true)} responses...")
    true_emb = model.encode(all_true, convert_to_numpy=True, show_progress_bar=False)
    cold_emb = model.encode(all_cold, convert_to_numpy=True, show_progress_bar=False)

    print("  Running PCA...")
    pca_true = PCA(n_components=50)
    pca_cold = PCA(n_components=50)

    pca_true.fit(true_emb)
    pca_cold.fit(cold_emb)

    # Effective dimensionality
    def effective_dim(var_ratio):
        p = var_ratio / np.sum(var_ratio)
        H = -np.sum(p * np.log(p + 1e-10))
        return np.exp(H)

    eff_dim_true = effective_dim(pca_true.explained_variance_ratio_)
    eff_dim_cold = effective_dim(pca_cold.explained_variance_ratio_)

    pca_results[model_name] = {
        'true_var_ratio': pca_true.explained_variance_ratio_,
        'cold_var_ratio': pca_cold.explained_variance_ratio_,
        'eff_dim_true': eff_dim_true,
        'eff_dim_cold': eff_dim_cold
    }

    print(f"  TRUE: First 5 PCs = {np.sum(pca_true.explained_variance_ratio_[:5]):.2%}")
    print(f"  COLD: First 5 PCs = {np.sum(pca_cold.explained_variance_ratio_[:5]):.2%}")
    print(f"  Effective dims: TRUE = {eff_dim_true:.2f}, COLD = {eff_dim_cold:.2f}")

# ============================================================================
# GENERATE PLOTS
# ============================================================================

print("\n" + "="*80)
print("GENERATING PLOTS")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# ===================================================================
# Plot 1: dRCI vs log(position)
# ===================================================================

ax = axes[0, 0]

for model_name, df in all_results.items():
    df_plot = df[(df['position'] > 1) & (df['position'] <= 29)].copy()
    df_plot['log_n'] = np.log(df_plot['position'] - 1)

    ax.scatter(df_plot['log_n'], df_plot['drci'], s=30, alpha=0.6, label=model_name)

    # Fit line
    slope, intercept, _, _, _ = linregress(df_plot['log_n'], df_plot['drci'])
    x_fit = np.linspace(df_plot['log_n'].min(), df_plot['log_n'].max(), 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, alpha=0.5, linewidth=1.5)

ax.set_xlabel('log(n) where n = position - 1', fontsize=11)
ax.set_ylabel('dRCI', fontsize=11)
ax.set_title('DeepSeek Log-Scale Trend', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ===================================================================
# Plot 2: dRCI vs Position (standard)
# ===================================================================

ax = axes[0, 1]

for model_name, df in all_results.items():
    ax.plot(df['position'], df['drci'], marker='o', markersize=4,
            alpha=0.7, linewidth=1.5, label=model_name)

ax.set_xlabel('Position', fontsize=11)
ax.set_ylabel('dRCI', fontsize=11)
ax.set_title('dRCI Across Positions', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 31)

# ===================================================================
# Plot 3: PCA Eigenvalues (log scale)
# ===================================================================

ax = axes[0, 2]

for model_name, pca_data in pca_results.items():
    ax.plot(range(1, 21), pca_data['true_var_ratio'][:20],
            marker='o', markersize=4, alpha=0.7, linewidth=1.5, label=f'{model_name} TRUE')
    ax.plot(range(1, 21), pca_data['cold_var_ratio'][:20],
            marker='s', markersize=4, alpha=0.7, linewidth=1.5,
            linestyle='--', label=f'{model_name} COLD')

ax.set_xlabel('Principal Component', fontsize=11)
ax.set_ylabel('Explained Variance Ratio', fontsize=11)
ax.set_title('PCA: TRUE vs COLD', fontsize=12, fontweight='bold')
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# ===================================================================
# Plot 4: Cumulative Variance
# ===================================================================

ax = axes[1, 0]

for model_name, pca_data in pca_results.items():
    cum_true = np.cumsum(pca_data['true_var_ratio'])
    cum_cold = np.cumsum(pca_data['cold_var_ratio'])

    ax.plot(range(1, 51), cum_true, marker='o', markersize=3,
            alpha=0.7, linewidth=1.5, label=f'{model_name} TRUE')
    ax.plot(range(1, 51), cum_cold, marker='s', markersize=3,
            alpha=0.7, linewidth=1.5, linestyle='--', label=f'{model_name} COLD')

ax.axhline(0.95, color='red', linestyle=':', alpha=0.5, linewidth=1)
ax.set_xlabel('Number of Components', fontsize=11)
ax.set_ylabel('Cumulative Variance Explained', fontsize=11)
ax.set_title('PCA: Cumulative Variance', fontsize=12, fontweight='bold')
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

# ===================================================================
# Plot 5: Effective Dimensionality
# ===================================================================

ax = axes[1, 1]

model_names = list(pca_results.keys())
eff_dims_true = [pca_results[m]['eff_dim_true'] for m in model_names]
eff_dims_cold = [pca_results[m]['eff_dim_cold'] for m in model_names]

x = np.arange(len(model_names))
width = 0.35

ax.bar(x - width/2, eff_dims_true, width, label='TRUE', color='steelblue', alpha=0.8)
ax.bar(x + width/2, eff_dims_cold, width, label='COLD', color='crimson', alpha=0.8)

ax.set_ylabel('Effective Dimensionality', fontsize=11)
ax.set_title('Information Content: TRUE vs COLD', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([m.split(' ')[0] for m in model_names], rotation=45, ha='right', fontsize=9)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# ===================================================================
# Plot 6: Log-fit parameters (alpha, beta)
# ===================================================================

ax = axes[1, 2]

models_short = [name.split(' ')[0] for name in log_df['model']]
x = np.arange(len(models_short))
width = 0.35

ax.bar(x - width/2, log_df['alpha'], width, label='alpha (slope)', color='green', alpha=0.8)
ax2 = ax.twinx()
ax2.bar(x + width/2, log_df['beta'], width, label='beta (intercept)', color='orange', alpha=0.8)

ax.set_ylabel('Alpha (slope)', fontsize=11, color='green')
ax2.set_ylabel('Beta (intercept)', fontsize=11, color='orange')
ax.set_title('Log-fit Parameters', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models_short, rotation=45, ha='right', fontsize=9)
ax.tick_params(axis='y', labelcolor='green')
ax2.tick_params(axis='y', labelcolor='orange')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
output_path = "c:/Users/barla/mch_experiments/analysis/deepseek_theory_simple.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"[OK] Plot saved: {output_path}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save log-fit parameters
log_df.to_csv('c:/Users/barla/mch_experiments/analysis/deepseek_logfit_params.csv', index=False)
print("[OK] Log-fit parameters: deepseek_logfit_params.csv")

# Save all position data
combined_df = pd.concat([df.assign(model=name) for name, df in all_results.items()])
combined_df.to_csv('c:/Users/barla/mch_experiments/analysis/deepseek_position_data.csv', index=False)
print("[OK] Position data: deepseek_position_data.csv")

# ============================================================================
# FINAL REPORT
# ============================================================================

print("\n" + "="*80)
print("DEEPSEEK THEORY - FINDINGS")
print("="*80)

print("\n1. LOG-SCALE TREND (alpha * log(n) + beta):")
for _, row in log_df.iterrows():
    print(f"   {row['model']:25s}: alpha={row['alpha']:+.4f}, beta={row['beta']:+.4f}, R²={row['R2']:.4f}")

avg_R2 = log_df['R2'].mean()
if avg_R2 > 0.50:
    print(f"\n   ✓ Log-scale model fits well (mean R² = {avg_R2:.4f})")
else:
    print(f"\n   ✗ Log-scale model fits poorly (mean R² = {avg_R2:.4f})")

print("\n2. PCA DIMENSIONALITY (TRUE vs COLD):")
for model_name in pca_results.keys():
    eff_true = pca_results[model_name]['eff_dim_true']
    eff_cold = pca_results[model_name]['eff_dim_cold']
    diff = eff_true - eff_cold
    print(f"   {model_name:25s}: TRUE={eff_true:.1f}, COLD={eff_cold:.1f}, diff={diff:+.1f}")

print("\n3. INTERPRETATION:")
print("   DeepSeek's theory suggests:")
print("   - dRCI should scale as alpha * log(n) + beta")
print("   - TRUE embeddings should have lower effective dimensionality (more structured)")
print("   - Information content concentrated in fewer dimensions with context")

mean_eff_diff = np.mean([pca_results[m]['eff_dim_true'] - pca_results[m]['eff_dim_cold']
                          for m in pca_results.keys()])

if mean_eff_diff < -2:
    print(f"\n   ✓ TRUE is more structured (mean diff = {mean_eff_diff:.2f} dims)")
elif mean_eff_diff > 2:
    print(f"\n   ✗ COLD is more structured (mean diff = {mean_eff_diff:.2f} dims)")
else:
    print(f"\n   ~ Similar structure (mean diff = {mean_eff_diff:.2f} dims)")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
