#!/usr/bin/env python3
"""
Test DeepSeek's Theoretical Framework Against Empirical ΔRCI Data

DeepSeek's Proposal:
- dRCI ~ 1 - (H(Y|X) / H(Y))
- Entanglement scales as alpha * log(n) + beta

Tests:
1. Calculate H(Y|X) and H(Y) from embeddings
2. Test correlation with empirical dRCI
3. Plot dRCI vs position on log scale
4. PCA on TRUE vs COLD embeddings
5. Compare eigenvalue distributions
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from scipy.stats import pearsonr, entropy, linregress
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import seaborn as sns

print("="*80)
print("DEEPSEEK THEORETICAL FRAMEWORK VALIDATION")
print("="*80)

# Load embedding model
print("\nLoading embedding model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Philosophy closed models (have full response data)
test_models = {
    'GPT-4o-mini': 'c:/Users/barla/mch_experiments/data/closed_model_philosophy_rerun/mch_results_gpt4o_mini_philosophy_50trials.json',
    'GPT-4o': 'c:/Users/barla/mch_experiments/data/closed_model_philosophy_rerun/mch_results_gpt4o_philosophy_50trials.json',
    'Claude Haiku': 'c:/Users/barla/mch_experiments/data/closed_model_philosophy_rerun/mch_results_claude_haiku_philosophy_50trials.json',
    'Gemini Flash': 'c:/Users/barla/mch_experiments/data/closed_model_philosophy_rerun/mch_results_gemini_flash_philosophy_50trials.json',
}

def estimate_entropy_knn(embeddings, k=3):
    """
    Estimate entropy using k-NN method
    H(X) ≈ -ψ(k) + ψ(N) + log(c_d) + d * mean(log(ρ_k))
    where ρ_k is distance to k-th nearest neighbor
    """
    N, d = embeddings.shape

    # Fit k-NN
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)

    # Distance to k-th nearest neighbor (exclude self at index 0)
    rho_k = distances[:, k]

    # Entropy estimate (simplified Kozachenko-Leonenko estimator)
    # H ≈ d * mean(log(rho_k)) + log(volume of unit ball in d dimensions) + log(N) - ψ(k)
    log_rho = np.log(rho_k + 1e-10)  # avoid log(0)

    # Digamma function approximation: ψ(k) ≈ log(k) - 1/(2k)
    psi_k = np.log(k) - 1/(2*k)
    psi_N = np.log(N) - 1/(2*N)

    # Volume constant for unit ball in d dimensions (log)
    log_c_d = (d/2) * np.log(np.pi) - np.log(np.math.gamma(d/2 + 1))

    H = -psi_k + psi_N + log_c_d + d * np.mean(log_rho)

    return H

def estimate_conditional_entropy(embeddings_Y, embeddings_X, k=3):
    """
    Estimate H(Y|X) using k-NN method
    H(Y|X) ≈ H(X,Y) - H(X)
    """
    # Joint distribution: concatenate X and Y
    joint = np.concatenate([embeddings_X, embeddings_Y], axis=1)

    H_joint = estimate_entropy_knn(joint, k=k)
    H_X = estimate_entropy_knn(embeddings_X, k=k)

    H_Y_given_X = H_joint - H_X

    return H_Y_given_X

def compute_position_analysis(file_path, model_name, n_positions=30):
    """
    For each position:
    1. Compute empirical ΔRCI
    2. Compute H(Y|X) and H(Y) from embeddings
    3. Test DeepSeek's formula: ΔRCI ≈ 1 - (H(Y|X) / H(Y))
    """
    print(f"\n{model_name}:")
    print(f"  Loading {file_path.split('/')[-1]}...")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    n_trials = data['n_trials']

    results = []

    for pos in range(n_positions):
        print(f"  Position {pos+1}/30...", end='', flush=True)

        # Extract responses at this position
        true_responses = []
        cold_responses = []

        for trial in data['trials']:
            true_responses.append(trial['responses']['true'][pos])
            cold_responses.append(trial['responses']['cold'][pos])

        # Compute embeddings
        true_emb = model.encode(true_responses, convert_to_numpy=True, show_progress_bar=False)
        cold_emb = model.encode(cold_responses, convert_to_numpy=True, show_progress_bar=False)

        # ===================================================================
        # EMPIRICAL ΔRCI (our standard calculation)
        # ===================================================================
        rci_true = []
        rci_cold = []

        for i in range(n_trials):
            other_idx = [j for j in range(n_trials) if j != i]

            # TRUE RCI
            true_sims = np.dot(true_emb[i], true_emb[other_idx].T)
            rci_true.append(np.mean(true_sims))

            # COLD RCI
            cold_sims = np.dot(cold_emb[i], cold_emb[other_idx].T)
            rci_cold.append(np.mean(cold_sims))

        empirical_drci = np.mean(rci_true) - np.mean(rci_cold)

        # ===================================================================
        # THEORETICAL CALCULATION: DeepSeek's Formula
        # ===================================================================
        # Y = TRUE embeddings (response space)
        # X = COLD embeddings (baseline/context-free space)

        # H(Y): Entropy of TRUE responses
        H_Y = estimate_entropy_knn(true_emb, k=3)

        # H(Y|X): Conditional entropy - how much uncertainty in TRUE given COLD structure
        # We approximate this by treating COLD as providing "context" structure
        H_Y_given_X = estimate_conditional_entropy(true_emb, cold_emb, k=3)

        # DeepSeek's formula: ΔRCI ≈ 1 - (H(Y|X) / H(Y))
        theoretical_drci = 1 - (H_Y_given_X / H_Y)

        # Also calculate H(X) for reference
        H_X = estimate_entropy_knn(cold_emb, k=3)

        # Mutual information: I(X;Y) = H(Y) - H(Y|X)
        mutual_info = H_Y - H_Y_given_X

        results.append({
            'position': pos + 1,
            'empirical_drci': empirical_drci,
            'theoretical_drci': theoretical_drci,
            'H_Y': H_Y,
            'H_X': H_X,
            'H_Y_given_X': H_Y_given_X,
            'mutual_info': mutual_info,
            'entropy_ratio': H_Y_given_X / H_Y if H_Y > 0 else np.nan
        })

        print(f" empirical={empirical_drci:.4f}, theoretical={theoretical_drci:.4f}")

    return pd.DataFrame(results)

# ============================================================================
# ANALYZE ALL MEDICAL MODELS
# ============================================================================

all_results = {}

for model_name, file_path in test_models.items():
    try:
        df = compute_position_analysis(file_path, model_name)
        all_results[model_name] = df
    except Exception as e:
        print(f"  ERROR: {e}")

# ============================================================================
# TEST 1: CORRELATION BETWEEN EMPIRICAL AND THEORETICAL ΔRCI
# ============================================================================

print("\n" + "="*80)
print("TEST 1: CORRELATION BETWEEN EMPIRICAL AND THEORETICAL dRCI")
print("="*80)

for model_name, df in all_results.items():
    # Remove any NaN values
    valid = ~(df['empirical_drci'].isna() | df['theoretical_drci'].isna())
    emp = df.loc[valid, 'empirical_drci'].values
    theo = df.loc[valid, 'theoretical_drci'].values

    r, p = pearsonr(emp, theo)

    print(f"\n{model_name}:")
    print(f"  Pearson r = {r:+.4f}, p = {p:.4e}")
    print(f"  R² = {r**2:.4f}")

    # Linear fit
    slope, intercept, _, _, _ = linregress(emp, theo)
    print(f"  Linear fit: theoretical = {slope:.4f} * empirical {intercept:+.4f}")

# ============================================================================
# TEST 2: LOG-SCALE POSITION TREND
# ============================================================================

print("\n" + "="*80)
print("TEST 2: LOG-SCALE POSITION TREND (DeepSeek's alpha * log(n) + beta)")
print("="*80)

for model_name, df in all_results.items():
    # Exclude position 1 (log(0) undefined) and position 30 (outlier)
    df_fit = df[(df['position'] > 1) & (df['position'] <= 29)].copy()

    # log(n) where n = position - 1 (number of prior exchanges)
    df_fit['log_n'] = np.log(df_fit['position'] - 1)

    # Fit: empirical_drci ~ α * log(n) + β
    slope, intercept, r, p, _ = linregress(df_fit['log_n'], df_fit['empirical_drci'])

    print(f"\n{model_name}:")
    print(f"  Empirical ΔRCI = {slope:.4f} * log(n) {intercept:+.4f}")
    print(f"  R² = {r**2:.4f}, p = {p:.4e}")

    # Also fit theoretical
    slope_th, intercept_th, r_th, p_th, _ = linregress(df_fit['log_n'], df_fit['theoretical_drci'])
    print(f"  Theoretical ΔRCI = {slope_th:.4f} * log(n) {intercept_th:+.4f}")
    print(f"  R² = {r_th**2:.4f}, p = {p_th:.4e}")

# ============================================================================
# TEST 3: PCA ON TRUE VS COLD EMBEDDINGS
# ============================================================================

print("\n" + "="*80)
print("TEST 3: PCA ANALYSIS - TRUE VS COLD EIGENVALUE DISTRIBUTIONS")
print("="*80)

pca_results = {}

for model_name, file_path in test_models.items():
    print(f"\n{model_name}:")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Collect all responses across all positions
    all_true = []
    all_cold = []

    for trial in data['trials']:
        all_true.extend(trial['responses']['true'])
        all_cold.extend(trial['responses']['cold'])

    print(f"  Encoding {len(all_true)} TRUE and {len(all_cold)} COLD responses...")

    # Encode
    true_emb = model.encode(all_true, convert_to_numpy=True, show_progress_bar=False)
    cold_emb = model.encode(all_cold, convert_to_numpy=True, show_progress_bar=False)

    # PCA
    print("  Running PCA...")
    pca_true = PCA(n_components=50)
    pca_cold = PCA(n_components=50)

    pca_true.fit(true_emb)
    pca_cold.fit(cold_emb)

    # Store results
    pca_results[model_name] = {
        'true_var': pca_true.explained_variance_,
        'cold_var': pca_cold.explained_variance_,
        'true_var_ratio': pca_true.explained_variance_ratio_,
        'cold_var_ratio': pca_cold.explained_variance_ratio_
    }

    # Summary stats
    print(f"  TRUE: First 5 PCs explain {np.sum(pca_true.explained_variance_ratio_[:5]):.2%} of variance")
    print(f"  COLD: First 5 PCs explain {np.sum(pca_cold.explained_variance_ratio_[:5]):.2%} of variance")

    # Effective dimensionality (Shannon entropy of eigenvalue distribution)
    def effective_dim(var_ratio):
        # Normalize to sum to 1
        p = var_ratio / np.sum(var_ratio)
        H = -np.sum(p * np.log(p + 1e-10))
        return np.exp(H)

    eff_dim_true = effective_dim(pca_true.explained_variance_ratio_)
    eff_dim_cold = effective_dim(pca_cold.explained_variance_ratio_)

    print(f"  Effective dimensionality: TRUE = {eff_dim_true:.2f}, COLD = {eff_dim_cold:.2f}")

# ============================================================================
# GENERATE PLOTS
# ============================================================================

print("\n" + "="*80)
print("GENERATING PLOTS")
print("="*80)

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# ===================================================================
# Plot 1: Empirical vs Theoretical ΔRCI
# ===================================================================

ax1 = fig.add_subplot(gs[0, 0])

for model_name, df in all_results.items():
    valid = ~(df['empirical_drci'].isna() | df['theoretical_drci'].isna())
    ax1.scatter(df.loc[valid, 'empirical_drci'],
                df.loc[valid, 'theoretical_drci'],
                alpha=0.6, s=30, label=model_name)

# Diagonal line
lim_min = min([df['empirical_drci'].min() for df in all_results.values()])
lim_max = max([df['empirical_drci'].max() for df in all_results.values()])
ax1.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.5, linewidth=2, label='Perfect Match')

ax1.set_xlabel('Empirical ΔRCI', fontsize=11)
ax1.set_ylabel('Theoretical ΔRCI\n1 - H(Y|X)/H(Y)', fontsize=11)
ax1.set_title('DeepSeek Formula vs Empirical', fontsize=12, fontweight='bold')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# ===================================================================
# Plot 2: ΔRCI vs log(position-1)
# ===================================================================

ax2 = fig.add_subplot(gs[0, 1])

for model_name, df in all_results.items():
    # Exclude position 1 and 30
    df_plot = df[(df['position'] > 1) & (df['position'] <= 29)].copy()
    df_plot['log_n'] = np.log(df_plot['position'] - 1)

    ax2.scatter(df_plot['log_n'], df_plot['empirical_drci'],
                alpha=0.6, s=30, label=model_name)

    # Fit line
    slope, intercept, _, _, _ = linregress(df_plot['log_n'], df_plot['empirical_drci'])
    x_fit = np.linspace(df_plot['log_n'].min(), df_plot['log_n'].max(), 100)
    y_fit = slope * x_fit + intercept
    ax2.plot(x_fit, y_fit, alpha=0.5, linewidth=1.5)

ax2.set_xlabel('log(n) where n = position - 1', fontsize=11)
ax2.set_ylabel('Empirical ΔRCI', fontsize=11)
ax2.set_title('Log-Scale Position Trend', fontsize=12, fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# ===================================================================
# Plot 3: Entropy Ratio vs Position
# ===================================================================

ax3 = fig.add_subplot(gs[0, 2])

for model_name, df in all_results.items():
    ax3.plot(df['position'], df['entropy_ratio'],
             marker='o', markersize=4, alpha=0.7, linewidth=1.5, label=model_name)

ax3.set_xlabel('Position', fontsize=11)
ax3.set_ylabel('H(Y|X) / H(Y)', fontsize=11)
ax3.set_title('Entropy Ratio Across Positions', fontsize=12, fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 31)

# ===================================================================
# Plot 4: Mutual Information vs Position
# ===================================================================

ax4 = fig.add_subplot(gs[1, 0])

for model_name, df in all_results.items():
    ax4.plot(df['position'], df['mutual_info'],
             marker='o', markersize=4, alpha=0.7, linewidth=1.5, label=model_name)

ax4.set_xlabel('Position', fontsize=11)
ax4.set_ylabel('I(X;Y) = H(Y) - H(Y|X)', fontsize=11)
ax4.set_title('Mutual Information Across Positions', fontsize=12, fontweight='bold')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, 31)

# ===================================================================
# Plot 5: Empirical vs Theoretical (Position-colored)
# ===================================================================

ax5 = fig.add_subplot(gs[1, 1])

# Concatenate all models
all_emp = []
all_theo = []
all_pos = []

for model_name, df in all_results.items():
    valid = ~(df['empirical_drci'].isna() | df['theoretical_drci'].isna())
    all_emp.extend(df.loc[valid, 'empirical_drci'].values)
    all_theo.extend(df.loc[valid, 'theoretical_drci'].values)
    all_pos.extend(df.loc[valid, 'position'].values)

scatter = ax5.scatter(all_emp, all_theo, c=all_pos, cmap='viridis',
                      s=30, alpha=0.6, edgecolors='none')

# Diagonal
lim_min = min(all_emp)
lim_max = max(all_emp)
ax5.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.5, linewidth=2)

ax5.set_xlabel('Empirical ΔRCI', fontsize=11)
ax5.set_ylabel('Theoretical ΔRCI', fontsize=11)
ax5.set_title('All Models (Color = Position)', fontsize=12, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax5)
cbar.set_label('Position', fontsize=10)
ax5.grid(True, alpha=0.3)

# ===================================================================
# Plot 6: PCA Eigenvalue Distributions
# ===================================================================

ax6 = fig.add_subplot(gs[1, 2])

for model_name, pca_data in pca_results.items():
    ax6.plot(range(1, 21), pca_data['true_var_ratio'][:20],
             marker='o', markersize=4, alpha=0.7, linewidth=1.5,
             label=f'{model_name} (TRUE)')
    ax6.plot(range(1, 21), pca_data['cold_var_ratio'][:20],
             marker='s', markersize=4, alpha=0.7, linewidth=1.5,
             linestyle='--', label=f'{model_name} (COLD)')

ax6.set_xlabel('Principal Component', fontsize=11)
ax6.set_ylabel('Explained Variance Ratio', fontsize=11)
ax6.set_title('PCA: TRUE vs COLD Eigenvalues', fontsize=12, fontweight='bold')
ax6.legend(fontsize=7, ncol=2)
ax6.grid(True, alpha=0.3)
ax6.set_yscale('log')

# ===================================================================
# Plot 7: H(Y) vs H(X) by Position
# ===================================================================

ax7 = fig.add_subplot(gs[2, 0])

for model_name, df in all_results.items():
    ax7.scatter(df['H_X'], df['H_Y'], c=df['position'], cmap='viridis',
                s=30, alpha=0.6, label=model_name)

ax7.set_xlabel('H(X) = H(COLD)', fontsize=11)
ax7.set_ylabel('H(Y) = H(TRUE)', fontsize=11)
ax7.set_title('Entropy: TRUE vs COLD', fontsize=12, fontweight='bold')
ax7.legend(fontsize=8)
ax7.grid(True, alpha=0.3)

# ===================================================================
# Plot 8: Residuals (Empirical - Theoretical)
# ===================================================================

ax8 = fig.add_subplot(gs[2, 1])

for model_name, df in all_results.items():
    residuals = df['empirical_drci'] - df['theoretical_drci']
    ax8.plot(df['position'], residuals,
             marker='o', markersize=4, alpha=0.7, linewidth=1.5, label=model_name)

ax8.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=2)
ax8.set_xlabel('Position', fontsize=11)
ax8.set_ylabel('Residual (Empirical - Theoretical)', fontsize=11)
ax8.set_title('Prediction Errors', fontsize=12, fontweight='bold')
ax8.legend(fontsize=8)
ax8.grid(True, alpha=0.3)
ax8.set_xlim(0, 31)

# ===================================================================
# Plot 9: Cumulative Variance Explained (PCA)
# ===================================================================

ax9 = fig.add_subplot(gs[2, 2])

for model_name, pca_data in pca_results.items():
    cum_true = np.cumsum(pca_data['true_var_ratio'])
    cum_cold = np.cumsum(pca_data['cold_var_ratio'])

    ax9.plot(range(1, 51), cum_true,
             marker='o', markersize=3, alpha=0.7, linewidth=1.5,
             label=f'{model_name} (TRUE)')
    ax9.plot(range(1, 51), cum_cold,
             marker='s', markersize=3, alpha=0.7, linewidth=1.5,
             linestyle='--', label=f'{model_name} (COLD)')

ax9.axhline(0.95, color='red', linestyle=':', alpha=0.5, linewidth=1)
ax9.set_xlabel('Number of Components', fontsize=11)
ax9.set_ylabel('Cumulative Variance Explained', fontsize=11)
ax9.set_title('PCA: Cumulative Variance', fontsize=12, fontweight='bold')
ax9.legend(fontsize=7, ncol=2)
ax9.grid(True, alpha=0.3)

plt.tight_layout()
output_path = "c:/Users/barla/mch_experiments/analysis/deepseek_theory_validation.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"[OK] Plot saved: {output_path}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Combine all results
combined_df = pd.concat([df.assign(model=name) for name, df in all_results.items()])
combined_df.to_csv('c:/Users/barla/mch_experiments/analysis/deepseek_theory_results.csv', index=False)
print("[OK] Results saved: deepseek_theory_results.csv")

# ============================================================================
# FINAL REPORT
# ============================================================================

print("\n" + "="*80)
print("FINAL REPORT: DeepSeek's Theory vs Empirical Data")
print("="*80)

# Overall correlation across all models
all_emp_flat = []
all_theo_flat = []

for df in all_results.values():
    valid = ~(df['empirical_drci'].isna() | df['theoretical_drci'].isna())
    all_emp_flat.extend(df.loc[valid, 'empirical_drci'].values)
    all_theo_flat.extend(df.loc[valid, 'theoretical_drci'].values)

r_overall, p_overall = pearsonr(all_emp_flat, all_theo_flat)

print(f"\n1. OVERALL CORRELATION:")
print(f"   Empirical vs Theoretical ΔRCI: r = {r_overall:+.4f}, p = {p_overall:.4e}")
print(f"   R² = {r_overall**2:.4f}")

if r_overall**2 > 0.70:
    print(f"   ✓ STRONG MATCH: DeepSeek's formula explains {r_overall**2*100:.1f}% of variance")
elif r_overall**2 > 0.50:
    print(f"   ~ MODERATE MATCH: DeepSeek's formula explains {r_overall**2*100:.1f}% of variance")
else:
    print(f"   ✗ WEAK MATCH: DeepSeek's formula explains only {r_overall**2*100:.1f}% of variance")

print(f"\n2. LOG-SCALE POSITION TREND:")
for model_name, df in all_results.items():
    df_fit = df[(df['position'] > 1) & (df['position'] <= 29)].copy()
    df_fit['log_n'] = np.log(df_fit['position'] - 1)
    slope, intercept, r, p, _ = linregress(df_fit['log_n'], df_fit['empirical_drci'])

    print(f"   {model_name}: α = {slope:.4f}, β = {intercept:+.4f}, R² = {r**2:.4f}")

print(f"\n3. PCA DIMENSIONALITY:")
for model_name, pca_data in pca_results.items():
    # Effective dimensionality
    def effective_dim(var_ratio):
        p = var_ratio / np.sum(var_ratio)
        H = -np.sum(p * np.log(p + 1e-10))
        return np.exp(H)

    eff_dim_true = effective_dim(pca_data['true_var_ratio'])
    eff_dim_cold = effective_dim(pca_data['cold_var_ratio'])

    print(f"   {model_name}: TRUE = {eff_dim_true:.2f} dims, COLD = {eff_dim_cold:.2f} dims")
    print(f"      Dimensionality increase: {eff_dim_true - eff_dim_cold:+.2f} dims")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
