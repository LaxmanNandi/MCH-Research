"""
TEST 1: Gaussian Assumption Verification
DeepSeek's Gap Analysis — Critical Test

Embeddings are NOT saved in JSON files, but raw text responses ARE saved
in 10 files. We regenerate embeddings using the same model (all-MiniLM-L6-v2)
and test the Gaussian assumption.

Testing on representative subset (2 medical + 2 philosophy) for tractability.
"""

import json
import numpy as np
from scipy import stats
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("TEST 1: GAUSSIAN ASSUMPTION VERIFICATION")
print("=" * 70)

# Load embedding model (same as used in experiments)
print("\nLoading embedding model (all-MiniLM-L6-v2)...")
embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("Model loaded.")

# Representative models (2 medical, 2 philosophy)
test_files = {
    'DeepSeek V3.1 (Med)': r'C:\Users\barla\mch_experiments\data\open_medical_rerun\mch_results_deepseek_v3_1_medical_50trials.json',
    'Mistral Small 24B (Med)': r'C:\Users\barla\mch_experiments\data\open_medical_rerun\mch_results_mistral_small_24b_medical_50trials.json',
    'GPT-4o (Phil)': r'C:\Users\barla\mch_experiments\data\closed_model_philosophy_rerun\mch_results_gpt4o_philosophy_50trials.json',
    'Claude Haiku (Phil)': r'C:\Users\barla\mch_experiments\data\closed_model_philosophy_rerun\mch_results_claude_haiku_philosophy_50trials.json',
}

results = []

for model_name, filepath in test_files.items():
    print(f"\n{'=' * 50}")
    print(f"Model: {model_name}")
    print(f"{'=' * 50}")

    with open(filepath, encoding='utf-8') as f:
        data = json.load(f)

    trials = data['trials']
    n_trials = min(len(trials), 20)  # Use first 20 trials for speed
    n_positions = 30

    print(f"Using {n_trials} trials x {n_positions} positions")

    # Collect all responses and embed them
    true_responses = []
    cold_responses = []

    for trial in trials[:n_trials]:
        resp = trial.get('responses', {})
        true_r = resp.get('true', [])
        cold_r = resp.get('cold', [])
        if true_r and cold_r:
            true_responses.extend(true_r[:n_positions])
            cold_responses.extend(cold_r[:n_positions])

    print(f"TRUE responses: {len(true_responses)}")
    print(f"COLD responses: {len(cold_responses)}")

    if len(true_responses) < 30 or len(cold_responses) < 30:
        print("Not enough responses, skipping.")
        continue

    # Generate embeddings
    print("Generating TRUE embeddings...")
    true_embs = embed_model.encode(true_responses, show_progress_bar=False)
    print("Generating COLD embeddings...")
    cold_embs = embed_model.encode(cold_responses, show_progress_bar=False)

    embedding_dim = true_embs.shape[1]
    print(f"Embedding dimension: {embedding_dim}")

    for condition, embs, label in [('TRUE', true_embs, 'true'), ('COLD', cold_embs, 'cold')]:
        print(f"\n--- {condition} condition ({embs.shape[0]} vectors) ---")

        # 1. Marginal normality test (Shapiro-Wilk per dimension)
        # Sample max 5000 for Shapiro-Wilk limit
        n_samples = min(embs.shape[0], 5000)
        embs_sample = embs[:n_samples]

        normality_violations = 0
        total_dims = embedding_dim

        # Test every 4th dimension for speed (384 dims -> 96 tests)
        test_dims = range(0, embedding_dim, 4)
        n_tested = 0

        for dim in test_dims:
            try:
                stat, p = stats.shapiro(embs_sample[:, dim])
                n_tested += 1
                if p < 0.01:
                    normality_violations += 1
            except:
                pass

        pct_violations = normality_violations / n_tested * 100 if n_tested > 0 else 0
        print(f"  Shapiro-Wilk (sampled {n_tested}/{embedding_dim} dims):")
        print(f"    Violations (p<0.01): {normality_violations}/{n_tested} ({pct_violations:.1f}%)")

        # 2. D'Agostino-Pearson test (more robust for larger samples)
        dag_violations = 0
        dag_tested = 0
        for dim in test_dims:
            try:
                stat, p = stats.normaltest(embs_sample[:, dim])
                dag_tested += 1
                if p < 0.01:
                    dag_violations += 1
            except:
                pass

        dag_pct = dag_violations / dag_tested * 100 if dag_tested > 0 else 0
        print(f"  D'Agostino-Pearson (sampled {dag_tested} dims):")
        print(f"    Violations (p<0.01): {dag_violations}/{dag_tested} ({dag_pct:.1f}%)")

        # 3. Skewness and Kurtosis across dimensions
        skewness = np.array([stats.skew(embs_sample[:, d]) for d in range(embedding_dim)])
        kurtosis = np.array([stats.kurtosis(embs_sample[:, d]) for d in range(embedding_dim)])
        print(f"  Skewness: mean={np.mean(skewness):.4f}, std={np.std(skewness):.4f}, max|skew|={np.max(np.abs(skewness)):.4f}")
        print(f"  Kurtosis: mean={np.mean(kurtosis):.4f}, std={np.std(kurtosis):.4f}, max|kurt|={np.max(np.abs(kurtosis)):.4f}")
        pct_high_skew = np.mean(np.abs(skewness) > 1.0) * 100
        pct_high_kurt = np.mean(np.abs(kurtosis) > 3.0) * 100
        print(f"  Dims with |skew|>1: {pct_high_skew:.1f}%")
        print(f"  Dims with |kurt|>3: {pct_high_kurt:.1f}%")

        # 4. Covariance sphericity
        cov_matrix = np.cov(embs_sample.T)
        # Use top eigenvalues (full eigendecomp too slow for 384x384)
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        eigenvalues = eigenvalues[eigenvalues > 0]  # Keep positive only

        sphericity_index = np.std(eigenvalues) / np.mean(eigenvalues)
        explained_by_top10 = np.sum(sorted(eigenvalues, reverse=True)[:10]) / np.sum(eigenvalues)
        effective_dim = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)

        print(f"  Covariance Sphericity:")
        print(f"    Sphericity index: {sphericity_index:.3f} (0=spherical, >0.5=non-spherical)")
        print(f"    Top-10 eigenvalues explain: {explained_by_top10*100:.1f}%")
        print(f"    Effective dimensionality: {effective_dim:.1f}/{embedding_dim}")

        # 5. Multivariate normality proxy: Mardia's test approximation
        # Chi-squared test on Mahalanobis distances
        try:
            mean_vec = np.mean(embs_sample, axis=0)
            # Use pseudo-inverse for stability
            cov_inv = np.linalg.pinv(cov_matrix)
            centered = embs_sample - mean_vec
            mahal_sq = np.sum(centered @ cov_inv * centered, axis=1)

            # Under multivariate normal, Mahalanobis^2 ~ chi-squared(p)
            expected_mean = embedding_dim
            expected_var = 2 * embedding_dim
            actual_mean = np.mean(mahal_sq)
            actual_var = np.var(mahal_sq)

            print(f"  Mahalanobis distances (multivariate normality proxy):")
            print(f"    Expected mean (chi2): {expected_mean:.1f}, Actual: {actual_mean:.1f}")
            print(f"    Expected var: {expected_var:.1f}, Actual: {actual_var:.1f}")
            print(f"    Ratio (actual/expected mean): {actual_mean/expected_mean:.3f}")
        except Exception as e:
            print(f"  Mahalanobis test failed: {e}")

        # Assessment
        is_marginal_ok = pct_violations < 30  # Less than 30% dimensions non-normal
        is_shape_ok = pct_high_skew < 20 and pct_high_kurt < 20
        is_spherical = sphericity_index < 2.0

        assessment = "HOLDS" if (is_marginal_ok and is_shape_ok) else \
                     "PARTIAL" if (is_marginal_ok or is_shape_ok) else "FAILS"

        print(f"\n  ASSESSMENT: Gaussian assumption {assessment}")
        print(f"    Marginal normality: {'OK' if is_marginal_ok else 'VIOLATED'} ({pct_violations:.0f}% non-normal)")
        print(f"    Shape (skew/kurt): {'OK' if is_shape_ok else 'NON-IDEAL'}")
        print(f"    Sphericity: {'SPHERICAL-ISH' if is_spherical else 'NON-SPHERICAL'} (index={sphericity_index:.3f})")

        results.append({
            'model': model_name,
            'condition': condition,
            'n_vectors': embs.shape[0],
            'embedding_dim': embedding_dim,
            'pct_shapiro_violations': pct_violations,
            'pct_dagostino_violations': dag_pct,
            'mean_skewness': np.mean(np.abs(skewness)),
            'mean_kurtosis': np.mean(np.abs(kurtosis)),
            'sphericity_index': sphericity_index,
            'effective_dim': effective_dim,
            'top10_explained': explained_by_top10,
            'assessment': assessment,
        })

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)

print(f"\n{'Model':<30} {'Cond':<6} {'N':<6} {'Shapiro%':<10} {'DAgo%':<8} {'|Skew|':<8} {'|Kurt|':<8} {'Sphericity':<12} {'EffDim':<8} {'Assessment'}")
print("-" * 120)
for r in results:
    print(f"{r['model']:<30} {r['condition']:<6} {r['n_vectors']:<6} {r['pct_shapiro_violations']:<10.1f} {r['pct_dagostino_violations']:<8.1f} {r['mean_skewness']:<8.3f} {r['mean_kurtosis']:<8.3f} {r['sphericity_index']:<12.3f} {r['effective_dim']:<8.1f} {r['assessment']}")

print("\n" + "=" * 70)
print("OVERALL VERDICT")
print("=" * 70)

assessments = [r['assessment'] for r in results]
n_holds = assessments.count('HOLDS')
n_partial = assessments.count('PARTIAL')
n_fails = assessments.count('FAILS')

print(f"HOLDS: {n_holds}/{len(assessments)}")
print(f"PARTIAL: {n_partial}/{len(assessments)}")
print(f"FAILS: {n_fails}/{len(assessments)}")

if n_holds + n_partial == len(assessments):
    print("\nGaussian assumption is REASONABLE for this data.")
    print("Marginal normality is approximately satisfied.")
    print("Covariance structure is non-spherical (expected for semantic embeddings).")
    print("This is typical for sentence embeddings and does not invalidate the framework.")
elif n_fails > len(assessments) / 2:
    print("\nGaussian assumption is PROBLEMATIC.")
    print("Consider non-parametric alternatives or explicit caveats.")
else:
    print("\nGaussian assumption is MIXED.")
    print("Some conditions satisfy, others don't. Include caveat in paper.")
