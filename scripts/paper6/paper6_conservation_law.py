#!/usr/bin/env python3
"""
Paper 6: Conservation Law Verification
=======================================
Tests whether ΔRCI and Var_Ratio are connected through mutual information
via a conservation-like relationship.

Theory (from DeepSeek):
  pred_ΔRCI = 1 - exp(-2·MI)
  pred_conservation = 1 - 2·MI
  obs_product = ΔRCI · Var_Ratio

Tests:
  Test 1: r(ΔRCI, pred_ΔRCI) — does MI predict context sensitivity?
  Test 2: r(obs_product, pred_conservation) — does a conservation law hold?

Uses only runs with saved response text (~13 model-domain runs).
Computes MI via KSG entropy estimator (k=3) with bootstrap CIs.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy import stats
from scipy.spatial import cKDTree
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding='utf-8')

BASE = Path("C:/Users/barla/mch_experiments")
OUTPUT_DIR = BASE / "data" / "paper6" / "conservation_law_verification"
FIGURE_DIR = BASE / "docs" / "figures" / "paper6"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# PHASE A: Data Preparation
# ============================================================================

def get_runs_with_text():
    """Identify all model-domain runs with saved response text."""
    dirs = [
        (BASE / "data" / "medical" / "closed_models", "Medical"),
        (BASE / "data" / "medical" / "open_models", "Medical"),
        (BASE / "data" / "medical" / "gemini_flash", "Medical"),
        (BASE / "data" / "philosophy" / "closed_models", "Philosophy"),
        (BASE / "data" / "philosophy" / "open_models", "Philosophy"),
    ]

    runs = []
    seen = set()  # deduplicate

    for d, domain in dirs:
        if not d.exists():
            continue
        for f in sorted(d.iterdir()):
            if not f.suffix == '.json':
                continue
            if 'checkpoint' in f.name or 'metrics_only' in f.name or 'BACKUP' in f.name:
                continue

            try:
                with open(f, encoding='utf-8') as fh:
                    data = json.load(fh)
                trials = data.get('trials', [])
                if not trials:
                    continue

                # Check for response text
                t = trials[0]
                if 'responses' not in t:
                    continue
                r = t['responses']
                if not (isinstance(r, dict) and 'true' in r):
                    continue
                if not (isinstance(r['true'], list) and len(r['true']) > 0):
                    continue
                if not (isinstance(r['true'][0], str) and len(r['true'][0]) > 10):
                    continue

                model = data.get('model', f.stem)
                key = f"{model}_{domain}"
                if key in seen:
                    continue
                seen.add(key)

                runs.append({
                    'model': model,
                    'domain': domain,
                    'path': str(f),
                    'n_trials': len(trials),
                })
            except Exception as e:
                print(f"  Skipping {f.name}: {e}")

    return runs


def compute_embeddings_for_run(run, embedder):
    """Extract TRUE and COLD response embeddings for a single run."""
    with open(run['path'], encoding='utf-8') as f:
        data = json.load(f)

    trials = data['trials']
    n_trials = len(trials)
    n_prompts = len(trials[0]['responses']['true'])

    # Collect all TRUE and COLD responses
    true_texts = []
    cold_texts = []
    true_alignments = []
    cold_alignments = []

    for t in trials:
        true_texts.extend(t['responses']['true'])
        cold_texts.extend(t['responses']['cold'])
        # Also grab pre-computed alignments if available
        if 'alignments' in t:
            true_alignments.extend(t['alignments'].get('true', []))
            cold_alignments.extend(t['alignments'].get('cold', []))

    # Compute embeddings
    print(f"  Embedding {len(true_texts)} TRUE + {len(cold_texts)} COLD responses...", flush=True)
    true_embs = embedder.encode(true_texts, show_progress_bar=False, convert_to_numpy=True)
    cold_embs = embedder.encode(cold_texts, show_progress_bar=False, convert_to_numpy=True)

    # Compute ΔRCI from alignments
    if true_alignments and cold_alignments:
        mean_true = np.mean(true_alignments)
        mean_cold = np.mean(cold_alignments)
        drci = mean_true - mean_cold
    else:
        # Compute from embeddings (Paper 2 method)
        # Per-trial: cosine(true[i], cold[i]) for each prompt position
        all_cosines = []
        for i in range(len(true_texts)):
            cos = np.dot(true_embs[i], cold_embs[i]) / (
                np.linalg.norm(true_embs[i]) * np.linalg.norm(cold_embs[i])
            )
            all_cosines.append(cos)
        drci = 1.0 - np.mean(all_cosines)

    # Compute Var_Ratio per position, then average
    # Var_Ratio = Var(TRUE embeddings at position p) / Var(COLD embeddings at position p)
    true_embs_by_trial = true_embs.reshape(n_trials, n_prompts, -1)
    cold_embs_by_trial = cold_embs.reshape(n_trials, n_prompts, -1)

    var_ratios = []
    for p in range(n_prompts):
        true_at_p = true_embs_by_trial[:, p, :]  # (n_trials, 384)
        cold_at_p = cold_embs_by_trial[:, p, :]

        var_true = np.mean(np.var(true_at_p, axis=0))
        var_cold = np.mean(np.var(cold_at_p, axis=0))

        if var_cold > 1e-10:
            vr = var_true / var_cold
        else:
            vr = 1.0
        var_ratios.append(vr)

    mean_var_ratio = np.mean(var_ratios)

    return {
        'true_embs': true_embs,
        'cold_embs': cold_embs,
        'n_trials': n_trials,
        'n_prompts': n_prompts,
        'drci': float(drci),
        'var_ratio': float(mean_var_ratio),
    }


# ============================================================================
# PHASE B: KSG Mutual Information Estimator
# ============================================================================

def ksg_entropy(X, k=3):
    """
    KSG (Kraskov-Stögbauer-Grassberger) entropy estimator.
    Uses k-nearest neighbor distances to estimate differential entropy.

    H(X) ≈ ψ(N) - ψ(k) + d·log(2) + (d/N)·Σ log(ε_i)

    where ε_i is the distance to the k-th nearest neighbor,
    d is dimensionality, N is sample size.
    """
    from scipy.special import digamma

    N, d = X.shape

    # Build KD-tree for efficient neighbor search
    tree = cKDTree(X)

    # Find k+1 nearest neighbors (includes self)
    distances, _ = tree.query(X, k=k+1, p=2)  # Euclidean
    # k-th neighbor distance (index k, since index 0 is self with dist 0)
    eps = distances[:, k]

    # Avoid log(0)
    eps = np.maximum(eps, 1e-10)

    # KSG entropy estimate
    # H = ψ(N) - ψ(k) + log(V_d) + (d/N) * Σ log(ε_i)
    # where V_d = π^(d/2) / Γ(d/2 + 1) is the volume of unit ball
    # But for MI computation, V_d cancels, so we use:
    # H ≈ ψ(N) - ψ(k) + d * <log(2·ε_i)>
    H = digamma(N) - digamma(k) + d * np.mean(np.log(2.0 * eps))

    return H


def compute_mutual_information(true_embs, cold_embs, k=3):
    """
    Compute mutual information between TRUE and COLD embedding distributions.

    MI = H(all) - (H(true) + H(cold)) / 2

    Uses dimensionality reduction via PCA to top 20 components
    (384D is too high for KSG to be reliable).
    """
    from sklearn.decomposition import PCA

    # Subsample if too many embeddings (KSG is O(N²))
    max_samples = 200
    if len(true_embs) > max_samples:
        idx_t = np.random.choice(len(true_embs), max_samples, replace=False)
        true_sub = true_embs[idx_t]
    else:
        true_sub = true_embs

    if len(cold_embs) > max_samples:
        idx_c = np.random.choice(len(cold_embs), max_samples, replace=False)
        cold_sub = cold_embs[idx_c]
    else:
        cold_sub = cold_embs

    # Combine and reduce dimensionality
    combined = np.vstack([true_sub, cold_sub])
    n_components = min(20, combined.shape[0] - 1, combined.shape[1])
    pca = PCA(n_components=n_components)
    combined_pca = pca.fit_transform(combined)

    true_pca = combined_pca[:len(true_sub)]
    cold_pca = combined_pca[len(true_sub):]

    # Compute entropies
    H_all = ksg_entropy(combined_pca, k=k)
    H_true = ksg_entropy(true_pca, k=k)
    H_cold = ksg_entropy(cold_pca, k=k)

    # MI = H_all - (H_true + H_cold) / 2
    MI = H_all - (H_true + H_cold) / 2.0

    return MI, H_all, H_true, H_cold


def bootstrap_mi(true_embs, cold_embs, n_bootstrap=1000, k=3):
    """Bootstrap confidence intervals for MI estimate."""
    mi_values = []

    for b in range(n_bootstrap):
        # Resample with replacement
        idx_t = np.random.choice(len(true_embs), len(true_embs), replace=True)
        idx_c = np.random.choice(len(cold_embs), len(cold_embs), replace=True)

        mi, _, _, _ = compute_mutual_information(
            true_embs[idx_t], cold_embs[idx_c], k=k
        )
        mi_values.append(mi)

    mi_values = np.array(mi_values)
    ci_lower = np.percentile(mi_values, 2.5)
    ci_upper = np.percentile(mi_values, 97.5)
    mi_mean = np.mean(mi_values)
    mi_std = np.std(mi_values)

    return mi_mean, mi_std, ci_lower, ci_upper


# ============================================================================
# PHASE C: Predictions and Tests
# ============================================================================

def compute_predictions(results):
    """Compute theory predictions and observed products."""
    for r in results:
        mi = r['mi']
        r['pred_drci'] = 1.0 - np.exp(-2.0 * mi)
        r['pred_conservation'] = 1.0 - 2.0 * mi
        r['obs_product'] = r['drci'] * r['var_ratio']
    return results


def run_correlation_tests(results):
    """Run Test 1 and Test 2 correlation analyses."""
    drci = np.array([r['drci'] for r in results])
    pred_drci = np.array([r['pred_drci'] for r in results])
    obs_product = np.array([r['obs_product'] for r in results])
    pred_conservation = np.array([r['pred_conservation'] for r in results])

    # Test 1: r(ΔRCI, pred_ΔRCI)
    r1, p1 = stats.pearsonr(drci, pred_drci)
    slope1, intercept1, _, _, se1 = stats.linregress(pred_drci, drci)
    rho1, rho_p1 = stats.spearmanr(drci, pred_drci)

    # Test 2: r(obs_product, pred_conservation)
    r2, p2 = stats.pearsonr(obs_product, pred_conservation)
    slope2, intercept2, _, _, se2 = stats.linregress(pred_conservation, obs_product)
    rho2, rho_p2 = stats.spearmanr(obs_product, pred_conservation)

    return {
        'test1': {
            'pearson_r': r1, 'pearson_p': p1,
            'spearman_rho': rho1, 'spearman_p': rho_p1,
            'slope': slope1, 'intercept': intercept1, 'se': se1,
        },
        'test2': {
            'pearson_r': r2, 'pearson_p': p2,
            'spearman_rho': rho2, 'spearman_p': rho_p2,
            'slope': slope2, 'intercept': intercept2, 'se': se2,
        },
    }


# ============================================================================
# PHASE D: Figures and Report
# ============================================================================

def generate_figures(results, test_results):
    """Generate scatter plots with regression lines."""

    # Color scheme by domain
    colors = {'Medical': '#e74c3c', 'Philosophy': '#3498db'}
    markers = {'Medical': 'o', 'Philosophy': 's'}

    # --- Figure 1: Test 1 (ΔRCI vs pred_ΔRCI) ---
    fig, ax = plt.subplots(figsize=(10, 8))

    for r in results:
        ax.scatter(r['pred_drci'], r['drci'],
                   c=colors[r['domain']], marker=markers[r['domain']],
                   s=100, edgecolors='black', linewidths=0.5, zorder=5)
        ax.annotate(r['model'].replace('_', ' ').title(),
                    (r['pred_drci'], r['drci']),
                    fontsize=7, ha='left', va='bottom',
                    xytext=(5, 3), textcoords='offset points')

    # Regression line
    pred_drci = np.array([r['pred_drci'] for r in results])
    drci = np.array([r['drci'] for r in results])
    t1 = test_results['test1']
    x_line = np.linspace(pred_drci.min() - 0.02, pred_drci.max() + 0.02, 100)
    y_line = t1['slope'] * x_line + t1['intercept']
    ax.plot(x_line, y_line, 'k--', alpha=0.7, linewidth=1.5)

    # Perfect prediction line
    lims = [min(pred_drci.min(), drci.min()) - 0.02,
            max(pred_drci.max(), drci.max()) + 0.02]
    ax.plot(lims, lims, ':', color='gray', alpha=0.5, label='Perfect prediction')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
               markersize=10, label='Medical'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#3498db',
               markersize=10, label='Philosophy'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11)

    ax.set_xlabel('Predicted ΔRCI = 1 - exp(-2·MI)', fontsize=12)
    ax.set_ylabel('Observed ΔRCI', fontsize=12)
    ax.set_title(f'Test 1: MI Predicts Context Sensitivity\n'
                 f'r = {t1["pearson_r"]:.3f}, p = {t1["pearson_p"]:.2e}, '
                 f'slope = {t1["slope"]:.3f}',
                 fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / 'fig_test1_drci_vs_pred.png', dpi=200)
    plt.close()
    print(f"  Saved: fig_test1_drci_vs_pred.png")

    # --- Figure 2: Test 2 (Conservation Law) ---
    fig, ax = plt.subplots(figsize=(10, 8))

    for r in results:
        ax.scatter(r['pred_conservation'], r['obs_product'],
                   c=colors[r['domain']], marker=markers[r['domain']],
                   s=100, edgecolors='black', linewidths=0.5, zorder=5)
        ax.annotate(r['model'].replace('_', ' ').title(),
                    (r['pred_conservation'], r['obs_product']),
                    fontsize=7, ha='left', va='bottom',
                    xytext=(5, 3), textcoords='offset points')

    # Regression line
    pred_cons = np.array([r['pred_conservation'] for r in results])
    obs_prod = np.array([r['obs_product'] for r in results])
    t2 = test_results['test2']
    x_line = np.linspace(pred_cons.min() - 0.05, pred_cons.max() + 0.05, 100)
    y_line = t2['slope'] * x_line + t2['intercept']
    ax.plot(x_line, y_line, 'k--', alpha=0.7, linewidth=1.5)

    # Perfect conservation line
    lims = [min(pred_cons.min(), obs_prod.min()) - 0.05,
            max(pred_cons.max(), obs_prod.max()) + 0.05]
    ax.plot(lims, lims, ':', color='gray', alpha=0.5, label='Perfect conservation')

    ax.legend(handles=legend_elements, loc='best', fontsize=11)
    ax.set_xlabel('Predicted Conservation = 1 - 2·MI', fontsize=12)
    ax.set_ylabel('Observed Product = ΔRCI × Var_Ratio', fontsize=12)
    ax.set_title(f'Test 2: Conservation Law Verification\n'
                 f'r = {t2["pearson_r"]:.3f}, p = {t2["pearson_p"]:.2e}, '
                 f'slope = {t2["slope"]:.3f}',
                 fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / 'fig_test2_conservation_law.png', dpi=200)
    plt.close()
    print(f"  Saved: fig_test2_conservation_law.png")


def generate_report(results, test_results):
    """Generate Paper6_Conservation_Law_Report.md"""

    t1 = test_results['test1']
    t2 = test_results['test2']

    # Determine verdict
    if t1['pearson_p'] < 0.01 and t2['pearson_p'] < 0.01:
        if abs(t1['pearson_r']) > 0.7 and abs(t2['pearson_r']) > 0.7:
            verdict = "SUPPORTED"
            verdict_text = "Both tests show strong, significant correlations. The conservation law theory is supported by the data."
        else:
            verdict = "PARTIAL"
            verdict_text = "Both tests show significant but moderate correlations. The theory captures real structure but may be incomplete."
    elif t1['pearson_p'] < 0.05 or t2['pearson_p'] < 0.05:
        verdict = "PARTIAL"
        verdict_text = "One or both tests show marginal significance. The theory has partial empirical support."
    else:
        verdict = "FALSIFIED"
        verdict_text = "Neither test shows significant correlation. The conservation law theory is not supported by the data."

    report = f"""# Paper 6: Conservation Law Verification Report

**Date:** {datetime.now().strftime('%B %d, %Y')}
**Author:** Dr. Laxman M M, MBBS | Primary Health Centre Manchi, Karnataka
**Status:** Empirical verification complete

---

## 1. Theory Under Test

The conservation law hypothesis proposes that ΔRCI and Var_Ratio are linked through
mutual information (MI) between TRUE and COLD response embedding distributions:

```
Prediction 1:  ΔRCI ≈ 1 - exp(-2·MI)
Prediction 2:  ΔRCI × Var_Ratio ≈ 1 - 2·MI    (conservation law)
```

If these hold, it would establish an information-theoretic foundation connecting
context sensitivity (ΔRCI) and output variance (Var_Ratio) through a single
underlying quantity (MI).

---

## 2. Data

{len(results)} model-domain runs with saved response text:

| # | Model | Domain | Trials | ΔRCI | Var_Ratio | MI | MI 95% CI |
|---|-------|--------|--------|------|-----------|-----|-----------|
"""

    for i, r in enumerate(results, 1):
        report += f"| {i} | {r['model']} | {r['domain']} | {r['n_trials']} | "
        report += f"{r['drci']:.4f} | {r['var_ratio']:.4f} | {r['mi']:.4f} | "
        report += f"[{r['mi_ci_lower']:.4f}, {r['mi_ci_upper']:.4f}] |\n"

    report += f"""
---

## 3. Test 1: MI Predicts Context Sensitivity

**Hypothesis:** ΔRCI = 1 - exp(-2·MI)

| Metric | Value |
|--------|-------|
| Pearson r | {t1['pearson_r']:.4f} |
| Pearson p | {t1['pearson_p']:.2e} |
| Spearman ρ | {t1['spearman_rho']:.4f} |
| Spearman p | {t1['spearman_p']:.2e} |
| Regression slope | {t1['slope']:.4f} ± {t1['se']:.4f} |
| Regression intercept | {t1['intercept']:.4f} |

A perfect prediction would yield slope = 1.0, intercept = 0.0.

![Test 1: MI Predicts ΔRCI](../../docs/figures/paper6/fig_test1_drci_vs_pred.png)

---

## 4. Test 2: Conservation Law

**Hypothesis:** ΔRCI × Var_Ratio = 1 - 2·MI

| Metric | Value |
|--------|-------|
| Pearson r | {t2['pearson_r']:.4f} |
| Pearson p | {t2['pearson_p']:.2e} |
| Spearman ρ | {t2['spearman_rho']:.4f} |
| Spearman p | {t2['spearman_p']:.2e} |
| Regression slope | {t2['slope']:.4f} ± {t2['se']:.4f} |
| Regression intercept | {t2['intercept']:.4f} |

A perfect conservation law would yield slope = 1.0, intercept = 0.0.

![Test 2: Conservation Law](../../docs/figures/paper6/fig_test2_conservation_law.png)

---

## 5. Predictions Table

| Model | Domain | ΔRCI | Var_Ratio | MI | pred_ΔRCI | obs_product | pred_conservation |
|-------|--------|------|-----------|-----|-----------|-------------|-------------------|
"""

    for r in results:
        report += f"| {r['model']} | {r['domain']} | {r['drci']:.4f} | {r['var_ratio']:.4f} | "
        report += f"{r['mi']:.4f} | {r['pred_drci']:.4f} | {r['obs_product']:.4f} | "
        report += f"{r['pred_conservation']:.4f} |\n"

    report += f"""
---

## 6. Verdict

### **{verdict}**

{verdict_text}

### Test Summary

| Test | r | p | Significant? |
|------|---|---|-------------|
| Test 1 (MI → ΔRCI) | {t1['pearson_r']:.4f} | {t1['pearson_p']:.2e} | {'Yes' if t1['pearson_p'] < 0.05 else 'No'} |
| Test 2 (Conservation) | {t2['pearson_r']:.4f} | {t2['pearson_p']:.2e} | {'Yes' if t2['pearson_p'] < 0.05 else 'No'} |

### Interpretation

- **Slope deviation from 1.0** indicates the theory's quantitative predictions need calibration
- **Intercept deviation from 0.0** indicates systematic offset
- The relationship between ΔRCI, Var_Ratio, and MI is {'well-captured' if verdict == 'SUPPORTED' else 'partially captured' if verdict == 'PARTIAL' else 'not captured'} by the proposed conservation law

---

## 7. Methods

### Mutual Information Estimation
- **Algorithm:** KSG (Kraskov-Stögbauer-Grassberger) entropy estimator, k=3
- **Dimensionality reduction:** PCA to 20 components (384D too high for KSG)
- **Subsampling:** Max 200 samples per condition for computational tractability
- **Bootstrap:** 1000 iterations for 95% confidence intervals

### Embedding Model
- all-MiniLM-L6-v2 (384-dimensional), consistent with Papers 1–5

### Var_Ratio Computation
- Per-position variance across 50 trials: Var(TRUE) / Var(COLD)
- Averaged across all 30 positions

---

**Report generated:** {datetime.now().isoformat()}
**Script:** scripts/analysis/paper6_conservation_law.py
"""

    report_path = OUTPUT_DIR / "Paper6_Conservation_Law_Report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  Saved: {report_path}")

    return report_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("PAPER 6: CONSERVATION LAW VERIFICATION")
    print("=" * 70)
    print()

    # Phase A: Identify runs
    print("PHASE A: Data Preparation")
    print("-" * 40)
    runs = get_runs_with_text()
    print(f"  Found {len(runs)} runs with response text:")
    for r in runs:
        print(f"    {r['domain']:>10} | {r['model']:<25} | {r['n_trials']} trials")
    print()

    # Load embedding model
    print("  Loading embedding model...", flush=True)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    print("  Embedding model loaded.", flush=True)
    print()

    # Phase A+B: Compute embeddings and MI for each run
    print("PHASE B: Mutual Information Computation")
    print("-" * 40)

    results = []
    for i, run in enumerate(runs):
        print(f"\n  [{i+1}/{len(runs)}] {run['model']} ({run['domain']})", flush=True)

        # Compute embeddings, dRCI, Var_Ratio
        emb_data = compute_embeddings_for_run(run, embedder)

        # Compute MI with bootstrap
        print(f"  Computing MI (KSG, k=3)...", flush=True)
        mi, H_all, H_true, H_cold = compute_mutual_information(
            emb_data['true_embs'], emb_data['cold_embs'], k=3
        )
        print(f"    MI = {mi:.4f}, H_all = {H_all:.4f}, H_true = {H_true:.4f}, H_cold = {H_cold:.4f}")

        print(f"  Bootstrap (n=1000)...", flush=True)
        mi_mean, mi_std, ci_lower, ci_upper = bootstrap_mi(
            emb_data['true_embs'], emb_data['cold_embs'], n_bootstrap=1000, k=3
        )
        print(f"    MI = {mi_mean:.4f} ± {mi_std:.4f}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"    ΔRCI = {emb_data['drci']:.4f}, Var_Ratio = {emb_data['var_ratio']:.4f}")

        results.append({
            'model': run['model'],
            'domain': run['domain'],
            'n_trials': run['n_trials'],
            'drci': emb_data['drci'],
            'var_ratio': emb_data['var_ratio'],
            'mi': mi_mean,
            'mi_std': mi_std,
            'mi_ci_lower': ci_lower,
            'mi_ci_upper': ci_upper,
            'H_all': H_all,
            'H_true': H_true,
            'H_cold': H_cold,
        })

    # Save intermediate results
    results_path = OUTPUT_DIR / "conservation_law_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved intermediate results: {results_path}")

    # Phase C: Predictions and correlations
    print("\n\nPHASE C: Predictions and Correlation Tests")
    print("-" * 40)
    results = compute_predictions(results)
    test_results = run_correlation_tests(results)

    t1 = test_results['test1']
    t2 = test_results['test2']
    print(f"\n  Test 1 (MI → ΔRCI):")
    print(f"    Pearson r = {t1['pearson_r']:.4f}, p = {t1['pearson_p']:.2e}")
    print(f"    Slope = {t1['slope']:.4f}, Intercept = {t1['intercept']:.4f}")
    print(f"\n  Test 2 (Conservation Law):")
    print(f"    Pearson r = {t2['pearson_r']:.4f}, p = {t2['pearson_p']:.2e}")
    print(f"    Slope = {t2['slope']:.4f}, Intercept = {t2['intercept']:.4f}")

    # Phase D: Figures and report
    print("\n\nPHASE D: Report Generation")
    print("-" * 40)
    generate_figures(results, test_results)
    report_path = generate_report(results, test_results)

    # Final verdict
    if t1['pearson_p'] < 0.01 and t2['pearson_p'] < 0.01:
        if abs(t1['pearson_r']) > 0.7 and abs(t2['pearson_r']) > 0.7:
            verdict = "SUPPORTED"
        else:
            verdict = "PARTIAL"
    elif t1['pearson_p'] < 0.05 or t2['pearson_p'] < 0.05:
        verdict = "PARTIAL"
    else:
        verdict = "FALSIFIED"

    print(f"\n{'=' * 70}")
    print(f"VERDICT: {verdict}")
    print(f"{'=' * 70}")
    print(f"\nReport: {report_path}")
    print(f"Figures: {FIGURE_DIR}")
    print(f"Data: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
