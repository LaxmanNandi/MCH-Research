#!/usr/bin/env python3
"""
COLD Condition Analysis: The Prior Voice
==========================================
Analyzes COLD (no-context) responses across all models to reveal
intrinsic model "personality" — their default voice without context.

Analyses:
1. Var_COLD across all models (ranking table)
2. Var_COLD by domain (Medical vs Philosophy, Mann-Whitney U)
3. Var_COLD by model family (vendor clustering)
4. Var_COLD vs model size (Spearman correlation)
5. COLD embedding clustering (t-SNE)
6. Failure mode classification (WILD/EMPTY)
7. Prior voice centroids (default answer signatures)

Output: data/paper7_cold_analysis/ + docs/figures/paper7/
"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

BASE = Path("C:/Users/barla/mch_experiments")
OUTPUT_DIR = BASE / "data" / "paper7_cold_analysis"
FIGURE_DIR = BASE / "docs" / "figures" / "paper7"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Model metadata
# ------------------------------------------------------------------
MODEL_META = {
    'gemini_flash':      {'vendor': 'Google',    'family': 'Gemini',  'params': None,   'params_label': 'Undisclosed'},
    'deepseek_v3_1':     {'vendor': 'DeepSeek',  'family': 'DeepSeek','params': 671,    'params_label': '671B (37B active)'},
    'kimi_k2':           {'vendor': 'Moonshot',  'family': 'Kimi',    'params': 1000,   'params_label': '~1T (32B active)'},
    'llama_4_maverick':  {'vendor': 'Meta',      'family': 'Llama',   'params': 400,    'params_label': '400B (17B active)'},
    'llama_4_scout':     {'vendor': 'Meta',      'family': 'Llama',   'params': 109,    'params_label': '109B (17B active)'},
    'ministral_14b':     {'vendor': 'Mistral',   'family': 'Mistral', 'params': 14,     'params_label': '14B'},
    'mistral_small_24b': {'vendor': 'Mistral',   'family': 'Mistral', 'params': 24,     'params_label': '24B'},
    'qwen3_235b':        {'vendor': 'Alibaba',   'family': 'Qwen',    'params': 235,    'params_label': '235B (22B active)'},
    'claude_haiku':      {'vendor': 'Anthropic', 'family': 'Claude',  'params': None,   'params_label': 'Undisclosed'},
    'gpt4o':             {'vendor': 'OpenAI',    'family': 'GPT',     'params': None,   'params_label': 'Undisclosed'},
    'gpt4o_mini':        {'vendor': 'OpenAI',    'family': 'GPT',     'params': None,   'params_label': 'Undisclosed'},
}

# ------------------------------------------------------------------
# Data discovery (reuse from robustness script)
# ------------------------------------------------------------------
def get_runs_with_text():
    dirs = [
        (BASE / "data" / "medical" / "closed_models",    "Medical"),
        (BASE / "data" / "medical" / "open_models",       "Medical"),
        (BASE / "data" / "medical" / "gemini_flash",      "Medical"),
        (BASE / "data" / "philosophy" / "closed_models",  "Philosophy"),
        (BASE / "data" / "philosophy" / "open_models",    "Philosophy"),
    ]
    runs = []
    seen = set()
    for d, domain in dirs:
        if not d.exists():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix != '.json':
                continue
            if any(x in f.name for x in ('checkpoint', 'metrics_only', 'BACKUP', 'rerun')):
                continue
            try:
                with open(f, encoding='utf-8') as fh:
                    data = json.load(fh)
                trials = data.get('trials', [])
                if not trials:
                    continue
                r = trials[0].get('responses', {})
                if not isinstance(r, dict) or 'cold' not in r:
                    continue
                if not (isinstance(r['cold'], list) and len(r['cold']) > 0 and
                        isinstance(r['cold'][0], str) and len(r['cold'][0]) > 10):
                    continue
                model = data.get('model', f.stem)
                key = f"{model}_{domain}"
                if key in seen:
                    continue
                seen.add(key)
                runs.append({
                    'model': model, 'domain': domain,
                    'path': str(f), 'n_trials': len(trials),
                    'n_prompts': len(r['cold']),
                })
            except Exception as e:
                print(f"  Skipping {f.name}: {e}")
    return runs


# ------------------------------------------------------------------
# Embed COLD responses
# ------------------------------------------------------------------
def embed_cold_responses(runs, embedder):
    """Embed all COLD responses and return structured data."""
    results = []
    for i, run in enumerate(runs, 1):
        model = run['model']
        domain = run['domain']
        print(f"[{i}/{len(runs)}] {model} ({domain})", flush=True)

        with open(run['path'], encoding='utf-8') as f:
            data = json.load(f)

        trials = data['trials']
        n_trials = len(trials)
        n_prompts = len(trials[0]['responses']['cold'])

        # Collect COLD texts
        cold_texts = []
        for t in trials:
            cold_texts.extend(t['responses']['cold'])

        print(f"    Embedding {len(cold_texts)} COLD texts ...", flush=True)
        cold_embs = embedder.encode(cold_texts, show_progress_bar=False,
                                     convert_to_numpy=True, batch_size=64)

        # Reshape: (n_trials, n_prompts, embedding_dim)
        cold_by_pos = cold_embs.reshape(n_trials, n_prompts, -1)

        # Var_COLD: mean across positions of mean-dimension variance
        var_per_pos = []
        for p in range(n_prompts):
            v = float(np.mean(np.var(cold_by_pos[:, p, :], axis=0)))
            var_per_pos.append(v)
        var_cold = float(np.mean(var_per_pos))

        # Also compute per-position variance for detailed analysis
        # Mean embedding centroid (prior voice)
        centroid = np.mean(cold_embs, axis=0)
        centroid_norm = centroid / np.linalg.norm(centroid)

        # Intra-model spread: mean pairwise cosine distance
        norms = np.linalg.norm(cold_embs, axis=1, keepdims=True)
        normed = cold_embs / norms
        # Sample 500 pairs for efficiency
        n_samples = min(500, len(cold_embs))
        idx = np.random.choice(len(cold_embs), n_samples, replace=False)
        sampled = normed[idx]
        cos_matrix = sampled @ sampled.T
        upper = cos_matrix[np.triu_indices(n_samples, k=1)]
        mean_intra_cos = float(np.mean(upper))
        intra_spread = float(1.0 - mean_intra_cos)

        meta = MODEL_META.get(model, {})
        results.append({
            'model': model,
            'domain': domain,
            'vendor': meta.get('vendor', 'Unknown'),
            'family': meta.get('family', 'Unknown'),
            'params': meta.get('params'),
            'params_label': meta.get('params_label', 'Unknown'),
            'n_trials': n_trials,
            'n_prompts': n_prompts,
            'var_cold': var_cold,
            'var_per_pos': var_per_pos,
            'intra_spread': intra_spread,
            'centroid': centroid_norm.tolist(),
            'all_embeddings': cold_embs,  # keep for clustering
        })
        print(f"    Var_COLD={var_cold:.6f}  Spread={intra_spread:.4f}")

    return results


# ------------------------------------------------------------------
# Analysis 1: Var_COLD ranking table
# ------------------------------------------------------------------
def analysis_var_cold_ranking(results):
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Var_COLD RANKING ACROSS ALL MODELS")
    print("=" * 70)

    sorted_r = sorted(results, key=lambda x: x['var_cold'], reverse=True)
    print(f"\n{'Rank':<5} {'Model':<25} {'Domain':<12} {'Vendor':<12} {'Var_COLD':>10} {'Spread':>8}")
    print("-" * 75)
    for rank, r in enumerate(sorted_r, 1):
        print(f"{rank:<5} {r['model']:<25} {r['domain']:<12} {r['vendor']:<12} "
              f"{r['var_cold']:>10.6f} {r['intra_spread']:>8.4f}")

    return sorted_r


# ------------------------------------------------------------------
# Analysis 2: Var_COLD by domain (Mann-Whitney U)
# ------------------------------------------------------------------
def analysis_domain_comparison(results):
    from scipy import stats

    print("\n" + "=" * 70)
    print("ANALYSIS 2: Var_COLD BY DOMAIN")
    print("=" * 70)

    med_vars = [r['var_cold'] for r in results if r['domain'] == 'Medical']
    phil_vars = [r['var_cold'] for r in results if r['domain'] == 'Philosophy']

    print(f"\n  Medical (N={len(med_vars)}):    mean={np.mean(med_vars):.6f}, "
          f"SD={np.std(med_vars, ddof=1):.6f}")
    print(f"  Philosophy (N={len(phil_vars)}): mean={np.mean(phil_vars):.6f}, "
          f"SD={np.std(phil_vars, ddof=1):.6f}")

    u_stat, p_val = stats.mannwhitneyu(med_vars, phil_vars, alternative='two-sided')
    print(f"\n  Mann-Whitney U = {u_stat:.1f}, p = {p_val:.4f}")

    # Effect size (rank-biserial)
    n1, n2 = len(med_vars), len(phil_vars)
    r_rb = 1 - (2 * u_stat) / (n1 * n2)
    print(f"  Rank-biserial r = {r_rb:.3f}")

    if p_val < 0.05:
        direction = "Medical < Philosophy" if np.mean(med_vars) < np.mean(phil_vars) else "Medical > Philosophy"
        print(f"  Result: SIGNIFICANT — {direction}")
    else:
        print(f"  Result: Not significant")

    return {'med_vars': med_vars, 'phil_vars': phil_vars, 'U': u_stat, 'p': p_val, 'r_rb': r_rb}


# ------------------------------------------------------------------
# Analysis 3: Var_COLD by model family
# ------------------------------------------------------------------
def analysis_family_clustering(results):
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Var_COLD BY MODEL FAMILY")
    print("=" * 70)

    families = defaultdict(list)
    for r in results:
        families[r['family']].append(r)

    print(f"\n{'Family':<15} {'N':>3} {'Mean Var_COLD':>14} {'SD':>10} {'Models'}")
    print("-" * 75)
    for fam in sorted(families.keys()):
        members = families[fam]
        vars = [m['var_cold'] for m in members]
        models = ', '.join(f"{m['model']}({m['domain'][:3]})" for m in members)
        sd = np.std(vars, ddof=1) if len(vars) > 1 else 0
        print(f"{fam:<15} {len(members):>3} {np.mean(vars):>14.6f} {sd:>10.6f} {models}")

    return families


# ------------------------------------------------------------------
# Analysis 4: Var_COLD vs model size (Spearman)
# ------------------------------------------------------------------
def analysis_size_correlation(results):
    from scipy import stats

    print("\n" + "=" * 70)
    print("ANALYSIS 4: Var_COLD vs MODEL SIZE (Spearman)")
    print("=" * 70)

    # Only models with known parameter counts
    sized = [(r['params'], r['var_cold'], r['model'], r['domain'])
             for r in results if r['params'] is not None]

    if len(sized) < 4:
        print("  Insufficient models with known parameter counts.")
        return None

    params = [s[0] for s in sized]
    vars = [s[1] for s in sized]

    rho, p_val = stats.spearmanr(params, vars)
    print(f"\n  Models with known size: {len(sized)}")
    for p, v, m, d in sorted(sized):
        print(f"    {m:<25} {d:<12} {p:>6}B  Var_COLD={v:.6f}")
    print(f"\n  Spearman ρ = {rho:.3f}, p = {p_val:.4f}")
    if p_val < 0.05:
        print(f"  Result: SIGNIFICANT — {'larger=more variable' if rho > 0 else 'larger=less variable'}")
    else:
        print(f"  Result: Not significant")

    return {'rho': rho, 'p': p_val, 'n': len(sized)}


# ------------------------------------------------------------------
# Analysis 5: COLD embedding clustering (t-SNE)
# ------------------------------------------------------------------
def analysis_clustering(results):
    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("\n" + "=" * 70)
    print("ANALYSIS 5: COLD EMBEDDING CLUSTERING (t-SNE)")
    print("=" * 70)

    # Collect all embeddings with labels
    all_embs = []
    labels_model = []
    labels_domain = []
    labels_vendor = []

    for r in results:
        embs = r['all_embeddings']
        # Sample 100 per model-domain for visualization
        n_sample = min(100, len(embs))
        idx = np.random.choice(len(embs), n_sample, replace=False)
        sampled = embs[idx]
        all_embs.append(sampled)
        labels_model.extend([r['model']] * n_sample)
        labels_domain.extend([r['domain']] * n_sample)
        labels_vendor.extend([r['vendor']] * n_sample)

    all_embs = np.vstack(all_embs)
    print(f"  Total embeddings for t-SNE: {len(all_embs)}")

    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    coords = tsne.fit_transform(all_embs)

    # Plot by vendor
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Color by domain
    ax = axes[0]
    domain_colors = {'Medical': '#e74c3c', 'Philosophy': '#3498db'}
    for domain in ['Medical', 'Philosophy']:
        mask = [d == domain for d in labels_domain]
        ax.scatter(coords[mask, 0], coords[mask, 1], c=domain_colors[domain],
                   alpha=0.3, s=5, label=domain)
    ax.set_title('COLD Embeddings by Domain', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')

    # Color by vendor
    ax = axes[1]
    vendors = list(set(labels_vendor))
    vendor_cmap = plt.cm.Set2(np.linspace(0, 1, len(vendors)))
    vendor_colors = {v: vendor_cmap[i] for i, v in enumerate(sorted(vendors))}
    for vendor in sorted(vendors):
        mask = [v == vendor for v in labels_vendor]
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[vendor_colors[vendor]],
                   alpha=0.3, s=5, label=vendor)
    ax.set_title('COLD Embeddings by Vendor', fontsize=14)
    ax.legend(fontsize=9, ncol=2)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')

    plt.tight_layout()
    fig_path = FIGURE_DIR / "cold_tsne_clustering.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    # Also plot by model (separate figure)
    fig, ax = plt.subplots(figsize=(12, 8))
    models = sorted(set(labels_model))
    model_cmap = plt.cm.tab20(np.linspace(0, 1, len(models)))
    for i, model in enumerate(models):
        mask = [m == model for m in labels_model]
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[model_cmap[i]],
                   alpha=0.4, s=8, label=model)
    ax.set_title('COLD Embeddings by Model', fontsize=14)
    ax.legend(fontsize=7, ncol=3, loc='upper right')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.tight_layout()
    fig_path2 = FIGURE_DIR / "cold_tsne_by_model.png"
    plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path2}")

    return coords


# ------------------------------------------------------------------
# Analysis 6: Failure mode classification
# ------------------------------------------------------------------
def analysis_failure_modes(results):
    print("\n" + "=" * 70)
    print("ANALYSIS 6: FAILURE MODE CLASSIFICATION")
    print("=" * 70)
    print("\n  Classification based on Var_COLD level:")
    print("    HIGH Var_COLD (> median) + context doesn't help = WILD")
    print("    LOW Var_COLD (< median)  + context doesn't help = EMPTY")
    print("    LOW Var_COLD + context helps = IDEAL")
    print("    HIGH Var_COLD + context helps = RICH")

    # Median split
    all_vars = [r['var_cold'] for r in results]
    median_var = np.median(all_vars)
    print(f"\n  Median Var_COLD = {median_var:.6f}")

    print(f"\n{'Model':<25} {'Domain':<12} {'Var_COLD':>10} {'Level':>8} {'Class':>10}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x['var_cold'], reverse=True):
        level = "HIGH" if r['var_cold'] > median_var else "LOW"
        # Classify based on Var_COLD level
        if level == "HIGH":
            cls = "WILD/RICH"
        else:
            cls = "EMPTY/IDEAL"
        print(f"{r['model']:<25} {r['domain']:<12} {r['var_cold']:>10.6f} {level:>8} {cls:>10}")


# ------------------------------------------------------------------
# Analysis 7: Prior voice centroids
# ------------------------------------------------------------------
def analysis_prior_voice(results):
    print("\n" + "=" * 70)
    print("ANALYSIS 7: PRIOR VOICE CENTROIDS")
    print("=" * 70)
    print("\n  Cosine similarity between model centroids (default voice signatures)")

    # Compute centroid distances
    centroids = {}
    for r in results:
        key = f"{r['model']}_{r['domain'][:3]}"
        centroids[key] = np.array(r['centroid'])

    keys = sorted(centroids.keys())
    n = len(keys)

    # Cosine similarity matrix
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = np.dot(centroids[keys[i]], centroids[keys[j]])

    # Print most similar pairs
    print(f"\n  Most similar prior voices (top 10):")
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((keys[i], keys[j], sim_matrix[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)

    for k1, k2, sim in pairs[:10]:
        print(f"    {k1:<30} ↔ {k2:<30} cos={sim:.4f}")

    print(f"\n  Most different prior voices (bottom 5):")
    for k1, k2, sim in pairs[-5:]:
        print(f"    {k1:<30} ↔ {k2:<30} cos={sim:.4f}")

    # Cross-domain: same model, different domain
    print(f"\n  Same model, cross-domain centroid similarity:")
    dual_models = set()
    for r in results:
        dual_models.add(r['model'])
    for model in sorted(dual_models):
        med_key = f"{model}_Med"
        phil_key = f"{model}_Phi"
        if med_key in centroids and phil_key in centroids:
            sim = float(np.dot(centroids[med_key], centroids[phil_key]))
            print(f"    {model:<25} Medical ↔ Philosophy: cos={sim:.4f}")

    # Heatmap
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(sim_matrix, cmap='RdBu_r', vmin=0.5, vmax=1.0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(keys, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(keys, fontsize=8)
    plt.colorbar(im, label='Cosine Similarity')
    ax.set_title('Prior Voice Centroid Similarity Matrix', fontsize=14)
    plt.tight_layout()
    fig_path = FIGURE_DIR / "prior_voice_heatmap.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {fig_path}")

    return sim_matrix, keys


# ------------------------------------------------------------------
# Summary figures
# ------------------------------------------------------------------
def generate_summary_figures(results, domain_stats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Figure 1: Var_COLD ranking bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    sorted_r = sorted(results, key=lambda x: x['var_cold'], reverse=True)
    names = [f"{r['model']}\n({r['domain'][:3]})" for r in sorted_r]
    vals = [r['var_cold'] for r in sorted_r]
    colors = ['#e74c3c' if r['domain'] == 'Medical' else '#3498db' for r in sorted_r]
    bars = ax.bar(range(len(vals)), vals, color=colors, alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Var_COLD (mean embedding variance)', fontsize=12)
    ax.set_title('Model Prior Voice Variability: Var_COLD Ranking', fontsize=14)
    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#e74c3c', label='Medical'),
                       Patch(color='#3498db', label='Philosophy')],
              fontsize=10)
    plt.tight_layout()
    fig_path = FIGURE_DIR / "var_cold_ranking.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    # Figure 2: Var_COLD by domain boxplot
    fig, ax = plt.subplots(figsize=(8, 6))
    med_vars = [r['var_cold'] for r in results if r['domain'] == 'Medical']
    phil_vars = [r['var_cold'] for r in results if r['domain'] == 'Philosophy']
    bp = ax.boxplot([med_vars, phil_vars], labels=['Medical', 'Philosophy'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('#e74c3c')
    bp['boxes'][1].set_facecolor('#3498db')
    for box in bp['boxes']:
        box.set_alpha(0.6)
    ax.set_ylabel('Var_COLD', fontsize=12)
    ax.set_title('COLD Condition Variance by Domain', fontsize=14)
    p_val = domain_stats['p']
    ax.text(0.5, 0.95, f"Mann-Whitney p={p_val:.4f}", transform=ax.transAxes,
            ha='center', fontsize=11, style='italic')
    plt.tight_layout()
    fig_path = FIGURE_DIR / "var_cold_domain_boxplot.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    # Figure 3: Var_COLD by vendor
    fig, ax = plt.subplots(figsize=(10, 6))
    vendors = defaultdict(list)
    for r in results:
        vendors[r['vendor']].append(r['var_cold'])
    vendor_names = sorted(vendors.keys())
    vendor_vals = [vendors[v] for v in vendor_names]
    positions = range(len(vendor_names))
    bp = ax.boxplot(vendor_vals, positions=positions, labels=vendor_names,
                     patch_artist=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(vendor_names)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    # Overlay individual points
    for i, (vn, vv) in enumerate(zip(vendor_names, vendor_vals)):
        ax.scatter([i] * len(vv), vv, c='black', alpha=0.5, s=30, zorder=5)
    ax.set_ylabel('Var_COLD', fontsize=12)
    ax.set_title('COLD Condition Variance by Vendor', fontsize=14)
    plt.tight_layout()
    fig_path = FIGURE_DIR / "var_cold_by_vendor.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    np.random.seed(42)

    print("=" * 70)
    print("COLD CONDITION ANALYSIS: THE PRIOR VOICE")
    print("Revealing intrinsic model personality from no-context responses")
    print("=" * 70)
    print()

    # Discover runs
    print("Discovering runs with COLD response text ...", flush=True)
    runs = get_runs_with_text()
    print(f"Found {len(runs)} runs:")
    for r in runs:
        print(f"  {r['domain']:>10} | {r['model']:<30} | {r['n_trials']} trials")
    print()

    # Load embedder (use same MiniLM as Papers 1-5 for consistency)
    from sentence_transformers import SentenceTransformer
    print("Loading all-MiniLM-L6-v2 (384D) ...", flush=True)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    dim = embedder.get_sentence_embedding_dimension()
    print(f"Model ready: {dim}D\n", flush=True)

    # Embed all COLD responses
    results = embed_cold_responses(runs, embedder)

    # Run all analyses
    ranking = analysis_var_cold_ranking(results)
    domain_stats = analysis_domain_comparison(results)
    families = analysis_family_clustering(results)
    size_corr = analysis_size_correlation(results)
    coords = analysis_clustering(results)
    analysis_failure_modes(results)
    sim_matrix, keys = analysis_prior_voice(results)

    # Generate summary figures
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)
    generate_summary_figures(results, domain_stats)

    # Save results (exclude large numpy arrays)
    save_results = []
    for r in results:
        save_r = {k: v for k, v in r.items() if k != 'all_embeddings'}
        save_results.append(save_r)

    output = {
        'analysis': 'COLD Prior Voice Analysis',
        'embedding_model': 'all-MiniLM-L6-v2',
        'embedding_dim': dim,
        'n_runs': len(results),
        'runs': save_results,
        'domain_comparison': {
            'medical_mean_var_cold': float(np.mean(domain_stats['med_vars'])),
            'philosophy_mean_var_cold': float(np.mean(domain_stats['phil_vars'])),
            'mann_whitney_U': float(domain_stats['U']),
            'mann_whitney_p': float(domain_stats['p']),
            'rank_biserial_r': float(domain_stats['r_rb']),
        },
        'size_correlation': size_corr,
        'centroid_similarity_keys': keys,
        'centroid_similarity_matrix': sim_matrix.tolist(),
    }

    out_path = OUTPUT_DIR / "cold_prior_voice_results.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"  Results: {out_path}")
    print(f"  Figures: {FIGURE_DIR}")


if __name__ == '__main__':
    main()
