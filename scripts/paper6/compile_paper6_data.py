#!/usr/bin/env python3
"""
Compile all Paper 6 manuscript data into a single structured JSON file.
Extracts from: conservation_law_results.json, conservation_product_test.csv,
robustness files (LaBSE, MPNet), legal/ethics trial data, and README DOIs.

For Legal and Ethics domains, re-embeds response text with all-MiniLM-L6-v2
to compute embedding-based Var_Ratio (same method as conservation_law_verification).
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import csv
import os
import re
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats as scipy_stats

# Import shared path helper (no hardcoded absolute paths)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.paths import repo_root  # noqa: E402

BASE = repo_root()

# ============================================================================
# 0. Load embedding model (needed for Legal + Ethics)
# ============================================================================
print("=== Loading embedding model ===")
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print(f"  Loaded all-MiniLM-L6-v2 (dim={embedder.get_sentence_embedding_dimension()})")

# ============================================================================
# 1. Load pre-computed MiniLM 384D conservation data (Medical + Philosophy, N=14)
# ============================================================================
print("\n=== Loading MiniLM 384D conservation data (Med+Phil) ===")

with open(BASE / "data/paper6/conservation_law_verification/conservation_law_results.json") as f:
    conservation_results = json.load(f)

# Build the baseline MiniLM runs from conservation_law_results.json
minilm_runs = []
for r in conservation_results:
    K = r['drci'] * r['var_ratio']
    minilm_runs.append({
        'model': r['model'],
        'domain': r['domain'],
        'embedding': 'all-MiniLM-L6-v2',
        'dim': 384,
        'drci': round(r['drci'], 6),
        'var_ratio': round(r['var_ratio'], 6),
        'K': round(K, 6),
        'mi': round(r.get('mi', 0), 6),
        'n_trials': r.get('n_trials', 50),
        'source': 'conservation_law_verification'
    })

for r in minilm_runs:
    print(f"  {r['model']:25s} {r['domain']:12s} dRCI={r['drci']:.4f} VR={r['var_ratio']:.4f} K={r['K']:.4f}")

# ============================================================================
# 2. Compute metrics from trial files using embedding-based Var_Ratio
# ============================================================================

def compute_metrics_from_trials(filepath, embedder):
    """
    Compute dRCI, Var_Ratio, entanglement, exploration arc, content fraction
    from trial data using embedding-based variance computation.

    Var_Ratio = mean across positions of [ mean(Var(TRUE_embs_at_p, axis=0)) / mean(Var(COLD_embs_at_p, axis=0)) ]
    This is the same method used in paper6_conservation_law.py.
    """
    with open(filepath, encoding='utf-8') as f:
        data = json.load(f)

    trials = data.get('trials', [])
    model = data.get('model', Path(filepath).stem)
    domain = data.get('domain', 'unknown')
    n_trials = len(trials)
    n_prompts = data.get('n_prompts', 30)

    # Collect per-trial dRCI (cold)
    trial_drcis = []
    for t in trials:
        dr = t.get('delta_rci', {})
        if isinstance(dr, dict) and 'cold' in dr:
            trial_drcis.append(dr['cold'])
    mean_drci = float(np.mean(trial_drcis)) if trial_drcis else 0.0

    # Check for response text
    has_responses = ('responses' in trials[0]) if trials else False
    if not has_responses:
        return {
            'model': model, 'domain': domain, 'embedding': 'all-MiniLM-L6-v2',
            'dim': 384, 'drci': round(mean_drci, 6), 'var_ratio': None, 'K': None,
            'n_trials': n_trials, 'has_responses': False,
            'source': str(Path(filepath).name), 'note': 'No response text, metrics only'
        }

    # Collect all TRUE, COLD, SCRAMBLED response texts
    true_texts = []
    cold_texts = []
    scram_texts = []
    for t in trials:
        resp = t['responses']
        true_texts.extend(resp['true'])
        cold_texts.extend(resp['cold'])
        if 'scrambled' in resp:
            scram_texts.extend(resp['scrambled'])

    # Embed all responses
    print(f"    Embedding {len(true_texts)} TRUE + {len(cold_texts)} COLD responses...", flush=True)
    true_embs = embedder.encode(true_texts, show_progress_bar=False, convert_to_numpy=True)
    cold_embs = embedder.encode(cold_texts, show_progress_bar=False, convert_to_numpy=True)
    scram_embs = None
    if scram_texts:
        scram_embs = embedder.encode(scram_texts, show_progress_bar=False, convert_to_numpy=True)

    # Reshape: (n_trials, n_prompts, dim)
    dim = true_embs.shape[1]
    true_embs_3d = true_embs.reshape(n_trials, n_prompts, dim)
    cold_embs_3d = cold_embs.reshape(n_trials, n_prompts, dim)
    scram_embs_3d = None
    if scram_embs is not None:
        scram_embs_3d = scram_embs.reshape(n_trials, n_prompts, dim)

    # --- Var_Ratio per position ---
    # Var_Ratio_p = mean(Var(TRUE_embs[:, p, :], axis=0)) / mean(Var(COLD_embs[:, p, :], axis=0))
    var_ratios = []
    for p in range(n_prompts):
        true_at_p = true_embs_3d[:, p, :]  # (n_trials, dim)
        cold_at_p = cold_embs_3d[:, p, :]
        var_true = np.mean(np.var(true_at_p, axis=0))
        var_cold = np.mean(np.var(cold_at_p, axis=0))
        if var_cold > 1e-10:
            var_ratios.append(var_true / var_cold)
        else:
            var_ratios.append(1.0)
    mean_var_ratio = float(np.mean(var_ratios))

    # --- Entanglement: position-level Pearson(dRCI_p, VRI_p) ---
    # dRCI_p = mean_true_cosine_p - mean_cold_cosine_p (from alignments in file)
    # VRI_p = 1 - Var_Ratio_p
    pos_drci = []
    pos_vri = []

    # Use alignments for per-position dRCI
    true_align_by_pos = defaultdict(list)
    cold_align_by_pos = defaultdict(list)
    for t in trials:
        aligns = t.get('alignments', {})
        for p in range(min(len(aligns.get('true', [])), n_prompts)):
            true_align_by_pos[p].append(aligns['true'][p])
        for p in range(min(len(aligns.get('cold', [])), n_prompts)):
            cold_align_by_pos[p].append(aligns['cold'][p])

    for p in range(n_prompts):
        if p in true_align_by_pos and p in cold_align_by_pos:
            mt = np.mean(true_align_by_pos[p])
            mc = np.mean(cold_align_by_pos[p])
            pos_drci.append(mt - mc)
            pos_vri.append(1.0 - var_ratios[p])

    ent_r = 0.0
    ent_p_val = 1.0
    if len(pos_drci) > 2:
        try:
            r_val, p_val = scipy_stats.pearsonr(pos_drci, pos_vri)
            if not np.isnan(r_val):
                ent_r = float(r_val)
                ent_p_val = float(p_val)
        except Exception:
            pass

    # --- Exploration Arc: Var(TRUE_embs at P_last) / Var(TRUE_embs at P1) ---
    # Using mean variance across embedding dimensions
    var_first = np.mean(np.var(true_embs_3d[:, 0, :], axis=0))
    var_last = np.mean(np.var(true_embs_3d[:, n_prompts-1, :], axis=0))
    arc = float(var_last / var_first) if var_first > 1e-10 else 0.0

    # --- Content Fraction at last position ---
    content_fraction = None
    if scram_embs_3d is not None:
        # Content Fraction = mean(SCRAM-COLD) / mean(TRUE-COLD) * 100%
        # Using cosine-based alignment means from file
        scram_align_by_pos = defaultdict(list)
        for t in trials:
            aligns = t.get('alignments', {})
            scram_a = aligns.get('scrambled', [])
            for p in range(min(len(scram_a), n_prompts)):
                scram_align_by_pos[p].append(scram_a[p])

        last_p = n_prompts - 1
        if last_p in true_align_by_pos and last_p in cold_align_by_pos and last_p in scram_align_by_pos:
            mean_true_last = np.mean(true_align_by_pos[last_p])
            mean_cold_last = np.mean(cold_align_by_pos[last_p])
            mean_scram_last = np.mean(scram_align_by_pos[last_p])
            denom = mean_true_last - mean_cold_last
            if abs(denom) > 1e-6:
                content_fraction = float(((mean_scram_last - mean_cold_last) / denom) * 100.0)

    result = {
        'model': model,
        'domain': domain,
        'embedding': 'all-MiniLM-L6-v2',
        'dim': 384,
        'drci': round(mean_drci, 6),
        'var_ratio': round(mean_var_ratio, 6),
        'K': round(mean_drci * mean_var_ratio, 6),
        'entanglement_r': round(ent_r, 4),
        'entanglement_p': round(ent_p_val, 6),
        'exploration_arc': round(arc, 4),
        'n_trials': n_trials,
        'has_responses': True,
        'source': str(Path(filepath).name)
    }
    if content_fraction is not None:
        result['content_fraction'] = round(content_fraction, 2)

    return result


# ============================================================================
# 3. Compute Legal + Ethics data from trial files with embedding-based VR
# ============================================================================
print("\n=== Computing Legal domain data (embedding-based VR) ===")

legal_dir = BASE / "data/legal/open_models"
legal_runs = []
legal_excluded = []

for f in sorted(legal_dir.glob("*.json")):
    if 'checkpoint' in f.name:
        continue
    try:
        result = compute_metrics_from_trials(f, embedder)
        if 'legal' in result['domain'].lower():
            result['domain'] = 'Legal'

        model = result['model']
        if model == 'kimi_k2_5':
            legal_excluded.append({**result, 'exclusion_reason': 'COLD refusal + 21% empty responses'})
        elif model == 'glm_5':
            legal_excluded.append({**result, 'exclusion_reason': '86% empty responses'})
        else:
            legal_runs.append(result)
        print(f"  {result['model']:25s} {result['domain']:8s} dRCI={result['drci']:.4f} VR={result.get('var_ratio', 'N/A')} K={result.get('K', 'N/A')}")
    except Exception as e:
        print(f"  Error processing {f.name}: {e}")

print("\n=== Computing Ethics domain data (embedding-based VR) ===")

ethics_dir = BASE / "data/ethics/open_models"
ethics_runs = []

for f in sorted(ethics_dir.glob("*.json")):
    if 'checkpoint' in f.name:
        continue
    try:
        result = compute_metrics_from_trials(f, embedder)
        if 'ethics' in result['domain'].lower():
            result['domain'] = 'Ethics'
        ethics_runs.append(result)
        print(f"  {result['model']:25s} {result['domain']:8s} dRCI={result['drci']:.4f} VR={result['var_ratio']:.4f} K={result['K']:.4f}")
    except Exception as e:
        print(f"  Error processing {f.name}: {e}")

# ============================================================================
# 4. Load robustness data
# ============================================================================
print("\n=== Loading robustness data ===")

with open(BASE / "data/paper6/robustness/paper6_robustness_LaBSE.json") as f:
    labse_data = json.load(f)

with open(BASE / "data/paper6/robustness/paper6_768d_summary.json") as f:
    mpnet_data = json.load(f)

with open(BASE / "data/paper6/embedding_robustness/robustness_results.json") as f:
    mpnet_full = json.load(f)

print(f"  LaBSE: {len(labse_data['results'])} runs, {len(labse_data['summary'])} domains")
print(f"  MPNet summary: {len(mpnet_data['results'])} runs, {len(mpnet_data['summary'])} domains")
print(f"  MPNet full: {len(mpnet_full['runs'])} runs")

# ============================================================================
# 5. Compute domain-level statistics
# ============================================================================
print("\n=== Domain-level statistics ===")

all_runs = minilm_runs + legal_runs + ethics_runs
domain_groups = defaultdict(list)
for r in all_runs:
    if r.get('K') is not None:
        domain_groups[r['domain']].append(r)

domain_stats = {}
for domain in ['Medical', 'Philosophy', 'Legal', 'Ethics']:
    runs = domain_groups.get(domain, [])
    if not runs:
        continue
    ks = [r['K'] for r in runs]
    drcis = [r['drci'] for r in runs]
    vrs = [r['var_ratio'] for r in runs if r.get('var_ratio') is not None]

    k_mean = np.mean(ks)
    k_std = np.std(ks, ddof=1) if len(ks) > 1 else 0
    k_cv = k_std / k_mean if k_mean > 0 else 0

    domain_stats[domain] = {
        'N': len(runs),
        'K_mean': round(float(k_mean), 4),
        'K_std': round(float(k_std), 4),
        'K_CV': round(float(k_cv), 4),
        'K_min': round(float(min(ks)), 4),
        'K_max': round(float(max(ks)), 4),
        'dRCI_mean': round(float(np.mean(drcis)), 4),
        'VR_mean': round(float(np.mean(vrs)), 4) if vrs else None,
        'models': [r['model'] for r in runs]
    }
    print(f"  {domain:12s}: N={len(runs)}, K={k_mean:.4f} +/- {k_std:.4f} (CV={k_cv:.3f})")

# ============================================================================
# 6. Pairwise Mann-Whitney U tests between domains
# ============================================================================
print("\n=== Pairwise Mann-Whitney U tests ===")

domain_pairs = []
domains = [d for d in ['Medical', 'Philosophy', 'Legal', 'Ethics'] if d in domain_groups]

for i in range(len(domains)):
    for j in range(i+1, len(domains)):
        d1, d2 = domains[i], domains[j]
        k1 = [r['K'] for r in domain_groups[d1]]
        k2 = [r['K'] for r in domain_groups[d2]]

        U, p = scipy_stats.mannwhitneyu(k1, k2, alternative='two-sided')

        # Cohen's d
        s1 = np.std(k1, ddof=1) if len(k1) > 1 else 0.001
        s2 = np.std(k2, ddof=1) if len(k2) > 1 else 0.001
        pooled_std = np.sqrt((s1**2 + s2**2) / 2)
        cohens_d = (np.mean(k1) - np.mean(k2)) / pooled_std if pooled_std > 0 else 0

        pair = {
            'domain_1': d1,
            'domain_2': d2,
            'U': float(U),
            'p': round(float(p), 6),
            'cohens_d': round(float(cohens_d), 3),
            'n1': len(k1),
            'n2': len(k2)
        }
        domain_pairs.append(pair)
        print(f"  {d1} vs {d2}: U={U:.0f}, p={p:.4f}, Cohen's d={cohens_d:.3f}")

# ============================================================================
# 7. Three-embedding comparison
# ============================================================================
print("\n=== Three-embedding comparison ===")

embedding_comparison = {
    'MiniLM_384D': {},
    'MPNet_768D': {},
    'LaBSE_768D': {}
}

for domain in domain_stats:
    embedding_comparison['MiniLM_384D'][domain] = {
        'K_mean': domain_stats[domain]['K_mean'],
        'K_CV': domain_stats[domain]['K_CV'],
        'N': domain_stats[domain]['N']
    }

for domain, vals in mpnet_data.get('summary', {}).items():
    embedding_comparison['MPNet_768D'][domain] = {
        'K_mean': round(vals['K_mean'], 4),
        'K_CV': round(vals['CV'], 4),
        'N': vals['N']
    }

for domain, vals in labse_data.get('summary', {}).items():
    embedding_comparison['LaBSE_768D'][domain] = {
        'K_mean': round(vals['K_mean'], 4),
        'K_CV': round(vals['CV'], 4),
        'N': vals['N']
    }

# Also capture the MPNet full robustness (Med+Phil only, with CI)
mpnet_full_stats = {}
if 'domain_stats' in mpnet_full:
    for domain, vals in mpnet_full['domain_stats'].items():
        if 'mpnet' in vals:
            mpnet_full_stats[domain] = {
                'K_mean': round(vals['mpnet']['mean'], 4),
                'K_CV': round(vals['mpnet']['cv'], 4),
                'N': vals['mpnet']['N'],
                'K_ci_lo': round(vals['mpnet']['ci_lo'], 4),
                'K_ci_hi': round(vals['mpnet']['ci_hi'], 4),
            }

for emb, domains_data in embedding_comparison.items():
    print(f"  {emb}:")
    for domain, vals in domains_data.items():
        print(f"    {domain}: K={vals['K_mean']:.4f} (CV={vals['K_CV']:.3f}, N={vals['N']})")

# ============================================================================
# 8. Extract DOIs from README.md
# ============================================================================
print("\n=== Extracting DOIs ===")

dois = {}
with open(BASE / "README.md", encoding='utf-8') as f:
    readme = f.read()

doi_pattern = r'\*\*(\d+)\*\*.*?DOI:\s*(10\.\d{5}/[^\s|)+]+)'
for m in re.finditer(doi_pattern, readme):
    paper_num = int(m.group(1))
    doi = m.group(2).rstrip('|').strip()
    dois[f"Paper_{paper_num}"] = doi

zenodo = re.search(r'Zenodo DOI:\s*(10\.\d{5}/[^\s|)]+)', readme)
if zenodo:
    dois['Paper_9_Zenodo'] = zenodo.group(1)

for k, v in sorted(dois.items()):
    print(f"  {k}: {v}")

# ============================================================================
# 9. Compile everything
# ============================================================================
print("\n=== Compiling output ===")

output = {
    "metadata": {
        "description": "Paper 6: Conservation Constraint - Complete Manuscript Data",
        "repository": "LaxmanNandi/MCH-Research",
        "author": "Dr. Laxman M M, MBBS",
        "generated_by": "compile_paper6_data.py",
        "baseline_embedding": "all-MiniLM-L6-v2 (384D)",
        "robustness_embeddings": ["all-mpnet-base-v2 (768D)", "LaBSE (768D)"],
        "n_domains": len(domain_stats),
        "total_runs": len(all_runs),
        "note": "Medical+Philosophy K values from conservation_law_verification (pre-computed embedding-based). Legal+Ethics K values computed fresh with embedding-based Var_Ratio."
    },
    "model_runs": {
        "Medical": [r for r in all_runs if r['domain'] == 'Medical'],
        "Philosophy": [r for r in all_runs if r['domain'] == 'Philosophy'],
        "Legal": [r for r in legal_runs if r.get('K') is not None],
        "Ethics": ethics_runs,
    },
    "excluded_runs": {
        "Legal": legal_excluded,
    },
    "domain_statistics": domain_stats,
    "pairwise_tests": domain_pairs,
    "embedding_comparison": {
        "MiniLM_384D": {
            "model": "all-MiniLM-L6-v2",
            "dim": 384,
            "per_domain": embedding_comparison['MiniLM_384D']
        },
        "MPNet_768D": {
            "model": "all-mpnet-base-v2",
            "dim": 768,
            "per_domain": embedding_comparison.get('MPNet_768D', {}),
            "per_model": mpnet_data.get('results', [])
        },
        "MPNet_768D_full_robustness": {
            "model": "all-mpnet-base-v2",
            "dim": 768,
            "note": "Med+Phil only, with CI, from embedding_robustness/robustness_results.json",
            "per_domain": mpnet_full_stats,
            "per_model": mpnet_full.get('runs', [])
        },
        "LaBSE_768D": {
            "model": "sentence-transformers/LaBSE",
            "dim": 768,
            "per_domain": embedding_comparison.get('LaBSE_768D', {}),
            "per_model": labse_data.get('results', [])
        }
    },
    "programme_dois": dois,
}

# Save
outpath = BASE / "data/paper6/paper6_manuscript_data.json"
with open(outpath, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\nSaved to: {outpath}")

# ============================================================================
# 10. Summary table
# ============================================================================
print("\n" + "="*100)
print("SUMMARY TABLE: All model-domain runs with K values")
print("="*100)
print(f"{'Model':<28s} {'Domain':<12s} {'dRCI':>8s} {'VR':>8s} {'K':>8s}  Source")
print("-"*100)

for domain in ['Medical', 'Philosophy', 'Legal', 'Ethics']:
    runs = [r for r in all_runs if r['domain'] == domain and r.get('K') is not None]
    for r in sorted(runs, key=lambda x: -x['K']):
        src = r.get('source', '')[:35]
        vr_str = f"{r['var_ratio']:8.4f}" if r.get('var_ratio') is not None else "     N/A"
        k_str = f"{r['K']:8.4f}" if r.get('K') is not None else "     N/A"
        print(f"  {r['model']:<26s} {r['domain']:<12s} {r['drci']:8.4f} {vr_str} {k_str}  {src}")
    if runs:
        ks = [r['K'] for r in runs]
        k_mean = np.mean(ks)
        k_cv = np.std(ks, ddof=1) / k_mean if k_mean > 0 and len(ks) > 1 else 0
        print(f"  {'--- DOMAIN MEAN ---':<26s} {domain:<12s} {'':>8s} {'':>8s} {k_mean:8.4f}  (N={len(runs)}, CV={k_cv:.3f})")
    print()

print("="*100)
print("DOMAIN SUMMARY (MiniLM 384D baseline)")
print("="*100)
for domain in ['Medical', 'Philosophy', 'Legal', 'Ethics']:
    if domain in domain_stats:
        ds = domain_stats[domain]
        print(f"  {domain:<12s}: K = {ds['K_mean']:.4f} +/- {ds['K_std']:.4f} (CV={ds['K_CV']:.3f}, N={ds['N']})")

print("\n" + "="*100)
print("PAIRWISE COMPARISONS")
print("="*100)
for p in domain_pairs:
    sig = "***" if p['p'] < 0.001 else ("**" if p['p'] < 0.01 else ("*" if p['p'] < 0.05 else "ns"))
    print(f"  {p['domain_1']} vs {p['domain_2']}: U={p['U']:.0f}, p={p['p']:.4f} {sig}, Cohen's d={p['cohens_d']:.3f}")

print("\n" + "="*100)
print("THREE-EMBEDDING COMPARISON")
print("="*100)
print(f"  {'Domain':<12s} {'MiniLM K':>10s} {'MPNet K':>10s} {'LaBSE K':>10s}")
for domain in ['Medical', 'Philosophy', 'Legal', 'Ethics']:
    ml = embedding_comparison['MiniLM_384D'].get(domain, {}).get('K_mean', '-')
    mp = embedding_comparison['MPNet_768D'].get(domain, {}).get('K_mean', '-')
    lb = embedding_comparison['LaBSE_768D'].get(domain, {}).get('K_mean', '-')
    ml_s = f"{ml:.4f}" if isinstance(ml, (float, int)) else str(ml)
    mp_s = f"{mp:.4f}" if isinstance(mp, (float, int)) else str(mp)
    lb_s = f"{lb:.4f}" if isinstance(lb, (float, int)) else str(lb)
    print(f"  {domain:<12s} {ml_s:>10s} {mp_s:>10s} {lb_s:>10s}")

print("\nDone.")
