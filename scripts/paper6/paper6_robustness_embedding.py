#!/usr/bin/env python3
"""
Paper 6: Embedding Robustness Check
=====================================
Recomputes ΔRCI and Var_Ratio for Medical and Philosophy domains
using all-mpnet-base-v2 (768D) instead of the original all-MiniLM-L6-v2 (384D).

Purpose: Preempt reviewer objection that the conservation constraint
is an artifact of the specific embedding space geometry.

Expected result: K(Medical) ≈ 0.429, K(Philosophy) ≈ 0.301 under new embedding.
If K values are similar with CV < 0.20, the constraint is embedding-robust.

Output: data/paper6/embedding_robustness/robustness_results.json + report
"""

import sys
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding='utf-8')

BASE = Path("C:/Users/barla/mch_experiments")
OUTPUT_DIR = BASE / "data" / "paper6" / "embedding_robustness"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Known K values from original MiniLM-L6-v2 (Papers 1-5)
# ------------------------------------------------------------------
ORIGINAL_K = {
    'Medical':    {'K': 0.429, 'CV': 0.170, 'N': 8},
    'Philosophy': {'K': 0.301, 'CV': 0.166, 'N': 6},
}

# ------------------------------------------------------------------
# Data file discovery
# ------------------------------------------------------------------

def get_runs_with_text():
    dirs = [
        (BASE / "data" / "medical" / "closed_models",  "Medical"),
        (BASE / "data" / "medical" / "open_models",    "Medical"),
        (BASE / "data" / "medical" / "gemini_flash",   "Medical"),
        (BASE / "data" / "philosophy" / "closed_models", "Philosophy"),
        (BASE / "data" / "philosophy" / "open_models",   "Philosophy"),
    ]

    runs = []
    seen = set()

    for d, domain in dirs:
        if not d.exists():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix != '.json':
                continue
            # Skip backups, checkpoints, metrics-only files
            if any(x in f.name for x in ('checkpoint', 'metrics_only', 'BACKUP', 'rerun')):
                continue

            try:
                with open(f, encoding='utf-8') as fh:
                    data = json.load(fh)

                trials = data.get('trials', [])
                if not trials:
                    continue

                t = trials[0]
                r = t.get('responses', {})
                if not isinstance(r, dict) or 'true' not in r:
                    continue
                if not (isinstance(r['true'], list) and
                        len(r['true']) > 0 and
                        isinstance(r['true'][0], str) and
                        len(r['true'][0]) > 10):
                    continue

                model = data.get('model', f.stem)
                key = f"{model}_{domain}"
                if key in seen:
                    continue
                seen.add(key)

                runs.append({
                    'model':    model,
                    'domain':   domain,
                    'path':     str(f),
                    'n_trials': len(trials),
                    'n_prompts': len(r['true']),
                })

            except Exception as e:
                print(f"  Skipping {f.name}: {e}")

    return runs


# ------------------------------------------------------------------
# Core computation (embedding-agnostic)
# ------------------------------------------------------------------

def compute_metrics_from_text(run, embedder):
    """
    Load response text from JSON, embed with given embedder,
    compute ΔRCI and Var_Ratio from scratch (ignores stored alignments).

    ΔRCI  = 1 - mean(cosine_sim(true_emb[i], cold_emb[i]))
            averaged over all 50 trials × 30 positions

    Var_Ratio = mean over positions of:
                  mean_dim_var(TRUE at position p) / mean_dim_var(COLD at position p)
    """
    with open(run['path'], encoding='utf-8') as f:
        data = json.load(f)

    trials    = data['trials']
    n_trials  = len(trials)
    n_prompts = len(trials[0]['responses']['true'])

    # Collect all response texts
    true_texts = []
    cold_texts = []
    for t in trials:
        true_texts.extend(t['responses']['true'])
        cold_texts.extend(t['responses']['cold'])

    total = len(true_texts)
    print(f"    Embedding {total} TRUE + {total} COLD texts ...", flush=True)

    true_embs = embedder.encode(true_texts, show_progress_bar=False,
                                 convert_to_numpy=True, batch_size=64)
    cold_embs = embedder.encode(cold_texts, show_progress_bar=False,
                                 convert_to_numpy=True, batch_size=64)

    # --- ΔRCI ---
    # Cosine similarity between paired true/cold embeddings at each position
    norms_t = np.linalg.norm(true_embs, axis=1, keepdims=True)
    norms_c = np.linalg.norm(cold_embs, axis=1, keepdims=True)
    cos_sims = np.sum(
        (true_embs / norms_t) * (cold_embs / norms_c),
        axis=1
    )
    drci = float(1.0 - np.mean(cos_sims))

    # --- Var_Ratio ---
    # Shape: (n_trials, n_prompts, embedding_dim)
    true_by_pos = true_embs.reshape(n_trials, n_prompts, -1)
    cold_by_pos = cold_embs.reshape(n_trials, n_prompts, -1)

    vr_per_pos = []
    for p in range(n_prompts):
        var_true = float(np.mean(np.var(true_by_pos[:, p, :], axis=0)))
        var_cold = float(np.mean(np.var(cold_by_pos[:, p, :], axis=0)))
        if var_cold > 1e-10:
            vr_per_pos.append(var_true / var_cold)
        else:
            vr_per_pos.append(1.0)

    var_ratio = float(np.mean(vr_per_pos))
    product   = drci * var_ratio

    return {
        'model':     run['model'],
        'domain':    run['domain'],
        'n_trials':  n_trials,
        'n_prompts': n_prompts,
        'drci':      drci,
        'var_ratio': var_ratio,
        'product':   product,
    }


# ------------------------------------------------------------------
# Summary statistics
# ------------------------------------------------------------------

def domain_stats(results, domain):
    prods = [r['product'] for r in results if r['domain'] == domain]
    if len(prods) < 2:
        return None
    arr = np.array(prods)
    mean = float(np.mean(arr))
    std  = float(np.std(arr, ddof=1))
    cv   = std / mean
    n    = len(arr)
    # 95% CI (t-distribution)
    from scipy import stats as sc
    t_crit = sc.t.ppf(0.975, df=n-1)
    ci_lo  = mean - t_crit * std / np.sqrt(n)
    ci_hi  = mean + t_crit * std / np.sqrt(n)
    return {
        'N': n, 'mean': mean, 'std': std, 'cv': cv,
        'ci_lo': ci_lo, 'ci_hi': ci_hi,
        'min': float(np.min(arr)), 'max': float(np.max(arr)),
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    print("=" * 70)
    print("PAPER 6: EMBEDDING ROBUSTNESS CHECK")
    print("Embedding: all-mpnet-base-v2 (768D)  vs  all-MiniLM-L6-v2 (384D original)")
    print("=" * 70)
    print()

    # Discover runs
    print("Discovering runs with response text ...", flush=True)
    runs = get_runs_with_text()
    print(f"Found {len(runs)} runs:")
    for r in runs:
        print(f"  {r['domain']:>10} | {r['model']:<30} | {r['n_trials']} trials")
    print()

    # Load new embedding model
    print("Loading all-mpnet-base-v2 ...", flush=True)
    embedder = SentenceTransformer('all-mpnet-base-v2')
    dim = embedder.get_sentence_embedding_dimension()
    print(f"Model ready: {dim}D\n", flush=True)

    # Process each run
    results = []
    for i, run in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {run['model']} ({run['domain']})", flush=True)
        res = compute_metrics_from_text(run, embedder)
        print(f"    ΔRCI={res['drci']:.4f}  Var_Ratio={res['var_ratio']:.4f}  "
              f"Product={res['product']:.4f}")
        results.append(res)
        print()

    # Print comparison table
    print("=" * 70)
    print("RESULTS TABLE")
    print("=" * 70)
    print(f"{'Model':<30} {'Domain':<12} {'ΔRCI':>8} {'VR':>8} {'Product':>9}")
    print("-" * 70)
    results.sort(key=lambda x: (x['domain'], -x['product']))
    for r in results:
        print(f"{r['model']:<30} {r['domain']:<12} "
              f"{r['drci']:>8.4f} {r['var_ratio']:>8.4f} {r['product']:>9.4f}")

    # Domain summaries + comparison
    print()
    print("=" * 70)
    print("DOMAIN COMPARISON: all-mpnet-base-v2  vs  all-MiniLM-L6-v2 (original)")
    print("=" * 70)

    comparison = {}
    for domain in ['Medical', 'Philosophy']:
        s = domain_stats(results, domain)
        if s is None:
            print(f"{domain}: insufficient data")
            continue

        orig = ORIGINAL_K[domain]
        delta_k = s['mean'] - orig['K']
        verdict = "CONSISTENT" if s['cv'] < 0.20 else ("WEAK" if s['cv'] < 0.30 else "FAILS")

        print(f"\n  {domain} (N={s['N']})")
        print(f"    new  K = {s['mean']:.4f}  (CV={s['cv']:.3f}, "
              f"95% CI [{s['ci_lo']:.3f}, {s['ci_hi']:.3f}])")
        print(f"    orig K = {orig['K']:.4f}  (CV={orig['CV']:.3f})  -- MiniLM-L6-v2")
        print(f"    ΔK     = {delta_k:+.4f}  ({delta_k/orig['K']*100:+.1f}%)")
        print(f"    Verdict: {verdict}")

        comparison[domain] = {
            'mpnet': s,
            'minilm': orig,
            'delta_k': delta_k,
            'delta_k_pct': delta_k / orig['K'] * 100,
            'verdict': verdict,
        }

    # Overall verdict
    all_consistent = all(
        v['verdict'] == 'CONSISTENT' for v in comparison.values()
    )
    print()
    print("=" * 70)
    if all_consistent:
        print("OVERALL: CONSTRAINT IS EMBEDDING-ROBUST")
        print("K values are consistent across embedding architectures.")
        print("The shared embedding space objection is REFUTED.")
    else:
        print("OVERALL: EMBEDDING DEPENDENCE DETECTED")
        print("K values shift significantly with embedding model.")
        print("Report as a limitation and qualify the conservation claim.")
    print("=" * 70)

    # Save JSON results
    output = {
        'embedding_model': 'all-mpnet-base-v2',
        'embedding_dim': dim,
        'original_embedding': 'all-MiniLM-L6-v2',
        'original_embedding_dim': 384,
        'runs': results,
        'domain_stats': {
            d: {
                'mpnet': comparison[d]['mpnet'],
                'minilm': comparison[d]['minilm'],
                'delta_k': comparison[d]['delta_k'],
                'delta_k_pct': comparison[d]['delta_k_pct'],
                'verdict': comparison[d]['verdict'],
            }
            for d in comparison
        },
        'overall_verdict': 'EMBEDDING_ROBUST' if all_consistent else 'EMBEDDING_DEPENDENT',
    }

    out_path = OUTPUT_DIR / "robustness_results.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
