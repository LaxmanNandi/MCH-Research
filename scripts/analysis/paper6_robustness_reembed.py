#!/usr/bin/env python3
"""
Paper 6 Robustness Check: Re-embed all domains with higher-dimensional models.
=============================================================================
Re-computes K, Var_Ratio, entanglement, and arc for all model-domain runs
using all-mpnet-base-v2 (768D) to validate that conservation law holds
independent of embedding model choice.

Current baseline: all-MiniLM-L6-v2 (384D)
Robustness check: all-mpnet-base-v2 (768D)
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from scipy.stats import pearsonr

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

EMBEDDING_MODELS = [
    ("all-mpnet-base-v2", 768),
]

BASE_DATA = "C:/Users/barla/mch_experiments/data"
OUTPUT_DIR = "C:/Users/barla/mch_experiments/data/paper6/robustness"

DOMAINS = {
    "medical_closed": {
        "path": "medical/closed_models",
        "domain": "Medical",
        "truth_type": "Discovered",
    },
    "medical_open": {
        "path": "medical/open_models",
        "domain": "Medical",
        "truth_type": "Discovered",
    },
    "philosophy_closed": {
        "path": "philosophy/closed_models",
        "domain": "Philosophy",
        "truth_type": "Explored",
    },
    "philosophy_open": {
        "path": "philosophy/open_models",
        "domain": "Philosophy",
        "truth_type": "Explored",
    },
    "legal": {
        "path": "legal/open_models",
        "domain": "Legal",
        "truth_type": "Argued",
    },
    "ethics": {
        "path": "ethics/open_models",
        "domain": "Ethics",
        "truth_type": "Felt",
    },
}

# Excluded models
EXCLUDED = ["kimi_k2_5_legal", "glm", "ministral_14b_legal"]

# ============================================================================
# FUNCTIONS
# ============================================================================

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def compute_metrics(all_true_embs, all_cold_embs, n_trials, n_positions, original_drci):
    """Compute K, Var_Ratio, entanglement, arc from re-embedded data."""

    pos_true_means = []
    pos_true_vars = []
    pos_cold_means = []
    pos_cold_vars = []

    for pos in range(n_positions):
        true_at_pos = [all_true_embs[t][pos] for t in range(n_trials)]
        cold_at_pos = [all_cold_embs[t][pos] for t in range(n_trials)]

        true_sims = [cosine_similarity(true_at_pos[i], true_at_pos[j])
                     for i in range(n_trials) for j in range(i+1, n_trials)]
        cold_sims = [cosine_similarity(cold_at_pos[i], cold_at_pos[j])
                     for i in range(n_trials) for j in range(i+1, n_trials)]

        pos_true_means.append(np.mean(true_sims))
        pos_true_vars.append(np.var(true_sims))
        pos_cold_means.append(np.mean(cold_sims))
        pos_cold_vars.append(np.var(cold_sims))

    var_ratio = np.mean(pos_true_vars) / np.mean(pos_cold_vars) if np.mean(pos_cold_vars) > 0 else float('inf')

    # K using original dRCI (from per-trial computation) + re-embedded Var_Ratio
    K = original_drci * var_ratio

    # Entanglement
    pos_drci = [pos_true_means[p] - pos_cold_means[p] for p in range(n_positions)]
    pos_vri = [1 - (pos_true_vars[p] / pos_cold_vars[p]) if pos_cold_vars[p] > 0 else 0
               for p in range(n_positions)]
    r_ent, p_ent = pearsonr(pos_drci, pos_vri)

    # Exploration arc
    p_last = n_positions - 1
    last_sims = [cosine_similarity(all_true_embs[i][p_last], all_true_embs[j][p_last])
                 for i in range(n_trials) for j in range(i+1, n_trials)]
    first_sims = [cosine_similarity(all_true_embs[i][0], all_true_embs[j][0])
                  for i in range(n_trials) for j in range(i+1, n_trials)]
    arc = np.var(last_sims) / np.var(first_sims) if np.var(first_sims) > 0 else float('inf')

    return {
        "var_ratio": float(var_ratio),
        "K": float(K),
        "entanglement_r": float(r_ent),
        "entanglement_p": float(p_ent),
        "arc": float(arc),
        "mean_true_sim": float(np.mean(pos_true_means)),
        "mean_cold_sim": float(np.mean(pos_cold_means)),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    from sentence_transformers import SentenceTransformer

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for emb_name, emb_dim in EMBEDDING_MODELS:
        print(f"\n{'='*70}")
        print(f"EMBEDDING MODEL: {emb_name} ({emb_dim}D)")
        print(f"{'='*70}")

        print(f"Loading {emb_name}...", flush=True)
        embedder = SentenceTransformer(emb_name)
        print("Loaded.", flush=True)

        all_results = []

        for domain_key, domain_info in DOMAINS.items():
            data_path = os.path.join(BASE_DATA, domain_info["path"])
            if not os.path.exists(data_path):
                print(f"\n  {domain_key}: PATH NOT FOUND, skipping")
                continue

            files = [f for f in os.listdir(data_path)
                     if f.endswith('.json') and '50trials' in f and 'checkpoint' not in f]

            for fname in sorted(files):
                # Skip excluded models
                if any(excl in fname.lower() for excl in EXCLUDED):
                    print(f"\n  SKIP (excluded): {fname[:60]}")
                    continue

                fpath = os.path.join(data_path, fname)

                try:
                    with open(fpath) as fh:
                        d = json.load(fh)
                except:
                    print(f"\n  SKIP (load error): {fname[:60]}")
                    continue

                trials = d.get('trials', d.get('results', []))
                if len(trials) == 0:
                    print(f"\n  SKIP (0 trials): {fname[:60]}")
                    continue

                # Check if responses exist
                if not isinstance(trials[0], dict) or 'responses' not in trials[0]:
                    print(f"\n  SKIP (no responses): {fname[:60]}")
                    continue

                n_trials = len(trials)
                n_positions = len(trials[0]['responses'].get('true', []))

                if n_positions == 0:
                    print(f"\n  SKIP (0 positions): {fname[:60]}")
                    continue

                # Get original dRCI
                if 'delta_rci' in trials[0]:
                    original_drcis = [t['delta_rci']['cold'] for t in trials]
                    original_drci = float(np.mean(original_drcis))
                elif 'alignments' in trials[0]:
                    original_drcis = [t['alignments']['mean_true'] - t['alignments']['mean_cold'] for t in trials]
                    original_drci = float(np.mean(original_drcis))
                else:
                    print(f"\n  SKIP (no dRCI): {fname[:60]}")
                    continue

                # Extract model name
                model_name = fname.replace('mch_results_', '').replace('_50trials.json', '').replace(f'_{domain_info["domain"].lower()}', '')

                print(f"\n  {domain_info['domain']:12s} | {model_name[:30]:30s} | {n_trials} trials | {n_positions} positions", flush=True)
                print(f"    Re-embedding with {emb_name}...", end="", flush=True)

                # Re-embed
                all_true_embs = []
                all_cold_embs = []

                for t_idx, trial in enumerate(trials):
                    true_resps = trial['responses']['true']
                    cold_resps = trial['responses']['cold']

                    true_embs = [embedder.encode(r) for r in true_resps]
                    cold_embs = [embedder.encode(r) for r in cold_resps]

                    all_true_embs.append(true_embs)
                    all_cold_embs.append(cold_embs)

                    if (t_idx + 1) % 10 == 0:
                        print(f" {t_idx+1}", end="", flush=True)

                print(" done.", flush=True)

                # Compute metrics
                metrics = compute_metrics(all_true_embs, all_cold_embs, n_trials, n_positions, original_drci)

                result = {
                    "model": model_name,
                    "domain": domain_info["domain"],
                    "truth_type": domain_info["truth_type"],
                    "embedding_model": emb_name,
                    "embedding_dim": emb_dim,
                    "n_trials": n_trials,
                    "n_positions": n_positions,
                    "original_drci": original_drci,
                    **metrics,
                }

                all_results.append(result)

                ent_sig = "***" if metrics["entanglement_p"] < 0.001 else "**" if metrics["entanglement_p"] < 0.01 else "*" if metrics["entanglement_p"] < 0.05 else "ns"
                print(f"    dRCI={original_drci:.4f}  VR={metrics['var_ratio']:.4f}  K={metrics['K']:.4f}  ent_r={metrics['entanglement_r']:.3f}{ent_sig}  arc={metrics['arc']:.2f}")

        # ================================================================
        # SUMMARY
        # ================================================================
        print(f"\n{'='*70}")
        print(f"SUMMARY: {emb_name} ({emb_dim}D)")
        print(f"{'='*70}")

        # Group by domain
        domains_found = {}
        for r in all_results:
            dom = r["domain"]
            if dom not in domains_found:
                domains_found[dom] = []
            domains_found[dom].append(r)

        print(f"\n{'Domain':12s} {'Truth Type':12s} {'N':>3s} {'K_mean':>8s} {'K_std':>8s} {'CV':>8s} {'K_384D':>8s}")
        print("-" * 70)

        # Reference K values from 384D
        ref_K = {"Medical": 0.429, "Philosophy": 0.301, "Legal": 0.348, "Ethics": 0.193}

        for dom in ["Medical", "Philosophy", "Legal", "Ethics"]:
            if dom in domains_found:
                ks = [r["K"] for r in domains_found[dom]]
                n = len(ks)
                mean_k = np.mean(ks)
                std_k = np.std(ks)
                cv = std_k / mean_k if mean_k > 0 else float('inf')
                ref = ref_K.get(dom, "—")
                print(f"{dom:12s} {domains_found[dom][0]['truth_type']:12s} {n:3d} {mean_k:8.4f} {std_k:8.4f} {cv:8.4f} {ref:>8s}")

        # Save results
        output = {
            "embedding_model": emb_name,
            "embedding_dim": emb_dim,
            "timestamp": datetime.now().isoformat(),
            "baseline_model": "all-MiniLM-L6-v2 (384D)",
            "results": all_results,
            "domain_summary": {}
        }

        for dom in domains_found:
            ks = [r["K"] for r in domains_found[dom]]
            output["domain_summary"][dom] = {
                "N": len(ks),
                "K_mean": float(np.mean(ks)),
                "K_std": float(np.std(ks)),
                "CV": float(np.std(ks) / np.mean(ks)) if np.mean(ks) > 0 else None,
                "K_range": [float(np.min(ks)), float(np.max(ks))],
                "K_384D_reference": ref_K.get(dom),
            }

        outfile = os.path.join(OUTPUT_DIR, f"paper6_robustness_{emb_name.replace('-','_')}.json")
        with open(outfile, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {outfile}")

        # Per-model detail table
        print(f"\n{'Model':30s} {'Domain':10s} {'dRCI':>8s} {'VR':>8s} {'K':>8s} {'Ent r':>8s} {'Arc':>8s}")
        print("-" * 85)
        for r in all_results:
            ent_sig = "***" if r["entanglement_p"] < 0.001 else "**" if r["entanglement_p"] < 0.01 else "*" if r["entanglement_p"] < 0.05 else "ns"
            print(f"{r['model']:30s} {r['domain']:10s} {r['original_drci']:8.4f} {r['var_ratio']:8.4f} {r['K']:8.4f} {r['entanglement_r']:7.3f}{ent_sig:3s} {r['arc']:8.2f}")


if __name__ == "__main__":
    main()
