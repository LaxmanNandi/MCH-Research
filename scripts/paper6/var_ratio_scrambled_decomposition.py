#!/usr/bin/env python3
"""
Var_Ratio Decomposition: Content vs Order
==========================================
Extends Paper 4/6 Var_Ratio to SCRAMBLED condition.

Decomposes:
  Var_Ratio_Total   = Var(TRUE) / Var(COLD)
  Var_Ratio_Content = Var(SCRAMBLED) / Var(COLD)
  Var_Ratio_Order   = Var(TRUE) / Var(SCRAMBLED)

Uses same re-embedding method as Paper 6 conservation law verification.
Embedding: all-MiniLM-L6-v2 (384D), variance of embedding dimensions across trials.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')

BASE = Path("C:/Users/barla/mch_experiments")

# Load embedding model
print("Loading embedding model: all-MiniLM-L6-v2...")
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("Loaded.\n")


def get_runs_with_scrambled():
    """Find all model-domain runs that have TRUE, COLD, and SCRAMBLED response text."""
    dirs = [
        (BASE / "data" / "medical" / "closed_models", "Medical"),
        (BASE / "data" / "medical" / "open_models", "Medical"),
        (BASE / "data" / "medical" / "gemini_flash", "Medical"),
        (BASE / "data" / "philosophy" / "closed_models", "Philosophy"),
        (BASE / "data" / "philosophy" / "open_models", "Philosophy"),
        (BASE / "data" / "legal" / "open_models", "Legal"),
    ]

    runs = []
    seen = set()

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

                t = trials[0]
                if 'responses' not in t:
                    continue
                r = t['responses']
                # Need all three conditions
                if not (isinstance(r, dict) and 'true' in r and 'cold' in r and 'scrambled' in r):
                    continue
                if not (isinstance(r['true'], list) and len(r['true']) > 0):
                    continue
                if not (isinstance(r['true'][0], str) and len(r['true'][0]) > 10):
                    continue
                # Verify scrambled has text too
                if not (isinstance(r['scrambled'], list) and len(r['scrambled']) > 0):
                    continue
                if not (isinstance(r['scrambled'][0], str) and len(r['scrambled'][0]) > 10):
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


def compute_var_ratio_decomposition(run):
    """Compute Var_Ratio for TRUE, COLD, SCRAMBLED using embedding variance."""
    with open(run['path'], encoding='utf-8') as f:
        data = json.load(f)

    trials = data['trials']
    n_trials = len(trials)
    n_prompts = len(trials[0]['responses']['true'])

    # Collect all responses
    true_texts = []
    cold_texts = []
    scram_texts = []

    for t in trials:
        true_texts.extend(t['responses']['true'])
        cold_texts.extend(t['responses']['cold'])
        scram_texts.extend(t['responses']['scrambled'])

    # Embed all three conditions
    print(f"  Embedding {n_trials}x{n_prompts} responses (TRUE+COLD+SCRAMBLED)...", flush=True)
    true_embs = embedder.encode(true_texts, show_progress_bar=False, convert_to_numpy=True)
    cold_embs = embedder.encode(cold_texts, show_progress_bar=False, convert_to_numpy=True)
    scram_embs = embedder.encode(scram_texts, show_progress_bar=False, convert_to_numpy=True)

    # Reshape by trial and position
    true_by_trial = true_embs.reshape(n_trials, n_prompts, -1)
    cold_by_trial = cold_embs.reshape(n_trials, n_prompts, -1)
    scram_by_trial = scram_embs.reshape(n_trials, n_prompts, -1)

    # Per-position variance ratios
    vr_total = []
    vr_content = []
    vr_order = []
    var_true_all = []
    var_cold_all = []
    var_scram_all = []

    for p in range(n_prompts):
        true_at_p = true_by_trial[:, p, :]   # (n_trials, 384)
        cold_at_p = cold_by_trial[:, p, :]
        scram_at_p = scram_by_trial[:, p, :]

        var_true = np.mean(np.var(true_at_p, axis=0))
        var_cold = np.mean(np.var(cold_at_p, axis=0))
        var_scram = np.mean(np.var(scram_at_p, axis=0))

        var_true_all.append(var_true)
        var_cold_all.append(var_cold)
        var_scram_all.append(var_scram)

        if var_cold > 1e-10:
            vr_t = var_true / var_cold
            vr_c = var_scram / var_cold
        else:
            vr_t = 1.0
            vr_c = 1.0

        if var_scram > 1e-10:
            vr_o = var_true / var_scram
        else:
            vr_o = 1.0

        vr_total.append(vr_t)
        vr_content.append(vr_c)
        vr_order.append(vr_o)

    return {
        'model': run['model'],
        'domain': run['domain'],
        'n_trials': n_trials,
        'n_prompts': n_prompts,
        'vr_total': np.mean(vr_total),
        'vr_content': np.mean(vr_content),
        'vr_order': np.mean(vr_order),
        'vr_total_per_pos': vr_total,
        'vr_content_per_pos': vr_content,
        'vr_order_per_pos': vr_order,
        'var_true_per_pos': var_true_all,
        'var_cold_per_pos': var_cold_all,
        'var_scram_per_pos': var_scram_all,
        'product_check': np.mean(vr_content) * np.mean(vr_order),  # should ≈ vr_total
    }


def main():
    print("=" * 70)
    print("VAR_RATIO DECOMPOSITION: CONTENT vs ORDER")
    print("Var(TRUE)/Var(COLD) = [Var(SCRAM)/Var(COLD)] × [Var(TRUE)/Var(SCRAM)]")
    print("=" * 70)

    runs = get_runs_with_scrambled()
    print(f"\nFound {len(runs)} runs with SCRAMBLED response text:\n")
    for r in runs:
        print(f"  {r['model']:30s} {r['domain']:12s} ({r['n_trials']} trials)")

    # Compute decomposition for each run
    results = []
    for r in runs:
        print(f"\n--- {r['model']} ({r['domain']}) ---")
        res = compute_var_ratio_decomposition(r)
        results.append(res)

    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY: Var_Ratio Decomposition")
    print("=" * 100)
    print(f"{'Model':<30s} {'Domain':<12s} {'VR_Total':>10s} {'VR_Content':>12s} {'VR_Order':>10s} {'Product':>10s}")
    print("-" * 100)

    med_results = []
    phil_results = []

    for r in results:
        print(f"{r['model']:<30s} {r['domain']:<12s} {r['vr_total']:10.4f} {r['vr_content']:12.4f} {r['vr_order']:10.4f} {r['product_check']:10.4f}")
        if r['domain'] == 'Medical':
            med_results.append(r)
        else:
            phil_results.append(r)

    # Domain summaries
    print("\n" + "=" * 70)
    print("DOMAIN SUMMARIES")
    print("=" * 70)

    for label, group in [("Medical", med_results), ("Philosophy", phil_results)]:
        if not group:
            continue
        vr_t = [r['vr_total'] for r in group]
        vr_c = [r['vr_content'] for r in group]
        vr_o = [r['vr_order'] for r in group]
        print(f"\n{label} (N={len(group)} runs):")
        print(f"  VR_Total:   {np.mean(vr_t):.4f} ± {np.std(vr_t):.4f}")
        print(f"  VR_Content: {np.mean(vr_c):.4f} ± {np.std(vr_c):.4f}")
        print(f"  VR_Order:   {np.mean(vr_o):.4f} ± {np.std(vr_o):.4f}")

    # Statistical tests
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS")
    print("=" * 70)

    if med_results and phil_results:
        med_vrc = [r['vr_content'] for r in med_results]
        phil_vrc = [r['vr_content'] for r in phil_results]
        med_vro = [r['vr_order'] for r in med_results]
        phil_vro = [r['vr_order'] for r in phil_results]

        u1, p1 = stats.mannwhitneyu(med_vrc, phil_vrc, alternative='two-sided')
        u2, p2 = stats.mannwhitneyu(med_vro, phil_vro, alternative='two-sided')
        print(f"\nVR_Content: Medical vs Philosophy")
        print(f"  Mann-Whitney U={u1}, p={p1:.6f}")
        print(f"  Medical mean={np.mean(med_vrc):.4f}, Phil mean={np.mean(phil_vrc):.4f}")

        print(f"\nVR_Order: Medical vs Philosophy")
        print(f"  Mann-Whitney U={u2}, p={p2:.6f}")
        print(f"  Medical mean={np.mean(med_vro):.4f}, Phil mean={np.mean(phil_vro):.4f}")

    # Content fraction of variance
    print("\n" + "=" * 70)
    print("CONTENT FRACTION OF VARIANCE")
    print("(VR_Content / VR_Total) — what fraction of variance amplification comes from content?")
    print("=" * 70)

    for r in results:
        frac = r['vr_content'] / r['vr_total'] if r['vr_total'] > 1e-10 else 0
        print(f"  {r['model']:<30s} {r['domain']:<12s} Content fraction: {frac:.1%}")

    if med_results and phil_results:
        med_frac = [r['vr_content'] / r['vr_total'] for r in med_results if r['vr_total'] > 1e-10]
        phil_frac = [r['vr_content'] / r['vr_total'] for r in phil_results if r['vr_total'] > 1e-10]
        print(f"\n  Medical mean content fraction:    {np.mean(med_frac):.1%} ± {np.std(med_frac):.1%}")
        print(f"  Philosophy mean content fraction:  {np.mean(phil_frac):.1%} ± {np.std(phil_frac):.1%}")
        u3, p3 = stats.mannwhitneyu(med_frac, phil_frac, alternative='two-sided')
        print(f"  Mann-Whitney U={u3}, p={p3:.6f}")

    # Llama spotlight
    print("\n" + "=" * 70)
    print("LLAMA MODELS SPOTLIGHT")
    print("(Testing if high Var_Ratio comes from content or order)")
    print("=" * 70)

    for r in results:
        if 'llama' in r['model'].lower() or 'maverick' in r['model'].lower() or 'scout' in r['model'].lower():
            print(f"\n  {r['model']} ({r['domain']}):")
            print(f"    VR_Total={r['vr_total']:.4f}  VR_Content={r['vr_content']:.4f}  VR_Order={r['vr_order']:.4f}")
            frac = r['vr_content'] / r['vr_total'] if r['vr_total'] > 1e-10 else 0
            print(f"    Content fraction: {frac:.1%}")
            # P30 specifically
            if len(r['vr_total_per_pos']) >= 30:
                print(f"    P30: VR_Total={r['vr_total_per_pos'][29]:.4f}  VR_Content={r['vr_content_per_pos'][29]:.4f}  VR_Order={r['vr_order_per_pos'][29]:.4f}")

    # Multiplicative decomposition check
    print("\n" + "=" * 70)
    print("MULTIPLICATIVE DECOMPOSITION CHECK")
    print("VR_Content × VR_Order ≈ VR_Total?")
    print("=" * 70)

    for r in results:
        ratio = r['product_check'] / r['vr_total'] if r['vr_total'] > 1e-10 else 0
        print(f"  {r['model']:<30s} {r['domain']:<12s} Product/Total = {ratio:.4f}")

    print("\nDone.")


if __name__ == '__main__':
    main()
