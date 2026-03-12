#!/usr/bin/env python3
"""
Compare Paper 4 vs Paper 6 Var_Ratio methods on the same data.

Paper 4 method: RCI per trial = mean cosine sim to all other trials.
  Var_Ratio = Var(RCI_TRUE) / Var(RCI_COLD)

Paper 6 method: Embedding dimension variance across trials.
  Var_Ratio = mean(Var_dim(TRUE)) / mean(Var_dim(COLD))

Run on the same 15 model-domain files to see if they agree.
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

print("Loading embedding model...")
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("Loaded.\n")


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def paper4_rci(embeddings):
    """Paper 4: RCI per trial = mean cosine sim to all other trials."""
    n = len(embeddings)
    rci_values = []
    for i in range(n):
        sims = []
        for j in range(n):
            if i != j:
                sims.append(cosine_similarity(embeddings[i], embeddings[j]))
        rci_values.append(np.mean(sims))
    return np.array(rci_values)


def get_runs():
    dirs = [
        (BASE / "data" / "medical" / "closed_models", "Medical"),
        (BASE / "data" / "medical" / "open_models", "Medical"),
        (BASE / "data" / "medical" / "gemini_flash", "Medical"),
        (BASE / "data" / "philosophy" / "closed_models", "Philosophy"),
        (BASE / "data" / "philosophy" / "open_models", "Philosophy"),
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
                if not (isinstance(r, dict) and 'true' in r and 'cold' in r and 'scrambled' in r):
                    continue
                if not (isinstance(r['true'][0], str) and len(r['true'][0]) > 10):
                    continue
                if not (isinstance(r['scrambled'][0], str) and len(r['scrambled'][0]) > 10):
                    continue
                model = data.get('model', f.stem)
                key = f"{model}_{domain}"
                if key in seen:
                    continue
                seen.add(key)
                runs.append({'model': model, 'domain': domain, 'path': str(f), 'n_trials': len(trials)})
            except:
                pass
    return runs


def analyze_run(run):
    with open(run['path'], encoding='utf-8') as f:
        data = json.load(f)

    trials = data['trials']
    n_trials = len(trials)
    n_prompts = len(trials[0]['responses']['true'])

    # Paper 4 method: per-position pairwise RCI
    p4_vr_true_cold = []
    p4_vr_scram_cold = []
    p4_vr_true_scram = []

    # Paper 6 method: embedding dimension variance
    p6_vr_true_cold = []
    p6_vr_scram_cold = []
    p6_vr_true_scram = []

    for pos in range(n_prompts):
        true_texts = [t['responses']['true'][pos] for t in trials]
        cold_texts = [t['responses']['cold'][pos] for t in trials]
        scram_texts = [t['responses']['scrambled'][pos] for t in trials]

        true_embs = embedder.encode(true_texts, show_progress_bar=False, convert_to_numpy=True)
        cold_embs = embedder.encode(cold_texts, show_progress_bar=False, convert_to_numpy=True)
        scram_embs = embedder.encode(scram_texts, show_progress_bar=False, convert_to_numpy=True)

        # === Paper 4 method ===
        rci_true = paper4_rci(true_embs)
        rci_cold = paper4_rci(cold_embs)
        rci_scram = paper4_rci(scram_embs)

        var_t = np.var(rci_true, ddof=1)
        var_c = np.var(rci_cold, ddof=1)
        var_s = np.var(rci_scram, ddof=1)

        if var_c > 1e-10:
            p4_vr_true_cold.append(var_t / var_c)
            p4_vr_scram_cold.append(var_s / var_c)
        if var_s > 1e-10:
            p4_vr_true_scram.append(var_t / var_s)

        # === Paper 6 method ===
        var_true_dim = np.mean(np.var(true_embs, axis=0))
        var_cold_dim = np.mean(np.var(cold_embs, axis=0))
        var_scram_dim = np.mean(np.var(scram_embs, axis=0))

        if var_cold_dim > 1e-10:
            p6_vr_true_cold.append(var_true_dim / var_cold_dim)
            p6_vr_scram_cold.append(var_scram_dim / var_cold_dim)
        if var_scram_dim > 1e-10:
            p6_vr_true_scram.append(var_true_dim / var_scram_dim)

    return {
        'model': run['model'],
        'domain': run['domain'],
        'p4_total': np.mean(p4_vr_true_cold),
        'p4_content': np.mean(p4_vr_scram_cold),
        'p4_order': np.mean(p4_vr_true_scram),
        'p6_total': np.mean(p6_vr_true_cold),
        'p6_content': np.mean(p6_vr_scram_cold),
        'p6_order': np.mean(p6_vr_true_scram),
    }


def main():
    runs = get_runs()
    print(f"Found {len(runs)} runs\n")

    results = []
    for i, r in enumerate(runs):
        print(f"[{i+1}/{len(runs)}] {r['model']} ({r['domain']})...", flush=True)
        res = analyze_run(r)
        results.append(res)

    print("\n" + "=" * 110)
    print("COMPARISON: Paper 4 (pairwise RCI) vs Paper 6 (embedding dim variance)")
    print("=" * 110)
    print(f"{'Model':<25s} {'Domain':<10s} | {'P4_Total':>9s} {'P4_Content':>11s} {'P4_Order':>9s} | {'P6_Total':>9s} {'P6_Content':>11s} {'P6_Order':>9s}")
    print("-" * 110)

    p4_totals = []
    p6_totals = []
    p4_contents = []
    p6_contents = []

    for r in results:
        print(f"{r['model']:<25s} {r['domain']:<10s} | {r['p4_total']:9.4f} {r['p4_content']:11.4f} {r['p4_order']:9.4f} | {r['p6_total']:9.4f} {r['p6_content']:11.4f} {r['p6_order']:9.4f}")
        p4_totals.append(r['p4_total'])
        p6_totals.append(r['p6_total'])
        p4_contents.append(r['p4_content'])
        p6_contents.append(r['p6_content'])

    # Correlation between methods
    r_total, p_total = stats.pearsonr(p4_totals, p6_totals)
    r_content, p_content = stats.pearsonr(p4_contents, p6_contents)

    print(f"\nCorrelation between methods:")
    print(f"  VR_Total:   r={r_total:.4f}, p={p_total:.6f}")
    print(f"  VR_Content: r={r_content:.4f}, p={p_content:.6f}")

    print("\nDone.")


if __name__ == '__main__':
    main()
