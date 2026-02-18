#!/usr/bin/env python3
"""
Compute Context Utilization Depth (CUD) from pilot results.

CUD = minimum K where dRCI_TRUNCATED(K) >= threshold * dRCI_TRUE

Uses per-trial dRCI_TRUE as reference (not K=max approximation).
"""

import os
import json
import numpy as np
import pandas as pd

def load_results(filepath):
    """Load results JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)

def compute_cud(data, threshold=0.90):
    """
    Compute CUD for a single model-domain result file.

    Returns dict with CUD value, dRCI curves, and diagnostics.
    """
    model = data["model"]
    domain = data["domain"]
    position = data["position"]
    trials = data["trials"]

    # Collect per-trial data
    k_values = set()
    trial_data = []

    for t in trials:
        drci_true = t["drci_true"]
        row = {"trial": t["trial"], "drci_true": drci_true}

        for K_str, kr in t["k_results"].items():
            K = int(K_str)
            k_values.add(K)
            row[f"drci_K{K}"] = kr["drci_truncated"]
            row[f"sim_true_K{K}"] = kr["sim_trunc_true"]

        trial_data.append(row)

    df = pd.DataFrame(trial_data)
    k_values = sorted(k_values)

    # Compute mean dRCI by K across trials
    mean_drci_true = df["drci_true"].mean()
    std_drci_true = df["drci_true"].std()

    drci_by_k = {}
    std_by_k = {}
    ratio_by_k = {}  # dRCI_TRUNCATED(K) / dRCI_TRUE

    for K in k_values:
        col = f"drci_K{K}"
        if col in df.columns:
            drci_by_k[K] = df[col].mean()
            std_by_k[K] = df[col].std()
            # Per-trial ratio, then average (more robust than ratio of means)
            ratios = df[col] / df["drci_true"]
            ratio_by_k[K] = ratios.mean()

    # Compute sim_trunc_true by K
    sim_by_k = {}
    sim_std_by_k = {}
    for K in k_values:
        col = f"sim_true_K{K}"
        if col in df.columns:
            sim_by_k[K] = df[col].mean()
            sim_std_by_k[K] = df[col].std()

    # Find CUD (dRCI-based): minimum K where ratio >= threshold
    cud_drci = f">{max(k_values)}"
    for K in k_values:
        if K in ratio_by_k and ratio_by_k[K] >= threshold:
            cud_drci = K
            break

    # Find CUD (sim_trunc_true-based): minimum K where sim >= threshold
    cud_sim = f">{max(k_values)}"
    for K in k_values:
        if K in sim_by_k and sim_by_k[K] >= threshold:
            cud_sim = K
            break

    return {
        "model": model,
        "domain": domain,
        "position": position,
        "n_trials": len(trials),
        "drci_true_mean": mean_drci_true,
        "drci_true_std": std_drci_true,
        "drci_by_k": drci_by_k,
        "std_by_k": std_by_k,
        "ratio_by_k": ratio_by_k,
        "sim_by_k": sim_by_k,
        "sim_std_by_k": sim_std_by_k,
        "cud_90_drci": cud_drci,
        "cud_90_sim": cud_sim,
        "k_values": k_values,
    }

def main():
    """Process all result files in results/raw/."""
    results_dir = "results/raw"
    all_cud = []

    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith(".json") and not fname.endswith("_checkpoint.json"):
            filepath = os.path.join(results_dir, fname)
            data = load_results(filepath)
            if not data.get("trials"):
                print(f"  Skipping {fname} (0 trials)")
                continue
            cud = compute_cud(data)
            all_cud.append(cud)

    # Display results
    print("\n" + "=" * 70)
    print("CONTEXT UTILIZATION DEPTH (CUD) — PILOT RESULTS")
    print("=" * 70)

    for r in all_cud:
        print(f"\n{r['model']} | {r['domain']} | P{r['position']} | n={r['n_trials']}")
        print(f"  dRCI_TRUE: {r['drci_true_mean']:.4f} (SD={r['drci_true_std']:.4f})")
        print(f"  CUD_dRCI (90% threshold): K={r['cud_90_drci']}")
        print(f"  CUD_SIM  (90% threshold): K={r['cud_90_sim']}")
        print(f"  {'K':>4s}  {'dRCI':>8s}  {'%TRUE':>6s}  {'sim->TRUE':>8s}")
        print(f"  {'-'*4}  {'-'*8}  {'-'*6}  {'-'*8}")
        for K in r["k_values"]:
            drci = r["drci_by_k"].get(K, 0)
            ratio = r["ratio_by_k"].get(K, 0)
            sim = r["sim_by_k"].get(K, 0)
            print(f"  K={K:2d}  {drci:.4f}   {ratio:.1%}   {sim:.4f}")

    # Save summary CSV
    rows = []
    for r in all_cud:
        for K in r["k_values"]:
            rows.append({
                "model": r["model"],
                "domain": r["domain"],
                "position": r["position"],
                "K": K,
                "drci_truncated": r["drci_by_k"].get(K, np.nan),
                "drci_std": r["std_by_k"].get(K, np.nan),
                "ratio_to_true": r["ratio_by_k"].get(K, np.nan),
                "sim_trunc_true": r["sim_by_k"].get(K, np.nan),
                "sim_std": r["sim_std_by_k"].get(K, np.nan),
                "drci_true": r["drci_true_mean"],
                "cud_90_drci": r["cud_90_drci"],
                "cud_90_sim": r["cud_90_sim"],
            })

    df_out = pd.DataFrame(rows)
    os.makedirs("results/processed", exist_ok=True)
    df_out.to_csv("results/processed/cud_summary.csv", index=False)
    print(f"\nSaved: results/processed/cud_summary.csv")

if __name__ == "__main__":
    main()
