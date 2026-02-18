#!/usr/bin/env python3
"""
Plot CUD dual-metric curves for Paper 7 pilot.

Panel 1: ΔRCI_TRUNCATED(K) vs K — context sensitivity (distance from COLD)
Panel 2: sim_trunc_true(K) vs K — context accuracy (convergence to TRUE)

Key insight: sensitivity saturates early, accuracy requires more context.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

def load_results(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

def plot_cud_curves():
    """Plot dual-metric CUD curves for all model-domain conditions."""

    results_dir = "results/raw"
    conditions = []

    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith(".json") and not fname.endswith("_checkpoint.json"):
            filepath = os.path.join(results_dir, fname)
            data = load_results(filepath)
            if data.get("trials"):
                conditions.append(data)

    if not conditions:
        print("No result files with data found in results/raw/")
        return

    n_conditions = len(conditions)

    # Two rows per condition: ΔRCI (top) and sim_trunc_true (bottom)
    fig, axes = plt.subplots(2, n_conditions, figsize=(7 * n_conditions, 10),
                              squeeze=False)

    for idx, c in enumerate(conditions):
        model = c["model"]
        domain = c["domain"]
        position = c["position"]
        trials = c["trials"]

        # Collect K values
        k_values = set()
        for t in trials:
            for K_str in t["k_results"]:
                k_values.add(int(K_str))
        k_values = sorted(k_values)

        # --- Compute ΔRCI stats ---
        mean_drci = []
        std_drci = []
        for K in k_values:
            vals = [t["k_results"][str(K)]["drci_truncated"]
                    for t in trials if str(K) in t["k_results"]
                    and t["k_results"][str(K)] is not None]
            mean_drci.append(np.mean(vals))
            std_drci.append(np.std(vals))

        drci_true_vals = [t["drci_true"] for t in trials]
        drci_true_mean = np.mean(drci_true_vals)
        drci_true_std = np.std(drci_true_vals)

        # --- Compute sim_trunc_true stats ---
        mean_sim = []
        std_sim = []
        for K in k_values:
            vals = [t["k_results"][str(K)]["sim_trunc_true"]
                    for t in trials if str(K) in t["k_results"]
                    and t["k_results"][str(K)] is not None]
            mean_sim.append(np.mean(vals))
            std_sim.append(np.std(vals))

        # ==========================================
        # Panel 1: ΔRCI (context sensitivity)
        # ==========================================
        ax1 = axes[0, idx]

        ax1.errorbar(k_values, mean_drci, yerr=std_drci,
                     marker='o', capsize=5, linewidth=2, markersize=8,
                     color='#2196F3', label='ΔRCI_TRUNCATED(K)')

        ax1.axhline(drci_true_mean, color='green', linestyle='--', linewidth=1.5,
                    label=f'ΔRCI_TRUE: {drci_true_mean:.3f}')
        ax1.axhline(0.9 * drci_true_mean, color='orange', linestyle=':',
                    linewidth=1.5, label=f'90% threshold: {0.9 * drci_true_mean:.3f}')

        ax1.axhspan(drci_true_mean - drci_true_std, drci_true_mean + drci_true_std,
                    alpha=0.1, color='green')

        ax1.set_ylabel("ΔRCI (sensitivity)", fontsize=12)
        ax1.set_title(f"{model} — {domain} (P{position})\nContext Sensitivity",
                      fontsize=13, fontweight='bold')
        ax1.legend(loc='lower right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(k_values)
        ax1.set_ylim(bottom=0)

        # ==========================================
        # Panel 2: sim_trunc_true (context accuracy)
        # ==========================================
        ax2 = axes[1, idx]

        ax2.errorbar(k_values, mean_sim, yerr=std_sim,
                     marker='s', capsize=5, linewidth=2, markersize=8,
                     color='#E91E63', label='sim(TRUNC, TRUE)')

        ax2.axhline(1.0, color='green', linestyle='--', linewidth=1.5,
                    label='Perfect match (1.0)')
        ax2.axhline(0.9, color='orange', linestyle=':',
                    linewidth=1.5, label='90% convergence')

        ax2.set_xlabel("K (last K message pairs)", fontsize=12)
        ax2.set_ylabel("Cosine Similarity to TRUE", fontsize=12)
        ax2.set_title(f"Context Accuracy (convergence to TRUE)",
                      fontsize=13, fontweight='bold')
        ax2.legend(loc='lower right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(k_values)
        ax2.set_ylim(0.4, 1.05)

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/cud_dual_metric.png", dpi=150, bbox_inches='tight')
    print("Saved: figures/cud_dual_metric.png")
    plt.close()

if __name__ == "__main__":
    plot_cud_curves()
