#!/usr/bin/env python3
"""
MCH Position-Dependent ΔRCI Analysis + Disruption Sensitivity

Insight 1: Does ΔRCI vary by prompt position within conversation?
  - Hypothesis: Early prompts (1-10) lower ΔRCI, late prompts (21-30) higher ΔRCI
  - Method: Compute per-position ΔRCI across all trials, ANOVA across bins

Insight 4: Disruption Sensitivity (SCRAMBLED vs COLD)
  - Does wrong-order context hurt more than no context?
  - Disruption Sensitivity = ΔRCI_scrambled - ΔRCI_cold
  - If > 0: scrambled hurts more than cold (wrong context actively harmful)
  - If < 0: scrambled helps more than cold (any context helps)

Outputs:
  - Console: Statistical results, ANOVA tables
  - Plots: Position-dependent ΔRCI curves, disruption sensitivity
  - CSV: Per-position data for further analysis
"""

import os
import sys
import json
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8')

# ============================================================================
# DATA LOADING
# ============================================================================

BASE_DIR = "C:/Users/barla/mch_experiments/data"
OUTPUT_DIR = "C:/Users/barla/mch_experiments/analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define all data sources with their metadata
DATA_SOURCES = {
    # Philosophy - Closed Models (Paper 2 rerun, 50 trials)
    "gpt4o_mini_phil": {
        "path": f"{BASE_DIR}/closed_model_philosophy_rerun/mch_results_gpt4o_mini_philosophy_50trials.json",
        "label": "GPT-4o-mini", "domain": "philosophy", "type": "closed"
    },
    "gpt4o_phil": {
        "path": f"{BASE_DIR}/closed_model_philosophy_rerun/mch_results_gpt4o_philosophy_50trials.json",
        "label": "GPT-4o", "domain": "philosophy", "type": "closed"
    },
    "claude_haiku_phil": {
        "path": f"{BASE_DIR}/closed_model_philosophy_rerun/mch_results_claude_haiku_philosophy_50trials.json",
        "label": "Claude Haiku", "domain": "philosophy", "type": "closed"
    },
    "gemini_flash_phil": {
        "path": f"{BASE_DIR}/closed_model_philosophy_rerun/mch_results_gemini_flash_philosophy_50trials.json",
        "label": "Gemini Flash", "domain": "philosophy", "type": "closed"
    },
    # Philosophy - Open Models (50 trials)
    "deepseek_phil": {
        "path": f"{BASE_DIR}/open_model_results/mch_results_deepseek_v3_1_philosophy_50trials.json",
        "label": "DeepSeek V3.1", "domain": "philosophy", "type": "open"
    },
    "llama_maverick_phil": {
        "path": f"{BASE_DIR}/open_model_results/mch_results_llama_4_maverick_philosophy_50trials.json",
        "label": "Llama 4 Maverick", "domain": "philosophy", "type": "open"
    },
    "llama_scout_phil": {
        "path": f"{BASE_DIR}/open_model_results/mch_results_llama_4_scout_philosophy_50trials.json",
        "label": "Llama 4 Scout", "domain": "philosophy", "type": "open"
    },
    "qwen3_phil": {
        "path": f"{BASE_DIR}/open_model_results/mch_results_qwen3_235b_philosophy_50trials.json",
        "label": "Qwen3 235B", "domain": "philosophy", "type": "open"
    },
    "mistral_small_phil": {
        "path": f"{BASE_DIR}/open_model_results/mch_results_mistral_small_24b_philosophy_50trials.json",
        "label": "Mistral Small 24B", "domain": "philosophy", "type": "open"
    },
    "ministral_phil": {
        "path": f"{BASE_DIR}/open_model_results/mch_results_ministral_14b_philosophy_50trials.json",
        "label": "Ministral 14B", "domain": "philosophy", "type": "open"
    },
    "kimi_k2_phil": {
        "path": f"{BASE_DIR}/open_model_results/mch_results_kimi_k2_philosophy_50trials.json",
        "label": "Kimi K2", "domain": "philosophy", "type": "open"
    },
    # Medical - Closed Models
    "gemini_flash_med": {
        "path": f"{BASE_DIR}/gemini_flash_medical_rerun/mch_results_gemini_flash_medical_50trials.json",
        "label": "Gemini Flash", "domain": "medical", "type": "closed"
    },
    "claude_haiku_med": {
        "path": f"{BASE_DIR}/medical_results/mch_results_claude_haiku_medical_50trials.json",
        "label": "Claude Haiku", "domain": "medical", "type": "closed"
    },
    "gpt4o_med": {
        "path": f"{BASE_DIR}/medical_results/mch_results_gpt4o_medical_50trials.json",
        "label": "GPT-4o", "domain": "medical", "type": "closed"
    },
    "gpt4o_mini_med": {
        "path": f"{BASE_DIR}/medical_results/mch_results_gpt4o_mini_medical_50trials.json",
        "label": "GPT-4o-mini", "domain": "medical", "type": "closed"
    },
}


def load_dataset(key, info):
    """Load a dataset and extract per-position alignment data."""
    path = info["path"]
    if not os.path.exists(path):
        print(f"  SKIP {key}: file not found")
        return None

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    trials = data.get("trials", [])
    if not trials:
        print(f"  SKIP {key}: no trials")
        return None

    # Check first trial has per-prompt alignments
    t0 = trials[0]
    if "alignments" not in t0:
        print(f"  SKIP {key}: no alignments in trial data")
        return None

    aligns = t0["alignments"]
    if not isinstance(aligns.get("cold"), list):
        print(f"  SKIP {key}: cold alignments not a list")
        return None

    n_prompts = len(aligns["cold"])
    n_trials = len(trials)

    # Extract per-position data: shape (n_trials, n_prompts)
    cold_matrix = np.zeros((n_trials, n_prompts))
    scrambled_matrix = np.zeros((n_trials, n_prompts))

    for i, trial in enumerate(trials):
        a = trial["alignments"]
        cold_matrix[i, :] = a["cold"]
        scrambled_matrix[i, :] = a["scrambled"]

    # ΔRCI per position = 1.0 - alignment (since TRUE = 1.0)
    drci_cold_matrix = 1.0 - cold_matrix
    drci_scrambled_matrix = 1.0 - scrambled_matrix

    return {
        "key": key,
        "label": info["label"],
        "domain": info["domain"],
        "type": info["type"],
        "n_trials": n_trials,
        "n_prompts": n_prompts,
        "cold_matrix": cold_matrix,           # (trials, positions)
        "scrambled_matrix": scrambled_matrix,   # (trials, positions)
        "drci_cold_matrix": drci_cold_matrix,  # (trials, positions)
        "drci_scrambled_matrix": drci_scrambled_matrix,
    }


# ============================================================================
# INSIGHT 1: POSITION-DEPENDENT ΔRCI
# ============================================================================

def analyze_position_drci(datasets):
    """Compute and analyze position-dependent ΔRCI for all models."""
    print("\n" + "=" * 70)
    print("INSIGHT 1: POSITION-DEPENDENT ΔRCI ANALYSIS")
    print("=" * 70)

    results = {}

    for ds in datasets:
        key = ds["key"]
        label = ds["label"]
        domain = ds["domain"]
        n_trials = ds["n_trials"]
        n_prompts = ds["n_prompts"]

        # Mean ΔRCI per position across all trials
        mean_drci_by_pos = np.mean(ds["drci_cold_matrix"], axis=0)  # shape (30,)
        std_drci_by_pos = np.std(ds["drci_cold_matrix"], axis=0)
        sem_drci_by_pos = std_drci_by_pos / np.sqrt(n_trials)

        # Bin into early/mid/late
        early = mean_drci_by_pos[:10]   # positions 1-10
        mid = mean_drci_by_pos[10:20]   # positions 11-20
        late = mean_drci_by_pos[20:30]  # positions 21-30

        # Per-trial bin means for ANOVA
        early_trials = np.mean(ds["drci_cold_matrix"][:, :10], axis=1)
        mid_trials = np.mean(ds["drci_cold_matrix"][:, 10:20], axis=1)
        late_trials = np.mean(ds["drci_cold_matrix"][:, 20:30], axis=1)

        # One-way ANOVA across bins
        f_stat, p_value = stats.f_oneway(early_trials, mid_trials, late_trials)

        # Linear trend: Spearman correlation of position vs ΔRCI
        positions = np.arange(1, n_prompts + 1)
        rho, p_trend = stats.spearmanr(positions, mean_drci_by_pos)

        # Effect size: difference between late and early
        late_minus_early = np.mean(late) - np.mean(early)

        results[key] = {
            "label": label,
            "domain": domain,
            "mean_drci_by_pos": mean_drci_by_pos,
            "std_drci_by_pos": std_drci_by_pos,
            "sem_drci_by_pos": sem_drci_by_pos,
            "early_mean": np.mean(early),
            "mid_mean": np.mean(mid),
            "late_mean": np.mean(late),
            "anova_f": f_stat,
            "anova_p": p_value,
            "trend_rho": rho,
            "trend_p": p_trend,
            "late_minus_early": late_minus_early,
        }

        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        trend_sig = "***" if p_trend < 0.001 else "**" if p_trend < 0.01 else "*" if p_trend < 0.05 else "ns"

        print(f"\n  {label} ({domain}):")
        print(f"    Early (1-10):  {np.mean(early):.4f}")
        print(f"    Mid   (11-20): {np.mean(mid):.4f}")
        print(f"    Late  (21-30): {np.mean(late):.4f}")
        print(f"    Late - Early:  {late_minus_early:+.4f}")
        print(f"    ANOVA: F={f_stat:.2f}, p={p_value:.4e} {sig}")
        print(f"    Trend: rho={rho:.3f}, p={p_trend:.4e} {trend_sig}")

    return results


def plot_position_drci(datasets, results, domain_filter=None, title_suffix=""):
    """Plot ΔRCI by position for multiple models."""
    filtered = [ds for ds in datasets if domain_filter is None or ds["domain"] == domain_filter]
    if not filtered:
        return

    domain_label = domain_filter.title() if domain_filter else "All"
    n_models = len(filtered)

    # Color maps
    colors_closed = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    colors_open = ['#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a']

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

    # Top: Individual model curves
    ax1 = axes[0]
    positions = np.arange(1, 31)

    for i, ds in enumerate(filtered):
        key = ds["key"]
        r = results[key]
        color = colors_closed[i % len(colors_closed)] if ds["type"] == "closed" else colors_open[i % len(colors_open)]
        linestyle = '-' if ds["type"] == "closed" else '--'

        ax1.plot(positions, r["mean_drci_by_pos"], label=f'{r["label"]}',
                 color=color, linewidth=1.5, linestyle=linestyle, alpha=0.8)
        ax1.fill_between(positions,
                         r["mean_drci_by_pos"] - r["sem_drci_by_pos"],
                         r["mean_drci_by_pos"] + r["sem_drci_by_pos"],
                         alpha=0.1, color=color)

    # Vertical lines for bins
    ax1.axvline(x=10.5, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(x=20.5, color='gray', linestyle=':', alpha=0.5)
    ax1.text(5.5, ax1.get_ylim()[1] * 0.95 if ax1.get_ylim()[1] > 0 else 0.5, 'EARLY', ha='center', fontsize=9, color='gray')
    ax1.text(15.5, ax1.get_ylim()[1] * 0.95 if ax1.get_ylim()[1] > 0 else 0.5, 'MID', ha='center', fontsize=9, color='gray')
    ax1.text(25.5, ax1.get_ylim()[1] * 0.95 if ax1.get_ylim()[1] > 0 else 0.5, 'LATE', ha='center', fontsize=9, color='gray')

    ax1.set_xlabel('Prompt Position', fontsize=12)
    ax1.set_ylabel('ΔRCI (1.0 - cold alignment)', fontsize=12)
    ax1.set_title(f'Position-Dependent ΔRCI — {domain_label} Domain{title_suffix}', fontsize=14)
    ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)
    ax1.set_xlim(1, 30)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax1.grid(True, alpha=0.3)

    # Bottom: Grand mean across all models
    ax2 = axes[1]
    all_drci = np.array([results[ds["key"]]["mean_drci_by_pos"] for ds in filtered])
    grand_mean = np.mean(all_drci, axis=0)
    grand_sem = np.std(all_drci, axis=0) / np.sqrt(n_models)

    ax2.plot(positions, grand_mean, color='black', linewidth=2, label='Grand Mean')
    ax2.fill_between(positions, grand_mean - grand_sem, grand_mean + grand_sem,
                     alpha=0.2, color='black')
    ax2.axvline(x=10.5, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(x=20.5, color='gray', linestyle=':', alpha=0.5)

    # Fit linear trend to grand mean
    slope, intercept, r_val, p_val, std_err = stats.linregress(positions, grand_mean)
    trend_line = slope * positions + intercept
    ax2.plot(positions, trend_line, color='red', linestyle='--', linewidth=1.5,
             label=f'Linear trend (slope={slope:.4f}, r={r_val:.3f}, p={p_val:.3e})')

    ax2.set_xlabel('Prompt Position', fontsize=12)
    ax2.set_ylabel('Grand Mean ΔRCI', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.set_xlim(1, 30)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"position_drci_{domain_label.lower()}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {OUTPUT_DIR}/{fname}")


# ============================================================================
# INSIGHT 4: DISRUPTION SENSITIVITY
# ============================================================================

def analyze_disruption_sensitivity(datasets):
    """Compare SCRAMBLED vs COLD to measure disruption sensitivity."""
    print("\n" + "=" * 70)
    print("INSIGHT 4: DISRUPTION SENSITIVITY (SCRAMBLED vs COLD)")
    print("=" * 70)
    print("  Disruption Sensitivity = ΔRCI_scrambled - ΔRCI_cold")
    print("  > 0: wrong-order context HURTS more than no context")
    print("  < 0: wrong-order context HELPS (any context > no context)")
    print("  ≈ 0: order doesn't matter beyond having/not-having context")

    results = {}

    for ds in datasets:
        key = ds["key"]
        label = ds["label"]
        domain = ds["domain"]
        n_trials = ds["n_trials"]

        # Per-trial overall ΔRCI
        trial_drci_cold = np.mean(ds["drci_cold_matrix"], axis=1)       # (n_trials,)
        trial_drci_scrambled = np.mean(ds["drci_scrambled_matrix"], axis=1)

        # Disruption sensitivity per trial
        disruption = trial_drci_scrambled - trial_drci_cold  # (n_trials,)
        mean_disruption = np.mean(disruption)
        std_disruption = np.std(disruption)
        sem_disruption = std_disruption / np.sqrt(n_trials)

        # One-sample t-test: is disruption significantly different from 0?
        t_stat, p_value = stats.ttest_1samp(disruption, 0)

        # Per-position disruption sensitivity
        pos_drci_cold = np.mean(ds["drci_cold_matrix"], axis=0)       # (30,)
        pos_drci_scrambled = np.mean(ds["drci_scrambled_matrix"], axis=0)
        pos_disruption = pos_drci_scrambled - pos_drci_cold  # (30,)

        # Paired t-test: cold vs scrambled at trial level
        t_paired, p_paired = stats.ttest_rel(trial_drci_cold, trial_drci_scrambled)

        results[key] = {
            "label": label,
            "domain": domain,
            "mean_drci_cold": np.mean(trial_drci_cold),
            "mean_drci_scrambled": np.mean(trial_drci_scrambled),
            "mean_disruption": mean_disruption,
            "std_disruption": std_disruption,
            "sem_disruption": sem_disruption,
            "t_stat": t_stat,
            "p_value": p_value,
            "pos_disruption": pos_disruption,
            "pos_drci_cold": pos_drci_cold,
            "pos_drci_scrambled": pos_drci_scrambled,
        }

        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        direction = "HARMFUL" if mean_disruption > 0 else "HELPFUL" if mean_disruption < 0 else "NEUTRAL"

        print(f"\n  {label} ({domain}):")
        print(f"    ΔRCI_cold:      {np.mean(trial_drci_cold):.4f}")
        print(f"    ΔRCI_scrambled: {np.mean(trial_drci_scrambled):.4f}")
        print(f"    Disruption:     {mean_disruption:+.4f} ± {std_disruption:.4f}")
        print(f"    t={t_stat:.3f}, p={p_value:.4e} {sig}")
        print(f"    Wrong-order context is: {direction}")

    return results


def plot_disruption_sensitivity(datasets, results):
    """Plot disruption sensitivity comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Separate by domain
    for ax_idx, domain in enumerate(["philosophy", "medical"]):
        ax = axes[ax_idx]
        domain_data = [(ds["key"], results[ds["key"]]) for ds in datasets
                       if ds["domain"] == domain and ds["key"] in results]

        if not domain_data:
            ax.text(0.5, 0.5, f'No {domain} data', ha='center', va='center')
            continue

        # Sort by disruption sensitivity
        domain_data.sort(key=lambda x: x[1]["mean_disruption"])

        labels = [r["label"] for _, r in domain_data]
        disruptions = [r["mean_disruption"] for _, r in domain_data]
        sems = [r["sem_disruption"] for _, r in domain_data]
        p_values = [r["p_value"] for _, r in domain_data]

        colors = ['#d62728' if d > 0 else '#2ca02c' for d in disruptions]
        y_pos = np.arange(len(labels))

        bars = ax.barh(y_pos, disruptions, xerr=sems, color=colors, alpha=0.7,
                       edgecolor='black', linewidth=0.5, capsize=3)

        # Add significance markers
        for i, (d, p) in enumerate(zip(disruptions, p_values)):
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            offset = 0.005 if d >= 0 else -0.005
            ax.text(d + offset, i, sig, va='center',
                    ha='left' if d >= 0 else 'right', fontsize=10, fontweight='bold')

        ax.axvline(x=0, color='black', linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel('Disruption Sensitivity\n(ΔRCI_scrambled − ΔRCI_cold)', fontsize=11)
        ax.set_title(f'{domain.title()} Domain', fontsize=13)
        ax.grid(True, alpha=0.3, axis='x')

        # Add interpretive labels
        xlim = ax.get_xlim()
        ax.text(xlim[0] + 0.002, len(labels) - 0.5,
                '← Wrong order HELPS', fontsize=8, color='#2ca02c', ha='left')
        ax.text(xlim[1] - 0.002, len(labels) - 0.5,
                'Wrong order HURTS →', fontsize=8, color='#d62728', ha='right')

    plt.suptitle('Disruption Sensitivity: Does Wrong-Order Context Help or Hurt?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "disruption_sensitivity.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {OUTPUT_DIR}/disruption_sensitivity.png")


def plot_position_disruption(datasets, disruption_results):
    """Plot per-position disruption sensitivity."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    positions = np.arange(1, 31)

    for ax_idx, domain in enumerate(["philosophy", "medical"]):
        ax = axes[ax_idx]
        domain_ds = [ds for ds in datasets if ds["domain"] == domain]

        if not domain_ds:
            continue

        for ds in domain_ds:
            key = ds["key"]
            if key not in disruption_results:
                continue
            r = disruption_results[key]
            ax.plot(positions, r["pos_disruption"], label=r["label"], alpha=0.6, linewidth=1)

        # Grand mean
        all_pos = [disruption_results[ds["key"]]["pos_disruption"]
                   for ds in domain_ds if ds["key"] in disruption_results]
        if all_pos:
            grand = np.mean(all_pos, axis=0)
            ax.plot(positions, grand, color='black', linewidth=2.5, label='Grand Mean')

        ax.axhline(y=0, color='red', linewidth=1, linestyle='--')
        ax.axvline(x=10.5, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(x=20.5, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Prompt Position', fontsize=11)
        ax.set_ylabel('Disruption Sensitivity\n(ΔRCI_scram − ΔRCI_cold)', fontsize=10)
        ax.set_title(f'{domain.title()} Domain', fontsize=13)
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
        ax.set_xlim(1, 30)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Per-Position Disruption Sensitivity', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "position_disruption.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/position_disruption.png")


# ============================================================================
# SUMMARY TABLE + CSV EXPORT
# ============================================================================

def print_summary_table(position_results, disruption_results):
    """Print combined summary table."""
    print("\n" + "=" * 70)
    print("COMBINED SUMMARY TABLE")
    print("=" * 70)

    header = f"{'Model':<22} {'Domain':<10} {'Early':<8} {'Mid':<8} {'Late':<8} {'L-E':<8} {'ANOVA p':<10} {'Disrupt':<10} {'Disrupt p':<10}"
    print(header)
    print("-" * len(header))

    for key in position_results:
        pr = position_results[key]
        dr = disruption_results.get(key, {})
        disrupt = dr.get("mean_disruption", float('nan'))
        disrupt_p = dr.get("p_value", float('nan'))

        print(f"{pr['label']:<22} {pr['domain']:<10} "
              f"{pr['early_mean']:<8.4f} {pr['mid_mean']:<8.4f} {pr['late_mean']:<8.4f} "
              f"{pr['late_minus_early']:<+8.4f} {pr['anova_p']:<10.2e} "
              f"{disrupt:<+10.4f} {disrupt_p:<10.2e}")


def export_csv(datasets, position_results, disruption_results):
    """Export per-position data to CSV."""
    csv_path = os.path.join(OUTPUT_DIR, "position_drci_data.csv")
    with open(csv_path, 'w') as f:
        f.write("model,domain,type,position,mean_drci_cold,sem_drci_cold,"
                "mean_drci_scrambled,disruption_sensitivity\n")
        for ds in datasets:
            key = ds["key"]
            if key not in position_results:
                continue
            pr = position_results[key]
            dr = disruption_results.get(key, {})
            pos_scram = dr.get("pos_drci_scrambled", np.zeros(30))

            for i in range(30):
                f.write(f"{pr['label']},{ds['domain']},{ds['type']},"
                        f"{i+1},{pr['mean_drci_by_pos'][i]:.6f},{pr['sem_drci_by_pos'][i]:.6f},"
                        f"{pos_scram[i]:.6f},{dr.get('pos_disruption', np.zeros(30))[i]:.6f}\n")

    print(f"\n  Exported: {csv_path}")

    # Also export summary
    summary_path = os.path.join(OUTPUT_DIR, "position_analysis_summary.csv")
    with open(summary_path, 'w') as f:
        f.write("model,domain,type,early_drci,mid_drci,late_drci,late_minus_early,"
                "anova_f,anova_p,trend_rho,trend_p,"
                "disruption_mean,disruption_std,disruption_p\n")
        for ds in datasets:
            key = ds["key"]
            if key not in position_results:
                continue
            pr = position_results[key]
            dr = disruption_results.get(key, {})
            f.write(f"{pr['label']},{ds['domain']},{ds['type']},"
                    f"{pr['early_mean']:.6f},{pr['mid_mean']:.6f},{pr['late_mean']:.6f},"
                    f"{pr['late_minus_early']:.6f},"
                    f"{pr['anova_f']:.4f},{pr['anova_p']:.6e},"
                    f"{pr['trend_rho']:.4f},{pr['trend_p']:.6e},"
                    f"{dr.get('mean_disruption', float('nan')):.6f},"
                    f"{dr.get('std_disruption', float('nan')):.6f},"
                    f"{dr.get('p_value', float('nan')):.6e}\n")

    print(f"  Exported: {summary_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("MCH POSITION-DEPENDENT ΔRCI + DISRUPTION SENSITIVITY ANALYSIS")
    print("=" * 70)

    # Load all datasets
    print("\nLoading datasets...")
    datasets = []
    for key, info in DATA_SOURCES.items():
        ds = load_dataset(key, info)
        if ds:
            datasets.append(ds)
            print(f"  Loaded {key}: {ds['n_trials']} trials, {ds['n_prompts']} prompts")

    print(f"\nTotal datasets loaded: {len(datasets)}")
    phil_count = sum(1 for ds in datasets if ds["domain"] == "philosophy")
    med_count = sum(1 for ds in datasets if ds["domain"] == "medical")
    print(f"  Philosophy: {phil_count} models")
    print(f"  Medical:    {med_count} models")

    # INSIGHT 1: Position-dependent ΔRCI
    position_results = analyze_position_drci(datasets)

    # Generate plots
    print("\nGenerating position plots...")
    plot_position_drci(datasets, position_results, domain_filter="philosophy",
                       title_suffix=" (Consciousness)")
    plot_position_drci(datasets, position_results, domain_filter="medical",
                       title_suffix=" (STEMI Case)")

    # INSIGHT 4: Disruption sensitivity
    disruption_results = analyze_disruption_sensitivity(datasets)

    # Generate disruption plots
    print("\nGenerating disruption plots...")
    plot_disruption_sensitivity(datasets, disruption_results)
    plot_position_disruption(datasets, disruption_results)

    # Summary
    print_summary_table(position_results, disruption_results)

    # Export CSV
    export_csv(datasets, position_results, disruption_results)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
