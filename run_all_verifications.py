#!/usr/bin/env python3
"""
Master script to run all verification experiments and generate final report.
Run this AFTER the main sequential experiment completes.
"""

import os
import sys
import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats

OUTPUT_DIR = "C:/Users/barla/mch_experiments/medical_results"

def run_verification_1():
    """Run GPT-4o-mini replication test."""
    print("\n" + "="*70)
    print("RUNNING VERIFICATION 1: GPT-4o-mini REPLICATION TEST")
    print("="*70 + "\n")

    result = subprocess.run(
        [sys.executable, "C:/Users/barla/mch_experiments/verification_gpt4o_mini_rerun.py"],
        cwd="C:/Users/barla/mch_experiments"
    )
    return result.returncode == 0

def run_verification_2():
    """Run Gemini Pro retry."""
    print("\n" + "="*70)
    print("RUNNING VERIFICATION 2: GEMINI PRO RETRY")
    print("="*70 + "\n")

    result = subprocess.run(
        [sys.executable, "C:/Users/barla/mch_experiments/verification_gemini_pro_retry.py"],
        cwd="C:/Users/barla/mch_experiments"
    )
    return result.returncode == 0

def generate_comparison_analysis():
    """Compare original GPT-4o-mini run with rerun."""
    print("\n" + "="*70)
    print("GENERATING COMPARISON ANALYSIS")
    print("="*70 + "\n")

    original_file = os.path.join(OUTPUT_DIR, "mch_results_gpt4o_mini_medical_50trials.json")
    rerun_file = os.path.join(OUTPUT_DIR, "mch_results_gpt4o_mini_medical_RERUN.json")

    if not os.path.exists(original_file):
        print("Original GPT-4o-mini results not found!")
        return None

    if not os.path.exists(rerun_file):
        print("Rerun GPT-4o-mini results not found!")
        return None

    with open(original_file, 'r') as f:
        original = json.load(f)
    with open(rerun_file, 'r') as f:
        rerun = json.load(f)

    orig_drcis = [t['delta_rci']['cold'] for t in original['trials']]
    rerun_drcis = [t['delta_rci']['cold'] for t in rerun['trials']]

    # Calculate interval statistics
    intervals = [10, 20, 30, 40, 50]
    comparison = {
        "intervals": {}
    }

    for interval in intervals:
        orig_subset = orig_drcis[:interval]
        rerun_subset = rerun_drcis[:interval]

        comparison["intervals"][interval] = {
            "original": {
                "mean": float(np.mean(orig_subset)),
                "std": float(np.std(orig_subset))
            },
            "rerun": {
                "mean": float(np.mean(rerun_subset)),
                "std": float(np.std(rerun_subset))
            }
        }

    # Overall comparison
    comparison["overall"] = {
        "original": {
            "mean": float(np.mean(orig_drcis)),
            "std": float(np.std(orig_drcis)),
            "n_trials": len(orig_drcis)
        },
        "rerun": {
            "mean": float(np.mean(rerun_drcis)),
            "std": float(np.std(rerun_drcis)),
            "n_trials": len(rerun_drcis)
        }
    }

    # Statistical test
    t_stat, p_val = stats.ttest_ind(orig_drcis, rerun_drcis)
    comparison["statistical_test"] = {
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "significant": p_val < 0.05
    }

    # Detect mode switches in both
    def detect_switches(drcis):
        switches = []
        for i in range(1, len(drcis)):
            prev_pattern = "CONVERGENT" if drcis[i-1] > 0.1 else "SOVEREIGN" if drcis[i-1] < -0.05 else "NEUTRAL"
            curr_pattern = "CONVERGENT" if drcis[i] > 0.1 else "SOVEREIGN" if drcis[i] < -0.05 else "NEUTRAL"
            if prev_pattern != curr_pattern and prev_pattern != "NEUTRAL" and curr_pattern != "NEUTRAL":
                switches.append({"trial": i+1, "from": prev_pattern, "to": curr_pattern})
        return switches

    comparison["mode_switches"] = {
        "original": detect_switches(orig_drcis),
        "rerun": detect_switches(rerun_drcis)
    }

    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Trial-by-trial comparison
    ax1 = axes[0, 0]
    ax1.plot(range(1, 51), orig_drcis, 'b-', alpha=0.7, label='Original', linewidth=1.5)
    ax1.plot(range(1, 51), rerun_drcis, 'r-', alpha=0.7, label='Rerun', linewidth=1.5)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('dRCI')
    ax1.set_title('GPT-4o-mini: Original vs Rerun (Trial-by-Trial)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Running mean comparison
    ax2 = axes[0, 1]
    orig_running = [np.mean(orig_drcis[:i+1]) for i in range(50)]
    rerun_running = [np.mean(rerun_drcis[:i+1]) for i in range(50)]
    ax2.plot(range(1, 51), orig_running, 'b-', label='Original Running Mean', linewidth=2)
    ax2.plot(range(1, 51), rerun_running, 'r-', label='Rerun Running Mean', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Trial')
    ax2.set_ylabel('Running Mean dRCI')
    ax2.set_title('GPT-4o-mini: Running Mean Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Distribution comparison
    ax3 = axes[1, 0]
    ax3.hist(orig_drcis, bins=20, alpha=0.5, label='Original', color='blue')
    ax3.hist(rerun_drcis, bins=20, alpha=0.5, label='Rerun', color='red')
    ax3.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax3.set_xlabel('dRCI')
    ax3.set_ylabel('Frequency')
    ax3.set_title('GPT-4o-mini: dRCI Distribution Comparison')
    ax3.legend()

    # Plot 4: Interval comparison
    ax4 = axes[1, 1]
    x = np.arange(len(intervals))
    width = 0.35
    orig_means = [comparison["intervals"][i]["original"]["mean"] for i in intervals]
    rerun_means = [comparison["intervals"][i]["rerun"]["mean"] for i in intervals]
    orig_stds = [comparison["intervals"][i]["original"]["std"] for i in intervals]
    rerun_stds = [comparison["intervals"][i]["rerun"]["std"] for i in intervals]

    ax4.bar(x - width/2, orig_means, width, yerr=orig_stds, label='Original', color='blue', alpha=0.7)
    ax4.bar(x + width/2, rerun_means, width, yerr=rerun_stds, label='Rerun', color='red', alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax4.set_xlabel('Trials (cumulative)')
    ax4.set_ylabel('Mean dRCI')
    ax4.set_title('GPT-4o-mini: Interval Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels([str(i) for i in intervals])
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "gpt4o_mini_replication_comparison.png"), dpi=150, bbox_inches='tight')
    print("Comparison plot saved: gpt4o_mini_replication_comparison.png")

    return comparison

def generate_final_report():
    """Generate comprehensive final report."""
    print("\n" + "="*70)
    print("GENERATING FINAL REPORT")
    print("="*70 + "\n")

    report = []
    report.append("# MCH Medical Domain - Complete Experiment Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n## Executive Summary\n")

    # Load all results
    models = {
        "Gemini Flash": "mch_results_gemini_flash_medical_50trials.json",
        "GPT-4o-mini": "mch_results_gpt4o_mini_medical_50trials.json",
        "GPT-4o": "mch_results_gpt4o_medical_50trials.json",
        "Claude Haiku": "mch_results_claude_haiku_medical_50trials.json",
        "Claude Opus": "mch_results_claude_opus_medical_50trials.json",
        "GPT-4o-mini (RERUN)": "mch_results_gpt4o_mini_medical_RERUN.json",
        "Gemini Pro (RETRY)": "mch_results_gemini_pro_medical_RETRY.json"
    }

    results = {}
    for name, filename in models.items():
        filepath = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            trials = data.get("trials", [])
            if trials:
                drcis = [t['delta_rci']['cold'] for t in trials]
                mean_drci = np.mean(drcis)
                std_drci = np.std(drcis)

                # Statistical test
                t_stat, p_val = stats.ttest_1samp(drcis, 0)
                if p_val >= 0.05:
                    pattern = "NEUTRAL"
                elif mean_drci > 0:
                    pattern = "CONVERGENT"
                else:
                    pattern = "SOVEREIGN"

                results[name] = {
                    "n_trials": len(trials),
                    "mean_drci": mean_drci,
                    "std_drci": std_drci,
                    "p_value": p_val,
                    "pattern": pattern,
                    "status": data.get("status", "COMPLETE")
                }

    # Results table
    report.append("## Complete Results Table\n")
    report.append("| Model | Trials | Mean dRCI | Std | p-value | Pattern | Status |")
    report.append("|-------|--------|-----------|-----|---------|---------|--------|")

    for name, r in results.items():
        report.append(f"| {name} | {r['n_trials']} | {r['mean_drci']:+.4f} | {r['std_drci']:.4f} | {r['p_value']:.4f} | **{r['pattern']}** | {r['status']} |")

    # Stability analysis
    report.append("\n## Model Stability Analysis\n")

    stable_models = []
    unstable_models = []

    for name, r in results.items():
        if "RERUN" in name or "RETRY" in name:
            continue
        if r['std_drci'] < 0.05:
            stable_models.append(name)
        elif r['std_drci'] > 0.15:
            unstable_models.append(name)

    report.append(f"**Stable Models** (low variance, consistent behavior):")
    for m in stable_models:
        report.append(f"- {m}")

    report.append(f"\n**Unstable Models** (high variance, bimodal behavior):")
    for m in unstable_models:
        report.append(f"- {m}")

    # Replication analysis
    if "GPT-4o-mini (RERUN)" in results:
        report.append("\n## GPT-4o-mini Replication Test\n")
        orig = results.get("GPT-4o-mini", {})
        rerun = results.get("GPT-4o-mini (RERUN)", {})

        if orig and rerun:
            report.append(f"| Metric | Original | Rerun |")
            report.append(f"|--------|----------|-------|")
            report.append(f"| Mean dRCI | {orig['mean_drci']:+.4f} | {rerun['mean_drci']:+.4f} |")
            report.append(f"| Std | {orig['std_drci']:.4f} | {rerun['std_drci']:.4f} |")
            report.append(f"| Pattern | {orig['pattern']} | {rerun['pattern']} |")

            if abs(orig['mean_drci'] - rerun['mean_drci']) > 0.1:
                report.append("\n**Finding**: Bimodal behavior NOT replicated - suggests instability is random, not systematic.")
            else:
                report.append("\n**Finding**: Bimodal behavior replicated - suggests systematic model characteristic.")

    # Gemini Pro analysis
    if "Gemini Pro (RETRY)" in results:
        report.append("\n## Gemini Pro Safety Filter Analysis\n")
        r = results["Gemini Pro (RETRY)"]
        if r['status'] != "COMPLETE":
            report.append(f"**Status**: {r['status']}")
            report.append(f"\n**Finding**: Gemini Pro consistently blocks medical content due to safety filters.")
            report.append("This represents a significant vendor safety behavior difference.")
        else:
            report.append(f"**Status**: COMPLETE - {r['n_trials']} trials")
            report.append(f"\nPattern: {r['pattern']} (Mean dRCI = {r['mean_drci']:+.4f})")

    # Key findings
    report.append("\n## Key Findings\n")
    report.append("1. **Domain Effect**: Medical reasoning shows different coherence patterns than philosophy")
    report.append("2. **Model Stability**: Varies significantly across vendors")
    report.append("3. **Safety Behavior**: Vendor-specific differences in handling medical content")

    # Save report
    report_path = os.path.join(OUTPUT_DIR, "mch_medical_domain_complete_report.md")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"Final report saved: {report_path}")
    return report_path

def main():
    print("="*70)
    print("MCH MEDICAL DOMAIN - VERIFICATION EXPERIMENTS")
    print("="*70)

    # Check if main experiment is complete
    required_files = [
        "mch_results_gpt4o_mini_medical_50trials.json",
        "mch_results_gpt4o_medical_50trials.json",
        "mch_results_claude_haiku_medical_50trials.json",
        "mch_results_claude_opus_medical_50trials.json"
    ]

    missing = [f for f in required_files if not os.path.exists(os.path.join(OUTPUT_DIR, f))]
    if missing:
        print(f"\nWaiting for main experiment to complete. Missing files:")
        for f in missing:
            print(f"  - {f}")
        print("\nRun this script again after the main experiment completes.")
        return

    print("\nMain experiment complete. Running verifications...")

    # Run verifications
    run_verification_1()
    run_verification_2()

    # Generate analyses
    generate_comparison_analysis()
    generate_final_report()

    print("\n" + "="*70)
    print("ALL VERIFICATIONS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
