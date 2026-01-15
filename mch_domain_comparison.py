"""
MCH Cross-Domain Comparison: Philosophy vs Medical Reasoning
Compares ΔRCI patterns across domains for each model

Author: Dr. Laxman M M, MBBS
Date: 2026-01-12
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Fix Unicode encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

PHILOSOPHY_DIR = "C:/Users/barla/mch_experiments"
MEDICAL_DIR = "C:/Users/barla/mch_experiments/medical_results"
OUTPUT_DIR = "C:/Users/barla/mch_experiments/domain_comparison"

PHILOSOPHY_FILES = {
    'GPT-4o-mini': 'mch_results_gpt4o_mini_n100_merged.json',
    'GPT-4o': 'mch_results_gpt4o_100trials.json',
    'Gemini Flash': 'mch_results_gemini_flash_100trials.json',
    'Gemini Pro': 'mch_results_gemini_pro_100trials.json',
    'Claude Haiku': 'mch_results_claude_haiku_100trials.json',
    'Claude Opus': 'mch_results_claude_opus_100trials.json'
}

MEDICAL_FILES = {
    'GPT-4o-mini': 'mch_results_gpt4o_mini_medical_50trials.json',
    'GPT-4o': 'mch_results_gpt4o_medical_50trials.json',
    'Gemini Flash': 'mch_results_gemini_flash_medical_50trials.json',
    'Gemini Pro': 'mch_results_gemini_pro_medical_50trials.json',
    'Claude Haiku': 'mch_results_claude_haiku_medical_50trials.json',
    'Claude Opus': 'mch_results_claude_opus_medical_50trials.json'
}

MODEL_INFO = {
    'GPT-4o-mini': {'vendor': 'OpenAI', 'tier': 'Efficient', 'color': '#10a37f'},
    'GPT-4o': {'vendor': 'OpenAI', 'tier': 'Flagship', 'color': '#0d8a6f'},
    'Gemini Flash': {'vendor': 'Google', 'tier': 'Efficient', 'color': '#4285f4'},
    'Gemini Pro': {'vendor': 'Google', 'tier': 'Flagship', 'color': '#1a73e8'},
    'Claude Haiku': {'vendor': 'Anthropic', 'tier': 'Efficient', 'color': '#d4a574'},
    'Claude Opus': {'vendor': 'Anthropic', 'tier': 'Flagship', 'color': '#c9956c'}
}

# =============================================================================
# DATA LOADING
# =============================================================================

def load_philosophy_data(model_name, filename):
    """Load philosophy domain data (original structure)."""
    filepath = os.path.join(PHILOSOPHY_DIR, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        results = json.load(f)

    trials = results['trials']
    delta_rci_cold = [t['controls']['cold']['delta_rci'] for t in trials]
    return np.array(delta_rci_cold)

def load_medical_data(model_name, filename):
    """Load medical domain data (new structure)."""
    filepath = os.path.join(MEDICAL_DIR, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        results = json.load(f)

    trials = results['trials']
    delta_rci_cold = [t['delta_rci']['cold'] for t in trials]
    return np.array(delta_rci_cold)

def load_all_data():
    """Load data from both domains."""
    data = {}

    print("\nLoading Philosophy domain data...")
    for model_name, filename in PHILOSOPHY_FILES.items():
        try:
            drci = load_philosophy_data(model_name, filename)
            if model_name not in data:
                data[model_name] = {}
            data[model_name]['philosophy'] = drci
            print(f"  {model_name}: {len(drci)} trials")
        except Exception as e:
            print(f"  ERROR loading {model_name}: {e}")

    print("\nLoading Medical domain data...")
    for model_name, filename in MEDICAL_FILES.items():
        try:
            drci = load_medical_data(model_name, filename)
            if model_name not in data:
                data[model_name] = {}
            data[model_name]['medical'] = drci
            print(f"  {model_name}: {len(drci)} trials")
        except Exception as e:
            print(f"  ERROR loading {model_name}: {e}")

    return data

# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def run_domain_comparison(data):
    """Compare ΔRCI between domains for each model."""
    print("\n" + "=" * 70)
    print("CROSS-DOMAIN COMPARISON: PHILOSOPHY vs MEDICAL")
    print("=" * 70)

    results = {}

    print("\nModel               | Philosophy      | Medical         | Diff     | p-value  | Consistent?")
    print("-" * 95)

    for model_name in MODEL_INFO.keys():
        if model_name not in data:
            continue
        if 'philosophy' not in data[model_name] or 'medical' not in data[model_name]:
            continue

        phil = data[model_name]['philosophy']
        med = data[model_name]['medical']

        mean_phil = np.mean(phil)
        mean_med = np.mean(med)
        std_phil = np.std(phil)
        std_med = np.std(med)

        # Independent samples t-test
        t_stat, p_val = stats.ttest_ind(phil, med)

        # Determine patterns
        phil_pattern = "CONVERGENT" if mean_phil > 0.01 else ("SOVEREIGN" if mean_phil < -0.01 else "NEUTRAL")
        med_pattern = "CONVERGENT" if mean_med > 0.01 else ("SOVEREIGN" if mean_med < -0.01 else "NEUTRAL")

        # Check consistency (same direction)
        consistent = "YES" if (mean_phil * mean_med > 0) or (abs(mean_phil) < 0.01 and abs(mean_med) < 0.01) else "NO"

        results[model_name] = {
            'philosophy_mean': mean_phil,
            'philosophy_std': std_phil,
            'medical_mean': mean_med,
            'medical_std': std_med,
            'difference': mean_phil - mean_med,
            't_stat': t_stat,
            'p_value': p_val,
            'consistent': consistent,
            'phil_pattern': phil_pattern,
            'med_pattern': med_pattern
        }

        diff = mean_phil - mean_med
        print(f"{model_name:20s}| {mean_phil:+.4f}±{std_phil:.3f}  | {mean_med:+.4f}±{std_med:.3f}  | {diff:+.4f} | {p_val:.4f}   | {consistent}")

    # Overall correlation between domains
    print("\n" + "-" * 70)
    phil_means = [results[m]['philosophy_mean'] for m in results]
    med_means = [results[m]['medical_mean'] for m in results]

    r, p = stats.pearsonr(phil_means, med_means)
    print(f"\nCross-domain correlation (6 models): r = {r:.3f}, p = {p:.4f}")

    if r > 0.7:
        print(">>> STRONG POSITIVE CORRELATION: Domain-invariant patterns!")
    elif r > 0.4:
        print(">>> MODERATE CORRELATION: Some domain-invariance")
    else:
        print(">>> WEAK CORRELATION: Domain-specific patterns")

    return results

# =============================================================================
# FIGURE GENERATION
# =============================================================================

def create_domain_comparison_figure(data, results, output_dir):
    """Create side-by-side comparison of Philosophy vs Medical domains."""
    print("\nGenerating cross-domain comparison figure...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    models = list(results.keys())
    x = np.arange(len(models))
    width = 0.35

    # Panel 1: Bar chart comparison
    ax1 = axes[0]

    phil_means = [results[m]['philosophy_mean'] for m in models]
    phil_stds = [results[m]['philosophy_std'] for m in models]
    med_means = [results[m]['medical_mean'] for m in models]
    med_stds = [results[m]['medical_std'] for m in models]

    bars1 = ax1.bar(x - width/2, phil_means, width, yerr=phil_stds,
                    label='Philosophy', color='#8B4513', alpha=0.7, capsize=3)
    bars2 = ax1.bar(x + width/2, med_means, width, yerr=med_stds,
                    label='Medical', color='#DC143C', alpha=0.7, capsize=3)

    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    ax1.set_ylabel('ΔRCI (True - Cold)', fontsize=12)
    ax1.set_title('ΔRCI by Domain and Model', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add significance markers
    for i, model in enumerate(models):
        p = results[model]['p_value']
        if p < 0.05:
            y_max = max(abs(phil_means[i]), abs(med_means[i])) + max(phil_stds[i], med_stds[i]) + 0.01
            ax1.text(i, y_max, '*', ha='center', fontsize=14, fontweight='bold')

    # Panel 2: Scatter plot - Philosophy vs Medical
    ax2 = axes[1]

    colors = [MODEL_INFO[m]['color'] for m in models]

    for i, model in enumerate(models):
        ax2.scatter(phil_means[i], med_means[i],
                   c=colors[i], s=200, edgecolor='black', linewidth=1.5,
                   label=model, zorder=5)

    # Add diagonal line (perfect correlation)
    lims = [min(min(phil_means), min(med_means)) - 0.02,
            max(max(phil_means), max(med_means)) + 0.02]
    ax2.plot(lims, lims, 'k--', alpha=0.5, label='Perfect agreement')
    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(x=0, color='gray', linestyle=':', alpha=0.5)

    ax2.set_xlabel('Philosophy ΔRCI', fontsize=12)
    ax2.set_ylabel('Medical ΔRCI', fontsize=12)
    ax2.set_title('Cross-Domain Pattern Consistency', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Add correlation annotation
    r = stats.pearsonr(phil_means, med_means)[0]
    ax2.text(0.95, 0.05, f'r = {r:.3f}', transform=ax2.transAxes,
             fontsize=12, ha='right', bbox=dict(boxstyle='round', facecolor='wheat'))

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'domain_comparison_philosophy_medical.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")

    return filepath

def create_heatmap_figure(data, results, output_dir):
    """Create heatmap showing ΔRCI patterns across models and domains."""
    print("\nGenerating heatmap figure...")

    models = list(results.keys())

    # Create matrix
    matrix = np.array([
        [results[m]['philosophy_mean'] for m in models],
        [results[m]['medical_mean'] for m in models]
    ])

    fig, ax = plt.subplots(figsize=(12, 4))

    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-0.15, vmax=0.15)

    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks([0, 1])
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(['Philosophy', 'Medical'], fontsize=11)

    # Add text annotations
    for i in range(2):
        for j in range(len(models)):
            text = f'{matrix[i, j]:+.3f}'
            color = 'white' if abs(matrix[i, j]) > 0.08 else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=10, fontweight='bold')

    plt.colorbar(im, ax=ax, label='ΔRCI')
    ax.set_title('MCH ΔRCI Patterns: Philosophy vs Medical Reasoning', fontsize=14, fontweight='bold')

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'domain_heatmap.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")

    return filepath

# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(results, output_dir):
    """Generate cross-domain comparison report."""
    report_path = os.path.join(output_dir, 'mch_domain_comparison_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MCH CROSS-DOMAIN COMPARISON REPORT\n")
        f.write("Philosophy vs Medical Clinical Reasoning\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Generated: 2026-01-12\n")
        f.write(f"Philosophy: 100 trials per model\n")
        f.write(f"Medical: 50 trials per model\n\n")

        f.write("=" * 80 + "\n")
        f.write("SUMMARY TABLE\n")
        f.write("=" * 80 + "\n\n")

        f.write("Model           | Philosophy ΔRCI | Medical ΔRCI  | Difference | p-value  | Pattern Match\n")
        f.write("-" * 90 + "\n")

        for model, r in results.items():
            f.write(f"{model:15s} | {r['philosophy_mean']:+.4f}        | {r['medical_mean']:+.4f}       | {r['difference']:+.4f}    | {r['p_value']:.4f}   | {r['consistent']}\n")

        f.write("\n")

        # Cross-domain correlation
        phil_means = [results[m]['philosophy_mean'] for m in results]
        med_means = [results[m]['medical_mean'] for m in results]
        r, p = stats.pearsonr(phil_means, med_means)

        f.write("=" * 80 + "\n")
        f.write("CROSS-DOMAIN CORRELATION\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Pearson r = {r:.4f}, p = {p:.4f}\n\n")

        if r > 0.7:
            f.write("INTERPRETATION: Strong positive correlation indicates domain-invariant MCH patterns.\n")
            f.write("Models show consistent convergent/sovereign tendencies across reasoning domains.\n")
        elif r > 0.4:
            f.write("INTERPRETATION: Moderate correlation suggests partial domain-invariance.\n")
            f.write("Some models show consistent patterns, others are domain-specific.\n")
        else:
            f.write("INTERPRETATION: Weak correlation indicates domain-specific MCH patterns.\n")
            f.write("Model behavior varies significantly between philosophy and medical reasoning.\n")

        f.write("\n")

        # Pattern comparison
        f.write("=" * 80 + "\n")
        f.write("PATTERN COMPARISON BY MODEL\n")
        f.write("=" * 80 + "\n\n")

        for model, r_data in results.items():
            f.write(f"{model}:\n")
            f.write(f"  Philosophy: {r_data['phil_pattern']} (ΔRCI = {r_data['philosophy_mean']:+.4f})\n")
            f.write(f"  Medical:    {r_data['med_pattern']} (ΔRCI = {r_data['medical_mean']:+.4f})\n")
            f.write(f"  Consistent: {r_data['consistent']}\n\n")

        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"\nReport saved: {report_path}")
    return report_path

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 70)
    print("MCH CROSS-DOMAIN COMPARISON")
    print("Philosophy vs Medical Clinical Reasoning")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    data = load_all_data()

    if len(data) < 6:
        print(f"\nWARNING: Only loaded {len(data)} models. Expected 6.")

    # Run comparison
    results = run_domain_comparison(data)

    if len(results) == 0:
        print("\nERROR: No complete data for comparison.")
        return

    # Generate figures
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    create_domain_comparison_figure(data, results, OUTPUT_DIR)
    create_heatmap_figure(data, results, OUTPUT_DIR)

    # Generate report
    generate_report(results, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("CROSS-DOMAIN COMPARISON COMPLETE!")
    print("=" * 70)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
