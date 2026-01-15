"""
Regenerate publication figures with clean titles (no version numbers)
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Configuration
DATA_DIR = "C:/Users/barla/mch_experiments"
OUTPUT_DIR = "C:/Users/barla/mch_experiments/publication_analysis"

DATA_FILES = {
    'GPT-4o-mini': 'mch_results_gpt4o_mini_n100_merged.json',
    'GPT-4o': 'mch_results_gpt4o_100trials.json',
    'Gemini Flash': 'mch_results_gemini_flash_100trials.json',
    'Gemini Pro': 'mch_results_gemini_pro_100trials.json',
    'Claude Haiku': 'mch_results_claude_haiku_100trials.json',
    'Claude Opus': 'mch_results_claude_opus_100trials.json'
}

MODEL_INFO = {
    'GPT-4o-mini': {'vendor': 'OpenAI', 'tier': 'Efficient', 'color': '#10a37f'},
    'GPT-4o': {'vendor': 'OpenAI', 'tier': 'Flagship', 'color': '#0d8a6f'},
    'Gemini Flash': {'vendor': 'Google', 'tier': 'Efficient', 'color': '#4285f4'},
    'Gemini Pro': {'vendor': 'Google', 'tier': 'Flagship', 'color': '#1a73e8'},
    'Claude Haiku': {'vendor': 'Anthropic', 'tier': 'Efficient', 'color': '#d4a574'},
    'Claude Opus': {'vendor': 'Anthropic', 'tier': 'Flagship', 'color': '#c9956c'}
}

def load_all_data():
    """Load all experiment data."""
    data = {}
    for model_name, filename in DATA_FILES.items():
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            results = json.load(f)
        trials = results['trials']
        delta_rci_cold = [t['controls']['cold']['delta_rci'] for t in trials]
        data[model_name] = {
            'delta_rci_cold': np.array(delta_rci_cold),
            'vendor': MODEL_INFO[model_name]['vendor'],
            'tier': MODEL_INFO[model_name]['tier']
        }
        print(f"  Loaded {model_name}: {len(trials)} trials")
    return data

def create_figure1(data, output_dir):
    """Figure 1: Violin plots - Response Coherence by Model"""
    print("\nGenerating Figure 1: Response Coherence by Model...")

    fig, ax = plt.subplots(figsize=(14, 8))

    models = list(data.keys())
    positions = list(range(len(models)))

    plot_data = []
    colors = []

    for model in models:
        plot_data.append(data[model]['delta_rci_cold'])
        colors.append(MODEL_INFO[model]['color'])

    parts = ax.violinplot(plot_data, positions=positions, showmeans=True, showmedians=True)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)

    # Significance markers
    for i, model in enumerate(models):
        drci = data[model]['delta_rci_cold']
        t_stat, p_val = stats.ttest_1samp(drci, 0)

        if p_val < 0.001:
            sig = "***"
        elif p_val < 0.01:
            sig = "**"
        elif p_val < 0.05:
            sig = "*"
        else:
            sig = "ns"

        y_pos = max(drci) + 0.02
        ax.text(i, y_pos, sig, ha='center', fontsize=12, fontweight='bold')

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(models, rotation=30, ha='right', fontsize=11)
    ax.set_ylabel('ΔRCI (True - Cold)', fontsize=12)
    ax.set_title('Response Coherence by Model (ΔRCI Distribution)\n(Positive = Convergent, Negative = Sovereign)', fontsize=14)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#10a37f', label='OpenAI'),
        Patch(facecolor='#4285f4', label='Google'),
        Patch(facecolor='#d4a574', label='Anthropic')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'figure1_response_coherence.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")

def create_figure2(data, output_dir):
    """Figure 2: Forest plot - Effect Sizes with 95% CI"""
    print("\nGenerating Figure 2: Effect Sizes with 95% CI...")

    fig, ax = plt.subplots(figsize=(12, 8))

    models = list(data.keys())
    y_positions = list(range(len(models)))

    means = []
    ci_lows = []
    ci_highs = []
    colors = []

    for model in models:
        drci = data[model]['delta_rci_cold']
        n = len(drci)
        mean = np.mean(drci)
        sem = np.std(drci, ddof=1) / np.sqrt(n)
        ci = stats.t.interval(0.95, df=n-1, loc=mean, scale=sem)

        means.append(mean)
        ci_lows.append(ci[0])
        ci_highs.append(ci[1])
        colors.append(MODEL_INFO[model]['color'])

    for i, (model, mean, ci_low, ci_high, color) in enumerate(zip(models, means, ci_lows, ci_highs, colors)):
        ax.plot([ci_low, ci_high], [i, i], color=color, linewidth=2, alpha=0.8)
        ax.scatter([mean], [i], color=color, s=150, zorder=5, edgecolor='black', linewidth=1)
        ax.scatter([ci_low, ci_high], [i, i], color=color, s=50, marker='|', linewidth=2)

    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(models, fontsize=11)
    ax.set_xlabel('ΔRCI (95% CI)', fontsize=12)
    ax.set_title('Effect Sizes with 95% Confidence Intervals\n(CIs crossing zero = Neutral pattern)', fontsize=14)

    # Pattern labels
    for i, model in enumerate(models):
        drci = data[model]['delta_rci_cold']
        t_stat, p_val = stats.ttest_1samp(drci, 0)
        mean = np.mean(drci)

        if p_val < 0.05:
            pattern = "CONVERGENT" if mean > 0 else "SOVEREIGN"
        else:
            pattern = "NEUTRAL"

        ax.text(ax.get_xlim()[1] + 0.01, i, pattern, va='center', fontsize=10,
                fontweight='bold' if pattern != "NEUTRAL" else 'normal')

    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(ax.get_xlim()[0] - 0.02, ax.get_xlim()[1] + 0.08)

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'figure2_effect_sizes.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")

def create_figure3(data, output_dir):
    """Figure 3: Vendor x Tier box plots"""
    print("\nGenerating Figure 3: Vendor and Tier Effects...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    df_list = []
    for model_name, model_data in data.items():
        for drci in model_data['delta_rci_cold']:
            df_list.append({
                'delta_rci': drci,
                'vendor': model_data['vendor'],
                'tier': model_data['tier'],
                'model': model_name
            })
    df = pd.DataFrame(df_list)

    # Left panel: By Vendor
    ax1 = axes[0]
    vendor_order = ['OpenAI', 'Google', 'Anthropic']
    vendor_colors = {'OpenAI': '#10a37f', 'Google': '#4285f4', 'Anthropic': '#d4a574'}

    bp1 = ax1.boxplot([df[df['vendor'] == v]['delta_rci'].values for v in vendor_order],
                       tick_labels=vendor_order, patch_artist=True)

    for patch, vendor in zip(bp1['boxes'], vendor_order):
        patch.set_facecolor(vendor_colors[vendor])
        patch.set_alpha(0.7)

    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax1.set_ylabel('ΔRCI', fontsize=12)
    ax1.set_title('ΔRCI by Vendor', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    for i, vendor in enumerate(vendor_order):
        mean = df[df['vendor'] == vendor]['delta_rci'].mean()
        ax1.scatter([i+1], [mean], color='red', s=100, marker='D', zorder=5)

    # Right panel: By Tier
    ax2 = axes[1]
    tier_order = ['Efficient', 'Flagship']
    tier_colors = {'Efficient': '#90EE90', 'Flagship': '#FFB6C1'}

    bp2 = ax2.boxplot([df[df['tier'] == t]['delta_rci'].values for t in tier_order],
                       tick_labels=tier_order, patch_artist=True)

    for patch, tier in zip(bp2['boxes'], tier_order):
        patch.set_facecolor(tier_colors[tier])
        patch.set_alpha(0.7)

    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_ylabel('ΔRCI', fontsize=12)
    ax2.set_title('ΔRCI by Tier', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    for i, tier in enumerate(tier_order):
        mean = df[df['tier'] == tier]['delta_rci'].mean()
        ax2.scatter([i+1], [mean], color='red', s=100, marker='D', zorder=5)

    fig.suptitle('Vendor and Tier Effects on Response Coherence', fontsize=14, fontweight='bold')

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'figure3_vendor_tier.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")

def main():
    print("=" * 60)
    print("REGENERATING PUBLICATION FIGURES")
    print("=" * 60)

    print("\nLoading data...")
    data = load_all_data()

    create_figure1(data, OUTPUT_DIR)
    create_figure2(data, OUTPUT_DIR)
    create_figure3(data, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("ALL FIGURES REGENERATED!")
    print("=" * 60)
    print(f"\nFiles saved to: {OUTPUT_DIR}")
    print("  - figure1_response_coherence.png")
    print("  - figure2_effect_sizes.png")
    print("  - figure3_vendor_tier.png")

if __name__ == "__main__":
    main()
