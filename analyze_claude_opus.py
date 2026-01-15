"""
MCH v8.1 Comprehensive Analysis - Claude Opus 4.5 (100 Trials)
Generates 9-panel visualization and detailed statistics
"""

import json
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
import sys

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def load_results(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_claude_opus():
    print("=" * 70)
    print("MCH v8.1 COMPREHENSIVE ANALYSIS - CLAUDE OPUS 4.5")
    print("=" * 70)

    # Load results
    results = load_results("mch_results_claude_opus_100trials.json")
    trials = results['trials']

    # Extract metrics
    true_alignments = [t['true']['alignment'] for t in trials]
    true_iqs = [t['true']['insight_quality'] for t in trials]
    entanglements = [t['true']['entanglement'] for t in trials]

    cold_alignments = [t['controls']['cold']['alignment'] for t in trials]
    scrambled_alignments = [t['controls']['scrambled']['alignment'] for t in trials]

    delta_rci_cold = [t['controls']['cold']['delta_rci'] for t in trials]
    delta_rci_scrambled = [t['controls']['scrambled']['delta_rci'] for t in trials]

    delta_iq_cold = [t['controls']['cold']['delta_iq'] for t in trials]
    delta_iq_scrambled = [t['controls']['scrambled']['delta_iq'] for t in trials]

    n = len(trials)

    # ============================================================
    # STATISTICAL ANALYSIS
    # ============================================================
    print(f"\nTotal Trials: {n}")
    print(f"Model: Claude Opus 4.5 (claude-opus-4-5-20251101)")
    print(f"Conditions: True (with history), Cold (no history), Scrambled")

    print("\n" + "=" * 70)
    print("PRIMARY HYPOTHESIS TEST: DELTA-RCI vs Cold")
    print("=" * 70)

    mean_drci = np.mean(delta_rci_cold)
    std_drci = np.std(delta_rci_cold, ddof=1)
    sem_drci = std_drci / np.sqrt(n)
    ci_95 = stats.t.interval(0.95, df=n-1, loc=mean_drci, scale=sem_drci)
    t_stat, p_val = stats.ttest_1samp(delta_rci_cold, 0)
    cohens_d = mean_drci / std_drci

    print(f"\nMean DELTA-RCI (True - Cold): {mean_drci:.4f}")
    print(f"Standard Deviation: {std_drci:.4f}")
    print(f"Standard Error: {sem_drci:.4f}")
    print(f"95% CI: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")
    print(f"\nt-statistic: {t_stat:.3f}")
    print(f"p-value: {p_val:.6f}")
    print(f"Cohen's d: {cohens_d:.3f}")

    # Classification
    if p_val < 0.05:
        if mean_drci > 0:
            classification = "CONVERGENT (Positive DELTA-RCI, p < 0.05)"
            pattern = "Convergent"
        else:
            classification = "SOVEREIGN (Negative DELTA-RCI, p < 0.05)"
            pattern = "Sovereign"
    else:
        classification = "NEUTRAL (p >= 0.05)"
        pattern = "Neutral"

    print(f"\n>>> CLASSIFICATION: {classification}")

    # Effect size interpretation
    if abs(cohens_d) < 0.2:
        effect_interp = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_interp = "small"
    elif abs(cohens_d) < 0.8:
        effect_interp = "medium"
    else:
        effect_interp = "large"
    print(f">>> Effect Size: {effect_interp} ({cohens_d:.3f})")

    # Scrambled condition
    print("\n" + "-" * 50)
    print("SECONDARY: DELTA-RCI vs Scrambled")
    print("-" * 50)

    mean_drci_s = np.mean(delta_rci_scrambled)
    std_drci_s = np.std(delta_rci_scrambled, ddof=1)
    sem_drci_s = std_drci_s / np.sqrt(n)
    ci_95_s = stats.t.interval(0.95, df=n-1, loc=mean_drci_s, scale=sem_drci_s)
    t_stat_s, p_val_s = stats.ttest_1samp(delta_rci_scrambled, 0)
    cohens_d_s = mean_drci_s / std_drci_s

    print(f"Mean DELTA-RCI (True - Scrambled): {mean_drci_s:.4f}")
    print(f"SD: {std_drci_s:.4f}, 95% CI: [{ci_95_s[0]:.4f}, {ci_95_s[1]:.4f}]")
    print(f"t({n-1}) = {t_stat_s:.3f}, p = {p_val_s:.4f}")
    print(f"Cohen's d: {cohens_d_s:.3f}")

    # ============================================================
    # FIRST 50 vs LAST 50 COMPARISON
    # ============================================================
    print("\n" + "=" * 70)
    print("TEMPORAL ANALYSIS: First 50 vs Last 50 Trials")
    print("=" * 70)

    first_50 = delta_rci_cold[:50]
    last_50 = delta_rci_cold[50:]

    mean_f50 = np.mean(first_50)
    mean_l50 = np.mean(last_50)
    std_f50 = np.std(first_50, ddof=1)
    std_l50 = np.std(last_50, ddof=1)

    t_temporal, p_temporal = stats.ttest_ind(first_50, last_50)
    pooled_std = np.sqrt(((49 * std_f50**2) + (49 * std_l50**2)) / 98)
    cohens_d_temporal = (mean_l50 - mean_f50) / pooled_std if pooled_std > 0 else 0

    print(f"\nFirst 50 Trials:")
    print(f"  Mean DELTA-RCI: {mean_f50:.4f} (SD: {std_f50:.4f})")

    print(f"\nLast 50 Trials:")
    print(f"  Mean DELTA-RCI: {mean_l50:.4f} (SD: {std_l50:.4f})")

    print(f"\nDifference (Last - First): {mean_l50 - mean_f50:.4f}")
    print(f"t({98}) = {t_temporal:.3f}, p = {p_temporal:.4f}")
    print(f"Cohen's d: {cohens_d_temporal:.3f}")

    if p_temporal < 0.05:
        if mean_l50 > mean_f50:
            temporal_trend = "Increasing coherence over time (significant)"
        else:
            temporal_trend = "Decreasing coherence over time (significant)"
    else:
        temporal_trend = "No significant temporal change"
    print(f">>> Temporal Trend: {temporal_trend}")

    # ============================================================
    # ADDITIONAL STATISTICS
    # ============================================================
    print("\n" + "=" * 70)
    print("ADDITIONAL METRICS")
    print("=" * 70)

    print(f"\nTrue Condition Alignments:")
    print(f"  Mean: {np.mean(true_alignments):.4f} (SD: {np.std(true_alignments):.4f})")
    print(f"  Range: [{min(true_alignments):.4f}, {max(true_alignments):.4f}]")

    print(f"\nCold Condition Alignments:")
    print(f"  Mean: {np.mean(cold_alignments):.4f} (SD: {np.std(cold_alignments):.4f})")

    print(f"\nScrambled Condition Alignments:")
    print(f"  Mean: {np.mean(scrambled_alignments):.4f} (SD: {np.std(scrambled_alignments):.4f})")

    print(f"\nEntanglement (E_t):")
    print(f"  Final E_t: {entanglements[-1]:.4f}")
    print(f"  Mean E_t: {np.mean(entanglements):.4f}")
    print(f"  Max E_t: {max(entanglements):.4f}")

    print(f"\nInsight Quality (True):")
    print(f"  Mean: {np.mean(true_iqs):.4f} (SD: {np.std(true_iqs):.4f})")

    # ============================================================
    # 9-PANEL VISUALIZATION
    # ============================================================
    print("\n" + "=" * 70)
    print("GENERATING 9-PANEL VISUALIZATION...")
    print("=" * 70)

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.suptitle(f'MCH v8.1 Analysis: Claude Opus 4.5 (n={n})\nPattern: {pattern} | Cohen\'s d = {cohens_d:.3f} | p = {p_val:.4f}',
                 fontsize=14, fontweight='bold')

    trial_nums = list(range(1, n + 1))

    # Panel 1: DELTA-RCI over trials (cold)
    ax1 = axes[0, 0]
    ax1.plot(trial_nums, delta_rci_cold, 'b-', alpha=0.6, linewidth=1)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax1.axhline(y=mean_drci, color='red', linestyle='-', linewidth=2, label=f'Mean = {mean_drci:.4f}')
    ax1.fill_between(trial_nums, ci_95[0], ci_95[1], alpha=0.2, color='red', label='95% CI')
    z = np.polyfit(trial_nums, delta_rci_cold, 1)
    p = np.poly1d(z)
    ax1.plot(trial_nums, p(trial_nums), 'g--', linewidth=2, label=f'Trend (slope={z[0]:.5f})')
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('DELTA-RCI (True - Cold)')
    ax1.set_title('DELTA-RCI Over Trials (vs Cold)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: DELTA-RCI histogram
    ax2 = axes[0, 1]
    ax2.hist(delta_rci_cold, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=2)
    ax2.axvline(x=mean_drci, color='red', linestyle='-', linewidth=2, label=f'Mean = {mean_drci:.4f}')
    ax2.set_xlabel('DELTA-RCI')
    ax2.set_ylabel('Frequency')
    ax2.set_title('DELTA-RCI Distribution')
    ax2.legend(fontsize=8)

    # Panel 3: Entanglement trajectory
    ax3 = axes[0, 2]
    ax3.plot(trial_nums, entanglements, 'purple', linewidth=2)
    ax3.fill_between(trial_nums, 0, entanglements, alpha=0.3, color='purple')
    ax3.set_xlabel('Trial')
    ax3.set_ylabel('Entanglement (E_t)')
    ax3.set_title('Cumulative Entanglement')
    ax3.grid(True, alpha=0.3)

    # Panel 4: True vs Cold alignments
    ax4 = axes[1, 0]
    ax4.scatter(trial_nums, true_alignments, alpha=0.6, label='True', c='blue', s=20)
    ax4.scatter(trial_nums, cold_alignments, alpha=0.6, label='Cold', c='orange', s=20)
    ax4.set_xlabel('Trial')
    ax4.set_ylabel('Alignment')
    ax4.set_title('True vs Cold Alignments')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # Panel 5: Boxplot comparison
    ax5 = axes[1, 1]
    bp = ax5.boxplot([delta_rci_cold, delta_rci_scrambled],
                      labels=['vs Cold', 'vs Scrambled'],
                      patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('coral')
    ax5.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax5.set_ylabel('DELTA-RCI')
    ax5.set_title('DELTA-RCI by Control Condition')

    # Panel 6: First 50 vs Last 50
    ax6 = axes[1, 2]
    positions = [1, 2]
    bp2 = ax6.boxplot([first_50, last_50], positions=positions, widths=0.6, patch_artist=True)
    bp2['boxes'][0].set_facecolor('lightgreen')
    bp2['boxes'][1].set_facecolor('lightcoral')
    ax6.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax6.set_xticks(positions)
    ax6.set_xticklabels(['First 50', 'Last 50'])
    ax6.set_ylabel('DELTA-RCI')
    ax6.set_title(f'Temporal Comparison (p={p_temporal:.4f})')

    # Panel 7: Insight Quality over time
    ax7 = axes[2, 0]
    ax7.plot(trial_nums, true_iqs, 'green', alpha=0.7, linewidth=1)
    ax7.axhline(y=np.mean(true_iqs), color='darkgreen', linestyle='--', linewidth=2, label=f'Mean = {np.mean(true_iqs):.3f}')
    ax7.set_xlabel('Trial')
    ax7.set_ylabel('Insight Quality')
    ax7.set_title('Insight Quality Over Trials')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)

    # Panel 8: Rolling mean DELTA-RCI (window=10)
    ax8 = axes[2, 1]
    window = 10
    rolling_mean = [np.mean(delta_rci_cold[max(0, i-window):i+1]) for i in range(n)]
    ax8.plot(trial_nums, rolling_mean, 'darkblue', linewidth=2)
    ax8.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax8.fill_between(trial_nums, 0, rolling_mean, where=[r > 0 for r in rolling_mean],
                     alpha=0.3, color='green', label='Positive')
    ax8.fill_between(trial_nums, 0, rolling_mean, where=[r <= 0 for r in rolling_mean],
                     alpha=0.3, color='red', label='Negative')
    ax8.set_xlabel('Trial')
    ax8.set_ylabel('Rolling Mean DELTA-RCI')
    ax8.set_title(f'Rolling Mean DELTA-RCI (window={window})')
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3)

    # Panel 9: Summary statistics table
    ax9 = axes[2, 2]
    ax9.axis('off')

    summary_text = f"""
    CLAUDE OPUS 4.5 SUMMARY
    ================================

    Classification: {pattern}

    DELTA-RCI vs Cold:
      Mean: {mean_drci:.4f}
      SD: {std_drci:.4f}
      95% CI: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]
      t({n-1}) = {t_stat:.3f}
      p-value: {p_val:.6f}
      Cohen's d: {cohens_d:.3f} ({effect_interp})

    Temporal Analysis:
      First 50: {mean_f50:.4f}
      Last 50: {mean_l50:.4f}
      Change: {mean_l50 - mean_f50:.4f}
      p = {p_temporal:.4f}

    Key Finding:
      Claude Opus 4.5 shows {pattern}
      pattern with context utilization.
    """

    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    output_file = 'claude_opus_mch_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved: {output_file}")

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"""
    CLAUDE OPUS 4.5 - MCH v8.1 EXPERIMENT RESULTS
    =============================================

    Pattern Classification: {classification}

    Statistical Evidence:
    - DELTA-RCI = {mean_drci:.4f}
    - p = {p_val:.6f}
    - Cohen's d = {cohens_d:.3f} ({effect_interp} effect)
    - 95% CI: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]

    Comparison Notes:
    - Claude Haiku: NEUTRAL pattern
    - Gemini Flash: SOVEREIGN pattern
    - Gemini Pro: SOVEREIGN pattern
    - Claude Opus 4.5: {pattern} pattern
    """)

    return {
        'pattern': pattern,
        'mean_drci': mean_drci,
        'std_drci': std_drci,
        'ci_95': ci_95,
        'p_value': p_val,
        'cohens_d': cohens_d,
        'first_50_mean': mean_f50,
        'last_50_mean': mean_l50,
        'temporal_p': p_temporal
    }

if __name__ == "__main__":
    analyze_claude_opus()
