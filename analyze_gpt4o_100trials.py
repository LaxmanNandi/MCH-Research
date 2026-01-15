"""
Comprehensive analysis of GPT-4o 100-trial MCH experiment
Includes First 50 vs Last 50 comparison and visualization
"""

import json
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import sys

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Load results
with open('mch_results_gpt4o_100trials.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

trials = results['trials']

# Extract ΔRCI values
delta_rci_cold = []
delta_rci_scrambled = []
true_alignments = []
cold_alignments = []
scrambled_alignments = []
entanglement_values = []

for trial in trials:
    if 'cold' in trial['controls']:
        delta_rci_cold.append(trial['controls']['cold']['delta_rci'])
        cold_alignments.append(trial['controls']['cold']['alignment'])
    if 'scrambled' in trial['controls']:
        delta_rci_scrambled.append(trial['controls']['scrambled']['delta_rci'])
        scrambled_alignments.append(trial['controls']['scrambled']['alignment'])
    true_alignments.append(trial['true']['alignment'])
    entanglement_values.append(trial['true']['entanglement'])

delta_rci_cold = np.array(delta_rci_cold)
delta_rci_scrambled = np.array(delta_rci_scrambled)
true_alignments = np.array(true_alignments)
cold_alignments = np.array(cold_alignments)
scrambled_alignments = np.array(scrambled_alignments)
entanglement_values = np.array(entanglement_values)

print("="*70)
print("GPT-4o 100-TRIAL MCH EXPERIMENT ANALYSIS")
print("="*70)

# Primary Statistics - Cold control
print("\n--- PRIMARY ANALYSIS: True vs Cold (No History) ---")
mean_cold = np.mean(delta_rci_cold)
std_cold = np.std(delta_rci_cold)
se_cold = stats.sem(delta_rci_cold)
ci95_cold = stats.t.interval(0.95, len(delta_rci_cold)-1, loc=mean_cold, scale=se_cold)
t_stat_cold, p_val_cold = stats.ttest_1samp(delta_rci_cold, 0)
cohens_d_cold = mean_cold / std_cold if std_cold > 0 else 0

print(f"N trials: {len(delta_rci_cold)}")
print(f"Mean ΔRCI: {mean_cold:.4f}")
print(f"SD: {std_cold:.4f}")
print(f"95% CI: [{ci95_cold[0]:.4f}, {ci95_cold[1]:.4f}]")
print(f"t({len(delta_rci_cold)-1}) = {t_stat_cold:.3f}")
print(f"p-value: {p_val_cold:.4f}")
print(f"Cohen's d: {cohens_d_cold:.3f}")

# Classification
if p_val_cold < 0.05:
    if mean_cold > 0:
        classification = "CONVERGENT"
        desc = "Context HELPS (positive ΔRCI)"
    else:
        classification = "SOVEREIGN"
        desc = "Context HURTS (negative ΔRCI)"
else:
    classification = "NEUTRAL"
    desc = "No significant context effect"

print(f"\nMCH CLASSIFICATION: {classification}")
print(f"Interpretation: {desc}")

# Scrambled control
print("\n--- SECONDARY: True vs Scrambled ---")
mean_scr = np.mean(delta_rci_scrambled)
std_scr = np.std(delta_rci_scrambled)
t_stat_scr, p_val_scr = stats.ttest_1samp(delta_rci_scrambled, 0)
cohens_d_scr = mean_scr / std_scr if std_scr > 0 else 0
print(f"Mean ΔRCI: {mean_scr:.4f} (SD: {std_scr:.4f})")
print(f"t({len(delta_rci_scrambled)-1}) = {t_stat_scr:.3f}, p = {p_val_scr:.4f}")
print(f"Cohen's d: {cohens_d_scr:.3f}")

# First 50 vs Last 50 Analysis
print("\n" + "="*70)
print("TEMPORAL ANALYSIS: First 50 vs Last 50 Trials")
print("="*70)

first_50_cold = delta_rci_cold[:50]
last_50_cold = delta_rci_cold[50:]

mean_first = np.mean(first_50_cold)
mean_last = np.mean(last_50_cold)
std_first = np.std(first_50_cold)
std_last = np.std(last_50_cold)

t_temporal, p_temporal = stats.ttest_ind(first_50_cold, last_50_cold)

print(f"First 50 trials: Mean ΔRCI = {mean_first:.4f} (SD: {std_first:.4f})")
print(f"Last 50 trials:  Mean ΔRCI = {mean_last:.4f} (SD: {std_last:.4f})")
print(f"Difference: {mean_last - mean_first:.4f}")
print(f"Independent t-test: t = {t_temporal:.3f}, p = {p_temporal:.4f}")

if p_temporal < 0.05:
    if mean_last > mean_first:
        print("SIGNIFICANT: Context utilization IMPROVED over time")
    else:
        print("SIGNIFICANT: Context utilization DECREASED over time")
else:
    print("No significant temporal change in context utilization")

# Entanglement analysis
print("\n--- ENTANGLEMENT TRAJECTORY ---")
print(f"Starting E: {entanglement_values[0]:.4f}")
print(f"Final E: {entanglement_values[-1]:.4f}")
print(f"Peak E: {np.max(entanglement_values):.4f} at trial {np.argmax(entanglement_values)}")
print(f"Mean E: {np.mean(entanglement_values):.4f}")

# Error analysis
print("\n--- ERROR ANALYSIS ---")
empty_true = [t['trial'] for t in trials if t['true']['response_length'] == 0]
print(f"Trials with empty TRUE responses: {len(empty_true)} - {empty_true if empty_true else 'None'}")

if 'error_log' in results:
    print(f"Total errors logged: {len(results['error_log'])}")

# Load GPT-4o-mini for comparison
print("\n" + "="*70)
print("COMPARISON: GPT-4o vs GPT-4o-mini")
print("="*70)

try:
    with open('mch_results_gpt4o_mini_n100_merged.json', 'r', encoding='utf-8') as f:
        mini_results = json.load(f)

    mini_trials = mini_results['trials']
    mini_delta_cold = [t['controls']['cold']['delta_rci'] for t in mini_trials if 'cold' in t['controls']]
    mini_delta_cold = np.array(mini_delta_cold)

    mean_mini = np.mean(mini_delta_cold)
    std_mini = np.std(mini_delta_cold)
    t_mini, p_mini = stats.ttest_1samp(mini_delta_cold, 0)
    cohens_d_mini = mean_mini / std_mini if std_mini > 0 else 0

    print(f"\nGPT-4o-mini (n={len(mini_delta_cold)}):")
    print(f"  Mean ΔRCI: {mean_mini:.4f} (SD: {std_mini:.4f})")
    print(f"  t = {t_mini:.3f}, p = {p_mini:.4f}")
    print(f"  Cohen's d: {cohens_d_mini:.3f}")

    print(f"\nGPT-4o (n={len(delta_rci_cold)}):")
    print(f"  Mean ΔRCI: {mean_cold:.4f} (SD: {std_cold:.4f})")
    print(f"  t = {t_stat_cold:.3f}, p = {p_val_cold:.4f}")
    print(f"  Cohen's d: {cohens_d_cold:.3f}")

    # Direct comparison
    t_compare, p_compare = stats.ttest_ind(delta_rci_cold, mini_delta_cold)
    diff = mean_cold - mean_mini
    print(f"\nDirect comparison (GPT-4o vs GPT-4o-mini):")
    print(f"  Difference in Mean ΔRCI: {diff:.4f}")
    print(f"  Independent t-test: t = {t_compare:.3f}, p = {p_compare:.4f}")

    if p_compare < 0.05:
        if diff > 0:
            print("  RESULT: GPT-4o shows BETTER context utilization than GPT-4o-mini")
        else:
            print("  RESULT: GPT-4o shows WORSE context utilization than GPT-4o-mini")
    else:
        print("  RESULT: No significant difference between GPT-4o and GPT-4o-mini")

    has_mini = True
except Exception as e:
    print(f"Could not load GPT-4o-mini results: {e}")
    has_mini = False

# Create visualization
print("\n" + "="*70)
print("GENERATING VISUALIZATION...")
print("="*70)

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('GPT-4o MCH Analysis (100 Trials)', fontsize=14, fontweight='bold')

# 1. ΔRCI Distribution (Cold)
ax1 = axes[0, 0]
ax1.hist(delta_rci_cold, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
ax1.axvline(x=mean_cold, color='green', linestyle='-', linewidth=2, label=f'Mean={mean_cold:.3f}')
ax1.set_xlabel('ΔRCI (True - Cold)')
ax1.set_ylabel('Frequency')
ax1.set_title('ΔRCI Distribution (vs Cold)')
ax1.legend()

# 2. ΔRCI over trials
ax2 = axes[0, 1]
ax2.plot(delta_rci_cold, color='steelblue', alpha=0.7, linewidth=1)
ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
window = 10
if len(delta_rci_cold) >= window:
    rolling_mean = np.convolve(delta_rci_cold, np.ones(window)/window, mode='valid')
    ax2.plot(range(window-1, len(delta_rci_cold)), rolling_mean, color='orange',
             linewidth=2, label=f'{window}-trial MA')
ax2.set_xlabel('Trial')
ax2.set_ylabel('ΔRCI')
ax2.set_title('ΔRCI Trajectory Over Trials')
ax2.legend()

# 3. Entanglement trajectory
ax3 = axes[0, 2]
ax3.plot(entanglement_values, color='purple', linewidth=2)
ax3.set_xlabel('Trial')
ax3.set_ylabel('Entanglement (E_t)')
ax3.set_title('Cumulative Entanglement')
ax3.fill_between(range(len(entanglement_values)), entanglement_values, alpha=0.3, color='purple')

# 4. True vs Cold alignment
ax4 = axes[1, 0]
ax4.scatter(cold_alignments, true_alignments, alpha=0.5, color='steelblue')
min_val = min(min(cold_alignments), min(true_alignments))
max_val = max(max(cold_alignments), max(true_alignments))
ax4.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
ax4.set_xlabel('Cold Alignment')
ax4.set_ylabel('True Alignment')
ax4.set_title('True vs Cold Alignment')
ax4.legend()

# 5. Box plot comparison
ax5 = axes[1, 1]
bp = ax5.boxplot([delta_rci_cold, delta_rci_scrambled],
                  tick_labels=['Cold', 'Scrambled'],
                  patch_artist=True)
bp['boxes'][0].set_facecolor('steelblue')
bp['boxes'][1].set_facecolor('coral')
ax5.axhline(y=0, color='red', linestyle='--')
ax5.set_ylabel('ΔRCI')
ax5.set_title('ΔRCI by Control Condition')

# 6. First 50 vs Last 50
ax6 = axes[1, 2]
positions = [1, 2]
bp2 = ax6.boxplot([first_50_cold, last_50_cold],
                   tick_labels=['First 50', 'Last 50'],
                   patch_artist=True)
bp2['boxes'][0].set_facecolor('lightblue')
bp2['boxes'][1].set_facecolor('darkblue')
ax6.axhline(y=0, color='red', linestyle='--')
ax6.set_ylabel('ΔRCI (vs Cold)')
ax6.set_title(f'Temporal Analysis (p={p_temporal:.4f})')

# 7. Alignment trajectories
ax7 = axes[2, 0]
ax7.plot(true_alignments, label='True', color='green', alpha=0.7)
ax7.plot(cold_alignments, label='Cold', color='blue', alpha=0.7)
ax7.set_xlabel('Trial')
ax7.set_ylabel('Alignment')
ax7.set_title('Alignment Trajectories')
ax7.legend()

# 8. Effect size comparison (if mini available)
ax8 = axes[2, 1]
if has_mini:
    models = ['GPT-4o', 'GPT-4o-mini']
    effect_sizes = [cohens_d_cold, cohens_d_mini]
    colors = ['steelblue', 'coral']
    bars = ax8.bar(models, effect_sizes, color=colors, edgecolor='black')
    ax8.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax8.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5)
    ax8.axhline(y=-0.2, color='gray', linestyle='--', alpha=0.5)
    ax8.set_ylabel("Cohen's d")
    ax8.set_title('Effect Size Comparison')
    for bar, val in zip(bars, effect_sizes):
        ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 if val >= 0 else bar.get_height() - 0.03,
                f'{val:.3f}', ha='center', fontsize=10)
else:
    ax8.text(0.5, 0.5, 'GPT-4o-mini\ndata not found', ha='center', va='center', transform=ax8.transAxes)
    ax8.set_title('Effect Size Comparison')

# 9. Summary statistics
ax9 = axes[2, 2]
ax9.axis('off')
summary_text = f"""MCH v8.1 Analysis Summary
═══════════════════════════════

Model: GPT-4o (gpt-4o)
Trials: {len(delta_rci_cold)}
Classification: {classification}

Primary Statistics (vs Cold):
  Mean ΔRCI: {mean_cold:.4f}
  SD: {std_cold:.4f}
  95% CI: [{ci95_cold[0]:.4f}, {ci95_cold[1]:.4f}]
  t({len(delta_rci_cold)-1}) = {t_stat_cold:.3f}
  p-value: {p_val_cold:.4f}
  Cohen's d: {cohens_d_cold:.3f}

Temporal Analysis:
  First 50 Mean: {mean_first:.4f}
  Last 50 Mean: {mean_last:.4f}
  Change: {mean_last - mean_first:.4f}
  p-value: {p_temporal:.4f}

Entanglement:
  Final E: {entanglement_values[-1]:.4f}
  Peak E: {np.max(entanglement_values):.4f}

Empty responses: {len(empty_true)} trials
"""
ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('gpt4o_mch_analysis.png', dpi=150, bbox_inches='tight')
print("Saved: gpt4o_mch_analysis.png")

# Final summary
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"""
GPT-4o MCH Experiment Results:
-----------------------------
Classification: {classification}
Mean ΔRCI: {mean_cold:.4f} (95% CI: [{ci95_cold[0]:.4f}, {ci95_cold[1]:.4f}])
p-value: {p_val_cold:.4f}
Cohen's d: {cohens_d_cold:.3f}

Interpretation:
{desc}

Temporal trend (First 50 vs Last 50): p = {p_temporal:.4f}
{'Significant improvement over time' if p_temporal < 0.05 and mean_last > mean_first else 'No significant temporal change'}

Note: {len(empty_true)} trials had empty TRUE responses due to connection errors.
""")

print("="*70)
print("ANALYSIS COMPLETE")
print("="*70)
