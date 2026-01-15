"""
Comprehensive analysis of Gemini 2.5 Pro 100-trial MCH experiment
Includes First 50 vs Last 50 comparison, visualization, and Gemini Pro vs Flash comparison
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
with open('mch_results_gemini_pro_100trials.json', 'r', encoding='utf-8') as f:
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
print("GEMINI 2.5 PRO 100-TRIAL MCH EXPERIMENT ANALYSIS")
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
print(f"p-value: {p_val_cold:.6f}")
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

# Load Gemini Flash for comparison
print("\n" + "="*70)
print("COMPARISON: Gemini 2.5 Pro vs Gemini 2.0 Flash")
print("="*70)

try:
    with open('mch_results_gemini_flash_100trials.json', 'r', encoding='utf-8') as f:
        flash_results = json.load(f)

    flash_trials = flash_results['trials']
    flash_delta_cold = [t['controls']['cold']['delta_rci'] for t in flash_trials if 'cold' in t['controls']]
    flash_delta_cold = np.array(flash_delta_cold)

    mean_flash = np.mean(flash_delta_cold)
    std_flash = np.std(flash_delta_cold)
    t_flash, p_flash = stats.ttest_1samp(flash_delta_cold, 0)
    cohens_d_flash = mean_flash / std_flash if std_flash > 0 else 0

    print(f"\nGemini 2.0 Flash (n={len(flash_delta_cold)}):")
    print(f"  Mean ΔRCI: {mean_flash:.4f} (SD: {std_flash:.4f})")
    print(f"  t = {t_flash:.3f}, p = {p_flash:.6f}")
    print(f"  Cohen's d: {cohens_d_flash:.3f}")

    # Classify Flash
    if p_flash < 0.05:
        flash_class = "SOVEREIGN" if mean_flash < 0 else "CONVERGENT"
    else:
        flash_class = "NEUTRAL"
    print(f"  Classification: {flash_class}")

    print(f"\nGemini 2.5 Pro (n={len(delta_rci_cold)}):")
    print(f"  Mean ΔRCI: {mean_cold:.4f} (SD: {std_cold:.4f})")
    print(f"  t = {t_stat_cold:.3f}, p = {p_val_cold:.6f}")
    print(f"  Cohen's d: {cohens_d_cold:.3f}")
    print(f"  Classification: {classification}")

    # Direct comparison
    t_compare, p_compare = stats.ttest_ind(delta_rci_cold, flash_delta_cold)
    diff = mean_cold - mean_flash
    print(f"\nDirect comparison (Pro vs Flash):")
    print(f"  Difference in Mean ΔRCI: {diff:.4f}")
    print(f"  Independent t-test: t = {t_compare:.3f}, p = {p_compare:.4f}")

    if p_compare < 0.05:
        if diff > 0:
            print("  RESULT: Gemini Pro shows BETTER context utilization than Flash")
        else:
            print("  RESULT: Gemini Pro shows WORSE context utilization than Flash")
    else:
        print("  RESULT: No significant difference between Pro and Flash")

    has_flash = True
except Exception as e:
    print(f"Could not load Gemini Flash results: {e}")
    has_flash = False

# Create visualization
print("\n" + "="*70)
print("GENERATING VISUALIZATION...")
print("="*70)

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('Gemini 2.5 Pro MCH Analysis (100 Trials)', fontsize=14, fontweight='bold')

# 1. ΔRCI Distribution (Cold)
ax1 = axes[0, 0]
ax1.hist(delta_rci_cold, bins=20, color='darkgreen', edgecolor='black', alpha=0.7)
ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
ax1.axvline(x=mean_cold, color='blue', linestyle='-', linewidth=2, label=f'Mean={mean_cold:.3f}')
ax1.set_xlabel('ΔRCI (True - Cold)')
ax1.set_ylabel('Frequency')
ax1.set_title('ΔRCI Distribution (vs Cold)')
ax1.legend()

# 2. ΔRCI over trials
ax2 = axes[0, 1]
ax2.plot(delta_rci_cold, color='darkgreen', alpha=0.7, linewidth=1)
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
ax4.scatter(cold_alignments, true_alignments, alpha=0.5, color='darkgreen')
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
bp['boxes'][0].set_facecolor('darkgreen')
bp['boxes'][1].set_facecolor('coral')
ax5.axhline(y=0, color='red', linestyle='--')
ax5.set_ylabel('ΔRCI')
ax5.set_title('ΔRCI by Control Condition')

# 6. First 50 vs Last 50
ax6 = axes[1, 2]
bp2 = ax6.boxplot([first_50_cold, last_50_cold],
                   tick_labels=['First 50', 'Last 50'],
                   patch_artist=True)
bp2['boxes'][0].set_facecolor('lightgreen')
bp2['boxes'][1].set_facecolor('darkgreen')
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

# 8. Effect size comparison (Pro vs Flash)
ax8 = axes[2, 1]
if has_flash:
    models = ['Gemini Pro', 'Gemini Flash']
    effect_sizes = [cohens_d_cold, cohens_d_flash]
    colors = ['darkgreen', 'gold']
    bars = ax8.bar(models, effect_sizes, color=colors, edgecolor='black')
    ax8.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax8.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5)
    ax8.axhline(y=-0.2, color='gray', linestyle='--', alpha=0.5)
    ax8.set_ylabel("Cohen's d")
    ax8.set_title('Effect Size Comparison')
    for bar, val in zip(bars, effect_sizes):
        ypos = bar.get_height() + 0.02 if val >= 0 else bar.get_height() - 0.05
        ax8.text(bar.get_x() + bar.get_width()/2, ypos,
                f'{val:.3f}', ha='center', fontsize=10)
else:
    ax8.text(0.5, 0.5, 'Gemini Flash\ndata not found', ha='center', va='center', transform=ax8.transAxes)
    ax8.set_title('Effect Size Comparison')

# 9. Summary statistics
ax9 = axes[2, 2]
ax9.axis('off')
summary_text = f"""MCH v8.1 Analysis Summary
===============================

Model: Gemini 2.5 Pro (Flagship)
Trials: {len(delta_rci_cold)}
Classification: {classification}

Primary Statistics (vs Cold):
  Mean ΔRCI: {mean_cold:.4f}
  SD: {std_cold:.4f}
  95% CI: [{ci95_cold[0]:.4f}, {ci95_cold[1]:.4f}]
  t({len(delta_rci_cold)-1}) = {t_stat_cold:.3f}
  p-value: {p_val_cold:.6f}
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
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.tight_layout()
plt.savefig('gemini_pro_mch_analysis.png', dpi=150, bbox_inches='tight')
print("Saved: gemini_pro_mch_analysis.png")

# Final summary
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"""
Gemini 2.5 Pro MCH Experiment Results:
--------------------------------------
Classification: {classification}
Mean ΔRCI: {mean_cold:.4f} (95% CI: [{ci95_cold[0]:.4f}, {ci95_cold[1]:.4f}])
p-value: {p_val_cold:.6f}
Cohen's d: {cohens_d_cold:.3f}

Interpretation:
{desc}

Key Finding: Gemini 2.5 Pro shows a SOVEREIGN pattern with NEGATIVE ΔRCI,
meaning responses are LESS aligned with prompts when conversation history
is provided compared to cold-start responses.

This suggests that Gemini Pro, like Gemini Flash, performs WORSE with
conversational context for these philosophical prompts.

Temporal trend (First 50 vs Last 50): p = {p_temporal:.4f}
{'Significant change over time' if p_temporal < 0.05 else 'No significant temporal change'}
""")

print("="*70)
print("ANALYSIS COMPLETE")
print("="*70)
