import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys

# Fix Unicode encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Load results
with open('mch_results_dr_laxman_mch_v8.1_20260109_143816.json', 'r') as f:
    data = json.load(f)

print("="*60)
print("DETAILED MCH v8.1 ANALYSIS")
print("="*60)

# Create comprehensive analysis plots
fig = plt.figure(figsize=(16, 12))

models = data['metadata']['config']['models']
colors = {'true': '#2E86AB', 'cold': '#A23B72', 'scrambled': '#F18F01'}

for model_idx, model in enumerate(models):
    model_trials = [t for t in data['trials'] if t['model'] == model]
    model_short = model.split('-')[0] + '-' + model.split('-')[1]

    print(f"\n{'='*60}")
    print(f"MODEL: {model}")
    print(f"{'='*60}")

    # Extract data
    trial_nums = [t['trial'] for t in model_trials]
    true_aligns = [t['true']['alignment'] for t in model_trials]
    cold_aligns = [t['controls']['cold']['alignment'] for t in model_trials]
    scrambled_aligns = [t['controls']['scrambled']['alignment'] for t in model_trials]
    entanglements = [t['true']['entanglement'] for t in model_trials]
    delta_rci_cold = [t['controls']['cold']['delta_rci'] for t in model_trials]
    delta_rci_scrambled = [t['controls']['scrambled']['delta_rci'] for t in model_trials]

    base_row = model_idx * 3

    # Plot 1: Alignment over trials
    ax1 = plt.subplot(len(models), 3, base_row + 1)
    ax1.plot(trial_nums, true_aligns, 'o-', color=colors['true'], label='True History', linewidth=2, markersize=6)
    ax1.plot(trial_nums, cold_aligns, 's-', color=colors['cold'], label='Cold Start', linewidth=2, markersize=6)
    ax1.plot(trial_nums, scrambled_aligns, '^-', color=colors['scrambled'], label='Scrambled', linewidth=2, markersize=6)
    ax1.set_xlabel('Trial Number')
    ax1.set_ylabel('Alignment Score')
    ax1.set_title(f'{model_short}: Alignment Over Trials')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cumulative ΔRCI
    ax2 = plt.subplot(len(models), 3, base_row + 2)

    # Calculate cumulative mean ΔRCI for cold
    cumulative_cold = [np.mean(delta_rci_cold[:i+1]) for i in range(len(delta_rci_cold))]
    cumulative_scrambled = [np.mean(delta_rci_scrambled[:i+1]) for i in range(len(delta_rci_scrambled))]

    ax2.plot(trial_nums, cumulative_cold, 'o-', color=colors['cold'], label='vs Cold', linewidth=2, markersize=6)
    ax2.plot(trial_nums, cumulative_scrambled, '^-', color=colors['scrambled'], label='vs Scrambled', linewidth=2, markersize=6)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax2.set_xlabel('Trial Number')
    ax2.set_ylabel('Cumulative Mean ΔRCI')
    ax2.set_title(f'{model_short}: Cumulative ΔRCI')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Find crossover points
    crossover_cold = None
    crossover_scrambled = None
    for i, val in enumerate(cumulative_cold):
        if val > 0 and crossover_cold is None:
            crossover_cold = i
    for i, val in enumerate(cumulative_scrambled):
        if val > 0 and crossover_scrambled is None:
            crossover_scrambled = i

    print(f"\nCumulative ΔRCI Analysis:")
    print(f"  vs Cold - crosses zero at trial: {crossover_cold if crossover_cold else 'Never'}")
    print(f"  vs Scrambled - crosses zero at trial: {crossover_scrambled if crossover_scrambled else 'Never'}")
    print(f"  Final cumulative ΔRCI (cold): {cumulative_cold[-1]:.4f}")
    print(f"  Final cumulative ΔRCI (scrambled): {cumulative_scrambled[-1]:.4f}")

    # Plot 3: Entanglement vs ΔRCI correlation
    ax3 = plt.subplot(len(models), 3, base_row + 3)
    ax3.scatter(entanglements, delta_rci_cold, color=colors['cold'], s=80, alpha=0.6, label='vs Cold', marker='o')
    ax3.scatter(entanglements, delta_rci_scrambled, color=colors['scrambled'], s=80, alpha=0.6, label='vs Scrambled', marker='^')

    # Calculate correlations
    corr_cold, p_cold = stats.pearsonr(entanglements, delta_rci_cold)
    corr_scrambled, p_scrambled = stats.pearsonr(entanglements, delta_rci_scrambled)

    # Add trend lines
    z_cold = np.polyfit(entanglements, delta_rci_cold, 1)
    p_cold_line = np.poly1d(z_cold)
    z_scrambled = np.polyfit(entanglements, delta_rci_scrambled, 1)
    p_scrambled_line = np.poly1d(z_scrambled)

    x_range = np.linspace(min(entanglements), max(entanglements), 100)
    ax3.plot(x_range, p_cold_line(x_range), color=colors['cold'], linestyle='--', alpha=0.5)
    ax3.plot(x_range, p_scrambled_line(x_range), color=colors['scrambled'], linestyle='--', alpha=0.5)

    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    ax3.set_xlabel('Entanglement State (E_t)')
    ax3.set_ylabel('ΔRCI')
    ax3.set_title(f'{model_short}: E_t vs ΔRCI Correlation')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)

    print(f"\nEntanglement Correlation Analysis:")
    print(f"  E_t vs ΔRCI (cold):      r = {corr_cold:.3f}, p = {p_cold:.4f}")
    print(f"  E_t vs ΔRCI (scrambled): r = {corr_scrambled:.3f}, p = {p_scrambled:.4f}")

    if abs(corr_cold) > 0.5 and p_cold < 0.05:
        print(f"  => Significant correlation with cold control!")
    if abs(corr_scrambled) > 0.5 and p_scrambled < 0.05:
        print(f"  => Significant correlation with scrambled control!")

    # Additional analysis
    print(f"\nTrial-by-Trial Statistics:")
    print(f"  Mean E_t: {np.mean(entanglements):.4f} (SD: {np.std(entanglements):.4f})")
    print(f"  E_t growth rate: {(entanglements[-1] - entanglements[0]) / len(entanglements):.4f} per trial")

    # Analyze variance in ΔRCI over time
    first_half_cold = delta_rci_cold[:5]
    second_half_cold = delta_rci_cold[5:]
    print(f"  ΔRCI (cold) - first half mean: {np.mean(first_half_cold):.4f}")
    print(f"  ΔRCI (cold) - second half mean: {np.mean(second_half_cold):.4f}")
    print(f"  ΔRCI (cold) - improvement: {np.mean(second_half_cold) - np.mean(first_half_cold):.4f}")

plt.tight_layout()
plt.savefig('mch_analysis_detailed.png', dpi=300, bbox_inches='tight')
print(f"\n{'='*60}")
print("Analysis saved: mch_analysis_detailed.png")
print(f"{'='*60}")

# Summary recommendations
print("\n" + "="*60)
print("RECOMMENDATIONS FOR FOLLOW-UP EXPERIMENTS")
print("="*60)
print("\n1. Sample Size:")
print("   - Current n=10 may be too small for statistical power")
print("   - Recommend n=30-50 for more reliable conclusions")

print("\n2. Task Type Effects:")
print("   - Test sequential/building prompts vs independent prompts")
print("   - Compare philosophical vs technical reasoning tasks")

print("\n3. Entanglement Dynamics:")
if any(abs(corr_cold) > 0.3 for corr_cold in [stats.pearsonr([t['true']['entanglement'] for t in data['trials'] if t['model'] == m], [t['controls']['cold']['delta_rci'] for t in data['trials'] if t['model'] == m])[0] for m in models]):
    print("   - E_t shows some correlation with ΔRCI")
    print("   - Longer sessions may reveal threshold effects")
else:
    print("   - E_t shows weak correlation with ΔRCI")
    print("   - May need different entanglement model or parameters")

print("\n4. Model Differences:")
print("   - Claude Opus 4 showed negative ΔRCI (unexpected)")
print("   - GPT-4o-mini showed positive trend (expected direction)")
print("   - Different models may benefit differently from history")

# Don't show plot in non-interactive mode
# plt.show()
