import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys

sys.stdout.reconfigure(encoding='utf-8')

data = json.load(open('mch_results_sequential_30trials.json'))

print('='*70)
print('CLAUDE OPUS 4 CROSSOVER & COLLAPSE ANALYSIS')
print('='*70)

# Get Claude trials
claude_trials = [t for t in data['trials'] if t['model'] == 'claude-opus-4-20250514']

# Extract all metrics
delta_cold = [t['controls']['cold']['delta_rci'] for t in claude_trials]
delta_scrambled = [t['controls']['scrambled']['delta_rci'] for t in claude_trials]
entanglements = [t['true']['entanglement'] for t in claude_trials]
true_aligns = [t['true']['alignment'] for t in claude_trials]
cold_aligns = [t['controls']['cold']['alignment'] for t in claude_trials]

# Calculate cumulative and rolling means
cumulative_cold = [np.mean(delta_cold[:i+1]) for i in range(len(delta_cold))]
rolling_cold = [np.mean(delta_cold[max(0,i-4):i+1]) for i in range(len(delta_cold))]

# Calculate E_t change rate
E_changes = [entanglements[i] - entanglements[i-1] if i > 0 else entanglements[0] for i in range(len(entanglements))]

print('\n--- Trial-by-Trial Analysis ---')
print('Trial | DRCI(cold) | Cumul. | Roll(5) | E_t    | E_change | Status')
print('-'*75)

crossover_cumulative = None
crossover_rolling = None
outliers = []
E_collapse_trial = None

for i in range(len(delta_cold)):
    status = ''

    # Check for outliers (|ΔRCI| > 0.3)
    if abs(delta_cold[i]) > 0.3:
        outliers.append((i+1, delta_cold[i]))
        status += ' [OUTLIER]'

    # Check cumulative crossover to positive
    if crossover_cumulative is None and cumulative_cold[i] > 0:
        crossover_cumulative = i + 1
        status += ' [Cumul>0]'

    # Check rolling crossover to positive
    if crossover_rolling is None and rolling_cold[i] > 0:
        crossover_rolling = i + 1
        status += ' [Roll>0]'

    # Check for E_t collapse (negative change after trial 10)
    if i > 10 and E_collapse_trial is None and E_changes[i] < -0.02:
        E_collapse_trial = i + 1
        status += ' [E_COLLAPSE]'

    print(f'{i+1:5} | {delta_cold[i]:+.4f}   | {cumulative_cold[i]:+.4f} | {rolling_cold[i]:+.4f} | {entanglements[i]:.4f} | {E_changes[i]:+.4f}  |{status}')

print('\n' + '='*70)
print('CROSSOVER SUMMARY')
print('='*70)
print(f'Cumulative mean crosses positive: {crossover_cumulative if crossover_cumulative else "NEVER"}')
print(f'Rolling mean (5) crosses positive: {crossover_rolling if crossover_rolling else "NEVER"}')

print('\n' + '='*70)
print('OUTLIER ANALYSIS')
print('='*70)
if outliers:
    for trial, val in outliers:
        direction = "negative" if val < 0 else "positive"
        print(f'  Trial {trial}: DRCI = {val:+.4f} (Extreme {direction})')
    print(f'Total outliers: {len(outliers)}')
else:
    print('  No extreme outliers (|DRCI| > 0.3) detected')

print('\n' + '='*70)
print('ENTANGLEMENT (E_t) TRAJECTORY ANALYSIS')
print('='*70)
print(f'Starting E_t (Trial 1):  {entanglements[0]:.4f}')
print(f'Peak E_t:                {max(entanglements):.4f} at Trial {entanglements.index(max(entanglements))+1}')
print(f'Final E_t (Trial 30):    {entanglements[-1]:.4f}')
print(f'E_t range:               {max(entanglements) - min(entanglements):.4f}')

# Find where E_t growth stalls
E_growth_rates = []
for i in range(5, len(entanglements)):
    rate = (entanglements[i] - entanglements[i-5]) / 5
    E_growth_rates.append((i+1, rate))

stall_trials = [t for t, r in E_growth_rates if r < 0.005]
print(f'\nE_t growth stalls (<0.005/trial) at trials: {stall_trials[:5] if stall_trials else "None"}')

# Analyze what happened around trial 15
print('\n' + '='*70)
print('TRIAL 15 DEEP DIVE (Suspected Collapse Point)')
print('='*70)
for i in range(12, 18):  # Trials 13-18
    print(f'Trial {i+1}: E_t={entanglements[i]:.4f}, DRCI={delta_cold[i]:+.4f}, True_align={true_aligns[i]:.4f}, Cold_align={cold_aligns[i]:.4f}')

print('\n' + '='*70)
print('FIRST HALF vs SECOND HALF')
print('='*70)
first_half_drci = delta_cold[:15]
second_half_drci = delta_cold[15:]
first_half_E = entanglements[:15]
second_half_E = entanglements[15:]

print(f'First 15 trials:')
print(f'  Mean DRCI: {np.mean(first_half_drci):+.4f} (SD: {np.std(first_half_drci):.4f})')
print(f'  Mean E_t:  {np.mean(first_half_E):.4f}')
print(f'  Positive DRCI trials: {sum(1 for d in first_half_drci if d > 0)}/15')

print(f'\nLast 15 trials:')
print(f'  Mean DRCI: {np.mean(second_half_drci):+.4f} (SD: {np.std(second_half_drci):.4f})')
print(f'  Mean E_t:  {np.mean(second_half_E):.4f}')
print(f'  Positive DRCI trials: {sum(1 for d in second_half_drci if d > 0)}/15')

print(f'\nChange (2nd - 1st half):')
print(f'  DRCI change: {np.mean(second_half_drci) - np.mean(first_half_drci):+.4f}')
print(f'  E_t change:  {np.mean(second_half_E) - np.mean(first_half_E):+.4f}')

# Correlation between E_t and DRCI
corr, p_val = stats.pearsonr(entanglements, delta_cold)
print(f'\nE_t vs DRCI correlation: r = {corr:.3f}, p = {p_val:.4f}')

# Final statistics
print('\n' + '='*70)
print('FINAL STATISTICS')
print('='*70)
print(f'Overall mean DRCI: {np.mean(delta_cold):+.4f}')
print(f'Trials with positive DRCI: {sum(1 for d in delta_cold if d > 0)}/30')
print(f'Trials with negative DRCI: {sum(1 for d in delta_cold if d < 0)}/30')
print(f'Most negative trial: Trial {delta_cold.index(min(delta_cold))+1} ({min(delta_cold):+.4f})')
print(f'Most positive trial: Trial {delta_cold.index(max(delta_cold))+1} ({max(delta_cold):+.4f})')

# Create visualization
fig, axes = plt.subplots(3, 2, figsize=(14, 14))
fig.suptitle('Claude Opus 4: DRCI Crossover & Entanglement Analysis (30 Sequential Trials)', fontsize=14, fontweight='bold')

# Plot 1: Individual ΔRCI per trial
ax1 = axes[0, 0]
colors = ['green' if d > 0 else 'red' for d in delta_cold]
ax1.bar(range(1, 31), delta_cold, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax1.set_xlabel('Trial Number')
ax1.set_ylabel('DRCI (True - Cold)')
ax1.set_title('Individual DRCI per Trial')
ax1.set_xlim(0, 31)
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Cumulative + Rolling means
ax2 = axes[0, 1]
ax2.plot(range(1, 31), cumulative_cold, 'b-o', linewidth=2, markersize=5, label='Cumulative Mean')
ax2.plot(range(1, 31), rolling_cold, 'g-s', linewidth=2, markersize=5, label='Rolling Mean (5)')
ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Zero Line')
ax2.fill_between(range(1, 31), cumulative_cold, 0,
                  where=[c > 0 for c in cumulative_cold], alpha=0.2, color='green')
ax2.fill_between(range(1, 31), cumulative_cold, 0,
                  where=[c <= 0 for c in cumulative_cold], alpha=0.2, color='red')
ax2.set_xlabel('Trial Number')
ax2.set_ylabel('Mean DRCI')
ax2.set_title('Cumulative & Rolling Mean DRCI')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

# Plot 3: Entanglement trajectory
ax3 = axes[1, 0]
ax3.plot(range(1, 31), entanglements, 'purple', linewidth=2.5, marker='o', markersize=6)
ax3.axvline(x=15, color='red', linestyle=':', linewidth=2, label='Trial 15 (suspected collapse)')
ax3.fill_between(range(1, 31), entanglements, alpha=0.3, color='purple')
ax3.set_xlabel('Trial Number')
ax3.set_ylabel('Entanglement State (E_t)')
ax3.set_title('Entanglement Trajectory Over Trials')
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)

# Plot 4: E_t vs DRCI scatter
ax4 = axes[1, 1]
colors_scatter = ['green' if d > 0 else 'red' for d in delta_cold]
ax4.scatter(entanglements, delta_cold, c=colors_scatter, s=80, alpha=0.7, edgecolors='black')
# Add trial numbers
for i in range(len(entanglements)):
    ax4.annotate(str(i+1), (entanglements[i], delta_cold[i]), fontsize=7, alpha=0.7)
ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
z = np.polyfit(entanglements, delta_cold, 1)
p = np.poly1d(z)
x_line = np.linspace(min(entanglements), max(entanglements), 100)
ax4.plot(x_line, p(x_line), 'b--', alpha=0.5, label=f'Trend (r={corr:.2f})')
ax4.set_xlabel('Entanglement (E_t)')
ax4.set_ylabel('DRCI')
ax4.set_title('E_t vs DRCI Relationship')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: First half vs Second half boxplot
ax5 = axes[2, 0]
bp = ax5.boxplot([first_half_drci, second_half_drci], positions=[1, 2], widths=0.6, patch_artist=True)
bp['boxes'][0].set_facecolor('salmon')
bp['boxes'][1].set_facecolor('lightsalmon')
ax5.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax5.set_xticks([1, 2])
ax5.set_xticklabels(['Trials 1-15', 'Trials 16-30'])
ax5.set_ylabel('DRCI')
ax5.set_title(f'First Half vs Second Half\n(Mean: {np.mean(first_half_drci):.4f} vs {np.mean(second_half_drci):.4f})')
ax5.grid(True, alpha=0.3, axis='y')
ax5.text(1, np.mean(first_half_drci) + 0.02, f'{np.mean(first_half_drci):.4f}', ha='center', fontweight='bold')
ax5.text(2, np.mean(second_half_drci) + 0.02, f'{np.mean(second_half_drci):.4f}', ha='center', fontweight='bold')

# Plot 6: True vs Cold alignment over time
ax6 = axes[2, 1]
ax6.plot(range(1, 31), true_aligns, 'b-o', linewidth=2, markersize=5, label='True History')
ax6.plot(range(1, 31), cold_aligns, 'r-s', linewidth=2, markersize=5, label='Cold Start')
ax6.fill_between(range(1, 31), true_aligns, cold_aligns,
                  where=[t > c for t, c in zip(true_aligns, cold_aligns)],
                  alpha=0.3, color='green', label='True > Cold')
ax6.fill_between(range(1, 31), true_aligns, cold_aligns,
                  where=[t <= c for t, c in zip(true_aligns, cold_aligns)],
                  alpha=0.3, color='red', label='Cold >= True')
ax6.set_xlabel('Trial Number')
ax6.set_ylabel('Alignment Score')
ax6.set_title('True vs Cold Alignment Over Trials')
ax6.legend(loc='best')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('claude_opus4_collapse_analysis.png', dpi=300, bbox_inches='tight')
print(f'\nVisualization saved: claude_opus4_collapse_analysis.png')
