import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Load both result files
print("Loading data...")
with open('mch_results_gpt4o_mini_replication.json', 'r') as f:
    original_data = json.load(f)

with open('mch_results_gpt4o_mini_trials_31_100.json', 'r') as f:
    extension_data = json.load(f)

# Merge trials
all_trials = original_data['trials'] + extension_data['trials']
print(f"Total trials: {len(all_trials)}")

# Extract ΔRCI values
delta_cold = [t['controls']['cold']['delta_rci'] for t in all_trials]
delta_scrambled = [t['controls']['scrambled']['delta_rci'] for t in all_trials]
entanglements = [t['true']['entanglement'] for t in all_trials]
true_aligns = [t['true']['alignment'] for t in all_trials]
cold_aligns = [t['controls']['cold']['alignment'] for t in all_trials]

print("\n" + "="*80)
print("MCH v8.1: GPT-4o-mini n=100 ANALYSIS")
print("="*80)

# Basic statistics
mean_drci = np.mean(delta_cold)
std_drci = np.std(delta_cold)
sem_drci = std_drci / np.sqrt(len(delta_cold))
ci_95 = 1.96 * sem_drci

print(f"\n### Full Dataset Statistics (n=100)")
print(f"Mean ΔRCI: {mean_drci:+.4f}")
print(f"SD: {std_drci:.4f}")
print(f"SEM: {sem_drci:.4f}")
print(f"95% CI: [{mean_drci - ci_95:+.4f}, {mean_drci + ci_95:+.4f}]")

# One-sample t-test
t_stat, p_val = stats.ttest_1samp(delta_cold, 0)
cohens_d = mean_drci / std_drci if std_drci > 0 else 0
print(f"\nOne-sample t-test against zero:")
print(f"  t({len(delta_cold)-1}) = {t_stat:.3f}")
print(f"  p = {p_val:.4f}")
print(f"  Cohen's d = {cohens_d:+.3f}")

# Positive trials
positive_count = sum(1 for d in delta_cold if d > 0)
print(f"\nPositive trials: {positive_count}/100 ({positive_count}%)")

# Quartile analysis
q1 = delta_cold[:25]
q2 = delta_cold[25:50]
q3 = delta_cold[50:75]
q4 = delta_cold[75:100]

print("\n" + "="*80)
print("QUARTILE ANALYSIS (25 trials each)")
print("="*80)

for i, (q, label) in enumerate([(q1, "Q1 (1-25)"), (q2, "Q2 (26-50)"),
                                 (q3, "Q3 (51-75)"), (q4, "Q4 (76-100)")]):
    q_mean = np.mean(q)
    q_std = np.std(q)
    q_t, q_p = stats.ttest_1samp(q, 0)
    positive = sum(1 for d in q if d > 0)
    print(f"\n{label}:")
    print(f"  Mean ΔRCI: {q_mean:+.4f} (SD: {q_std:.4f})")
    print(f"  t(24) = {q_t:.3f}, p = {q_p:.4f}")
    print(f"  Positive: {positive}/25")

# First 50 vs Last 50
first_50 = delta_cold[:50]
last_50 = delta_cold[50:]

print("\n" + "="*80)
print("FIRST HALF vs SECOND HALF (50 trials each)")
print("="*80)

f50_mean = np.mean(first_50)
f50_std = np.std(first_50)
f50_t, f50_p = stats.ttest_1samp(first_50, 0)

l50_mean = np.mean(last_50)
l50_std = np.std(last_50)
l50_t, l50_p = stats.ttest_1samp(last_50, 0)

print(f"\nFirst 50 trials (1-50):")
print(f"  Mean ΔRCI: {f50_mean:+.4f} (SD: {f50_std:.4f})")
print(f"  t(49) = {f50_t:.3f}, p = {f50_p:.4f}")
print(f"  Positive: {sum(1 for d in first_50 if d > 0)}/50")

print(f"\nLast 50 trials (51-100):")
print(f"  Mean ΔRCI: {l50_mean:+.4f} (SD: {l50_std:.4f})")
print(f"  t(49) = {l50_t:.3f}, p = {l50_p:.4f}")
print(f"  Positive: {sum(1 for d in last_50 if d > 0)}/50")

# Independent t-test for difference between halves
t_diff, p_diff = stats.ttest_ind(first_50, last_50)
print(f"\nIndependent t-test (First 50 vs Last 50):")
print(f"  t(98) = {t_diff:.3f}, p = {p_diff:.4f}")
print(f"  Change: {l50_mean - f50_mean:+.4f}")

# Cumulative and rolling means
cumulative_mean = [np.mean(delta_cold[:i+1]) for i in range(100)]
rolling_mean_5 = [np.mean(delta_cold[max(0,i-4):i+1]) for i in range(100)]
rolling_mean_10 = [np.mean(delta_cold[max(0,i-9):i+1]) for i in range(100)]

# Find crossover points
first_positive = None
sustained_positive = None
for i, c in enumerate(cumulative_mean):
    if c > 0 and first_positive is None:
        first_positive = i + 1
        break

print("\n" + "="*80)
print("CROSSOVER ANALYSIS")
print("="*80)
print(f"First positive cumulative mean: Trial {first_positive if first_positive else 'NEVER'}")

# Check for sustained positive (5 consecutive)
for i in range(len(cumulative_mean) - 4):
    if all(c > 0 for c in cumulative_mean[i:i+5]):
        sustained_positive = i + 1
        break
print(f"First sustained positive (5 consecutive): Trial {sustained_positive if sustained_positive else 'NEVER'}")

# Prompt-cycle analysis (30 prompts cycle)
print("\n" + "="*80)
print("PROMPT CYCLE ANALYSIS")
print("="*80)

cycle1 = delta_cold[:30]  # Trials 1-30
cycle2 = delta_cold[30:60]  # Trials 31-60
cycle3 = delta_cold[60:90]  # Trials 61-90
remaining = delta_cold[90:100]  # Trials 91-100

print(f"\nCycle 1 (Trials 1-30): Mean = {np.mean(cycle1):+.4f}")
print(f"Cycle 2 (Trials 31-60): Mean = {np.mean(cycle2):+.4f}")
print(f"Cycle 3 (Trials 61-90): Mean = {np.mean(cycle3):+.4f}")
print(f"Partial Cycle 4 (91-100): Mean = {np.mean(remaining):+.4f}")

# Same-prompt comparison across cycles
print("\n### Same-prompt comparison (averaged across cycles)")
prompt_avgs = []
for prompt_idx in range(30):
    values = [delta_cold[prompt_idx]]
    if prompt_idx + 30 < 100:
        values.append(delta_cold[prompt_idx + 30])
    if prompt_idx + 60 < 100:
        values.append(delta_cold[prompt_idx + 60])
    if prompt_idx + 90 < 100:
        values.append(delta_cold[prompt_idx + 90])
    prompt_avgs.append((prompt_idx, np.mean(values), len(values)))

# Sort by average ΔRCI
prompt_avgs_sorted = sorted(prompt_avgs, key=lambda x: x[1], reverse=True)
print("\nTop 5 prompts (highest avg ΔRCI):")
for idx, avg, n in prompt_avgs_sorted[:5]:
    print(f"  Prompt {idx+1}: {avg:+.4f} (n={n})")

print("\nBottom 5 prompts (lowest avg ΔRCI):")
for idx, avg, n in prompt_avgs_sorted[-5:]:
    print(f"  Prompt {idx+1}: {avg:+.4f} (n={n})")

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 16))

# Plot 1: ΔRCI per trial with trend line
ax1 = fig.add_subplot(3, 3, 1)
trials = np.arange(1, 101)
ax1.bar(trials, delta_cold, color=['green' if d > 0 else 'red' for d in delta_cold], alpha=0.6)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax1.axhline(y=mean_drci, color='blue', linestyle='--', linewidth=2, label=f'Mean = {mean_drci:+.4f}')

# Add trend line
z = np.polyfit(trials, delta_cold, 1)
p = np.poly1d(z)
ax1.plot(trials, p(trials), 'r-', linewidth=2, label=f'Trend: slope = {z[0]:.4f}')

ax1.set_xlabel('Trial Number')
ax1.set_ylabel('ΔRCI (True - Cold)')
ax1.set_title('GPT-4o-mini: ΔRCI per Trial (n=100)', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Cumulative mean
ax2 = fig.add_subplot(3, 3, 2)
ax2.plot(trials, cumulative_mean, 'b-', linewidth=2, label='Cumulative Mean')
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
ax2.fill_between(trials, 0, cumulative_mean,
                  where=[c > 0 for c in cumulative_mean], alpha=0.3, color='green', label='History Helps')
ax2.fill_between(trials, 0, cumulative_mean,
                  where=[c <= 0 for c in cumulative_mean], alpha=0.3, color='red', label='History Hurts')
ax2.set_xlabel('Trial Number')
ax2.set_ylabel('Cumulative Mean ΔRCI')
ax2.set_title('Cumulative Mean ΔRCI Over 100 Trials', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Rolling means
ax3 = fig.add_subplot(3, 3, 3)
ax3.plot(trials, rolling_mean_5, 'g-', linewidth=2, alpha=0.8, label='5-trial rolling')
ax3.plot(trials, rolling_mean_10, 'b-', linewidth=2, alpha=0.8, label='10-trial rolling')
ax3.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
ax3.set_xlabel('Trial Number')
ax3.set_ylabel('Rolling Mean ΔRCI')
ax3.set_title('Rolling Mean ΔRCI', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Quartile boxplots
ax4 = fig.add_subplot(3, 3, 4)
bp = ax4.boxplot([q1, q2, q3, q4], labels=['Q1 (1-25)', 'Q2 (26-50)', 'Q3 (51-75)', 'Q4 (76-100)'],
                  patch_artist=True)
colors = ['#ff9999', '#ffcc99', '#99ff99', '#99ccff']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax4.set_ylabel('ΔRCI')
ax4.set_title('ΔRCI by Quartile', fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# Add quartile means
for i, q in enumerate([q1, q2, q3, q4]):
    ax4.text(i+1, np.max(q) + 0.02, f'μ={np.mean(q):+.3f}', ha='center', fontsize=9)

# Plot 5: First 50 vs Last 50
ax5 = fig.add_subplot(3, 3, 5)
bp2 = ax5.boxplot([first_50, last_50], labels=['First 50 (1-50)', 'Last 50 (51-100)'],
                   patch_artist=True)
bp2['boxes'][0].set_facecolor('#ff9999')
bp2['boxes'][1].set_facecolor('#99ff99')
ax5.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax5.set_ylabel('ΔRCI')
ax5.set_title(f'First vs Last 50 Trials\n(Change: {l50_mean - f50_mean:+.4f})', fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

# Add means
for i, (d, label) in enumerate([(first_50, 'First 50'), (last_50, 'Last 50')]):
    ax5.text(i+1, np.max(d) + 0.02, f'μ={np.mean(d):+.3f}', ha='center', fontsize=10)

# Plot 6: Entanglement trajectory
ax6 = fig.add_subplot(3, 3, 6)
ax6.plot(trials, entanglements, 'purple', linewidth=2)
ax6.set_xlabel('Trial Number')
ax6.set_ylabel('Entanglement State (E_t)')
ax6.set_title('Entanglement Trajectory (n=100)', fontweight='bold')
ax6.grid(True, alpha=0.3)

# Plot 7: True vs Cold alignment
ax7 = fig.add_subplot(3, 3, 7)
ax7.plot(trials, true_aligns, 'g-', linewidth=1.5, alpha=0.7, label='True History')
ax7.plot(trials, cold_aligns, 'r--', linewidth=1.5, alpha=0.7, label='Cold Start')
ax7.fill_between(trials, true_aligns, cold_aligns,
                  where=[t > c for t, c in zip(true_aligns, cold_aligns)],
                  alpha=0.3, color='green', label='True > Cold')
ax7.fill_between(trials, true_aligns, cold_aligns,
                  where=[t <= c for t, c in zip(true_aligns, cold_aligns)],
                  alpha=0.3, color='red', label='Cold >= True')
ax7.set_xlabel('Trial Number')
ax7.set_ylabel('Alignment Score')
ax7.set_title('True vs Cold Alignment', fontweight='bold')
ax7.legend(fontsize=8)
ax7.grid(True, alpha=0.3)

# Plot 8: Distribution histogram
ax8 = fig.add_subplot(3, 3, 8)
ax8.hist(delta_cold, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
ax8.axvline(x=0, color='black', linestyle='-', linewidth=2)
ax8.axvline(x=mean_drci, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_drci:+.4f}')
ax8.set_xlabel('ΔRCI')
ax8.set_ylabel('Frequency')
ax8.set_title('ΔRCI Distribution (n=100)', fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3)

# Plot 9: Prompt cycle comparison
ax9 = fig.add_subplot(3, 3, 9)
cycle_means = [np.mean(cycle1), np.mean(cycle2), np.mean(cycle3), np.mean(remaining)]
cycle_stds = [np.std(cycle1), np.std(cycle2), np.std(cycle3), np.std(remaining)]
cycle_labels = ['Cycle 1\n(1-30)', 'Cycle 2\n(31-60)', 'Cycle 3\n(61-90)', 'Partial 4\n(91-100)']
bars = ax9.bar(range(4), cycle_means, yerr=cycle_stds, capsize=5,
               color=['#4c72b0', '#55a868', '#c44e52', '#8172b2'], alpha=0.8)
ax9.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax9.set_xticks(range(4))
ax9.set_xticklabels(cycle_labels)
ax9.set_ylabel('Mean ΔRCI')
ax9.set_title('ΔRCI by Prompt Cycle', fontweight='bold')
ax9.grid(True, alpha=0.3, axis='y')

plt.suptitle('MCH v8.1: GPT-4o-mini Extended Analysis (n=100 Trials)\n' +
             f'Overall: Mean ΔRCI = {mean_drci:+.4f}, p = {p_val:.4f}, Cohen\'s d = {cohens_d:+.3f}',
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('mch_gpt4o_mini_n100_analysis.png', dpi=300, bbox_inches='tight')
print('\nVisualization saved: mch_gpt4o_mini_n100_analysis.png')

# Save merged results
merged_results = {
    'metadata': {
        'description': 'GPT-4o-mini MCH v8.1 combined results (n=100)',
        'original_file': 'mch_results_gpt4o_mini_replication.json',
        'extension_file': 'mch_results_gpt4o_mini_trials_31_100.json',
        'total_trials': 100,
        'model': 'gpt-4o-mini'
    },
    'statistics': {
        'mean_drci': mean_drci,
        'std_drci': std_drci,
        'sem_drci': sem_drci,
        'ci_95_lower': mean_drci - ci_95,
        'ci_95_upper': mean_drci + ci_95,
        't_statistic': t_stat,
        'p_value': p_val,
        'cohens_d': cohens_d,
        'positive_trials': positive_count,
        'first_50_mean': f50_mean,
        'last_50_mean': l50_mean,
        'temporal_change': l50_mean - f50_mean
    },
    'trials': all_trials
}

with open('mch_results_gpt4o_mini_n100_merged.json', 'w', encoding='utf-8') as f:
    json.dump(merged_results, f, indent=2, ensure_ascii=False)
print('Merged results saved: mch_results_gpt4o_mini_n100_merged.json')

print("\n" + "="*80)
print("FINAL CONCLUSIONS")
print("="*80)

print(f"""
### GPT-4o-mini at n=100: Does "History Helps" Hold at Scale?

SUMMARY STATISTICS:
- Mean ΔRCI: {mean_drci:+.4f} (95% CI: [{mean_drci - ci_95:+.4f}, {mean_drci + ci_95:+.4f}])
- Statistical test: t(99) = {t_stat:.3f}, p = {p_val:.4f}
- Effect size: Cohen's d = {cohens_d:+.3f}
- Positive trials: {positive_count}/100 ({positive_count}%)

TEMPORAL DYNAMICS:
- First 50 trials: Mean = {f50_mean:+.4f}
- Last 50 trials: Mean = {l50_mean:+.4f}
- Change over time: {l50_mean - f50_mean:+.4f}

INTERPRETATION:
""")

if p_val < 0.05:
    if mean_drci > 0:
        print("- SIGNIFICANT POSITIVE EFFECT: Conversation history HELPS at n=100")
    else:
        print("- SIGNIFICANT NEGATIVE EFFECT: Conversation history HURTS at n=100")
else:
    print("- NO SIGNIFICANT EFFECT: Cannot confirm history helps or hurts at n=100")
    if l50_mean > f50_mean:
        print("- TREND: Performance appears to IMPROVE over time (but not significant)")
    else:
        print("- TREND: Performance appears to DECLINE over time (but not significant)")

print(f"""
- The 95% CI [{mean_drci - ci_95:+.4f}, {mean_drci + ci_95:+.4f}] {'excludes' if p_val < 0.05 else 'includes'} zero
- The effect is {'small' if abs(cohens_d) < 0.2 else 'medium' if abs(cohens_d) < 0.5 else 'large'} by conventional standards

COMPARISON WITH n=30:
- Original n=30 Mean ΔRCI: +0.0010 (p = 0.9548)
- Extended n=100 Mean ΔRCI: {mean_drci:+.4f} (p = {p_val:.4f})
- Direction: {'Same' if np.sign(mean_drci) == np.sign(0.0010) or (abs(mean_drci) < 0.01 and abs(0.0010) < 0.01) else 'Changed'}
""")
