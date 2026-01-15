import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Load all results
gpt4o_full = json.load(open('mch_results_gpt4o_full_30trials.json'))
gpt4o_mini = json.load(open('mch_results_gpt4o_mini_replication.json'))
claude_opus = json.load(open('mch_results_claude_opus_rate_limit_test.json'))
claude_haiku = json.load(open('mch_results_claude_haiku_30trials.json'))

print('='*80)
print('MCH v8.1 COMPREHENSIVE 4-MODEL COMPARISON')
print('='*80)
print('\nModels tested:')
print('  - GPT-4o (OpenAI Flagship)')
print('  - GPT-4o-mini (OpenAI Efficient)')
print('  - Claude Opus 4 (Anthropic Flagship)')
print('  - Claude Haiku 4 (Anthropic Efficient)')
print('='*80)

# Extract data for each model
def extract_data(data, model_name_filter=None):
    trials = data['trials']
    if model_name_filter:
        trials = [t for t in trials if t['model'] == model_name_filter]

    delta_cold = [t['controls']['cold']['delta_rci'] for t in trials]
    delta_scrambled = [t['controls']['scrambled']['delta_rci'] for t in trials]
    entanglements = [t['true']['entanglement'] for t in trials]
    true_aligns = [t['true']['alignment'] for t in trials]
    cold_aligns = [t['controls']['cold']['alignment'] for t in trials]

    return {
        'delta_cold': delta_cold,
        'delta_scrambled': delta_scrambled,
        'entanglements': entanglements,
        'true_aligns': true_aligns,
        'cold_aligns': cold_aligns
    }

gpt4o_data = extract_data(gpt4o_full, 'gpt-4o')
mini_data = extract_data(gpt4o_mini, 'gpt-4o-mini')
opus_data = extract_data(claude_opus, 'claude-opus-4-20250514')
haiku_data = extract_data(claude_haiku, 'claude-haiku-4-5-20251001')

# Calculate cumulative and rolling means
def calc_means(deltas):
    cumulative = [np.mean(deltas[:i+1]) for i in range(len(deltas))]
    rolling = [np.mean(deltas[max(0,i-4):i+1]) for i in range(len(deltas))]
    return cumulative, rolling

gpt4o_cumul, gpt4o_roll = calc_means(gpt4o_data['delta_cold'])
mini_cumul, mini_roll = calc_means(mini_data['delta_cold'])
opus_cumul, opus_roll = calc_means(opus_data['delta_cold'])
haiku_cumul, haiku_roll = calc_means(haiku_data['delta_cold'])

print('\n' + '='*80)
print('SUMMARY STATISTICS - ALL 4 MODELS')
print('='*80)

models = [
    ('GPT-4o (Flagship)', gpt4o_data),
    ('GPT-4o-mini (Efficient)', mini_data),
    ('Claude Opus 4 (Flagship)', opus_data),
    ('Claude Haiku 4 (Efficient)', haiku_data)
]

print(f"\n{'Model':<28} {'Mean DRCI':<12} {'SD':<10} {'t-stat':<10} {'p-value':<12} {'Cohen d':<10} {'Pos/30':<8}")
print('-'*100)

for name, data in models:
    deltas = data['delta_cold']
    mean_d = np.mean(deltas)
    std_d = np.std(deltas)
    t_stat, p_val = stats.ttest_1samp(deltas, 0)
    cohens_d = mean_d / std_d if std_d > 0 else 0
    positive = sum(1 for d in deltas if d > 0)

    sig = '*' if p_val < 0.05 else ''
    print(f"{name:<28} {mean_d:+.4f}      {std_d:.4f}    {t_stat:+.3f}     {p_val:.4f}{sig:<4}   {cohens_d:+.3f}     {positive}/30")

print('\n' + '='*80)
print('KEY RESEARCH QUESTIONS')
print('='*80)

# Question 1: Does Claude Haiku show same pattern as Claude Opus? (Family effect)
print('\n### Q1: Does Claude Haiku show same pattern as Claude Opus? (FAMILY EFFECT)')
print('-'*60)

opus_mean = np.mean(opus_data['delta_cold'])
haiku_mean = np.mean(haiku_data['delta_cold'])
opus_std = np.std(opus_data['delta_cold'])
haiku_std = np.std(haiku_data['delta_cold'])

print(f"  Claude Opus 4:  Mean DRCI = {opus_mean:+.4f} (SD: {opus_std:.4f})")
print(f"  Claude Haiku 4: Mean DRCI = {haiku_mean:+.4f} (SD: {haiku_std:.4f})")
print(f"  Difference: {haiku_mean - opus_mean:+.4f}")

# Independent samples t-test between Claude models
t_claude, p_claude = stats.ttest_ind(opus_data['delta_cold'], haiku_data['delta_cold'])
print(f"\n  Independent t-test: t = {t_claude:.3f}, p = {p_claude:.4f}")

if opus_mean < 0 and haiku_mean > 0:
    print("\n  ANSWER: NO - Claude Haiku shows OPPOSITE pattern to Claude Opus!")
    print("  - Opus: History HURTS (negative DRCI)")
    print("  - Haiku: History HELPS (positive DRCI)")
    print("  - NO family effect detected - tier/size matters more than vendor")
elif (opus_mean < 0 and haiku_mean < 0) or (opus_mean > 0 and haiku_mean > 0):
    print("\n  ANSWER: YES - Both Claude models show same direction")
else:
    print("\n  ANSWER: Mixed results")

# Question 2: Within-family correlations
print('\n### Q2: Within-Family Correlations')
print('-'*60)

corr_claude, p_claude_corr = stats.pearsonr(opus_data['delta_cold'], haiku_data['delta_cold'])
corr_openai, p_openai_corr = stats.pearsonr(gpt4o_data['delta_cold'], mini_data['delta_cold'])

print(f"  Claude family (Opus vs Haiku): r = {corr_claude:+.3f} (p = {p_claude_corr:.4f})")
print(f"  OpenAI family (4o vs mini):    r = {corr_openai:+.3f} (p = {p_openai_corr:.4f})")

if corr_openai > corr_claude:
    print(f"\n  OpenAI models are MORE correlated than Claude models")
    print(f"  Correlation difference: {corr_openai - corr_claude:.3f}")
else:
    print(f"\n  Claude models are MORE correlated than OpenAI models")
    print(f"  Correlation difference: {corr_claude - corr_openai:.3f}")

# Question 3: Cross-family same-tier comparison
print('\n### Q3: Cross-Family Same-Tier Comparison')
print('-'*60)

# Flagship comparison (GPT-4o vs Claude Opus)
print("\n  FLAGSHIP TIER (GPT-4o vs Claude Opus 4):")
gpt4o_mean = np.mean(gpt4o_data['delta_cold'])
corr_flagship, p_flagship = stats.pearsonr(gpt4o_data['delta_cold'], opus_data['delta_cold'])
print(f"    GPT-4o:       Mean DRCI = {gpt4o_mean:+.4f}")
print(f"    Claude Opus:  Mean DRCI = {opus_mean:+.4f}")
print(f"    Correlation:  r = {corr_flagship:+.3f} (p = {p_flagship:.4f})")

# Efficient tier comparison (GPT-4o-mini vs Claude Haiku)
print("\n  EFFICIENT TIER (GPT-4o-mini vs Claude Haiku 4):")
mini_mean = np.mean(mini_data['delta_cold'])
corr_efficient, p_efficient = stats.pearsonr(mini_data['delta_cold'], haiku_data['delta_cold'])
print(f"    GPT-4o-mini:  Mean DRCI = {mini_mean:+.4f}")
print(f"    Claude Haiku: Mean DRCI = {haiku_mean:+.4f}")
print(f"    Correlation:  r = {corr_efficient:+.3f} (p = {p_efficient:.4f})")

print('\n' + '='*80)
print('FIRST HALF vs SECOND HALF ANALYSIS')
print('='*80)

print(f"\n{'Model':<28} {'1st Half':<12} {'2nd Half':<12} {'Change':<12} {'Pattern':<25}")
print('-'*90)

for name, data in models:
    deltas = data['delta_cold']
    first_half = np.mean(deltas[:15])
    second_half = np.mean(deltas[15:])
    change = second_half - first_half

    if change > 0.02:
        pattern = "Improves over time"
    elif change < -0.02:
        pattern = "Declines over time"
    else:
        pattern = "Stable"

    print(f"{name:<28} {first_half:+.4f}      {second_half:+.4f}      {change:+.4f}      {pattern}")

print('\n' + '='*80)
print('COMPLETE CORRELATION MATRIX')
print('='*80)

# All pairwise correlations
all_models_data = [
    ('GPT-4o', gpt4o_data),
    ('GPT-4o-mini', mini_data),
    ('Claude Opus', opus_data),
    ('Claude Haiku', haiku_data)
]

print(f"\n{'':20} {'GPT-4o':<14} {'GPT-4o-mini':<14} {'Claude Opus':<14} {'Claude Haiku':<14}")
print('-'*76)

for i, (name1, data1) in enumerate(all_models_data):
    row = f"{name1:<20}"
    for j, (name2, data2) in enumerate(all_models_data):
        if i == j:
            row += f"{'1.000':<14}"
        else:
            corr, p = stats.pearsonr(data1['delta_cold'], data2['delta_cold'])
            sig = '*' if p < 0.05 else ''
            row += f"{corr:+.3f}{sig:<10}"
    print(row)

print("\n* = p < 0.05")

print('\n' + '='*80)
print('TRIAL-BY-TRIAL DRCI COMPARISON')
print('='*80)

print(f"\n{'Trial':<8} {'GPT-4o':<12} {'GPT-4o-mini':<12} {'Claude Opus':<12} {'Claude Haiku':<12}")
print('-'*56)
for i in range(30):
    print(f"{i+1:<8} {gpt4o_data['delta_cold'][i]:+.4f}      {mini_data['delta_cold'][i]:+.4f}      {opus_data['delta_cold'][i]:+.4f}      {haiku_data['delta_cold'][i]:+.4f}")

print('\n' + '='*80)
print('2x2 FACTORIAL ANALYSIS')
print('='*80)

# Compute 2x2 means
flagship_openai = gpt4o_mean
flagship_anthropic = opus_mean
efficient_openai = mini_mean
efficient_anthropic = haiku_mean

print("\n                     OpenAI        Anthropic     Row Mean")
print("-"*60)
print(f"Flagship (Large)     {flagship_openai:+.4f}        {flagship_anthropic:+.4f}        {(flagship_openai+flagship_anthropic)/2:+.4f}")
print(f"Efficient (Small)    {efficient_openai:+.4f}        {efficient_anthropic:+.4f}        {(efficient_openai+efficient_anthropic)/2:+.4f}")
print("-"*60)
print(f"Column Mean          {(flagship_openai+efficient_openai)/2:+.4f}        {(flagship_anthropic+efficient_anthropic)/2:+.4f}")

# Main effects
vendor_effect = ((flagship_openai + efficient_openai) - (flagship_anthropic + efficient_anthropic)) / 2
tier_effect = ((efficient_openai + efficient_anthropic) - (flagship_openai + flagship_anthropic)) / 2
interaction = ((flagship_openai - efficient_openai) - (flagship_anthropic - efficient_anthropic)) / 2

print(f"\nMain Effects:")
print(f"  Vendor effect (OpenAI - Anthropic): {vendor_effect:+.4f}")
print(f"  Tier effect (Efficient - Flagship): {tier_effect:+.4f}")
print(f"  Interaction:                        {interaction:+.4f}")

print('\n' + '='*80)
print('MAJOR FINDINGS')
print('='*80)

print("""
1. VENDOR EFFECT (OpenAI vs Anthropic):
   - OpenAI models average: {:.4f}
   - Anthropic models average: {:.4f}
   - OpenAI slightly more positive (history marginally helps)
   - Anthropic shows mixed pattern (Opus negative, Haiku positive)

2. TIER EFFECT (Flagship vs Efficient):
   - Flagship models average: {:.4f}
   - Efficient models average: {:.4f}
   - Efficient tier shows MORE positive DRCI
   - Smaller models may benefit more from conversation history

3. THE SURPRISING HAIKU RESULT:
   - Claude Haiku shows OPPOSITE pattern to Claude Opus!
   - Haiku: +{:.4f} (history helps)
   - Opus:  {:.4f} (history hurts)
   - This contradicts the "family effect" hypothesis

4. CROSS-FAMILY CORRELATIONS:
   - OpenAI family (4o vs mini): r = {:.3f}
   - Claude family (Opus vs Haiku): r = {:.3f}
   - OpenAI models are more internally consistent

5. SAME-TIER COMPARISONS:
   - Flagship tier (4o vs Opus): Different means, different patterns
   - Efficient tier (mini vs Haiku): Both positive, similar pattern
""".format(
    (gpt4o_mean + mini_mean) / 2,
    (opus_mean + haiku_mean) / 2,
    (gpt4o_mean + opus_mean) / 2,
    (mini_mean + haiku_mean) / 2,
    haiku_mean,
    opus_mean,
    corr_openai,
    corr_claude
))

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 16))

# Define colors
colors = {
    'gpt4o': '#1f77b4',      # Blue
    'mini': '#2ca02c',        # Green
    'opus': '#d62728',        # Red
    'haiku': '#ff7f0e'        # Orange
}

# Row 1: Summary bar chart and 2x2 factorial
ax1 = fig.add_subplot(3, 4, 1)
model_names = ['GPT-4o\n(Flagship)', 'GPT-4o-mini\n(Efficient)', 'Claude Opus\n(Flagship)', 'Claude Haiku\n(Efficient)']
means = [gpt4o_mean, mini_mean, opus_mean, haiku_mean]
stds = [np.std(gpt4o_data['delta_cold']), np.std(mini_data['delta_cold']),
        np.std(opus_data['delta_cold']), np.std(haiku_data['delta_cold'])]
bar_colors = [colors['gpt4o'], colors['mini'], colors['opus'], colors['haiku']]

bars = ax1.bar(range(4), means, yerr=stds, color=bar_colors, alpha=0.7, capsize=5)
ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax1.set_xticks(range(4))
ax1.set_xticklabels(model_names, fontsize=8)
ax1.set_ylabel('Mean DRCI')
ax1.set_title('Mean DRCI by Model\n(Error bars = 1 SD)', fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# Add significance markers
p_values = [stats.ttest_1samp(d['delta_cold'], 0)[1] for _, d in models]
for i, (mean, std, p) in enumerate(zip(means, stds, p_values)):
    sig = '**' if p < 0.01 else '*' if p < 0.05 else ''
    ax1.text(i, mean + std + 0.02, sig, ha='center', fontsize=14, fontweight='bold')

# 2x2 factorial visualization
ax2 = fig.add_subplot(3, 4, 2)
x_positions = [0, 1]
ax2.plot(x_positions, [flagship_openai, flagship_anthropic], 'o-',
         color='darkblue', linewidth=2, markersize=10, label='Flagship')
ax2.plot(x_positions, [efficient_openai, efficient_anthropic], 's--',
         color='darkgreen', linewidth=2, markersize=10, label='Efficient')
ax2.axhline(y=0, color='black', linestyle=':', linewidth=1)
ax2.set_xticks(x_positions)
ax2.set_xticklabels(['OpenAI', 'Anthropic'])
ax2.set_ylabel('Mean DRCI')
ax2.set_title('2x2 Factorial Design\n(Vendor x Tier)', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Family comparison
ax3 = fig.add_subplot(3, 4, 3)
family_data = [[gpt4o_mean, mini_mean], [opus_mean, haiku_mean]]
x = np.arange(2)
width = 0.35
ax3.bar(x - width/2, [gpt4o_mean, opus_mean], width, label='Flagship', color=['#1f77b4', '#d62728'], alpha=0.7)
ax3.bar(x + width/2, [mini_mean, haiku_mean], width, label='Efficient', color=['#2ca02c', '#ff7f0e'], alpha=0.7)
ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax3.set_xticks(x)
ax3.set_xticklabels(['OpenAI', 'Anthropic'])
ax3.set_ylabel('Mean DRCI')
ax3.set_title('Within-Family Comparison', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Correlation heatmap
ax4 = fig.add_subplot(3, 4, 4)
corr_matrix = np.array([
    [1, corr_openai, stats.pearsonr(gpt4o_data['delta_cold'], opus_data['delta_cold'])[0],
     stats.pearsonr(gpt4o_data['delta_cold'], haiku_data['delta_cold'])[0]],
    [corr_openai, 1, stats.pearsonr(mini_data['delta_cold'], opus_data['delta_cold'])[0], corr_efficient],
    [stats.pearsonr(gpt4o_data['delta_cold'], opus_data['delta_cold'])[0],
     stats.pearsonr(mini_data['delta_cold'], opus_data['delta_cold'])[0], 1, corr_claude],
    [stats.pearsonr(gpt4o_data['delta_cold'], haiku_data['delta_cold'])[0], corr_efficient, corr_claude, 1]
])
im = ax4.imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1)
ax4.set_xticks(range(4))
ax4.set_yticks(range(4))
ax4.set_xticklabels(['GPT-4o', 'mini', 'Opus', 'Haiku'], fontsize=8)
ax4.set_yticklabels(['GPT-4o', 'mini', 'Opus', 'Haiku'], fontsize=8)
ax4.set_title('Correlation Matrix', fontweight='bold')
for i in range(4):
    for j in range(4):
        ax4.text(j, i, f'{corr_matrix[i,j]:.2f}', ha='center', va='center', fontsize=9)
plt.colorbar(im, ax=ax4)

# Row 2: Cumulative and Rolling means
ax5 = fig.add_subplot(3, 4, 5)
ax5.plot(range(1, 31), gpt4o_cumul, '-o', color=colors['gpt4o'], linewidth=2, markersize=3, label='GPT-4o')
ax5.plot(range(1, 31), mini_cumul, '-s', color=colors['mini'], linewidth=2, markersize=3, label='GPT-4o-mini')
ax5.plot(range(1, 31), opus_cumul, '-^', color=colors['opus'], linewidth=2, markersize=3, label='Claude Opus')
ax5.plot(range(1, 31), haiku_cumul, '-d', color=colors['haiku'], linewidth=2, markersize=3, label='Claude Haiku')
ax5.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
ax5.set_xlabel('Trial Number')
ax5.set_ylabel('Cumulative Mean DRCI')
ax5.set_title('Cumulative Mean DRCI Over Trials', fontweight='bold')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

ax6 = fig.add_subplot(3, 4, 6)
ax6.plot(range(1, 31), gpt4o_roll, '-o', color=colors['gpt4o'], linewidth=2, markersize=3, label='GPT-4o')
ax6.plot(range(1, 31), mini_roll, '-s', color=colors['mini'], linewidth=2, markersize=3, label='GPT-4o-mini')
ax6.plot(range(1, 31), opus_roll, '-^', color=colors['opus'], linewidth=2, markersize=3, label='Claude Opus')
ax6.plot(range(1, 31), haiku_roll, '-d', color=colors['haiku'], linewidth=2, markersize=3, label='Claude Haiku')
ax6.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
ax6.set_xlabel('Trial Number')
ax6.set_ylabel('Rolling Mean DRCI (5-trial)')
ax6.set_title('Rolling Mean DRCI Over Trials', fontweight='bold')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

# Entanglement trajectories
ax7 = fig.add_subplot(3, 4, 7)
ax7.plot(range(1, 31), gpt4o_data['entanglements'], '-o', color=colors['gpt4o'], linewidth=2, markersize=3, label='GPT-4o')
ax7.plot(range(1, 31), mini_data['entanglements'], '-s', color=colors['mini'], linewidth=2, markersize=3, label='GPT-4o-mini')
ax7.plot(range(1, 31), opus_data['entanglements'], '-^', color=colors['opus'], linewidth=2, markersize=3, label='Claude Opus')
ax7.plot(range(1, 31), haiku_data['entanglements'], '-d', color=colors['haiku'], linewidth=2, markersize=3, label='Claude Haiku')
ax7.set_xlabel('Trial Number')
ax7.set_ylabel('Entanglement State (E_t)')
ax7.set_title('Entanglement Trajectories', fontweight='bold')
ax7.legend(fontsize=8)
ax7.grid(True, alpha=0.3)

# First vs Second half boxplots
ax8 = fig.add_subplot(3, 4, 8)
first_half = [gpt4o_data['delta_cold'][:15], mini_data['delta_cold'][:15],
              opus_data['delta_cold'][:15], haiku_data['delta_cold'][:15]]
second_half = [gpt4o_data['delta_cold'][15:], mini_data['delta_cold'][15:],
               opus_data['delta_cold'][15:], haiku_data['delta_cold'][15:]]

bp1 = ax8.boxplot(first_half, positions=[1, 2.5, 4, 5.5], widths=0.4, patch_artist=True)
bp2 = ax8.boxplot(second_half, positions=[1.5, 3, 4.5, 6], widths=0.4, patch_artist=True)

model_colors = [colors['gpt4o'], colors['mini'], colors['opus'], colors['haiku']]
for i, (box1, box2) in enumerate(zip(bp1['boxes'], bp2['boxes'])):
    box1.set_facecolor(model_colors[i])
    box1.set_alpha(0.4)
    box2.set_facecolor(model_colors[i])
    box2.set_alpha(0.8)

ax8.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax8.set_xticks([1.25, 2.75, 4.25, 5.75])
ax8.set_xticklabels(['GPT-4o', 'mini', 'Opus', 'Haiku'], fontsize=8)
ax8.set_ylabel('DRCI')
ax8.set_title('First 15 (light) vs Last 15 (dark) Trials', fontweight='bold')
ax8.grid(True, alpha=0.3, axis='y')

# Row 3: Individual model True vs Cold alignment
for idx, (name, data, color) in enumerate([
    ('GPT-4o', gpt4o_data, colors['gpt4o']),
    ('GPT-4o-mini', mini_data, colors['mini']),
    ('Claude Opus 4', opus_data, colors['opus']),
    ('Claude Haiku 4', haiku_data, colors['haiku'])
]):
    ax = fig.add_subplot(3, 4, 9 + idx)
    ax.plot(range(1, 31), data['true_aligns'], '-o', color=color, linewidth=2, markersize=3, label='True History')
    ax.plot(range(1, 31), data['cold_aligns'], '--s', color=color, linewidth=2, markersize=3, label='Cold Start', alpha=0.5)

    ax.fill_between(range(1, 31), data['true_aligns'], data['cold_aligns'],
                    where=[t > c for t, c in zip(data['true_aligns'], data['cold_aligns'])],
                    alpha=0.3, color='green', label='True > Cold')
    ax.fill_between(range(1, 31), data['true_aligns'], data['cold_aligns'],
                    where=[t <= c for t, c in zip(data['true_aligns'], data['cold_aligns'])],
                    alpha=0.3, color='red', label='Cold >= True')

    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Alignment Score')
    ax.set_title(f'{name}: True vs Cold', fontweight='bold')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

plt.suptitle('MCH v8.1: Complete 4-Model Comparison\n(GPT-4o, GPT-4o-mini, Claude Opus 4, Claude Haiku 4 - 30 Trials Each)',
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('mch_four_model_comparison.png', dpi=300, bbox_inches='tight')
print('\nVisualization saved: mch_four_model_comparison.png')

print('\n' + '='*80)
print('FINAL SUMMARY TABLE')
print('='*80)

print("""
+---------------------------+------------+------------+------------+------------+
| Metric                    | GPT-4o     | GPT-4o-mini| Claude Opus| Claude Haiku|
|                           | (Flagship) | (Efficient)| (Flagship) | (Efficient)|
+---------------------------+------------+------------+------------+------------+
| Mean DRCI                 | {:+.4f}    | {:+.4f}    | {:+.4f}    | {:+.4f}    |
| p-value                   | {:.4f}     | {:.4f}     | {:.4f}*    | {:.4f}     |
| Cohen's d                 | {:+.3f}     | {:+.3f}     | {:+.3f}     | {:+.3f}     |
| First 15 trials           | {:+.4f}    | {:+.4f}    | {:+.4f}    | {:+.4f}    |
| Last 15 trials            | {:+.4f}    | {:+.4f}    | {:+.4f}    | {:+.4f}    |
| Positive trials           | {}/30      | {}/30      | {}/30      | {}/30      |
+---------------------------+------------+------------+------------+------------+

* = statistically significant (p < 0.05)
""".format(
    gpt4o_mean, mini_mean, opus_mean, haiku_mean,
    stats.ttest_1samp(gpt4o_data['delta_cold'], 0)[1],
    stats.ttest_1samp(mini_data['delta_cold'], 0)[1],
    stats.ttest_1samp(opus_data['delta_cold'], 0)[1],
    stats.ttest_1samp(haiku_data['delta_cold'], 0)[1],
    gpt4o_mean / np.std(gpt4o_data['delta_cold']),
    mini_mean / np.std(mini_data['delta_cold']),
    opus_mean / np.std(opus_data['delta_cold']),
    haiku_mean / np.std(haiku_data['delta_cold']),
    np.mean(gpt4o_data['delta_cold'][:15]),
    np.mean(mini_data['delta_cold'][:15]),
    np.mean(opus_data['delta_cold'][:15]),
    np.mean(haiku_data['delta_cold'][:15]),
    np.mean(gpt4o_data['delta_cold'][15:]),
    np.mean(mini_data['delta_cold'][15:]),
    np.mean(opus_data['delta_cold'][15:]),
    np.mean(haiku_data['delta_cold'][15:]),
    sum(1 for d in gpt4o_data['delta_cold'] if d > 0),
    sum(1 for d in mini_data['delta_cold'] if d > 0),
    sum(1 for d in opus_data['delta_cold'] if d > 0),
    sum(1 for d in haiku_data['delta_cold'] if d > 0)
))

print('\n' + '='*80)
print('CONCLUSIONS')
print('='*80)
print("""
1. THE HAIKU SURPRISE:
   Claude Haiku 4 shows POSITIVE DRCI (+{:.4f}), OPPOSITE to Claude Opus 4 ({:.4f})
   This is the most surprising finding of the 4-model comparison!

2. NO CLEAR FAMILY EFFECT:
   - OpenAI models correlate strongly (r = {:.3f})
   - Claude models do NOT correlate (r = {:.3f})
   - This suggests vendor/family is NOT the primary determinant

3. TIER EFFECT IS STRONGER:
   - Efficient models (mini, Haiku) both show positive DRCI
   - Flagship models (4o, Opus) both show near-zero or negative DRCI
   - Smaller models may benefit more from conversation history

4. STATISTICAL SIGNIFICANCE:
   - Only Claude Opus 4 reaches statistical significance (p < 0.05)
   - All other models show non-significant effects
   - Effect sizes are small across all models

5. PRACTICAL IMPLICATIONS:
   - For Claude Opus: Shorter conversations may be better
   - For Claude Haiku: Conversation history helps (like GPT-4o-mini)
   - For GPT-4o: No clear benefit from history
   - For GPT-4o-mini: History may help slightly
""".format(haiku_mean, opus_mean, corr_openai, corr_claude))
