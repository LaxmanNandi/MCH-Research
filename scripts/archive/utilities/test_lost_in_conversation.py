#!/usr/bin/env python3
"""
Test "Lost in Conversation" Phenomenon (Laban et al. 2025)
Using MCH Research Program Data

Experiments:
1. Var_Ratio progression: Does variance increase with conversation length?
2. Convergent vs Divergent: Do divergent models show degradation pattern?
3. Recovery analysis: Can models recover after entering divergent regime?
4. Domain comparison: Which domain shows more "getting lost"?
5. Llama trajectory: Trace Llama models' descent into divergence

Hypothesis: "Lost in Conversation" = Divergent Entanglement (Var_Ratio > 1)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, linregress
from scipy.ndimage import gaussian_filter1d

# Load position-level data
print("="*80)
print("TESTING 'LOST IN CONVERSATION' PHENOMENON")
print("="*80)

df = pd.read_csv('c:/Users/barla/mch_experiments/analysis/entanglement_position_data.csv')

print(f"\nDataset: {len(df)} observations, {df['model'].nunique()} models")

# ============================================================================
# EXPERIMENT 1: Variance Ratio Progression Over Conversation
# ============================================================================

print("\n" + "="*80)
print("EXPERIMENT 1: Does Var_Ratio increase with conversation length?")
print("="*80)
print("\nHypothesis: If 'Lost in Conversation' is real, Var_Ratio should")
print("increase as position increases (models become more unstable).")

progression_results = []

for model_name in df['model'].unique():
    model_df = df[df['model'] == model_name].sort_values('position')

    # Compute trend: correlation between position and Var_Ratio
    r, p = pearsonr(model_df['position'], model_df['var_ratio'])

    # Linear regression slope
    slope, intercept, r_slope, p_slope, stderr = linregress(model_df['position'], model_df['var_ratio'])

    # Mean Var_Ratio in early (P1-10) vs late (P21-30) conversation
    early = model_df[model_df['position'] <= 10]['var_ratio'].mean()
    late = model_df[model_df['position'] >= 21]['var_ratio'].mean()
    delta = late - early

    # Count positions where divergent (Var_Ratio > 1)
    n_divergent = (model_df['var_ratio'] > 1.0).sum()
    pct_divergent = n_divergent / len(model_df) * 100

    domain = 'Medical' if '(Med)' in model_name else 'Philosophy'

    progression_results.append({
        'model': model_name,
        'domain': domain,
        'correlation_r': r,
        'correlation_p': p,
        'slope': slope,
        'slope_p': p_slope,
        'early_var_ratio': early,
        'late_var_ratio': late,
        'delta_early_late': delta,
        'n_divergent_positions': n_divergent,
        'pct_divergent': pct_divergent,
        'pattern': 'INCREASING' if slope > 0 else 'DECREASING'
    })

prog_df = pd.DataFrame(progression_results).sort_values('slope', ascending=False)

print("\nModels with INCREASING Var_Ratio (getting more unstable):")
print(prog_df[prog_df['slope'] > 0][['model', 'domain', 'slope', 'delta_early_late', 'pct_divergent']])

print("\nModels with DECREASING Var_Ratio (getting more stable):")
print(prog_df[prog_df['slope'] < 0][['model', 'domain', 'slope', 'delta_early_late', 'pct_divergent']])

# ============================================================================
# EXPERIMENT 2: Convergent vs Divergent Model Classes
# ============================================================================

print("\n" + "="*80)
print("EXPERIMENT 2: Do divergent models show degradation?")
print("="*80)

# Classify models by mean Var_Ratio
model_summary = df.groupby('model').agg({
    'var_ratio': 'mean',
    'drci': 'mean'
}).reset_index()

model_summary['class'] = model_summary['var_ratio'].apply(
    lambda x: 'DIVERGENT' if x > 1.0 else 'CONVERGENT'
)

print("\nModel Classification:")
print(model_summary.sort_values('var_ratio', ascending=False))

convergent = model_summary[model_summary['class'] == 'CONVERGENT']
divergent = model_summary[model_summary['class'] == 'DIVERGENT']

print(f"\nCONVERGENT models (n={len(convergent)}): mean Var_Ratio = {convergent['var_ratio'].mean():.3f}")
print(f"DIVERGENT models (n={len(divergent)}): mean Var_Ratio = {divergent['var_ratio'].mean():.3f}")

# ============================================================================
# EXPERIMENT 3: Recovery Analysis
# ============================================================================

print("\n" + "="*80)
print("EXPERIMENT 3: Can models recover after entering divergent regime?")
print("="*80)
print("\nLaban et al. claim: 'models get lost and do NOT recover'")
print("Test: If a model goes divergent (Var_Ratio > 1), does it return to convergent?")

recovery_results = []

for model_name in df['model'].unique():
    model_df = df[df['model'] == model_name].sort_values('position')

    # Find first divergent position
    divergent_positions = model_df[model_df['var_ratio'] > 1.0]['position'].values

    if len(divergent_positions) > 0:
        first_divergent = divergent_positions[0]

        # Check if model returns to convergent after first divergence
        after_divergence = model_df[model_df['position'] > first_divergent]
        convergent_after = (after_divergence['var_ratio'] <= 1.0).sum()
        total_after = len(after_divergence)

        recovery_rate = convergent_after / total_after * 100 if total_after > 0 else 0

        recovery_results.append({
            'model': model_name,
            'first_divergent_pos': first_divergent,
            'positions_after': total_after,
            'convergent_after': convergent_after,
            'recovery_rate': recovery_rate,
            'recovers': 'YES' if recovery_rate > 50 else 'NO'
        })

if recovery_results:
    recovery_df = pd.DataFrame(recovery_results).sort_values('recovery_rate')
    print("\nRecovery Analysis (models that went divergent):")
    print(recovery_df)

    recovers_yes = recovery_df[recovery_df['recovers'] == 'YES']
    recovers_no = recovery_df[recovery_df['recovers'] == 'NO']

    print(f"\nModels that RECOVER (>50% return to convergent): {len(recovers_yes)}")
    print(f"Models that DO NOT RECOVER: {len(recovers_no)}")

    if len(recovers_no) > 0:
        print(f"\nModels consistent with Laban et al. (do not recover):")
        print(recovers_no[['model', 'first_divergent_pos', 'recovery_rate']])

# ============================================================================
# EXPERIMENT 4: Domain Comparison
# ============================================================================

print("\n" + "="*80)
print("EXPERIMENT 4: Which domain shows more 'getting lost'?")
print("="*80)

domain_summary = df.groupby('model').agg({
    'var_ratio': 'mean',
    'drci': 'mean'
}).reset_index()

domain_summary['domain'] = domain_summary['model'].apply(
    lambda x: 'Medical' if '(Med)' in x else 'Philosophy'
)

domain_stats = domain_summary.groupby('domain').agg({
    'var_ratio': ['mean', 'std', 'median'],
    'drci': ['mean', 'std', 'median']
})

print("\nDomain-level statistics:")
print(domain_stats)

# Position-level progression by domain
domain_progression = df.groupby(['position']).agg({
    'var_ratio': 'mean',
    'drci': 'mean'
}).reset_index()

# Separate by domain
med_df = df[df['model'].str.contains('Med')]
phil_df = df[df['model'].str.contains('Phil')]

med_prog = med_df.groupby('position')['var_ratio'].mean()
phil_prog = phil_df.groupby('position')['var_ratio'].mean()

print("\nMean Var_Ratio by domain across positions:")
print(f"Medical: {med_prog.mean():.3f} (range: {med_prog.min():.3f} - {med_prog.max():.3f})")
print(f"Philosophy: {phil_prog.mean():.3f} (range: {phil_prog.min():.3f} - {phil_prog.max():.3f})")

# ============================================================================
# EXPERIMENT 5: Llama Trajectory (Extreme Divergence Case Study)
# ============================================================================

print("\n" + "="*80)
print("EXPERIMENT 5: Llama Models - Descent into Divergence")
print("="*80)

llama_models = [m for m in df['model'].unique() if 'Llama' in m]

print(f"\nLlama models: {llama_models}")

for llama in llama_models:
    llama_df = df[df['model'] == llama].sort_values('position')

    print(f"\n{llama}:")
    print(f"  Mean Var_Ratio: {llama_df['var_ratio'].mean():.3f}")
    print(f"  P30 Var_Ratio: {llama_df[llama_df['position'] == 30]['var_ratio'].values[0]:.3f}")
    print(f"  P1 Var_Ratio: {llama_df[llama_df['position'] == 1]['var_ratio'].values[0]:.3f}")
    print(f"  Max Var_Ratio: {llama_df['var_ratio'].max():.3f} at P{llama_df.loc[llama_df['var_ratio'].idxmax(), 'position']:.0f}")
    print(f"  Positions divergent (>1.0): {(llama_df['var_ratio'] > 1.0).sum()}/30")

    # Find turning point: when model goes from convergent to divergent
    convergent_positions = llama_df[llama_df['var_ratio'] <= 1.0]['position'].values
    divergent_positions = llama_df[llama_df['var_ratio'] > 1.0]['position'].values

    if len(convergent_positions) > 0 and len(divergent_positions) > 0:
        last_convergent = convergent_positions[-1] if convergent_positions[-1] < divergent_positions[0] else None
        first_divergent = divergent_positions[0]

        if last_convergent:
            print(f"  Turning point: P{last_convergent:.0f} (convergent) → P{first_divergent:.0f} (divergent)")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# ===================================================================
# Plot 1: Var_Ratio progression by model
# ===================================================================

ax = fig.add_subplot(gs[0, :])

for model in df['model'].unique():
    model_df = df[df['model'] == model].sort_values('position')
    color = 'crimson' if '(Med)' in model else 'steelblue'
    label = model.split(' (')[0]

    # Smooth with gaussian filter
    smoothed = gaussian_filter1d(model_df['var_ratio'].values, sigma=1.5)

    ax.plot(model_df['position'], smoothed,
            color=color, alpha=0.7, linewidth=1.5, label=label)

ax.axhline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.5,
           label='Convergent/Divergent Threshold')
ax.set_xlabel('Conversation Position', fontsize=12)
ax.set_ylabel('Var_Ratio (smoothed)', fontsize=12)
ax.set_title('"Lost in Conversation" Test: Var_Ratio Progression',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 31)

# ===================================================================
# Plot 2: Early vs Late Var_Ratio
# ===================================================================

ax = fig.add_subplot(gs[1, 0])

models = prog_df['model'].apply(lambda x: x.split(' (')[0])
colors = ['crimson' if '(Med)' in m else 'steelblue' for m in prog_df['model']]

x = np.arange(len(models))
ax.bar(x, prog_df['delta_early_late'], color=colors, alpha=0.7)
ax.axhline(0, color='black', linewidth=1, alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Δ Var_Ratio (Late - Early)', fontsize=11)
ax.set_title('Change from Early (P1-10) to Late (P21-30)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# ===================================================================
# Plot 3: Convergent vs Divergent Classes
# ===================================================================

ax = fig.add_subplot(gs[1, 1])

convergent_models = model_summary[model_summary['class'] == 'CONVERGENT']['model'].apply(lambda x: x.split(' (')[0])
divergent_models = model_summary[model_summary['class'] == 'DIVERGENT']['model'].apply(lambda x: x.split(' (')[0])

convergent_var = model_summary[model_summary['class'] == 'CONVERGENT']['var_ratio'].values
divergent_var = model_summary[model_summary['class'] == 'DIVERGENT']['var_ratio'].values

x_conv = np.arange(len(convergent_models))
x_div = np.arange(len(divergent_models))

ax.bar(x_conv, convergent_var, color='steelblue', alpha=0.7, label='Convergent')
ax.bar(x_div + len(x_conv) + 0.5, divergent_var, color='crimson', alpha=0.7, label='Divergent')
ax.axhline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.5)

all_labels = list(convergent_models) + [''] + list(divergent_models)
ax.set_xticks(np.arange(len(all_labels)))
ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Mean Var_Ratio', fontsize=11)
ax.set_title('Convergent vs Divergent Model Classes', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# ===================================================================
# Plot 4: Recovery Rate
# ===================================================================

if recovery_results:
    ax = fig.add_subplot(gs[1, 2])

    recovery_df_sorted = recovery_df.sort_values('recovery_rate')
    models_rec = recovery_df_sorted['model'].apply(lambda x: x.split(' (')[0])

    colors_rec = ['green' if r > 50 else 'red' for r in recovery_df_sorted['recovery_rate']]

    x = np.arange(len(models_rec))
    ax.barh(x, recovery_df_sorted['recovery_rate'], color=colors_rec, alpha=0.7)
    ax.axvline(50, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_yticks(x)
    ax.set_yticklabels(models_rec, fontsize=9)
    ax.set_xlabel('Recovery Rate (%)', fontsize=11)
    ax.set_title('Recovery After First Divergence', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

# ===================================================================
# Plot 5: Domain comparison trajectories
# ===================================================================

ax = fig.add_subplot(gs[2, 0])

med_mean = med_df.groupby('position')['var_ratio'].mean()
phil_mean = phil_df.groupby('position')['var_ratio'].mean()

med_sem = med_df.groupby('position')['var_ratio'].sem()
phil_sem = phil_df.groupby('position')['var_ratio'].sem()

ax.plot(med_mean.index, med_mean.values, color='crimson', linewidth=3, label='Medical', marker='o', markersize=5)
ax.fill_between(med_mean.index, med_mean - med_sem, med_mean + med_sem, color='crimson', alpha=0.2)

ax.plot(phil_mean.index, phil_mean.values, color='steelblue', linewidth=3, label='Philosophy', marker='s', markersize=5)
ax.fill_between(phil_mean.index, phil_mean - phil_sem, phil_mean + phil_sem, color='steelblue', alpha=0.2)

ax.axhline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax.set_xlabel('Position', fontsize=11)
ax.set_ylabel('Mean Var_Ratio', fontsize=11)
ax.set_title('Domain-Level Trajectories (mean ± SEM)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 31)

# ===================================================================
# Plot 6: Llama Descent into Divergence
# ===================================================================

ax = fig.add_subplot(gs[2, 1:])

for llama in llama_models:
    llama_df = df[df['model'] == llama].sort_values('position')
    label = llama.split(' (')[0]

    ax.plot(llama_df['position'], llama_df['var_ratio'],
            marker='o', markersize=6, linewidth=2.5, alpha=0.8, label=label)

    # Highlight P30
    p30_val = llama_df[llama_df['position'] == 30]['var_ratio'].values[0]
    ax.scatter([30], [p30_val], s=200, marker='*', edgecolor='black', linewidth=2, zorder=10)

ax.axhline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Threshold')
ax.axhline(3.0, color='orange', linestyle=':', linewidth=2, alpha=0.5, label='High-Risk (>3.0)')
ax.axvline(30, color='purple', linestyle=':', linewidth=2, alpha=0.3, label='P30 (Summarization)')

ax.set_xlabel('Position', fontsize=12)
ax.set_ylabel('Var_Ratio', fontsize=12)
ax.set_title('Llama Models: Descent into Extreme Divergence', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 31)
ax.set_ylim(bottom=0)

# Save
output_path = "c:/Users/barla/mch_experiments/analysis/lost_in_conversation_tests.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n[OK] Plot saved: {output_path}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*80)
print("SUMMARY: 'LOST IN CONVERSATION' PHENOMENON")
print("="*80)

print("\n1. PROGRESSION TEST:")
increasing = (prog_df['slope'] > 0).sum()
decreasing = (prog_df['slope'] < 0).sum()
print(f"   Models with INCREASING Var_Ratio: {increasing}/12 ({increasing/12*100:.1f}%)")
print(f"   Models with DECREASING Var_Ratio: {decreasing}/12 ({decreasing/12*100:.1f}%)")

print("\n2. CONVERGENT VS DIVERGENT:")
n_conv = (model_summary['var_ratio'] < 1.0).sum()
n_div = (model_summary['var_ratio'] >= 1.0).sum()
print(f"   CONVERGENT models: {n_conv}/12")
print(f"   DIVERGENT models: {n_div}/12")

if recovery_results:
    print("\n3. RECOVERY ANALYSIS:")
    n_recover = len(recovers_yes) if 'recovers_yes' in locals() else 0
    n_no_recover = len(recovers_no) if 'recovers_no' in locals() else 0
    print(f"   Models that RECOVER after divergence: {n_recover}")
    print(f"   Models that DO NOT recover: {n_no_recover}")
    if n_no_recover > 0:
        print(f"   [OK] Consistent with Laban et al. 'do not recover' claim")

print("\n4. DOMAIN COMPARISON:")
med_mean_var = df[df['model'].str.contains('Med')]['var_ratio'].mean()
phil_mean_var = df[df['model'].str.contains('Phil')]['var_ratio'].mean()
print(f"   Medical mean Var_Ratio: {med_mean_var:.3f}")
print(f"   Philosophy mean Var_Ratio: {phil_mean_var:.3f}")
if med_mean_var > phil_mean_var:
    print(f"   Medical shows MORE divergence tendency")

print("\n5. LLAMA ANOMALY:")
for llama in llama_models:
    llama_df = df[df['model'] == llama]
    p30_var = llama_df[llama_df['position'] == 30]['var_ratio'].values[0]
    esi = 1 / abs(1 - p30_var)
    print(f"   {llama.split('(')[0]}: Var_Ratio(P30) = {p30_var:.2f}, ESI = {esi:.2f}")
    if p30_var > 3.0:
        print(f"      [WARNING] EXTREME DIVERGENCE (Var_Ratio > 3.0)")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# Save summary
summary_data = {
    'experiment': ['Progression', 'Classification', 'Recovery', 'Domain', 'Llama'],
    'models_increasing_var': [increasing, None, None, None, None],
    'convergent_models': [None, n_conv, None, None, None],
    'divergent_models': [None, n_div, None, None, None],
    'models_no_recovery': [None, None, n_no_recover if recovery_results else None, None, None],
    'medical_mean_var': [None, None, None, med_mean_var, None],
    'philosophy_mean_var': [None, None, None, phil_mean_var, None]
}

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('c:/Users/barla/mch_experiments/analysis/lost_in_conversation_summary.csv', index=False)
print("\n[OK] Summary saved: lost_in_conversation_summary.csv")

# Save progression details
prog_df.to_csv('c:/Users/barla/mch_experiments/analysis/lost_in_conversation_progression.csv', index=False)
print("[OK] Progression data saved: lost_in_conversation_progression.csv")
