#!/usr/bin/env python3
"""
Generate Figure 7: Domain Dominance Visualization
Generate Table 4: Exploration Arc Data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr

# Load data
df = pd.read_csv('C:/Users/barla/mch_experiments/scripts/validate/exploration_arc_with_var_ratio.csv')

# Create Figure 7
fig, ax = plt.subplots(figsize=(10, 8))

# Separate by domain
phil_df = df[df['domain'] == 'philosophy']
med_df = df[df['domain'] == 'medical']

# Plot philosophy models (blue)
ax.scatter(phil_df['var_ratio'], phil_df['exploration_arc'],
          s=120, alpha=0.7, color='#2E86AB', label='Philosophy (open-ended)',
          edgecolors='black', linewidth=1.5, zorder=3)

# Plot medical models (red)
ax.scatter(med_df['var_ratio'], med_df['exploration_arc'],
          s=120, alpha=0.7, color='#C73E1D', label='Medical (task-convergent)',
          edgecolors='black', linewidth=1.5, zorder=3)

# Add domain means as larger diamond markers
phil_mean_vr = phil_df['var_ratio'].mean()
phil_mean_arc = phil_df['exploration_arc'].mean()
med_mean_vr = med_df['var_ratio'].mean()
med_mean_arc = med_df['exploration_arc'].mean()

ax.scatter([phil_mean_vr], [phil_mean_arc], s=300, marker='D',
          color='#2E86AB', edgecolors='black', linewidth=2,
          label='Philosophy mean', zorder=5, alpha=0.9)
ax.scatter([med_mean_vr], [med_mean_arc], s=300, marker='D',
          color='#C73E1D', edgecolors='black', linewidth=2,
          label='Medical mean', zorder=5, alpha=0.9)

# Add reference lines
ax.axhline(y=1, color='gray', linestyle='--', linewidth=1.5, alpha=0.6,
          label='Arc = 1.0 (no change)', zorder=1)
ax.axvline(x=1, color='gray', linestyle='--', linewidth=1.5, alpha=0.6,
          label='Var_Ratio = 1.0 (neutral)', zorder=1)

# Add model labels with custom positioning to avoid overlaps
label_offsets = {
    'GPT-4o': (8, 8),
    'GPT-4o-mini': (8, -12),
    'Claude Haiku': (8, 8),
    'Gemini Flash Phil': (8, -12),
    'DeepSeek V3.1': (-60, -8),
    'Kimi K2': (-40, -8),
    'Llama 4 Maverick': (8, 8),
    'Llama 4 Scout': (8, 8),
    'Mistral Small 24B': (-70, 8),
    'Ministral 14B': (-60, 8),
    'Qwen3 235B': (8, 8),
    'Gemini Flash Med': (8, -12)
}

for idx, row in df.iterrows():
    model = row['model_exp']
    # Shorten names for readability
    model_short = (model.replace('Gemini Flash Phil', 'GF-Phil')
                   .replace('Gemini Flash Med', 'GF-Med')
                   .replace('Llama 4 Maverick', 'L4-Mav')
                   .replace('Llama 4 Scout', 'L4-Scout')
                   .replace('Mistral Small 24B', 'Mistral')
                   .replace('DeepSeek V3.1', 'DeepSeek')
                   .replace('Claude Haiku', 'Haiku')
                   .replace('GPT-4o-mini', 'GPT4m')
                   .replace('GPT-4o', 'GPT4')
                   .replace('Ministral 14B', 'Mini14B')
                   .replace('Qwen3 235B', 'Qwen3')
                   .replace('Kimi K2', 'Kimi'))

    offset = label_offsets.get(model, (6, 6))
    ax.annotate(model_short, (row['var_ratio'], row['exploration_arc']),
               xytext=offset, textcoords='offset points',
               fontsize=9, alpha=0.8, fontweight='normal')

# Add quadrant labels (very subtle, repositioned to edges)
ax.text(0.91, 2.75, 'Convergent\nExplorers', fontsize=9, alpha=0.25,
       ha='center', va='top', style='italic', color='gray')
ax.text(1.65, 2.75, 'Divergent\nExplorers', fontsize=9, alpha=0.25,
       ha='right', va='top', style='italic', color='gray')
ax.text(0.91, 0.72, 'Convergent\nStable', fontsize=9, alpha=0.25,
       ha='center', va='bottom', style='italic', color='gray')
ax.text(1.65, 0.72, 'Divergent\nStable', fontsize=9, alpha=0.25,
       ha='right', va='bottom', style='italic', color='gray')

# Add statistics box
r_overall, p_overall = pearsonr(df['var_ratio'], df['exploration_arc'])
r_phil, p_phil = pearsonr(phil_df['var_ratio'], phil_df['exploration_arc'])
r_med, p_med = pearsonr(med_df['var_ratio'], med_df['exploration_arc'])

stats_text = (
    f"Correlations:\n"
    f"Overall:    r = {r_overall:.2f}, p = {p_overall:.2f}\n"
    f"Philosophy: r = {r_phil:.2f}, p = {p_phil:.2f}\n"
    f"Medical:    r = {r_med:.2f}, p = {p_med:.2f}"
)

ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
       fontsize=10, verticalalignment='bottom', horizontalalignment='right',
       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'),
       family='monospace')

# Labels and title
ax.set_xlabel('Var_Ratio (Variance TRUE / Variance COLD)', fontsize=13, fontweight='bold')
ax.set_ylabel('Exploration Arc (Diversity_Late / Diversity_Early)', fontsize=13, fontweight='bold')
ax.set_title('Domain Dominance: Task Structure Overrides Model Entanglement',
            fontsize=14, fontweight='bold', pad=15)

# Set axis limits
ax.set_xlim(0.90, 1.70)
ax.set_ylim(0.70, 2.85)

# Legend
ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.8)

plt.tight_layout()

# Save figure
fig_path = Path('C:/Users/barla/mch_experiments/papers/paper5_safety/figures/fig7_domain_dominance.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"[SAVED] Figure 7: {fig_path}")

# Also save as PDF for paper
fig_pdf = Path('C:/Users/barla/mch_experiments/papers/paper5_safety/figures/fig7_domain_dominance.pdf')
plt.savefig(fig_pdf, dpi=300, bbox_inches='tight', format='pdf')
print(f"[SAVED] Figure 7 (PDF): {fig_pdf}")

plt.close()

# Create Table 4
table4 = df[['domain', 'model_exp', 'var_ratio', 'diversity_early', 'diversity_late', 'exploration_arc']].copy()
table4.columns = ['Domain', 'Model', 'Var_Ratio', 'Diversity_Early', 'Diversity_Late', 'Exploration_Arc']

# Sort by domain, then by exploration arc descending
table4 = table4.sort_values(['Domain', 'Exploration_Arc'], ascending=[True, False])

# Round numeric columns
table4['Var_Ratio'] = table4['Var_Ratio'].round(3)
table4['Diversity_Early'] = table4['Diversity_Early'].round(4)
table4['Diversity_Late'] = table4['Diversity_Late'].round(4)
table4['Exploration_Arc'] = table4['Exploration_Arc'].round(3)

# Save table
table_path = Path('C:/Users/barla/mch_experiments/papers/paper5_safety/tables/table4_exploration_arc.csv')
table4.to_csv(table_path, index=False)
print(f"[SAVED] Table 4: {table_path}")

# Print preview
print("\n" + "="*80)
print("TABLE 4: EXPLORATION ARC BY DOMAIN AND MODEL")
print("="*80)
print(table4.to_string(index=False))

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print("\nPhilosophy Domain:")
print(f"  N models:        {len(phil_df)}")
print(f"  Var_Ratio:       {phil_df['var_ratio'].mean():.3f} +/- {phil_df['var_ratio'].std():.3f}")
print(f"  Exploration Arc: {phil_df['exploration_arc'].mean():.3f} +/- {phil_df['exploration_arc'].std():.3f}")
print(f"  All Arc > 1:     {(phil_df['exploration_arc'] > 1).all()}")

print("\nMedical Domain:")
print(f"  N models:        {len(med_df)}")
print(f"  Var_Ratio:       {med_df['var_ratio'].mean():.3f} +/- {med_df['var_ratio'].std():.3f}")
print(f"  Exploration Arc: {med_df['exploration_arc'].mean():.3f} +/- {med_df['exploration_arc'].std():.3f}")
print(f"  Arc < 1:         {(med_df['exploration_arc'] < 1).sum()} models")
print(f"  Arc ~ 1:         {((med_df['exploration_arc'] >= 0.95) & (med_df['exploration_arc'] <= 1.05)).sum()} models")
print(f"  Arc > 1:         {(med_df['exploration_arc'] > 1.05).sum()} models")

print("\n" + "="*80)
print("FIGURE CAPTION")
print("="*80)
print("""
Figure 7: Domain Dominance in Exploration Trajectories. Philosophy models (blue)
all show Exploration_Arc > 1.2, indicating diversity expansion regardless of
Var_Ratio. Medical models (red) cluster near Arc = 1.0, with task structure
constraining exploration even at high Var_Ratio. Diamond markers show domain means.
Domain structure determines exploration trajectory; Var_Ratio determines starting
position. Overall correlation r = -0.41 (p = 0.18), driven by domain separation
rather than within-domain effects.
""")

print("\n" + "="*80)
print("Complete! Figure 7 and Table 4 generated successfully.")
print("="*80)
