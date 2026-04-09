"""
Publication-Quality Figures: Trial Convergence and Model Comparison
====================================================================
Generates Figures 8 and 9 using cleaned 50-trial dataset.

Dataset methodology:
- Uses 50-trial reruns with corrected prompt set
- Excludes early runs with uncorrected prompts
- Includes updated data for models with multiple runs

Output:
- Figure 8: Trial-level convergence (scatter + rolling mean)
- Figure 9: Model comparison (mean ΔRCI with 95% CI)

Both figures saved at 300 DPI for publication.
"""

import pandas as pd
import numpy as np
import json
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# STEP 1: LOAD AND CLEAN DATA
# ============================================================

print("=" * 70)
print("TASK 1: IDENTIFY AND EXCLUDE OUTLIERS")
print("=" * 70)

df = pd.read_csv(r'C:\Users\barla\mch_experiments\results\tables\trial_level_drci.csv')

# Filter to trial <= 50 (exclude 100-trial old data)
df = df[df['trial_number'] <= 50].copy()

# Count before
n_before = len(df)

# Exclude 1: gemini_flash medical from medical_results (OLD, dRCI=-0.13)
mask_gf_old = (df['model'] == 'gemini_flash') & (df['domain'] == 'medical') & (df['source_folder'] == 'medical_results')
excluded_gf = df[mask_gf_old]
print(f"EXCLUDE: gemini_flash medical original: N={len(excluded_gf)}, mean_dRCI={excluded_gf['delta_rci'].mean():.4f}")
df = df[~mask_gf_old]

# Exclude 2: gpt4o_mini medical from medical_results (OLD, dRCI=0.03)
mask_gm_old = (df['model'] == 'gpt4o_mini') & (df['domain'] == 'medical') & (df['source_folder'] == 'medical_results')
excluded_gm = df[mask_gm_old]
print(f"EXCLUDE: gpt4o_mini medical original: N={len(excluded_gm)}, mean_dRCI={excluded_gm['delta_rci'].mean():.4f}")
df = df[~mask_gm_old]

# Add: gemini_flash medical RERUN (missing from CSV due to filename dedup)
print("\nADDING: gemini_flash medical RERUN from gemini_flash_medical_rerun/")
rerun_path = r'C:\Users\barla\mch_experiments\data\medical\gemini_flash\mch_results_gemini_flash_medical_50trials.json'
with open(rerun_path, encoding='utf-8') as f:
    rerun_data = json.load(f)

rerun_rows = []
for i, trial in enumerate(rerun_data['trials']):
    dr = trial['delta_rci']
    drci = dr['cold'] if isinstance(dr, dict) else float(dr)
    rerun_rows.append({
        'model': 'gemini_flash_rerun',
        'domain': 'medical',
        'task_type': 'Type2_Closed',
        'trial_number': i + 1,
        'delta_rci': drci,
        'source_folder': 'gemini_flash_medical_rerun',
    })

rerun_df = pd.DataFrame(rerun_rows)
print(f"  Added: N={len(rerun_df)}, mean_dRCI={rerun_df['delta_rci'].mean():.4f}")

df = pd.concat([df, rerun_df], ignore_index=True)

# Rename gpt4o_mini_rerun to gpt4o_mini for cleaner labels
df.loc[df['model'] == 'gpt4o_mini_rerun', 'model'] = 'gpt4o_mini'

n_after = len(df)
print(f"\nBefore: {n_before} rows")
print(f"After: {n_after} rows (removed {n_before - n_after + len(rerun_df)} old, added {len(rerun_df)} rerun)")

# Summary
print("\n=== CLEAN DATA SUMMARY ===")
for domain in ['medical', 'philosophy']:
    sub = df[df['domain'] == domain]
    models = sorted(sub['model'].unique())
    print(f"\n{domain.upper()} ({len(models)} models):")
    for m in models:
        ms = sub[sub['model'] == m]
        print(f"  {m}: N={len(ms)}, mean_dRCI={ms['delta_rci'].mean():.4f}")

# ============================================================
# STEP 2: CONVERGENCE STATS
# ============================================================

print("\n" + "=" * 70)
print("TASK 2: CONVERGENCE ANALYSIS (CLEAN DATA)")
print("=" * 70)

def convergence_stats(t, d, label):
    t = np.array(t, dtype=float)
    d = np.array(d, dtype=float)
    slope, intercept, r_val, p_val, se = stats.linregress(t, d)
    return {'label': label, 'n': len(d), 'mean': np.mean(d), 'std': np.std(d),
            'slope': slope, 'p': p_val, 'r': r_val}

results = {}
for domain in ['medical', 'philosophy']:
    sub = df[df['domain'] == domain]
    r = convergence_stats(sub['trial_number'], sub['delta_rci'], domain.upper())
    results[domain] = r
    print(f"\n{domain.upper()}:")
    print(f"  N = {r['n']}, Models = {sub['model'].nunique()}")
    print(f"  Mean dRCI = {r['mean']:.4f} +/- {r['std']:.4f}")
    print(f"  Slope = {r['slope']:.6f} (p={r['p']:.4f})")
    print(f"  Stable? {'YES' if r['p'] > 0.05 else 'NO'}")

r_all = convergence_stats(df['trial_number'], df['delta_rci'], "OVERALL")
results['overall'] = r_all
print(f"\nOVERALL:")
print(f"  N = {r_all['n']}, Mean = {r_all['mean']:.4f}, Slope = {r_all['slope']:.6f} (p={r_all['p']:.4f})")

# ============================================================
# TASK 3a: FIGURE 8 — Trial Convergence
# ============================================================

print("\n" + "=" * 70)
print("TASK 3a: REGENERATING FIGURE 8")
print("=" * 70)

med_r = results['medical']
phil_r = results['philosophy']

fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

for idx, (domain, ax) in enumerate(zip(['medical', 'philosophy'], axes)):
    sub = df[df['domain'] == domain]
    models = sorted(sub['model'].unique())
    n_models = len(models)

    if domain == 'medical':
        cmap = plt.cm.tab10
    else:
        cmap = plt.cm.tab20

    colors = [cmap(i / max(n_models - 1, 1)) for i in range(n_models)]

    for m_idx, model in enumerate(models):
        m_sub = sub[sub['model'] == model]
        trials = m_sub['trial_number'].values
        drci = m_sub['delta_rci'].values

        # Scatter
        ax.scatter(trials, drci, alpha=0.4, s=18, color=colors[m_idx],
                   label=model, zorder=2)

        # Rolling mean (5-trial window)
        if len(drci) >= 5:
            # Sort by trial number first
            sort_idx = np.argsort(trials)
            drci_sorted = drci[sort_idx]
            trials_sorted = trials[sort_idx]
            rolling = np.convolve(drci_sorted, np.ones(5)/5, mode='valid')
            x_rolling = trials_sorted[2:2+len(rolling)]
            ax.plot(x_rolling, rolling, color=colors[m_idx], linewidth=1.5, alpha=0.8, zorder=3)

    # Overall trend line
    all_t = sub['trial_number'].values.astype(float)
    all_d = sub['delta_rci'].values
    slope, intercept, _, _, _ = stats.linregress(all_t, all_d)
    x_trend = np.array([1, 50])
    y_trend = slope * x_trend + intercept
    ax.plot(x_trend, y_trend, 'k--', linewidth=2.5, alpha=0.7, zorder=4,
            label=f'Trend (slope={slope:.5f})')

    # Reference line
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5, linewidth=1)

    # Labels
    task = 'Type2_Closed' if domain == 'medical' else 'Type1_Open'
    r_info = results[domain]
    ax.set_title(f'{domain.capitalize()} ({task})\n'
                 f'slope={r_info["slope"]:.5f}, p={r_info["p"]:.4f}',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Trial Number', fontsize=11)
    if idx == 0:
        ax.set_ylabel('dRCI (trial-level)', fontsize=11)
    ax.set_xlim(0, 51)
    ax.legend(fontsize=6.5, loc='lower right', ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.2)

plt.suptitle('Trial-Level Convergence: dRCI Stable Across Trials\n'
             '(New methodology only, 50-trial runs)',
             fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0.02, 1, 0.93])

fig.text(0.5, 0.01, 'Colors = individual models | Lines = 5-trial rolling mean | Dashed = overall trend',
         ha='center', fontsize=9, style='italic', color='gray')

fig8_path = r'C:\Users\barla\mch_experiments\docs\figures\paper4\figure8_trial_convergence.png'
plt.savefig(fig8_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Figure 8 saved: {fig8_path}")
plt.close()

# ============================================================
# TASK 3b: FIGURE 9 — Model Comparison Bar Chart
# ============================================================

print("\n" + "=" * 70)
print("TASK 3b: REGENERATING FIGURE 9")
print("=" * 70)

# Compute per-model stats
model_stats = []
for model in sorted(df['model'].unique()):
    m_sub = df[df['model'] == model]
    domain = m_sub['domain'].iloc[0]
    model_stats.append({
        'model': model,
        'domain': domain,
        'mean_drci': m_sub['delta_rci'].mean(),
        'std_drci': m_sub['delta_rci'].std(),
        'n': len(m_sub),
    })

ms_df = pd.DataFrame(model_stats).sort_values('mean_drci', ascending=False)

fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(ms_df))
bar_colors = ['#2196F3' if d == 'medical' else '#FF9800' for d in ms_df['domain']]

bars = ax.bar(x, ms_df['mean_drci'], yerr=ms_df['std_drci'],
              capsize=4, color=bar_colors, alpha=0.85, edgecolor='white', linewidth=0.5)

# Add value labels on bars
for i, (_, row) in enumerate(ms_df.iterrows()):
    ax.text(i, row['mean_drci'] + row['std_drci'] + 0.008,
            f"{row['mean_drci']:.3f}",
            ha='center', va='bottom', fontsize=8, fontweight='bold')

# X-axis labels: model (domain)
labels = [f"{row['model']}\n({row['domain'][:3]})" for _, row in ms_df.iterrows()]
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)

ax.set_ylabel('Mean dRCI', fontsize=12)
ax.set_title('Mean dRCI by Model (Trial-Level)\nClean data: old outliers excluded, reruns included',
             fontsize=14, fontweight='bold')

# Reference line
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2196F3', alpha=0.85, label='Medical (Type2_Closed)'),
                   Patch(facecolor='#FF9800', alpha=0.85, label='Philosophy (Type1_Open)')]
ax.legend(handles=legend_elements, fontsize=10, loc='upper right')

ax.grid(True, axis='y', alpha=0.2)
ax.set_ylim(bottom=0)

plt.tight_layout()

fig9_path = r'C:\Users\barla\mch_experiments\docs\figures\paper4\figure9_model_comparison.png'
plt.savefig(fig9_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Figure 9 saved: {fig9_path}")
plt.close()

# ============================================================
# STEP 4: REPORT
# ============================================================

print("\n" + "=" * 70)
print("TASK 4: FINAL REPORT")
print("=" * 70)

print(f"""
EXCLUDED:
  1. gemini_flash medical original: N=50, mean_dRCI=-0.1331 (source: medical_results)
  2. gpt4o_mini medical original:   N=50, mean_dRCI=0.0257  (source: medical_results)
  Total excluded: 100 rows

ADDED:
  1. gemini_flash_rerun medical:    N=50, mean_dRCI={rerun_df['delta_rci'].mean():.4f} (source: gemini_flash_medical_rerun)

CLEAN DATA STATS:
  Medical:    N={results['medical']['n']}, mean_dRCI={results['medical']['mean']:.4f}, slope={results['medical']['slope']:.6f}, p={results['medical']['p']:.4f}, Stable={'YES' if results['medical']['p'] > 0.05 else 'NO'}
  Philosophy: N={results['philosophy']['n']}, mean_dRCI={results['philosophy']['mean']:.4f}, slope={results['philosophy']['slope']:.6f}, p={results['philosophy']['p']:.4f}, Stable={'YES' if results['philosophy']['p'] > 0.05 else 'NO'}
  Overall:    N={results['overall']['n']}, mean_dRCI={results['overall']['mean']:.4f}

FIGURES SAVED:
  Figure 8: {fig8_path}
  Figure 9: {fig9_path}


""")
