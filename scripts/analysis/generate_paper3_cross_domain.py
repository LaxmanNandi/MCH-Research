#!/usr/bin/env python3
"""
Paper 3: Cross-Domain AI Behavior Analysis
Philosophy vs Medical domains with complete response text data.

Philosophy (4 CLOSED models): GPT-4o, GPT-4o-mini, Claude Haiku, Gemini Flash
Medical (6 OPEN models): DeepSeek V3.1, Llama 4 Maverick/Scout, Mistral Small, Ministral, Qwen3
Total: 10 models, ~45,000 responses with full text
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

BASE_DIR = Path("C:/Users/barla/mch_experiments")
OUT_DIR = BASE_DIR / "docs" / "figures" / "paper3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA PATHS - Models with complete response text
# ============================================================================

PHILOSOPHY_MODELS = {
    "GPT-4o": BASE_DIR / "data/philosophy/closed_models/mch_results_gpt4o_philosophy_50trials.json",
    "GPT-4o-mini": BASE_DIR / "data/philosophy/closed_models/mch_results_gpt4o_mini_philosophy_50trials.json",
    "Claude Haiku": BASE_DIR / "data/philosophy/closed_models/mch_results_claude_haiku_philosophy_50trials.json",
    "Gemini Flash": BASE_DIR / "data/philosophy/closed_models/mch_results_gemini_flash_philosophy_50trials.json",
}

MEDICAL_MODELS = {
    "DeepSeek V3.1": BASE_DIR / "data/medical/open_models/mch_results_deepseek_v3_1_medical_50trials.json",
    "Llama 4 Maverick": BASE_DIR / "data/medical/open_models/mch_results_llama_4_maverick_medical_50trials.json",
    "Llama 4 Scout": BASE_DIR / "data/medical/open_models/mch_results_llama_4_scout_medical_50trials.json",
    "Mistral Small 24B": BASE_DIR / "data/medical/open_models/mch_results_mistral_small_24b_medical_50trials.json",
    "Ministral 14B": BASE_DIR / "data/medical/open_models/mch_results_ministral_14b_medical_50trials.json",
    "Qwen3 235B": BASE_DIR / "data/medical/open_models/mch_results_qwen3_235b_medical_50trials.json",
    "Kimi K2": BASE_DIR / "data/medical/open_models/mch_results_kimi_k2_medical_50trials.json",
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_model_data(filepath):
    """Load 50-trial JSON and extract position-level metrics."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    n_trials = len(data['trials'])
    n_positions = len(data['trials'][0]['prompts'])

    position_drci = np.zeros(n_positions)
    position_std = np.zeros(n_positions)
    disruption = np.zeros(n_positions)

    for pos in range(n_positions):
        trial_drcis_cold = []
        trial_drcis_scrambled = []

        for trial in data['trials']:
            mean_true = trial['alignments']['true'][pos]
            mean_cold = trial['alignments']['cold'][pos]
            mean_scrambled = trial['alignments']['scrambled'][pos]

            drci_cold = mean_true - mean_cold
            drci_scrambled = mean_true - mean_scrambled

            trial_drcis_cold.append(drci_cold)
            trial_drcis_scrambled.append(drci_scrambled)

        position_drci[pos] = np.mean(trial_drcis_cold)
        position_std[pos] = np.std(trial_drcis_cold)
        disruption[pos] = np.mean(trial_drcis_scrambled) - np.mean(trial_drcis_cold)

    return {
        'position_drci': position_drci,
        'position_std': position_std,
        'position_sem': position_std / np.sqrt(n_trials),
        'disruption': disruption,
        'n_trials': n_trials,
        'n_positions': n_positions
    }

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 80)
print("PAPER 3: CROSS-DOMAIN AI BEHAVIOR ANALYSIS")
print("=" * 80)
print(f"Philosophy models: {len(PHILOSOPHY_MODELS)} (closed-source APIs)")
print(f"Medical models: {len(MEDICAL_MODELS)} (open-source)")
print(f"Total: {len(PHILOSOPHY_MODELS) + len(MEDICAL_MODELS)} models")
print("=" * 80)

phil_data = {}
for name, path in PHILOSOPHY_MODELS.items():
    if path.exists():
        phil_data[name] = load_model_data(path)
        print(f"[OK] Loaded {name} (philosophy)")
    else:
        print(f"[!] WARNING: {name} not found")

print()

med_data = {}
for name, path in MEDICAL_MODELS.items():
    if path.exists():
        med_data[name] = load_model_data(path)
        print(f"[OK] Loaded {name} (medical)")
    else:
        print(f"[!] WARNING: {name} not found")

print()

# ============================================================================
# FIGURE 1: Position-Dependent dRCI by Domain
# ============================================================================

print("Generating Figure 1: Position-dependent dRCI by domain...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
positions = np.arange(1, 31)

# Philosophy domain
for model_name, data in phil_data.items():
    ax1.plot(positions, data['position_drci'], marker='o', markersize=3,
             label=model_name, alpha=0.6, linewidth=1.5)

# Grand mean with SEM
phil_grand_mean = np.mean([d['position_drci'] for d in phil_data.values()], axis=0)
phil_grand_sem = np.std([d['position_drci'] for d in phil_data.values()], axis=0) / np.sqrt(len(phil_data))

ax1.plot(positions, phil_grand_mean, 'k-', linewidth=3, label='Grand Mean', alpha=0.9)
ax1.fill_between(positions,
                  phil_grand_mean - phil_grand_sem,
                  phil_grand_mean + phil_grand_sem,
                  color='black', alpha=0.2)

ax1.set_xlabel("Position")
ax1.set_ylabel("dRCI")
ax1.set_title("Philosophy Domain (n=4 models)")
ax1.legend(fontsize=8, loc='best')
ax1.grid(True, alpha=0.3)

# Medical domain
for model_name, data in med_data.items():
    ax2.plot(positions, data['position_drci'], marker='o', markersize=3,
             label=model_name, alpha=0.6, linewidth=1.5)

# Grand mean with SEM
med_grand_mean = np.mean([d['position_drci'] for d in med_data.values()], axis=0)
med_grand_sem = np.std([d['position_drci'] for d in med_data.values()], axis=0) / np.sqrt(len(med_data))

ax2.plot(positions, med_grand_mean, 'k-', linewidth=3, label='Grand Mean', alpha=0.9)
ax2.fill_between(positions,
                  med_grand_mean - med_grand_sem,
                  med_grand_mean + med_grand_sem,
                  color='black', alpha=0.2)

ax2.set_xlabel("Position")
ax2.set_ylabel("dRCI")
ax2.set_title("Medical Domain (n=6 models)")
ax2.legend(fontsize=8, loc='best')
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUT_DIR / "fig1_position_drci_domains.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  [OK] Saved: fig1_position_drci_domains.png")

# ============================================================================
# FIGURE 2: Domain Grand Mean Comparison
# ============================================================================

print("Generating Figure 2: Domain grand mean comparison...")

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(positions, phil_grand_mean, 'o-', color='#1f77b4', linewidth=2.5,
        markersize=5, label='Philosophy (Inverted-U)', alpha=0.8)
ax.fill_between(positions,
                 phil_grand_mean - phil_grand_sem,
                 phil_grand_mean + phil_grand_sem,
                 color='#1f77b4', alpha=0.2)

ax.plot(positions, med_grand_mean, 's-', color='#d62728', linewidth=2.5,
        markersize=5, label='Medical (U-shaped)', alpha=0.8)
ax.fill_between(positions,
                 med_grand_mean - med_grand_sem,
                 med_grand_mean + med_grand_sem,
                 color='#d62728', alpha=0.2)

ax.set_xlabel("Position", fontsize=12)
ax.set_ylabel("dRCI (Mean ± SEM)", fontsize=12)
ax.set_title("Cross-Domain Comparison: Context Sensitivity Dynamics", fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUT_DIR / "fig2_domain_grand_mean.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  [OK] Saved: fig2_domain_grand_mean.png")

# ============================================================================
# FIGURE 3: Z-Score Analysis (P30 Outlier Detection)
# ============================================================================

print("Generating Figure 3: Z-scores (P30 outlier)...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Medical domain Z-scores
med_models_list = list(med_data.keys())
med_z_scores = []

for model_name in med_models_list:
    data = med_data[model_name]
    p1_29 = data['position_drci'][:29]
    p30 = data['position_drci'][29]

    mean_p1_29 = np.mean(p1_29)
    std_p1_29 = np.std(p1_29)

    z_score = (p30 - mean_p1_29) / std_p1_29 if std_p1_29 > 0 else 0
    med_z_scores.append(z_score)

ax1.barh(range(len(med_models_list)), med_z_scores, color='#d62728', alpha=0.7)
ax1.set_yticks(range(len(med_models_list)))
ax1.set_yticklabels(med_models_list, fontsize=9)
ax1.set_xlabel("Z-score (P30 vs P1-P29)")
ax1.set_title("Medical Domain: P30 Outlier Analysis")
ax1.axvline(x=0, color='black', linewidth=0.8)
ax1.axvline(x=2, color='orange', linewidth=0.8, linestyle='--', label='Z=2')
ax1.axvline(x=-2, color='orange', linewidth=0.8, linestyle='--')
ax1.grid(True, alpha=0.3, axis='x')
ax1.legend()

# Philosophy domain Z-scores
phil_models_list = list(phil_data.keys())
phil_z_scores = []

for model_name in phil_models_list:
    data = phil_data[model_name]
    p1_29 = data['position_drci'][:29]
    p30 = data['position_drci'][29]

    mean_p1_29 = np.mean(p1_29)
    std_p1_29 = np.std(p1_29)

    z_score = (p30 - mean_p1_29) / std_p1_29 if std_p1_29 > 0 else 0
    phil_z_scores.append(z_score)

ax2.barh(range(len(phil_models_list)), phil_z_scores, color='#1f77b4', alpha=0.7)
ax2.set_yticks(range(len(phil_models_list)))
ax2.set_yticklabels(phil_models_list, fontsize=9)
ax2.set_xlabel("Z-score (P30 vs P1-P29)")
ax2.set_title("Philosophy Domain: P30 Outlier Analysis")
ax2.axvline(x=0, color='black', linewidth=0.8)
ax2.axvline(x=2, color='orange', linewidth=0.8, linestyle='--', label='Z=2')
ax2.axvline(x=-2, color='orange', linewidth=0.8, linestyle='--')
ax2.grid(True, alpha=0.3, axis='x')
ax2.legend()

fig.tight_layout()
fig.savefig(OUT_DIR / "fig3_zscores.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  [OK] Saved: fig3_zscores.png")

# ============================================================================
# FIGURE 4: Three-Bin Analysis (Positions 1-29)
# ============================================================================

print("Generating Figure 4: Three-bin analysis...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Medical domain
med_early = []
med_mid = []
med_late = []

for model_name in med_models_list:
    data = med_data[model_name]
    early = np.mean(data['position_drci'][:10])
    mid = np.mean(data['position_drci'][10:20])
    late = np.mean(data['position_drci'][20:29])

    med_early.append(early)
    med_mid.append(mid)
    med_late.append(late)

x = np.arange(len(med_models_list))
width = 0.25

ax1.bar(x - width, med_early, width, label='Early (1-10)', color='#2ca02c', alpha=0.7)
ax1.bar(x, med_mid, width, label='Mid (11-20)', color='#ff7f0e', alpha=0.7)
ax1.bar(x + width, med_late, width, label='Late (21-29)', color='#d62728', alpha=0.7)

ax1.set_ylabel('Mean dRCI')
ax1.set_title('Medical: U-shaped Pattern (positions 1-29)')
ax1.set_xticks(x)
ax1.set_xticklabels([m.split()[0] for m in med_models_list], rotation=45, ha='right', fontsize=8)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Philosophy domain
phil_early = []
phil_mid = []
phil_late = []

for model_name in phil_models_list:
    data = phil_data[model_name]
    early = np.mean(data['position_drci'][:10])
    mid = np.mean(data['position_drci'][10:20])
    late = np.mean(data['position_drci'][20:29])

    phil_early.append(early)
    phil_mid.append(mid)
    phil_late.append(late)

x2 = np.arange(len(phil_models_list))

ax2.bar(x2 - width, phil_early, width, label='Early (1-10)', color='#2ca02c', alpha=0.7)
ax2.bar(x2, phil_mid, width, label='Mid (11-20)', color='#ff7f0e', alpha=0.7)
ax2.bar(x2 + width, phil_late, width, label='Late (21-29)', color='#d62728', alpha=0.7)

ax2.set_ylabel('Mean dRCI')
ax2.set_title('Philosophy: Inverted-U Pattern (positions 1-29)')
ax2.set_xticks(x2)
ax2.set_xticklabels([m.split()[0] for m in phil_models_list], rotation=45, ha='right', fontsize=8)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

fig.tight_layout()
fig.savefig(OUT_DIR / "fig4_three_bin_analysis.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  [OK] Saved: fig4_three_bin_analysis.png")

# ============================================================================
# FIGURE 5A: Disruption Sensitivity by Model
# ============================================================================

print("Generating Figure 5a: Disruption sensitivity...")

fig, ax = plt.subplots(figsize=(10, 6))

all_models = list(phil_data.keys()) + list(med_data.keys())
all_disruption = []
colors = []

for model_name in phil_data.keys():
    ds = np.mean(phil_data[model_name]['disruption'])
    all_disruption.append(ds)
    colors.append('#1f77b4')

for model_name in med_data.keys():
    ds = np.mean(med_data[model_name]['disruption'])
    all_disruption.append(ds)
    colors.append('#d62728')

x = np.arange(len(all_models))
ax.barh(x, all_disruption, color=colors, alpha=0.7)
ax.set_yticks(x)
ax.set_yticklabels(all_models, fontsize=9)
ax.set_xlabel("Disruption Sensitivity (dRCI_scrambled - dRCI_cold)")
ax.set_title("Disruption Sensitivity by Model (Negative = Presence > Order)")
ax.axvline(x=0, color='black', linewidth=0.8)
ax.grid(True, alpha=0.3, axis='x')

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#1f77b4', alpha=0.7, label='Philosophy'),
    Patch(facecolor='#d62728', alpha=0.7, label='Medical')
]
ax.legend(handles=legend_elements)

fig.tight_layout()
fig.savefig(OUT_DIR / "fig5a_disruption_sensitivity.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  [OK] Saved: fig5a_disruption_sensitivity.png")

# ============================================================================
# FIGURE 5B: Per-Position Disruption Sensitivity
# ============================================================================

print("Generating Figure 5b: Per-position disruption...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

positions = np.arange(1, 31)

# Medical domain
for model_name in med_models_list:
    data = med_data[model_name]
    ax1.plot(positions, data['disruption'], marker='o', markersize=3,
             label=model_name, alpha=0.7, linewidth=1.5)

ax1.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
ax1.set_ylabel("Disruption Sensitivity")
ax1.set_title("Medical Domain: Per-Position Disruption")
ax1.legend(fontsize=8, ncol=2)
ax1.grid(True, alpha=0.3)

# Philosophy domain
for model_name in phil_models_list:
    data = phil_data[model_name]
    ax2.plot(positions, data['disruption'], marker='o', markersize=3,
             label=model_name, alpha=0.7, linewidth=1.5)

ax2.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
ax2.set_xlabel("Position")
ax2.set_ylabel("Disruption Sensitivity")
ax2.set_title("Philosophy Domain: Per-Position Disruption")
ax2.legend(fontsize=8, ncol=2)
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUT_DIR / "fig5b_position_disruption.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  [OK] Saved: fig5b_position_disruption.png")

# ============================================================================
# FIGURE 7: Model Scaling (Slope vs Disruption)
# ============================================================================

print("Generating Figure 7: Model scaling...")

fig, ax = plt.subplots(figsize=(8, 6))

slopes = []
disruptions = []
model_names = []
colors_list = []

positions_29 = np.arange(1, 30)

# Philosophy models
for model_name in phil_models_list:
    data = phil_data[model_name]
    drci_29 = data['position_drci'][:29]

    slope, intercept, r_value, p_value, std_err = stats.linregress(positions_29, drci_29)
    mean_disruption = np.mean(data['disruption'])

    slopes.append(slope)
    disruptions.append(mean_disruption)
    model_names.append(model_name)
    colors_list.append('#1f77b4')

# Medical models
for model_name in med_models_list:
    data = med_data[model_name]
    drci_29 = data['position_drci'][:29]

    slope, intercept, r_value, p_value, std_err = stats.linregress(positions_29, drci_29)
    mean_disruption = np.mean(data['disruption'])

    slopes.append(slope)
    disruptions.append(mean_disruption)
    model_names.append(model_name)
    colors_list.append('#d62728')

ax.scatter(slopes, disruptions, c=colors_list, s=80, alpha=0.7)

for i, name in enumerate(model_names):
    ax.annotate(name.split()[0], (slopes[i], disruptions[i]),
                textcoords="offset points", xytext=(5, 5), fontsize=8, alpha=0.7)

ax.axhline(y=0, color='black', linewidth=0.8)
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_xlabel("Slope (positions 1-29)")
ax.set_ylabel("Mean Disruption Sensitivity")
ax.set_title("Model-Specific Scaling Patterns by Domain")
ax.grid(True, alpha=0.3)

legend_elements = [
    Patch(facecolor='#1f77b4', alpha=0.7, label='Philosophy'),
    Patch(facecolor='#d62728', alpha=0.7, label='Medical')
]
ax.legend(handles=legend_elements)

fig.tight_layout()
fig.savefig(OUT_DIR / "fig7_model_scaling.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  [OK] Saved: fig7_model_scaling.png")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("CROSS-DOMAIN SUMMARY STATISTICS")
print("=" * 80)

print(f"\nPhilosophy Domain (n={len(phil_data)} models):")
print(f"  Early (1-10):  {np.mean(phil_early):.3f} ± {np.std(phil_early):.3f}")
print(f"  Mid (11-20):   {np.mean(phil_mid):.3f} ± {np.std(phil_mid):.3f}")
print(f"  Late (21-29):  {np.mean(phil_late):.3f} ± {np.std(phil_late):.3f}")
print(f"  Pattern: {'Inverted-U' if np.mean(phil_mid) > np.mean(phil_early) and np.mean(phil_mid) > np.mean(phil_late) else 'Other'}")
print(f"  Mean Disruption Sensitivity: {np.mean([d for i, d in enumerate(all_disruption) if colors[i] == '#1f77b4']):.3f}")

print(f"\nMedical Domain (n={len(med_data)} models):")
print(f"  Early (1-10):  {np.mean(med_early):.3f} ± {np.std(med_early):.3f}")
print(f"  Mid (11-20):   {np.mean(med_mid):.3f} ± {np.std(med_mid):.3f}")
print(f"  Late (21-29):  {np.mean(med_late):.3f} ± {np.std(med_late):.3f}")
print(f"  Pattern: {'U-shaped' if np.mean(med_mid) < np.mean(med_early) and np.mean(med_mid) < np.mean(med_late) else 'Other'}")
print(f"  Mean Disruption Sensitivity: {np.mean([d for i, d in enumerate(all_disruption) if colors[i] == '#d62728']):.3f}")

print(f"\nP30 Outlier Analysis:")
print(f"  Medical Z-scores: {np.mean(med_z_scores):.2f} ± {np.std(med_z_scores):.2f} (range: {np.min(med_z_scores):.2f} to {np.max(med_z_scores):.2f})")
print(f"  Philosophy Z-scores: {np.mean(phil_z_scores):.2f} ± {np.std(phil_z_scores):.2f} (range: {np.min(phil_z_scores):.2f} to {np.max(phil_z_scores):.2f})")

print("\n" + "=" * 80)
print("ALL FIGURES GENERATED SUCCESSFULLY")
print(f"Output directory: {OUT_DIR}")
print("=" * 80)
