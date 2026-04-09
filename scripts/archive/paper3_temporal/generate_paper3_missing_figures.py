#!/usr/bin/env python3
"""
Generate Missing Paper 3 Figures (fig3, fig4, fig5a, fig5b, fig7)
Uses existing 50-trial JSON data only.
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

# Data paths - Philosophy CLOSED models only
PHIL_MODELS = {
    "GPT-4o": BASE_DIR / "data/philosophy/closed_models/mch_results_gpt4o_philosophy_50trials.json",
    "GPT-4o-mini": BASE_DIR / "data/philosophy/closed_models/mch_results_gpt4o_mini_philosophy_50trials.json",
    "Claude Haiku": BASE_DIR / "data/philosophy/closed_models/mch_results_claude_haiku_philosophy_50trials.json",
    "Gemini Flash": BASE_DIR / "data/gemini_flash/mch_results_gemini_flash_philosophy_50trials.json",
}

# Medical OPEN models only
MED_MODELS = {
    "Llama 4 Maverick": BASE_DIR / "data/medical/open_models/mch_results_llama_4_maverick_medical_50trials.json",
    "Llama 4 Scout": BASE_DIR / "data/medical/open_models/mch_results_llama_4_scout_medical_50trials.json",
    "Mistral Small 24B": BASE_DIR / "data/medical/open_models/mch_results_mistral_small_24b_medical_50trials.json",
    "Ministral 14B": BASE_DIR / "data/medical/open_models/mch_results_ministral_14b_medical_50trials.json",
    "DeepSeek V3.1": BASE_DIR / "data/medical/open_models/mch_results_deepseek_v3_1_medical_50trials.json",
    "Qwen3 235B": BASE_DIR / "data/medical/open_models/mch_results_qwen3_235b_medical_50trials.json",
}

def load_model_data(filepath):
    """Load and parse 50-trial JSON data."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Extract trial-level dRCI values across positions
    n_trials = len(data['trials'])
    n_positions = len(data['trials'][0]['prompts'])

    # Position-level dRCI (mean across trials)
    position_drci = np.zeros(n_positions)
    position_std = np.zeros(n_positions)

    # Disruption sensitivity per position
    disruption = np.zeros(n_positions)

    for pos in range(n_positions):
        trial_drcis_cold = []
        trial_drcis_scrambled = []

        for trial in data['trials']:
            # dRCI for this position in this trial
            mean_true = trial['alignments']['true'][pos]
            mean_cold = trial['alignments']['cold'][pos]
            mean_scrambled = trial['alignments']['scrambled'][pos]

            drci_cold = mean_true - mean_cold
            drci_scrambled = mean_true - mean_scrambled

            trial_drcis_cold.append(drci_cold)
            trial_drcis_scrambled.append(drci_scrambled)

        position_drci[pos] = np.mean(trial_drcis_cold)
        position_std[pos] = np.std(trial_drcis_cold)

        # Disruption sensitivity: scrambled - cold
        disruption[pos] = np.mean(trial_drcis_scrambled) - np.mean(trial_drcis_cold)

    return {
        'position_drci': position_drci,
        'position_std': position_std,
        'disruption': disruption,
        'n_trials': n_trials,
        'n_positions': n_positions
    }

# ---------------------------------------------------------------------
# Load all model data
# ---------------------------------------------------------------------

print("Loading model data...")
phil_data = {}
for name, path in PHIL_MODELS.items():
    if path.exists():
        phil_data[name] = load_model_data(path)
        print(f"  Loaded {name} (philosophy)")
    else:
        print(f"  WARNING: {name} file not found: {path}")

med_data = {}
for name, path in MED_MODELS.items():
    if path.exists():
        med_data[name] = load_model_data(path)
        print(f"  Loaded {name} (medical)")
    else:
        print(f"  WARNING: {name} file not found: {path}")

# ---------------------------------------------------------------------
# Figure 3: Z-score analysis (P30 outlier detection)
# ---------------------------------------------------------------------

print("\nGenerating Figure 3: Z-scores...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Medical domain - compute Z-scores for P30 relative to P1-P29
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

# Plot medical Z-scores
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

# Philosophy domain
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

# Plot philosophy Z-scores
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
print(f"  Saved: fig3_zscores.png")

# ---------------------------------------------------------------------
# Figure 4: Three-bin comparison (positions 1-10, 11-20, 21-29)
# ---------------------------------------------------------------------

print("Generating Figure 4: Three-bin analysis...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

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

ax1.set_ylabel('Mean ΔRCI')
ax1.set_title('Medical: Three-Bin Pattern (U-shaped)')
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

ax2.set_ylabel('Mean ΔRCI')
ax2.set_title('Philosophy: Three-Bin Pattern (Inverted-U)')
ax2.set_xticks(x2)
ax2.set_xticklabels([m.split()[0] for m in phil_models_list], rotation=45, ha='right', fontsize=8)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

fig.tight_layout()
fig.savefig(OUT_DIR / "fig4_three_bin_analysis.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: fig4_three_bin_analysis.png")

# ---------------------------------------------------------------------
# Figure 5a: Disruption Sensitivity by model
# ---------------------------------------------------------------------

print("Generating Figure 5a: Disruption sensitivity...")

fig, ax = plt.subplots(figsize=(10, 6))

# Combine all models
all_models = list(phil_data.keys()) + list(med_data.keys())
all_disruption = []
colors = []

for model_name in phil_data.keys():
    ds = np.mean(phil_data[model_name]['disruption'])
    all_disruption.append(ds)
    colors.append('#1f77b4')  # Blue for philosophy

for model_name in med_data.keys():
    ds = np.mean(med_data[model_name]['disruption'])
    all_disruption.append(ds)
    colors.append('#d62728')  # Red for medical

x = np.arange(len(all_models))
ax.barh(x, all_disruption, color=colors, alpha=0.7)
ax.set_yticks(x)
ax.set_yticklabels(all_models, fontsize=9)
ax.set_xlabel("Disruption Sensitivity (ΔRCI_scrambled - ΔRCI_cold)")
ax.set_title("Disruption Sensitivity by Model")
ax.axvline(x=0, color='black', linewidth=0.8)
ax.grid(True, alpha=0.3, axis='x')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#1f77b4', alpha=0.7, label='Philosophy'),
    Patch(facecolor='#d62728', alpha=0.7, label='Medical')
]
ax.legend(handles=legend_elements)

fig.tight_layout()
fig.savefig(OUT_DIR / "fig5a_disruption_sensitivity.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: fig5a_disruption_sensitivity.png")

# ---------------------------------------------------------------------
# Figure 5b: Per-position Disruption Sensitivity
# ---------------------------------------------------------------------

print("Generating Figure 5b: Position × disruption...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Medical domain - show all models
positions = np.arange(1, 31)

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
print(f"  Saved: fig5b_position_disruption.png")

# ---------------------------------------------------------------------
# Figure 7: Model scaling (positions 1-29 slope vs disruption)
# ---------------------------------------------------------------------

print("Generating Figure 7: Model scaling...")

fig, ax = plt.subplots(figsize=(8, 6))

# Compute slopes for positions 1-29
slopes = []
disruptions = []
model_names = []
colors_list = []

positions_29 = np.arange(1, 30)

# Philosophy models
for model_name in phil_models_list:
    data = phil_data[model_name]
    drci_29 = data['position_drci'][:29]

    # Linear regression
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

# Scatter plot
ax.scatter(slopes, disruptions, c=colors_list, s=80, alpha=0.7)

# Annotate points
for i, name in enumerate(model_names):
    ax.annotate(name.split()[0], (slopes[i], disruptions[i]),
                textcoords="offset points", xytext=(5, 5), fontsize=8, alpha=0.7)

ax.axhline(y=0, color='black', linewidth=0.8)
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_xlabel("Slope (positions 1-29)")
ax.set_ylabel("Mean Disruption Sensitivity")
ax.set_title("Model-Specific Scaling Patterns")
ax.grid(True, alpha=0.3)

# Legend
legend_elements = [
    Patch(facecolor='#1f77b4', alpha=0.7, label='Philosophy'),
    Patch(facecolor='#d62728', alpha=0.7, label='Medical')
]
ax.legend(handles=legend_elements)

fig.tight_layout()
fig.savefig(OUT_DIR / "fig7_model_scaling.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: fig7_model_scaling.png")

print("\n" + "="*70)
print("ALL FIGURES GENERATED SUCCESSFULLY")
print(f"Output directory: {OUT_DIR}")
print("="*70)
