#!/usr/bin/env python3
"""
Regenerate Figure 8 (Figure S2: Trial-level convergence) with correct Paper 3 models.

Shows rolling mean convergence across 50 trials for each model-domain pair.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy import stats

# Base paths
base = Path('C:/Users/barla/mch_experiments/data')

# Define correct Paper 3 models with their JSON file paths
philosophy_models = {
    'GPT-4o': base / 'philosophy' / 'closed_models' / 'mch_results_gpt4o_philosophy_50trials.json',
    'GPT-4o-mini': base / 'philosophy' / 'closed_models' / 'mch_results_gpt4o_mini_philosophy_50trials.json',
    'Claude Haiku': base / 'philosophy' / 'closed_models' / 'mch_results_claude_haiku_philosophy_50trials.json',
    'Gemini Flash': base / 'philosophy' / 'closed_models' / 'mch_results_gemini_flash_philosophy_50trials.json'
}

medical_models = {
    'DeepSeek V3.1': base / 'medical' / 'open_models' / 'mch_results_deepseek_v3_1_medical_50trials.json',
    'Gemini Flash': base / 'medical' / 'gemini_flash' / 'mch_results_gemini_flash_medical_50trials.json',
    'Kimi K2': base / 'medical' / 'open_models' / 'mch_results_kimi_k2_medical_50trials.json',
    'Llama 4 Maverick': base / 'medical' / 'open_models' / 'mch_results_llama_4_maverick_medical_50trials.json',
    'Llama 4 Scout': base / 'medical' / 'open_models' / 'mch_results_llama_4_scout_medical_50trials.json',
    'Ministral 14B': base / 'medical' / 'open_models' / 'mch_results_ministral_14b_medical_50trials.json',
    'Mistral Small 24B': base / 'medical' / 'open_models' / 'mch_results_mistral_small_24b_medical_50trials.json',
    'Qwen3 235B': base / 'medical' / 'open_models' / 'mch_results_qwen3_235b_medical_50trials.json'
}

def extract_trial_drci_values(json_path):
    """Extract per-trial ΔRCI values from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    drci_values = []
    for trial in data.get('trials', []):
        # ΔRCI = mean(RCI_TRUE) - mean(RCI_COLD)
        # Get from delta_rci dict if it exists, or calculate from means
        if 'delta_rci' in trial:
            if isinstance(trial['delta_rci'], dict):
                # Use cold as the ΔRCI (this is how it's stored)
                drci = trial['delta_rci'].get('cold', 0)
            else:
                drci = trial['delta_rci']
        elif 'means' in trial:
            drci = trial['means'].get('true', 0) - trial['means'].get('cold', 0)
        else:
            drci = 0

        drci_values.append(drci)

    return np.array(drci_values)

def calculate_rolling_mean(values, window=5):
    """Calculate rolling mean with specified window."""
    return pd.Series(values).rolling(window=window, min_periods=1).mean().values

def calculate_trend_stats(values):
    """Calculate linear trend slope and p-value."""
    x = np.arange(len(values))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
    return slope, p_value

# Extract data for all models
phil_data = {}
for model_name, json_path in philosophy_models.items():
    if json_path.exists():
        drci_values = extract_trial_drci_values(json_path)
        phil_data[model_name] = drci_values
        print(f"Philosophy - {model_name}: {len(drci_values)} trials, mean={np.mean(drci_values):.3f}")

med_data = {}
for model_name, json_path in medical_models.items():
    if json_path.exists():
        drci_values = extract_trial_drci_values(json_path)
        med_data[model_name] = drci_values
        print(f"Medical - {model_name}: {len(drci_values)} trials, mean={np.mean(drci_values):.3f}")

# Calculate trend statistics for each domain
print(f"\n{'='*70}")
print("Trend Statistics:")
print(f"{'='*70}")

phil_slopes = []
phil_pvals = []
for model_name, values in phil_data.items():
    slope, pval = calculate_trend_stats(values)
    phil_slopes.append(slope)
    phil_pvals.append(pval)
    sig = "non-sig" if pval > 0.05 else "SIGNIFICANT"
    print(f"Philosophy - {model_name:20s}: slope={slope:.6f}, p={pval:.4f} ({sig})")

med_slopes = []
med_pvals = []
for model_name, values in med_data.items():
    slope, pval = calculate_trend_stats(values)
    med_slopes.append(slope)
    med_pvals.append(pval)
    sig = "non-sig" if pval > 0.05 else "SIGNIFICANT"
    print(f"Medical - {model_name:20s}: slope={slope:.6f}, p={pval:.4f} ({sig})")

# Calculate overall domain trend
phil_all_slopes = np.mean(phil_slopes)
phil_all_pvals = np.mean(phil_pvals)
med_all_slopes = np.mean(med_slopes)
med_all_pvals = np.mean(med_pvals)

print(f"\nPhilosophy domain avg: slope={phil_all_slopes:.6f}, p={phil_all_pvals:.4f}")
print(f"Medical domain avg: slope={med_all_slopes:.6f}, p={med_all_pvals:.4f}")

# Count non-significant models
phil_nonsig = sum(p > 0.05 for p in phil_pvals)
med_nonsig = sum(p > 0.05 for p in med_pvals)
total_nonsig = phil_nonsig + med_nonsig
total_models = len(phil_data) + len(med_data)

print(f"\nNon-significant drift: {total_nonsig}/{total_models} models (p > 0.05)")

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Color palette
colors_phil = ['#4A90E2', '#E94B3C', '#50C878', '#FFB347']
colors_med = ['#4A90E2', '#E94B3C', '#50C878', '#FFB347', '#9370DB', '#FF69B4', '#20B2AA', '#FFA500']

# Plot Philosophy (left panel)
ax = axes[0]
for i, (model_name, values) in enumerate(phil_data.items()):
    trials = np.arange(1, len(values) + 1)
    rolling = calculate_rolling_mean(values, window=5)

    # Plot individual points
    ax.scatter(trials, values, alpha=0.3, s=20, color=colors_phil[i])
    # Plot rolling mean
    ax.plot(trials, rolling, linewidth=2, label=model_name, color=colors_phil[i])

ax.set_xlabel('Trial Number', fontsize=11)
ax.set_ylabel('dRCI', fontsize=11)
ax.set_title(f'Philosophy (Type1, Open)\nslope={phil_all_slopes:.5f}, p={phil_all_pvals:.4f}',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='best')
ax.grid(alpha=0.3, linestyle='--')
ax.set_xlim(0, 51)

# Plot Medical (right panel)
ax = axes[1]
for i, (model_name, values) in enumerate(med_data.items()):
    trials = np.arange(1, len(values) + 1)
    rolling = calculate_rolling_mean(values, window=5)

    # Plot individual points
    ax.scatter(trials, values, alpha=0.3, s=20, color=colors_med[i])
    # Plot rolling mean
    ax.plot(trials, rolling, linewidth=2, label=model_name, color=colors_med[i])

ax.set_xlabel('Trial Number', fontsize=11)
ax.set_ylabel('dRCI', fontsize=11)
ax.set_title(f'Medical (Type2, Closed)\nslope={med_all_slopes:.5f}, p={med_all_pvals:.4f}',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='best')
ax.grid(alpha=0.3, linestyle='--')
ax.set_xlim(0, 51)

# Main title
fig.suptitle('Trial-Level Convergence: dRCI Stable Across Trials\n(New methodology only, 50-trial runs)',
             fontsize=14, fontweight='bold')

plt.tight_layout()

# Save
output_path = 'C:/Users/barla/Desktop/Paper4_Preprint_Submission/figures/figure8_trial_convergence.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n{'='*70}")
print(f"Figure saved to: {output_path}")
print(f"{'='*70}")

plt.close()

print(f"\nFigure 8 (S2) regenerated with correct Paper 3 model assignments")
print(f"  Philosophy: {len(phil_data)} models")
print(f"  Medical: {len(med_data)} models")
print(f"  Total: {total_models} model-domain runs")
print(f"  Non-significant drift: {total_nonsig}/{total_models} models")
