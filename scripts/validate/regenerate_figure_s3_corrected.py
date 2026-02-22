#!/usr/bin/env python3
"""
Regenerate Figure S3 with correct values from JSON files ONLY.
Do NOT use paper3_correct_models.csv (has wrong Gemini Flash medical value).
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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

def get_mean_drci_from_json(json_path):
    """Extract mean ΔRCI from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Check if statistics has mean_drci
    if 'mean_drci' in data['statistics']:
        return data['statistics']['mean_drci']

    # Otherwise calculate from trials (for Gemini Flash medical)
    drci_values = []
    for trial in data['trials']:
        if isinstance(trial['delta_rci'], dict):
            drci_values.append(trial['delta_rci']['cold'])
        else:
            drci_values.append(trial['delta_rci'])

    return np.mean(drci_values)

# Extract data
phil_results = []
for model_name, json_path in philosophy_models.items():
    mean_drci = get_mean_drci_from_json(json_path)
    phil_results.append({'model': model_name, 'mean_drci': mean_drci, 'domain': 'Philosophy'})
    print(f'Philosophy - {model_name}: {mean_drci:.6f}')

med_results = []
for model_name, json_path in medical_models.items():
    mean_drci = get_mean_drci_from_json(json_path)
    med_results.append({'model': model_name, 'mean_drci': mean_drci, 'domain': 'Medical'})
    print(f'Medical - {model_name}: {mean_drci:.6f}')

# Create DataFrame
df_phil = pd.DataFrame(phil_results)
df_med = pd.DataFrame(med_results)

# Calculate overall domain means
phil_mean = df_phil['mean_drci'].mean()
med_mean = df_med['mean_drci'].mean()

print(f'\n{'='*70}')
print(f'Philosophy domain mean: {phil_mean:.6f} (rounds to {phil_mean:.3f})')
print(f'Medical domain mean: {med_mean:.6f} (rounds to {med_mean:.3f})')
print(f'{'='*70}')

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Separate by domain and sort
phil = df_phil.sort_values('mean_drci', ascending=False)
med = df_med.sort_values('mean_drci', ascending=False)

# Plot bars
x_phil = np.arange(len(phil))
x_med = np.arange(len(med)) + len(phil) + 1

bars_phil = ax.bar(x_phil, phil['mean_drci'], color='#4A90E2', alpha=0.8,
                    label='Philosophy (open-goal)')
bars_med = ax.bar(x_med, med['mean_drci'], color='#E94B3C', alpha=0.8,
                   label='Medical (closed-goal)')

# Add domain mean lines
ax.axhline(phil_mean, xmin=0, xmax=(len(phil))/(len(phil)+len(med)+1),
           color='#4A90E2', linestyle='--', linewidth=2, alpha=0.6)
ax.axhline(med_mean, xmin=(len(phil)+1)/(len(phil)+len(med)+1), xmax=1,
           color='#E94B3C', linestyle='--', linewidth=2, alpha=0.6)

# Set x-axis labels
all_labels = list(phil['model']) + [''] + list(med['model'])
ax.set_xticks(list(x_phil) + [len(phil)] + list(x_med))
ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=9)

# Labels and title
ax.set_ylabel('Mean ΔRCI', fontsize=12)
ax.set_xlabel('Model', fontsize=12)
ax.set_title('Model-Level Mean Context Sensitivity (ΔRCI) by Domain',
             fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Adjust layout
plt.tight_layout()

# Save
output_path = 'C:/Users/barla/Desktop/Paper4_Preprint_Submission/figures/figure9_model_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f'\nFigure saved to: {output_path}')

plt.close()

print(f'\nFigure S3 regenerated with CORRECTED values from JSON files')
print(f'  Philosophy: {phil_mean:.3f} (4 models)')
print(f'  Medical: {med_mean:.3f} (8 models)')
