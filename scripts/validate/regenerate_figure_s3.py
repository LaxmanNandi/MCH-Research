#!/usr/bin/env python3
"""
Regenerate Figure S3 (figure9_model_comparison.png) with correct Paper 3 model assignments.

Paper 3 models:
- Philosophy (4 models): GPT-4o, GPT-4o-mini, Claude Haiku, Gemini Flash
- Medical (8 models): DeepSeek V3.1, Gemini Flash, Kimi K2, Llama 4 Maverick,
                      Llama 4 Scout, Ministral 14B, Mistral Small 24B, Qwen3 235B
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load trial-level data
df = pd.read_csv('C:/Users/barla/mch_experiments/results/tables/trial_level_drci.csv')

# Define correct Paper 3 model-domain pairs
philosophy_models = [
    ('gpt4o', 'philosophy'),
    ('gpt4o_mini', 'philosophy'),
    ('claude_haiku', 'philosophy'),
    ('gemini_flash', 'philosophy')
]

medical_models = [
    ('deepseek_v3_1', 'medical'),
    ('gemini_flash', 'medical'),
    ('kimi_k2', 'medical'),
    ('llama_4_maverick', 'medical'),
    ('llama_4_scout', 'medical'),
    ('ministral_14b', 'medical'),
    ('mistral_small_24b', 'medical'),
    ('qwen3_235b', 'medical')
]

# Filter data for each domain
phil_data = pd.DataFrame()
for model, domain in philosophy_models:
    subset = df[(df['model'] == model) & (df['domain'] == domain)]
    phil_data = pd.concat([phil_data, subset])

med_data = pd.DataFrame()
for model, domain in medical_models:
    subset = df[(df['model'] == model) & (df['domain'] == domain)]
    med_data = pd.concat([med_data, subset])

# Calculate mean ΔRCI for each model
phil_means = phil_data.groupby('model')['delta_rci'].mean().reset_index()
phil_means['domain'] = 'Philosophy'
phil_means.columns = ['model', 'mean_drci', 'domain']

med_means = med_data.groupby('model')['delta_rci'].mean().reset_index()
med_means['domain'] = 'Medical'
med_means.columns = ['model', 'mean_drci', 'domain']

# Combine
all_means = pd.concat([phil_means, med_means])

# Calculate overall domain means
phil_overall = phil_data['delta_rci'].mean()
med_overall = med_data['delta_rci'].mean()

print(f"\nPhilosophy domain mean dRCI: {phil_overall:.3f} (N={len(phil_data)} trials, {len(phil_means)} models)")
print(f"Medical domain mean dRCI: {med_overall:.3f} (N={len(med_data)} trials, {len(med_means)} models)")
print("\nPer-model means:")
print(all_means.sort_values(['domain', 'mean_drci'], ascending=[True, False]))

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Separate by domain
phil = all_means[all_means['domain'] == 'Philosophy'].sort_values('mean_drci', ascending=False)
med = all_means[all_means['domain'] == 'Medical'].sort_values('mean_drci', ascending=False)

# Plot bars
x_phil = np.arange(len(phil))
x_med = np.arange(len(med)) + len(phil) + 1  # Add gap

bars_phil = ax.bar(x_phil, phil['mean_drci'], color='#4A90E2', alpha=0.8, label='Philosophy (open-goal)')
bars_med = ax.bar(x_med, med['mean_drci'], color='#E94B3C', alpha=0.8, label='Medical (closed-goal)')

# Add domain mean lines
ax.axhline(phil_overall, xmin=0, xmax=(len(phil))/(len(phil)+len(med)+1),
           color='#4A90E2', linestyle='--', linewidth=2, alpha=0.6)
ax.axhline(med_overall, xmin=(len(phil)+1)/(len(phil)+len(med)+1), xmax=1,
           color='#E94B3C', linestyle='--', linewidth=2, alpha=0.6)

# Format model names for display
def format_model_name(name):
    name_map = {
        'gpt4o': 'GPT-4o',
        'gpt4o_mini': 'GPT-4o-mini',
        'claude_haiku': 'Claude Haiku',
        'gemini_flash': 'Gemini Flash',
        'deepseek_v3_1': 'DeepSeek V3.1',
        'kimi_k2': 'Kimi K2',
        'llama_4_maverick': 'Llama 4 Maverick',
        'llama_4_scout': 'Llama 4 Scout',
        'ministral_14b': 'Ministral 14B',
        'mistral_small_24b': 'Mistral Small 24B',
        'qwen3_235b': 'Qwen3 235B'
    }
    return name_map.get(name, name)

# Set x-axis labels
all_labels = [format_model_name(m) for m in phil['model']] + [''] + [format_model_name(m) for m in med['model']]
ax.set_xticks(list(x_phil) + [len(phil)] + list(x_med))
ax.set_xticklabels(all_labels, rotation=45, ha='right')

# Labels and title
ax.set_ylabel('Mean ΔRCI', fontsize=12)
ax.set_xlabel('Model', fontsize=12)
ax.set_title('Model-Level Mean Context Sensitivity (ΔRCI) by Domain', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Adjust layout
plt.tight_layout()

# Save
output_path = 'C:/Users/barla/Desktop/Paper4_Preprint_Submission/figures/figure9_model_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {output_path}")

plt.close()

print(f"\nFigure S3 regenerated with correct Paper 3 model assignments")
print(f"  Philosophy: {phil_overall:.3f} (4 models)")
print(f"  Medical: {med_overall:.3f} (8 models)")
