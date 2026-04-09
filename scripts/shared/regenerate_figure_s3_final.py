#!/usr/bin/env python3
"""
Regenerate Figure S3 with correct Paper 3 model assignments.

Uses paper3_correct_models.csv + JSON files for missing models (Kimi K2, Ministral 14B).
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load base data from paper3_correct_models.csv
df = pd.read_csv('/tmp/paper3_correct_models.csv')

# Add Kimi K2 (medical) from JSON
kimi_json = 'C:/Users/barla/mch_experiments/data/medical/open_models/mch_results_kimi_k2_medical_50trials.json'
with open(kimi_json, 'r', encoding='utf-8') as f:
    kimi_data = json.load(f)
    kimi_mean = kimi_data['statistics']['mean_drci']

# Add Ministral 14B (medical) from JSON
ministral_json = 'C:/Users/barla/mch_experiments/data/medical/open_models/mch_results_ministral_14b_medical_50trials.json'
with open(ministral_json, 'r', encoding='utf-8') as f:
    ministral_data = json.load(f)
    ministral_mean = ministral_data['statistics']['mean_drci']

# Add missing models to dataframe
new_rows = pd.DataFrame([
    {'model': 'Kimi K2', 'domain': 'medical', 'mean_drci': kimi_mean, 'sem': 0, 'ci_95': 0},
    {'model': 'Ministral 14B', 'domain': 'medical', 'mean_drci': ministral_mean, 'sem': 0, 'ci_95': 0}
])

df = pd.concat([df, new_rows], ignore_index=True)

# Separate by domain
phil = df[df['domain'] == 'philosophy'].sort_values('mean_drci', ascending=False)
med = df[df['domain'] == 'medical'].sort_values('mean_drci', ascending=False)

# Calculate overall domain means
phil_mean = phil['mean_drci'].mean()
med_mean = med['mean_drci'].mean()

print(f"{'='*70}")
print(f"Paper 3 Model Assignments:")
print(f"{'='*70}")
print(f"\nPhilosophy (4 models): mean dRCI = {phil_mean:.3f}")
for _, row in phil.iterrows():
    print(f"  {row['model']:20s} {row['mean_drci']:.3f}")

print(f"\nMedical (8 models): mean dRCI = {med_mean:.3f}")
for _, row in med.iterrows():
    print(f"  {row['model']:20s} {row['mean_drci']:.3f}")

print(f"\n{'='*70}")

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

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
print(f"\nFigure saved to: {output_path}")

plt.close()

print(f"\nFigure S3 regenerated successfully")
print(f"Caption values: Medical = {med_mean:.3f}, Philosophy = {phil_mean:.3f}")
