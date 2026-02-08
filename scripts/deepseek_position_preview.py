#!/usr/bin/env python3
"""
DeepSeek V3.1 Medical - Position-Dependent dRCI Preview Analysis
Analyzes first 35 trials to check for position 30 spike effect
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

print("="*70)
print("DEEPSEEK V3.1 MEDICAL - POSITION-DEPENDENT PREVIEW")
print("Analyzing first 35 trials for position 30 spike effect")
print("="*70)

# Load embedding model
print("\nLoading embedding model: all-MiniLM-L6-v2...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("Model loaded.")

# Load checkpoint data
checkpoint_path = "c:/Users/barla/mch_experiments/data/open_medical_rerun/mch_results_deepseek_v3_1_medical_checkpoint.json"
print(f"\nLoading checkpoint: {checkpoint_path}")

with open(checkpoint_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

n_trials = data['n_trials']
print(f"Trials available: {n_trials}")

# Extract responses by position
print(f"\nExtracting responses from {n_trials} trials...")

# Initialize storage: position -> condition -> list of responses
position_responses = {}
for pos in range(30):
    position_responses[pos] = {'true': [], 'cold': [], 'scrambled': []}

# Extract from each trial
for trial in data['trials']:
    true_responses = trial['responses']['true']
    cold_responses = trial['responses']['cold']
    scrambled_responses = trial['responses']['scrambled']

    for pos in range(30):
        position_responses[pos]['true'].append(true_responses[pos])
        position_responses[pos]['cold'].append(cold_responses[pos])
        position_responses[pos]['scrambled'].append(scrambled_responses[pos])

print(f"Extracted {len(position_responses[0]['true'])} responses per position per condition")

# Compute position-dependent dRCI
print("\nComputing embeddings and dRCI for each position...")

position_drci = []

for pos in range(30):
    print(f"  Position {pos+1}/30...", end='', flush=True)

    true_resp = position_responses[pos]['true']
    cold_resp = position_responses[pos]['cold']

    # Compute embeddings
    true_emb = model.encode(true_resp, convert_to_numpy=True, show_progress_bar=False)
    cold_emb = model.encode(cold_resp, convert_to_numpy=True, show_progress_bar=False)

    # Compute RCI for each trial at this position
    rci_true = []
    rci_cold = []

    for i in range(n_trials):
        # TRUE: similarity to other TRUE responses at this position
        other_true = [j for j in range(n_trials) if j != i]
        true_sims = np.dot(true_emb[i], true_emb[other_true].T)
        rci_true.append(np.mean(true_sims))

        # COLD: similarity to other COLD responses at this position
        other_cold = [j for j in range(n_trials) if j != i]
        cold_sims = np.dot(cold_emb[i], cold_emb[other_cold].T)
        rci_cold.append(np.mean(cold_sims))

    # Mean dRCI for this position
    mean_rci_true = np.mean(rci_true)
    mean_rci_cold = np.mean(rci_cold)
    mean_drci = mean_rci_true - mean_rci_cold

    position_drci.append(mean_drci)
    print(f" dRCI={mean_drci:.4f}")

position_drci = np.array(position_drci)

# Analyze position 30 outlier effect
print("\n" + "="*70)
print("POSITION 30 OUTLIER ANALYSIS")
print("="*70)

# Compute Z-score for position 30 vs positions 1-29
drci_1_29 = position_drci[:29]
drci_30 = position_drci[29]

mean_1_29 = np.mean(drci_1_29)
std_1_29 = np.std(drci_1_29, ddof=1)
z_score = (drci_30 - mean_1_29) / std_1_29

is_outlier = abs(z_score) > 2.0

print(f"\nPositions 1-29:")
print(f"  Mean dRCI = {mean_1_29:.4f}")
print(f"  Std Dev   = {std_1_29:.4f}")

print(f"\nPosition 30:")
print(f"  dRCI      = {drci_30:.4f}")
print(f"  Z-score   = {z_score:+.3f}")
print(f"  Outlier?  = {'YES' if is_outlier else 'NO'} (threshold: |Z| > 2.0)")

if is_outlier:
    print(f"\n*** POSITION 30 SPIKE DETECTED! ***")
    print(f"DeepSeek V3.1 shows same pattern as closed models:")
    print(f"  - GPT-4o:       Z = +3.69")
    print(f"  - Claude Haiku: Z = +4.25")
    print(f"  - DeepSeek V3.1 (philosophy): Z = +2.04")
else:
    print(f"\nNo position 30 spike detected (yet).")
    print(f"This is UNEXPECTED given 100% spike rate in closed medical models.")

# Trend analysis
print("\n" + "="*70)
print("TREND ANALYSIS")
print("="*70)

positions = np.arange(1, 31)

# All 30 positions
r_all, p_all = pearsonr(positions, position_drci)
slope_all = np.polyfit(positions, position_drci, 1)[0]

# Positions 1-29 only
r_29, p_29 = pearsonr(positions[:29], drci_1_29)
slope_29 = np.polyfit(positions[:29], drci_1_29, 1)[0]

print(f"\nAll 30 positions:")
print(f"  Slope = {slope_all:+.5f}")
print(f"  r     = {r_all:+.3f}")
print(f"  p     = {p_all:.3e}")

print(f"\nPositions 1-29 only:")
print(f"  Slope = {slope_29:+.5f}")
print(f"  r     = {r_29:+.3f}")
print(f"  p     = {p_29:.3e}")

if abs(slope_all) > abs(slope_29):
    reduction = (abs(slope_all) - abs(slope_29)) / abs(slope_all) * 100
    print(f"\nPosition 30 effect on trend: {reduction:.1f}% reduction when excluded")

# Generate plot
print("\n" + "="*70)
print("GENERATING PREVIEW PLOT")
print("="*70)

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# Plot data
ax.plot(positions, position_drci, 'o-', color='steelblue', linewidth=2, markersize=6, label='DeepSeek V3.1')

# Highlight position 30
ax.plot(30, drci_30, 'o', color='red', markersize=12, label=f'Position 30 (Z={z_score:+.2f})', zorder=5)

# Reference line for mean of positions 1-29
ax.axhline(mean_1_29, color='gray', linestyle='--', linewidth=1, alpha=0.5, label=f'Mean 1-29 ({mean_1_29:.3f})')

# Z-score boundaries
ax.axhline(mean_1_29 + 2*std_1_29, color='red', linestyle=':', linewidth=1, alpha=0.3, label='±2 SD threshold')
ax.axhline(mean_1_29 - 2*std_1_29, color='red', linestyle=':', linewidth=1, alpha=0.3)

ax.set_xlabel('Position in Conversation', fontsize=12)
ax.set_ylabel('dRCI (TRUE - COLD)', fontsize=12)
ax.set_title('DeepSeek V3.1 Medical - Position-Dependent dRCI Preview (n=35 trials)', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 31)

plt.tight_layout()
output_path = "c:/Users/barla/mch_experiments/analysis/deepseek_position_preview.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Plot saved: {output_path}")

# Summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

print(f"\nEarly (1-10):  Mean dRCI = {np.mean(position_drci[:10]):.4f} (SD = {np.std(position_drci[:10]):.4f})")
print(f"Mid (11-20):   Mean dRCI = {np.mean(position_drci[10:20]):.4f} (SD = {np.std(position_drci[10:20]):.4f})")
print(f"Late (21-29):  Mean dRCI = {np.mean(position_drci[20:29]):.4f} (SD = {np.std(position_drci[20:29]):.4f})")
print(f"Position 30:   dRCI      = {drci_30:.4f}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)

print(f"\nNote: This preview uses {n_trials} trials.")
print(f"Full experiment will have 50 trials for increased statistical power.")
print(f"Position 30 outlier status: {'CONFIRMED' if is_outlier else 'NOT YET DETECTED'}")

if is_outlier:
    print(f"\n✓ DeepSeek V3.1 replicates position 30 spike effect!")
    print(f"  This supports the Type 2 (Task Enablement) hypothesis.")
    print(f"  The phenomenon appears to be model-agnostic.")
