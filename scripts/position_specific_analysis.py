#!/usr/bin/env python3
"""
Position-Specific dRCI Analysis
Extracts P10, P30 values and analyzes position-specific patterns
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from scipy.stats import ttest_rel, ttest_1samp
import seaborn as sns

print("="*80)
print("POSITION-SPECIFIC dRCI ANALYSIS")
print("="*80)

# Load embedding model
print("\nLoading embedding model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Define models and file paths
philosophy_models = {
    'GPT-4o-mini': 'c:/Users/barla/mch_experiments/data/closed_model_philosophy_rerun/mch_results_gpt4o_mini_philosophy_50trials.json',
    'GPT-4o': 'c:/Users/barla/mch_experiments/data/closed_model_philosophy_rerun/mch_results_gpt4o_philosophy_50trials.json',
    'Claude Haiku': 'c:/Users/barla/mch_experiments/data/closed_model_philosophy_rerun/mch_results_claude_haiku_philosophy_50trials.json',
    'Gemini Flash': 'c:/Users/barla/mch_experiments/data/closed_model_philosophy_rerun/mch_results_gemini_flash_philosophy_50trials.json',
    'DeepSeek V3.1': 'c:/Users/barla/mch_experiments/data/open_model_results/mch_results_deepseek_v3_1_philosophy_50trials.json',
    'Llama 4 Maverick': 'c:/Users/barla/mch_experiments/data/open_model_results/mch_results_llama_4_maverick_philosophy_50trials.json',
    'Llama 4 Scout': 'c:/Users/barla/mch_experiments/data/open_model_results/mch_results_llama_4_scout_philosophy_50trials.json',
    'Qwen3 235B': 'c:/Users/barla/mch_experiments/data/open_model_results/mch_results_qwen3_235b_philosophy_50trials.json',
    'Mistral Small 24B': 'c:/Users/barla/mch_experiments/data/open_model_results/mch_results_mistral_small_24b_philosophy_50trials.json',
    'Ministral 14B': 'c:/Users/barla/mch_experiments/data/open_model_results/mch_results_ministral_14b_philosophy_50trials.json',
    'Kimi K2': 'c:/Users/barla/mch_experiments/data/open_model_results/mch_results_kimi_k2_philosophy_50trials.json'
}

medical_models = {
    'Gemini Flash': 'c:/Users/barla/mch_experiments/data/medical_results/mch_results_gemini_flash_medical_50trials.json',
    'Claude Haiku': 'c:/Users/barla/mch_experiments/data/medical_results/mch_results_claude_haiku_medical_50trials.json',
    'GPT-4o': 'c:/Users/barla/mch_experiments/data/medical_results/mch_results_gpt4o_medical_50trials.json',
    'GPT-4o-mini': 'c:/Users/barla/mch_experiments/data/medical_results/mch_results_gpt4o_mini_medical_50trials.json',
    'DeepSeek V3.1': 'c:/Users/barla/mch_experiments/data/open_medical_rerun/mch_results_deepseek_v3_1_medical_50trials.json',
    'Llama 4 Maverick': 'c:/Users/barla/mch_experiments/data/open_medical_rerun/mch_results_llama_4_maverick_medical_50trials.json',
    'Llama 4 Scout': 'c:/Users/barla/mch_experiments/data/open_medical_rerun/mch_results_llama_4_scout_medical_50trials.json'
}

def compute_position_drci(file_path, n_positions=30):
    """Compute dRCI for each position"""
    print(f"  Loading {file_path.split('/')[-1]}...", end=' ', flush=True)

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    n_trials = data['n_trials']

    # Extract responses by position
    position_responses = {}
    for pos in range(n_positions):
        position_responses[pos] = {'true': [], 'cold': []}

    for trial in data['trials']:
        for pos in range(n_positions):
            position_responses[pos]['true'].append(trial['responses']['true'][pos])
            position_responses[pos]['cold'].append(trial['responses']['cold'][pos])

    # Compute dRCI for each position
    position_drci = []

    for pos in range(n_positions):
        true_resp = position_responses[pos]['true']
        cold_resp = position_responses[pos]['cold']

        # Compute embeddings
        true_emb = model.encode(true_resp, convert_to_numpy=True, show_progress_bar=False)
        cold_emb = model.encode(cold_resp, convert_to_numpy=True, show_progress_bar=False)

        # Compute RCI for each trial
        rci_true = []
        rci_cold = []

        for i in range(n_trials):
            other_idx = [j for j in range(n_trials) if j != i]
            true_sims = np.dot(true_emb[i], true_emb[other_idx].T)
            rci_true.append(np.mean(true_sims))

            cold_sims = np.dot(cold_emb[i], cold_emb[other_idx].T)
            rci_cold.append(np.mean(cold_sims))

        mean_drci = np.mean(rci_true) - np.mean(rci_cold)
        position_drci.append(mean_drci)

    print("OK")
    return np.array(position_drci)

# ============================================================================
# PHILOSOPHY MODELS: Extract P10 and P30
# ============================================================================

print("\n" + "="*80)
print("PHILOSOPHY MODELS: P10 vs P30")
print("="*80)

philosophy_data = []

for model_name, file_path in philosophy_models.items():
    try:
        position_drci = compute_position_drci(file_path)

        p10_drci = position_drci[9]   # Position 10 (0-indexed)
        p30_drci = position_drci[29]  # Position 30

        # Surrounding positions for P10 (P8-P12)
        p10_context = position_drci[7:12]  # Positions 8-12

        philosophy_data.append({
            'model': model_name,
            'p10_drci': p10_drci,
            'p30_drci': p30_drci,
            'p10_context_mean': np.mean(p10_context),
            'p10_context_std': np.std(p10_context, ddof=1)
        })

        print(f"  {model_name:20s}: P10={p10_drci:.4f}, P30={p30_drci:.4f}, Diff={p30_drci - p10_drci:+.4f}")

    except FileNotFoundError:
        print(f"  {model_name:20s}: FILE NOT FOUND")

philosophy_df = pd.DataFrame(philosophy_data)

print(f"\nPhilosophy Summary:")
print(f"  P10 mean: {philosophy_df['p10_drci'].mean():.4f} +/- {philosophy_df['p10_drci'].std():.4f}")
print(f"  P30 mean: {philosophy_df['p30_drci'].mean():.4f} +/- {philosophy_df['p30_drci'].std():.4f}")
print(f"  Mean difference (P30 - P10): {(philosophy_df['p30_drci'] - philosophy_df['p10_drci']).mean():+.4f}")

# Statistical test: Is P30 different from P10?
if len(philosophy_df) > 1:
    t_stat, p_val = ttest_rel(philosophy_df['p30_drci'], philosophy_df['p10_drci'])
    print(f"  Paired t-test (P30 vs P10): t={t_stat:.3f}, p={p_val:.4f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}")

# ============================================================================
# MEDICAL MODELS: Extract all positions 1-30
# ============================================================================

print("\n" + "="*80)
print("MEDICAL MODELS: Full Position Curves")
print("="*80)

medical_position_data = {}

for model_name, file_path in medical_models.items():
    try:
        position_drci = compute_position_drci(file_path)
        medical_position_data[model_name] = position_drci

        p29_drci = position_drci[28]  # Position 29
        p30_drci = position_drci[29]  # Position 30
        spike = p30_drci - p29_drci

        # Z-score for P30
        mean_1_29 = np.mean(position_drci[:29])
        std_1_29 = np.std(position_drci[:29], ddof=1)
        z_score = (p30_drci - mean_1_29) / std_1_29

        print(f"  {model_name:20s}: P29={p29_drci:.4f}, P30={p30_drci:.4f}, Spike={spike:+.4f}, Z={z_score:+.2f}")

    except FileNotFoundError:
        print(f"  {model_name:20s}: FILE NOT FOUND")

if medical_position_data:
    # Compute statistics across all medical models
    all_p29 = [medical_position_data[m][28] for m in medical_position_data]
    all_p30 = [medical_position_data[m][29] for m in medical_position_data]

    print(f"\nMedical Summary:")
    print(f"  P29 mean: {np.mean(all_p29):.4f} +/- {np.std(all_p29):.4f}")
    print(f"  P30 mean: {np.mean(all_p30):.4f} +/- {np.std(all_p30):.4f}")
    print(f"  Mean spike (P30 - P29): {np.mean(all_p30) - np.mean(all_p29):+.4f}")

    if len(all_p30) > 1:
        t_stat, p_val = ttest_rel(all_p30, all_p29)
        print(f"  Paired t-test (P30 vs P29): t={t_stat:.3f}, p={p_val:.4e} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}")

# ============================================================================
# PLOT 1: Philosophy P10 vs P30
# ============================================================================

print("\n" + "="*80)
print("GENERATING PLOTS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1A: Philosophy P10 vs P30 scatter
ax = axes[0, 0]
ax.scatter(philosophy_df['p10_drci'], philosophy_df['p30_drci'], s=100, alpha=0.7, c='steelblue')

for i, row in philosophy_df.iterrows():
    ax.annotate(row['model'], (row['p10_drci'], row['p30_drci']),
                fontsize=8, ha='right', va='bottom', alpha=0.7)

# Add diagonal reference line (P10 = P30)
lim_min = min(philosophy_df['p10_drci'].min(), philosophy_df['p30_drci'].min()) - 0.02
lim_max = max(philosophy_df['p10_drci'].max(), philosophy_df['p30_drci'].max()) + 0.02
ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.3, linewidth=1, label='P10 = P30')

ax.set_xlabel('Position 10 dRCI', fontsize=11)
ax.set_ylabel('Position 30 dRCI', fontsize=11)
ax.set_title('Philosophy: P10 vs P30 (Type 2 at different positions)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 1B: Philosophy P10-P30 differences
ax = axes[0, 1]
diffs = philosophy_df['p30_drci'] - philosophy_df['p10_drci']
colors = ['red' if d > 0 else 'blue' for d in diffs]
ax.barh(range(len(philosophy_df)), diffs, color=colors, alpha=0.7)
ax.set_yticks(range(len(philosophy_df)))
ax.set_yticklabels(philosophy_df['model'], fontsize=9)
ax.axvline(0, color='black', linewidth=1, linestyle='-')
ax.set_xlabel('dRCI Difference (P30 - P10)', fontsize=11)
ax.set_title('Philosophy: Type 2 Effect at P30 vs P10', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Plot 2A: Medical P30 vs P29 spike
ax = axes[1, 0]
if medical_position_data:
    models = list(medical_position_data.keys())
    p29_values = [medical_position_data[m][28] for m in models]
    p30_values = [medical_position_data[m][29] for m in models]

    x = np.arange(len(models))
    width = 0.35

    ax.bar(x - width/2, p29_values, width, label='Position 29', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, p30_values, width, label='Position 30', color='crimson', alpha=0.8)

    ax.set_ylabel('dRCI', fontsize=11)
    ax.set_title('Medical: Position 30 Spike vs Position 29', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

# Plot 2B: Medical full position curves
ax = axes[1, 1]
if medical_position_data:
    for model_name, position_drci in medical_position_data.items():
        ax.plot(range(1, 31), position_drci, marker='o', markersize=4, alpha=0.7, linewidth=1.5, label=model_name)

    # Highlight position 30
    ax.axvline(30, color='red', linestyle='--', alpha=0.3, linewidth=2, label='P30 (summarization)')
    ax.axvline(10, color='green', linestyle='--', alpha=0.3, linewidth=2, label='P10 (early summary)')

    ax.set_xlabel('Position in Conversation', fontsize=11)
    ax.set_ylabel('dRCI', fontsize=11)
    ax.set_title('Medical: Full Position Curves (1-30)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 31)

plt.tight_layout()
output_path = "c:/Users/barla/mch_experiments/analysis/position_specific_analysis.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n[OK] Plot saved: {output_path}")

# ============================================================================
# STATISTICAL ANALYSIS: Is philosophy P10 different from surrounding?
# ============================================================================

print("\n" + "="*80)
print("PHILOSOPHY P10 CONTEXT ANALYSIS")
print("="*80)

print("\nTesting if P10 differs from surrounding positions (P8-P12):")

for i, row in philosophy_df.iterrows():
    model_name = row['model']
    p10_drci = row['p10_drci']
    context_mean = row['p10_context_mean']
    context_std = row['p10_context_std']

    # Simple deviation check
    deviation = abs(p10_drci - context_mean) / context_std if context_std > 0 else 0

    print(f"  {model_name:20s}: P10={p10_drci:.4f}, Context={context_mean:.4f}+/-{context_std:.4f}, Z={deviation:+.2f}")

# ============================================================================
# SAVE DATA TO CSV
# ============================================================================

print("\n" + "="*80)
print("SAVING DATA")
print("="*80)

# Save philosophy data
philosophy_df.to_csv('c:/Users/barla/mch_experiments/analysis/philosophy_p10_p30_data.csv', index=False)
print("[OK] Philosophy data: analysis/philosophy_p10_p30_data.csv")

# Save medical position data
if medical_position_data:
    medical_df = pd.DataFrame(medical_position_data)
    medical_df.index = range(1, 31)
    medical_df.index.name = 'position'
    medical_df.to_csv('c:/Users/barla/mch_experiments/analysis/medical_position_curves.csv')
    print("[OK] Medical data: analysis/medical_position_curves.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
