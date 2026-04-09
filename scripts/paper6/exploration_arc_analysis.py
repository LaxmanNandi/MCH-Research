#!/usr/bin/env python3
"""
EXPLORATION ARC ANALYSIS

Tests whether Var_Ratio predicts exploration trajectory:
- Var_Ratio > 1 (divergent) → Diversity increases (Exploration_Arc > 1)
- Var_Ratio < 1 (convergent) → Diversity decreases (Exploration_Arc < 1)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def compute_diversity_at_position(responses):
    """
    Diversity = 1 - RCI
    RCI = mean pairwise cosine similarity
    Higher diversity = less similar responses
    """
    if len(responses) < 2:
        return 0.0

    embeddings = model.encode(responses, convert_to_numpy=True)

    # Compute all pairwise similarities
    n = len(embeddings)
    similarities = []
    for i in range(n):
        for j in range(i+1, n):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            similarities.append(sim)

    rci = np.mean(similarities)
    diversity = 1 - rci

    return diversity

def analyze_exploration_arc(filepath, model_name, domain):
    """
    Compute Exploration Arc for a model-domain run
    """
    print(f"\nAnalyzing {model_name} ({domain})...")

    with open(filepath) as f:
        data = json.load(f)

    n_trials = data['n_trials']
    n_positions = len(data['trials'][0]['responses']['true'])

    # Compute diversity at each position
    diversities = []

    for pos in range(n_positions):
        print(f"  Position {pos+1}/{n_positions}...", end='\r')

        # Collect TRUE responses at this position
        true_responses = [trial['responses']['true'][pos] for trial in data['trials']]

        # Compute diversity
        div = compute_diversity_at_position(true_responses)
        diversities.append(div)

    print()  # New line

    # Define windows
    early_window = diversities[0:5]  # P1-P5
    late_window = diversities[25:30]  # P26-P30

    diversity_early = np.mean(early_window)
    diversity_late = np.mean(late_window)

    exploration_arc = diversity_late / diversity_early if diversity_early > 0 else np.nan

    print(f"  Diversity Early (P1-P5):  {diversity_early:.4f}")
    print(f"  Diversity Late (P26-P30): {diversity_late:.4f}")
    print(f"  Exploration Arc:          {exploration_arc:.4f}")

    return {
        'model': model_name,
        'domain': domain,
        'n_trials': n_trials,
        'diversity_early': diversity_early,
        'diversity_late': diversity_late,
        'exploration_arc': exploration_arc,
        'diversities': diversities
    }

def main():
    # Paper 4 data files (12 models with response text)
    data_files = {
        # Philosophy (4 models)
        'GPT-4o': ('C:/Users/barla/mch_experiments/data/philosophy/closed_models/mch_results_gpt4o_philosophy_50trials.json', 'philosophy'),
        'GPT-4o-mini': ('C:/Users/barla/mch_experiments/data/philosophy/closed_models/mch_results_gpt4o_mini_philosophy_50trials.json', 'philosophy'),
        'Claude Haiku': ('C:/Users/barla/mch_experiments/data/philosophy/closed_models/mch_results_claude_haiku_philosophy_50trials.json', 'philosophy'),
        'Gemini Flash Phil': ('C:/Users/barla/mch_experiments/data/philosophy/closed_models/mch_results_gemini_flash_philosophy_50trials.json', 'philosophy'),

        # Medical (8 models)
        'DeepSeek V3.1': ('C:/Users/barla/mch_experiments/data/medical/open_models/mch_results_deepseek_v3_1_medical_50trials.json', 'medical'),
        'Kimi K2': ('C:/Users/barla/mch_experiments/data/medical/open_models/mch_results_kimi_k2_medical_50trials.json', 'medical'),
        'Llama 4 Maverick': ('C:/Users/barla/mch_experiments/data/medical/open_models/mch_results_llama_4_maverick_medical_50trials.json', 'medical'),
        'Llama 4 Scout': ('C:/Users/barla/mch_experiments/data/medical/open_models/mch_results_llama_4_scout_medical_50trials.json', 'medical'),
        'Mistral Small 24B': ('C:/Users/barla/mch_experiments/data/medical/open_models/mch_results_mistral_small_24b_medical_50trials.json', 'medical'),
        'Ministral 14B': ('C:/Users/barla/mch_experiments/data/medical/open_models/mch_results_ministral_14b_medical_50trials.json', 'medical'),
        'Qwen3 235B': ('C:/Users/barla/mch_experiments/data/medical/open_models/mch_results_qwen3_235b_medical_50trials.json', 'medical'),
        'Gemini Flash Med': ('C:/Users/barla/mch_experiments/data/medical/gemini_flash/mch_results_gemini_flash_medical_50trials.json', 'medical'),
    }

    results = []

    for model_name, (filepath, domain) in data_files.items():
        try:
            result = analyze_exploration_arc(filepath, model_name, domain)
            results.append(result)
        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Create DataFrame
    df = pd.DataFrame(results)

    # Load Var_Ratio from Paper 4 verification (if available)
    # For now, we'll need to compute it or use stored values
    print("\n" + "="*70)
    print("EXPLORATION ARC RESULTS")
    print("="*70)
    print(df[['model', 'domain', 'diversity_early', 'diversity_late', 'exploration_arc']])

    # Save data
    output_csv = Path('C:/Users/barla/mch_experiments/scripts/validate/exploration_arc_data.csv')
    df.to_csv(output_csv, index=False)
    print(f"\nData saved to: {output_csv}")

    # Compute statistics (without Var_Ratio for now)
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)

    phil_df = df[df['domain'] == 'philosophy']
    med_df = df[df['domain'] == 'medical']

    print(f"\nPhilosophy (n={len(phil_df)}):")
    print(f"  Mean Exploration Arc: {phil_df['exploration_arc'].mean():.4f}")
    print(f"  Range: {phil_df['exploration_arc'].min():.4f} - {phil_df['exploration_arc'].max():.4f}")

    print(f"\nMedical (n={len(med_df)}):")
    print(f"  Mean Exploration Arc: {med_df['exploration_arc'].mean():.4f}")
    print(f"  Range: {med_df['exploration_arc'].min():.4f} - {med_df['exploration_arc'].max():.4f}")

    # Note about Var_Ratio correlation
    print("\n" + "="*70)
    print("NOTE")
    print("="*70)
    print("To complete analysis:")
    print("1. Load Var_Ratio from Paper 4 entanglement data")
    print("2. Merge with Exploration Arc data")
    print("3. Compute correlation and create scatter plot")

if __name__ == '__main__':
    main()
