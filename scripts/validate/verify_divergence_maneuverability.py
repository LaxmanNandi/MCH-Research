#!/usr/bin/env python3
"""
Verify that divergence (higher Var_Ratio) correlates with response diversity/maneuverability
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import pdist

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def measure_response_diversity(responses):
    """
    Measure diversity of responses:
    1. Embedding spread (how far apart responses are)
    2. Mean pairwise distance
    """
    if len(responses) < 2:
        return 0.0, 0.0

    embeddings = model.encode(responses, convert_to_numpy=True)

    # Mean pairwise distance (maneuverability indicator)
    distances = pdist(embeddings, metric='euclidean')
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    return mean_distance, std_distance

def analyze_model_maneuverability(filepath, model_name):
    """Analyze response maneuverability for a model"""
    with open(filepath) as f:
        data = json.load(f)

    print(f"\n{'='*70}")
    print(f"{model_name}")
    print('='*70)

    # Sample a few positions for detailed analysis
    test_positions = [0, 9, 14, 19, 29]  # Early, mid, late positions

    for pos in test_positions:
        # Collect all responses at this position across trials
        true_responses = [trial['responses']['true'][pos] for trial in data['trials']]
        cold_responses = [trial['responses']['cold'][pos] for trial in data['trials']]

        # Measure diversity
        true_div, true_std = measure_response_diversity(true_responses)
        cold_div, cold_std = measure_response_diversity(cold_responses)

        print(f"\nPosition {pos+1}:")
        print(f"  TRUE diversity: {true_div:.4f} (SD: {true_std:.4f})")
        print(f"  COLD diversity: {cold_div:.4f} (SD: {cold_std:.4f})")
        print(f"  Delta Diversity (TRUE-COLD): {true_div - cold_div:.4f}")

def main():
    files = [
        ('C:/Users/barla/mch_experiments/data/philosophy/open_models/mch_results_llama_4_maverick_philosophy_50trials.json',
         'Llama 4 Maverick (Var_Ratio=0.77, CONVERGENT)'),
        ('C:/Users/barla/mch_experiments/data/philosophy/open_models/mch_results_deepseek_v3_1_philosophy_50trials.json',
         'DeepSeek V3.1 (Var_Ratio=1.03, NEUTRAL)'),
        ('C:/Users/barla/mch_experiments/data/philosophy/open_models/mch_results_qwen3_235b_philosophy_rerun_checkpoint.json',
         'Qwen3 235B (Var_Ratio=1.93, DIVERGENT)')
    ]

    print("="*70)
    print("DIVERGENCE = MANEUVERABILITY VERIFICATION")
    print("="*70)
    print("\nHypothesis: Higher Var_Ratio -> Greater response diversity")
    print("Measuring: Euclidean distance between response embeddings")
    print("  Higher distance = More diverse/maneuverable responses")

    for filepath, model_name in files:
        try:
            analyze_model_maneuverability(filepath, model_name)
        except Exception as e:
            print(f"\nError analyzing {model_name}: {e}")

    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("\nIf hypothesis is correct:")
    print("  - Llama (convergent) should show LOW diversity")
    print("  - Qwen3 (divergent) should show HIGH diversity")
    print("  - DeepSeek (neutral) should be in between")

if __name__ == '__main__':
    main()
