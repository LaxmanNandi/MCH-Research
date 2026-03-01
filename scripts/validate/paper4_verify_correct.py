#!/usr/bin/env python3
"""
Proper Paper 4 verification using STORED DRCI values and computing Var_Ratio from embeddings
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import pandas as pd

# Initialize embedding model
print("Loading embedding model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def compute_rci_for_condition(embeddings):
    """Compute RCI as average pairwise similarity"""
    n_trials = len(embeddings)
    rci_values = []

    for i in range(n_trials):
        similarities = []
        for j in range(n_trials):
            if i != j:
                sim = cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)

        trial_rci = np.mean(similarities) if similarities else 0.0
        rci_values.append(trial_rci)

    return np.array(rci_values)

def analyze_model(file_path, model_name):
    """Analyze model using stored DRCI and computed Var_Ratio"""
    print(f"\n=== Analyzing {model_name} ===")

    with open(file_path) as f:
        data = json.load(f)

    # Get stored DRCI
    mean_drci = data['statistics']['mean_drci']
    print(f"Stored mean DRCI: {mean_drci:.6f}")

    # Compute Var_Ratio at each position
    n_positions = len(data['trials'][0]['responses']['true'])
    results = []

    for pos in range(n_positions):
        print(f"Processing position {pos+1}/{n_positions}...", end='\r')

        true_responses = []
        cold_responses = []

        for trial in data['trials']:
            if pos < len(trial['responses']['true']) and pos < len(trial['responses']['cold']):
                true_responses.append(trial['responses']['true'][pos])
                cold_responses.append(trial['responses']['cold'][pos])

        # Generate embeddings
        true_embeddings = model.encode(true_responses, convert_to_numpy=True)
        cold_embeddings = model.encode(cold_responses, convert_to_numpy=True)

        # Compute RCI values per trial
        rci_true_values = compute_rci_for_condition(true_embeddings)
        rci_cold_values = compute_rci_for_condition(cold_embeddings)

        # Compute variance
        var_rci_true = np.var(rci_true_values, ddof=1) if len(rci_true_values) > 1 else 0.0
        var_rci_cold = np.var(rci_cold_values, ddof=1) if len(rci_cold_values) > 1 else 0.0

        var_ratio = var_rci_true / var_rci_cold if var_rci_cold > 0 else np.nan

        results.append({
            'model': model_name,
            'position': pos + 1,
            'var_ratio': var_ratio,
            'vri': 1 - var_ratio
        })

    print()
    df = pd.DataFrame(results)

    mean_var_ratio = df['var_ratio'].mean()
    print(f"Mean Var_Ratio: {mean_var_ratio:.6f}")

    # Conservation product
    K = mean_drci * mean_var_ratio
    print(f"Conservation product K = DRCI × Var_Ratio = {K:.6f}")

    return mean_drci, mean_var_ratio, K

def main():
    base_path = Path('C:/Users/barla/mch_experiments/data/philosophy/open_models')

    models = [
        (base_path / 'mch_results_deepseek_v3_1_philosophy_50trials.json', 'DeepSeek V3.1'),
        (base_path / 'mch_results_llama_4_maverick_philosophy_50trials.json', 'Llama 4 Maverick')
    ]

    results = []
    for filepath, model_name in models:
        drci, var_ratio, K = analyze_model(filepath, model_name)
        results.append({
            'model': model_name,
            'drci': drci,
            'var_ratio': var_ratio,
            'K': K
        })

    print("\n" + "="*70)
    print("CONSERVATION LAW TEST (Complete 50-trial models)")
    print("="*70)

    for r in results:
        print(f"\n{r['model']}:")
        print(f"  DRCI:      {r['drci']:.6f}")
        print(f"  Var_Ratio: {r['var_ratio']:.6f}")
        print(f"  K:         {r['K']:.6f}")

    mean_K = np.mean([r['K'] for r in results])
    print(f"\nMean K: {mean_K:.6f}")
    print(f"Paper 6 philosophy K: 0.301")
    print(f"Match: {abs(mean_K - 0.301) < 0.05}")

if __name__ == '__main__':
    main()
