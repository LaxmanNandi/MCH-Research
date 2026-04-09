#!/usr/bin/env python3
"""
Verify Paper 4 claims on philosophy open model reruns
Tests whether ΔRCI ~ VRI correlation holds for models not included in Paper 4
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from scipy.stats import pearsonr
import pandas as pd

# Initialize embedding model
print("Loading embedding model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def compute_rci_for_condition(embeddings):
    """
    Compute RCI for a condition at a position
    RCI = average pairwise cosine similarity among all responses
    Returns: array of RCI values per trial (for variance computation)
    """
    n_trials = len(embeddings)
    rci_values = []

    # For each trial, compute its similarity to all other trials
    for i in range(n_trials):
        similarities = []
        for j in range(n_trials):
            if i != j:  # Don't compare with itself
                sim = cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)

        # RCI for this trial is mean similarity to all others
        trial_rci = np.mean(similarities) if similarities else 0.0
        rci_values.append(trial_rci)

    return np.array(rci_values)

def analyze_model(file_path, model_name, domain):
    """
    Analyze a single model file
    Returns: DataFrame with position-level metrics
    """
    print(f"\n=== Analyzing {model_name} ({domain}) ===")

    with open(file_path) as f:
        data = json.load(f)

    n_trials = data['n_trials']
    print(f"Trials: {n_trials}")

    if not data.get('response_text_saved', False):
        print("ERROR: No response text saved!")
        return None

    # Get number of positions from first trial
    n_positions = len(data['trials'][0]['responses']['true'])
    print(f"Positions: {n_positions}")

    results = []

    for pos in range(n_positions):
        print(f"Processing position {pos+1}/{n_positions}...", end='\r')

        # Collect responses for this position
        true_responses = []
        cold_responses = []

        for trial in data['trials']:
            if pos < len(trial['responses']['true']) and pos < len(trial['responses']['cold']):
                true_responses.append(trial['responses']['true'][pos])
                cold_responses.append(trial['responses']['cold'][pos])

        # Generate embeddings
        true_embeddings = model.encode(true_responses, convert_to_numpy=True)
        cold_embeddings = model.encode(cold_responses, convert_to_numpy=True)

        # Compute RCI for each condition
        rci_true_values = compute_rci_for_condition(true_embeddings)
        rci_cold_values = compute_rci_for_condition(cold_embeddings)

        # Compute mean RCI and variance
        mean_rci_true = np.mean(rci_true_values)
        mean_rci_cold = np.mean(rci_cold_values)
        var_rci_true = np.var(rci_true_values, ddof=1) if len(rci_true_values) > 1 else 0.0
        var_rci_cold = np.var(rci_cold_values, ddof=1) if len(rci_cold_values) > 1 else 0.0

        # Compute metrics
        delta_rci = mean_rci_true - mean_rci_cold
        var_ratio = var_rci_true / var_rci_cold if var_rci_cold > 0 else np.nan
        vri = 1 - var_ratio

        results.append({
            'model': model_name,
            'domain': domain,
            'position': pos + 1,
            'n_trials': len(true_responses),
            'mean_rci_true': mean_rci_true,
            'mean_rci_cold': mean_rci_cold,
            'var_rci_true': var_rci_true,
            'var_rci_cold': var_rci_cold,
            'delta_rci': delta_rci,
            'var_ratio': var_ratio,
            'vri': vri
        })

    print()  # New line after progress
    df = pd.DataFrame(results)

    # Summary stats
    print(f"\nSummary for {model_name}:")
    print(f"  Mean dRCI: {df['delta_rci'].mean():.4f} (SD: {df['delta_rci'].std():.4f})")
    print(f"  Mean Var_Ratio: {df['var_ratio'].mean():.4f} (SD: {df['var_ratio'].std():.4f})")
    print(f"  Mean VRI: {df['vri'].mean():.4f} (SD: {df['vri'].std():.4f})")

    return df

def main():
    base_path = Path('C:/Users/barla/mch_experiments/data')

    # Models to analyze
    models = [
        {
            'path': base_path / 'philosophy/open_models/mch_results_deepseek_v3_1_philosophy_50trials.json',
            'name': 'DeepSeek V3.1',
            'domain': 'philosophy'
        },
        {
            'path': base_path / 'philosophy/open_models/mch_results_llama_4_maverick_philosophy_50trials.json',
            'name': 'Llama 4 Maverick',
            'domain': 'philosophy'
        },
        {
            'path': base_path / 'philosophy/open_models/mch_results_mistral_small_24b_philosophy_rerun_checkpoint.json',
            'name': 'Mistral Small 24B',
            'domain': 'philosophy'
        },
        {
            'path': base_path / 'philosophy/open_models/mch_results_qwen3_235b_philosophy_rerun_checkpoint.json',
            'name': 'Qwen3 235B',
            'domain': 'philosophy'
        }
    ]

    # Analyze each model
    all_results = []
    for model_info in models:
        if model_info['path'].exists():
            df = analyze_model(model_info['path'], model_info['name'], model_info['domain'])
            if df is not None:
                all_results.append(df)
        else:
            print(f"WARNING: File not found: {model_info['path']}")

    if not all_results:
        print("ERROR: No valid results!")
        return

    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)

    # Save results
    output_path = Path('C:/Users/barla/mch_experiments/scripts/validate/paper4_rerun_verification.csv')
    combined_df.to_csv(output_path, index=False)
    print(f"\n[SAVED] Results saved to: {output_path}")

    # Overall correlation analysis
    print("\n" + "="*60)
    print("PAPER 4 CLAIM VERIFICATION")
    print("="*60)

    # Remove any NaN values
    valid_data = combined_df.dropna(subset=['delta_rci', 'vri'])

    # Compute correlation
    r, p = pearsonr(valid_data['delta_rci'], valid_data['vri'])
    n = len(valid_data)

    print(f"\n1. dRCI ~ VRI Correlation:")
    print(f"   Paper 4 (original): r=0.76, p=2.37e-68, N=360")
    print(f"   Rerun models:       r={r:.3f}, p={p:.2e}, N={n}")
    print(f"   Status: {'REPLICATED' if r > 0.6 and p < 0.001 else 'DIFFERENT'}")

    print(f"\n2. Var_Ratio Patterns:")
    convergent = (valid_data['var_ratio'] < 1).sum()
    divergent = (valid_data['var_ratio'] > 1).sum()
    print(f"   Convergent (Var_Ratio < 1): {convergent} ({100*convergent/n:.1f}%)")
    print(f"   Divergent (Var_Ratio > 1):  {divergent} ({100*divergent/n:.1f}%)")

    print(f"\n3. Philosophy Domain Characteristics:")
    print(f"   Mean Var_Ratio: {valid_data['var_ratio'].mean():.3f}")
    print(f"   Paper 4 philosophy: ~1.01")
    print(f"   Status: {'CONSISTENT' if abs(valid_data['var_ratio'].mean() - 1.01) < 0.2 else 'DIFFERENT'}")

    print(f"\n4. Model-Level Summary:")
    model_summary = valid_data.groupby('model').agg({
        'delta_rci': ['mean', 'std'],
        'var_ratio': ['mean', 'std'],
        'vri': ['mean', 'std']
    }).round(4)
    print(model_summary)

    print("\n" + "="*60)
    print("Verification complete!")
    print("="*60)

if __name__ == '__main__':
    main()
