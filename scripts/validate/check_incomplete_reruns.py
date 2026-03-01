import json
import numpy as np

files = [
    ('C:/Users/barla/mch_experiments/data/philosophy/open_models/mch_results_mistral_small_24b_philosophy_rerun_checkpoint.json', 'Mistral Small 24B', 10),
    ('C:/Users/barla/mch_experiments/data/philosophy/open_models/mch_results_qwen3_235b_philosophy_rerun_checkpoint.json', 'Qwen3 235B', 35)
]

print("="*70)
print("INCOMPLETE RERUN MODELS - STORED DRCI")
print("="*70)

for filepath, model_name, n_trials in files:
    with open(filepath) as f:
        data = json.load(f)
    
    print(f"\n{model_name} ({n_trials} trials):")
    
    if 'statistics' in data and 'mean_drci' in data['statistics']:
        mean_drci = data['statistics']['mean_drci']
        print(f"  Stored mean DRCI: {mean_drci:.6f}")
    else:
        # Compute from trial-level delta_rci
        if 'delta_rci' in data['trials'][0]:
            drcis = [trial['delta_rci']['cold'] for trial in data['trials']]
            mean_drci = np.mean(drcis)
            print(f"  Computed mean DRCI from trials: {mean_drci:.6f}")
        else:
            print("  No DRCI found!")

print("\n" + "="*70)
print("CONSERVATION LAW FOR INCOMPLETE MODELS")
print("="*70)
print("\nFrom earlier Var_Ratio computation:")
print("  Mistral Small 24B: Var_Ratio = 2.71")
print("  Qwen3 235B:        Var_Ratio = 1.93")
print("\nNeed stored DRCI to compute K...")
