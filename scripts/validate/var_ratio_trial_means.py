import json
import numpy as np

files = [
    ('C:/Users/barla/mch_experiments/data/philosophy/open_models/mch_results_deepseek_v3_1_philosophy_50trials.json', 'DeepSeek V3.1'),
    ('C:/Users/barla/mch_experiments/data/philosophy/open_models/mch_results_llama_4_maverick_philosophy_50trials.json', 'Llama 4 Maverick')
]

print("="*70)
print("VAR_RATIO from trial-level MEANS")
print("="*70)

for filepath, model_name in files:
    with open(filepath) as f:
        data = json.load(f)
    
    print(f"\n{model_name}:")
    
    # Extract trial-level mean RCI values
    mean_true_per_trial = [trial['means']['true'] for trial in data['trials']]
    mean_cold_per_trial = [trial['means']['cold'] for trial in data['trials']]
    
    # Variance of trial means
    var_true = np.var(mean_true_per_trial, ddof=1)
    var_cold = np.var(mean_cold_per_trial, ddof=1)
    var_ratio = var_true / var_cold if var_cold > 0 else np.nan
    
    print(f"  Var(mean TRUE per trial): {var_true:.6f}")
    print(f"  Var(mean COLD per trial): {var_cold:.6f}")
    print(f"  Var_Ratio: {var_ratio:.6f}")
    
    mean_drci = data['statistics']['mean_drci']
    K = mean_drci * var_ratio
    print(f"  DRCI: {mean_drci:.6f}")
    print(f"  K = DRCI × Var_Ratio = {K:.6f}")
