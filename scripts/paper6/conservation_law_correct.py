import json
import numpy as np

files = [
    ('C:/Users/barla/mch_experiments/data/philosophy/open_models/mch_results_deepseek_v3_1_philosophy_50trials.json', 'DeepSeek V3.1'),
    ('C:/Users/barla/mch_experiments/data/philosophy/open_models/mch_results_llama_4_maverick_philosophy_50trials.json', 'Llama 4 Maverick')
]

print("="*70)
print("CONSERVATION LAW TEST (Using stored alignments)")
print("="*70)

results = []

for filepath, model_name in files:
    with open(filepath) as f:
        data = json.load(f)
    
    print(f"\n{model_name}:")
    
    # Get stored DRCI
    mean_drci = data['statistics']['mean_drci']
    print(f"  Mean DRCI (stored): {mean_drci:.6f}")
    
    # Compute Var_Ratio from alignments
    # Collect all position-level RCI values
    true_rcis = []
    cold_rcis = []
    
    for trial in data['trials']:
        true_rcis.extend(trial['alignments']['true'])
        cold_rcis.extend(trial['alignments']['cold'])
    
    var_true = np.var(true_rcis, ddof=1)
    var_cold = np.var(cold_rcis, ddof=1)
    var_ratio = var_true / var_cold
    
    print(f"  Var(TRUE):  {var_true:.6f}")
    print(f"  Var(COLD):  {var_cold:.6f}")
    print(f"  Var_Ratio:  {var_ratio:.6f}")
    
    K = mean_drci * var_ratio
    print(f"  K = DRCI × Var_Ratio = {K:.6f}")
    
    results.append({'model': model_name, 'drci': mean_drci, 'var_ratio': var_ratio, 'K': K})

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

mean_K = np.mean([r['K'] for r in results])
sd_K = np.std([r['K'] for r in results], ddof=1)
cv_K = sd_K / abs(mean_K) if mean_K != 0 else float('inf')

print(f"\nMean K (complete 50-trial models): {mean_K:.6f}")
print(f"SD K:   {sd_K:.6f}")
print(f"CV K:   {cv_K:.6f}")

print(f"\nPaper 6 philosophy: K=0.301, CV=0.166")
print(f"Conservation law holds: {abs(mean_K - 0.301) < 0.05}")
