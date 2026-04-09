import json
import numpy as np

# Load DeepSeek with response text
with open('C:/Users/barla/mch_experiments/data/philosophy/open_models/mch_results_deepseek_v3_1_philosophy_50trials.json') as f:
    deepseek = json.load(f)

# Load Llama with response text  
with open('C:/Users/barla/mch_experiments/data/philosophy/open_models/mch_results_llama_4_maverick_philosophy_50trials.json') as f:
    llama = json.load(f)

print("="*70)
print("STORED DELTA_RCI VALUES IN RESPONSE TEXT FILES")
print("="*70)

print("\nDeepSeek V3.1:")
deepseek_drcis = [trial['delta_rci'] for trial in deepseek['trials']]
print(f"  Trial-level DRCIs computed: {len(deepseek_drcis)}")
print(f"  Mean of trial DRCIs: {np.mean(deepseek_drcis):.6f}")
print(f"  Stored mean_drci:    {deepseek['statistics']['mean_drci']:.6f}")
print(f"  Match: {abs(np.mean(deepseek_drcis) - deepseek['statistics']['mean_drci']) < 0.00001}")

print("\nLlama 4 Maverick:")
llama_drcis = [trial['delta_rci'] for trial in llama['trials']]
print(f"  Trial-level DRCIs computed: {len(llama_drcis)}")
print(f"  Mean of trial DRCIs: {np.mean(llama_drcis):.6f}")
print(f"  Stored mean_drci:    {llama['statistics']['mean_drci']:.6f}")
print(f"  Match: {abs(np.mean(llama_drcis) - llama['statistics']['mean_drci']) < 0.00001}")

print("\n" + "="*70)
print("CONSERVATION LAW CHECK (using stored values)")
print("="*70)

# We need Var_Ratio for conservation check
# Check if we have variance data
if 'alignments' in deepseek['trials'][0]:
    print("\nAlignment data available - can compute Var_Ratio from scratch")
    print("But let me check if mean values are stored...")
    
if 'means' in deepseek['trials'][0]:
    print("\nMean values structure:")
    print(f"  Keys: {list(deepseek['trials'][0]['means'].keys())}")
