import json

# Load original metrics_only files
with open('C:/Users/barla/mch_experiments/data/philosophy/open_models/mch_results_deepseek_v3_1_philosophy_50trials_metrics_only.json') as f:
    deepseek_orig = json.load(f)

with open('C:/Users/barla/mch_experiments/data/philosophy/open_models/mch_results_llama_4_maverick_philosophy_50trials_metrics_only.json') as f:
    llama_orig = json.load(f)

print("="*70)
print("ORIGINAL Paper 2 DATA (metrics_only files)")
print("="*70)

print("\nDeepSeek V3.1 Philosophy:")
print(f"  DRCI: {deepseek_orig.get('drci', 'N/A')}")
print(f"  Mean RCI TRUE: {deepseek_orig.get('mean_rci_true', 'N/A')}")
print(f"  Mean RCI COLD: {deepseek_orig.get('mean_rci_cold', 'N/A')}")

print("\nLlama 4 Maverick Philosophy:")
print(f"  DRCI: {llama_orig.get('drci', 'N/A')}")
print(f"  Mean RCI TRUE: {llama_orig.get('mean_rci_true', 'N/A')}")  
print(f"  Mean RCI COLD: {llama_orig.get('mean_rci_cold', 'N/A')}")

print("\n" + "="*70)
print("COMPARISON TO RERUN VERIFICATION")
print("="*70)
print("\nDeepSeek V3.1:")
print(f"  Original DRCI: {deepseek_orig.get('drci', 'N/A')}")
print(f"  Rerun DRCI:    0.0021")
print(f"  Ratio:         {deepseek_orig.get('drci', 0) / 0.0021 if 0.0021 != 0 else 'inf'}x")

print("\nLlama 4 Maverick:")
print(f"  Original DRCI: {llama_orig.get('drci', 'N/A')}")  
print(f"  Rerun DRCI:    0.0330")
print(f"  Ratio:         {llama_orig.get('drci', 0) / 0.0330 if 0.0330 != 0 else 'inf'}x")
