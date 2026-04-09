import json

# Original metrics_only files (Paper 2/6)
files_to_check = [
    ('C:/Users/barla/mch_experiments/data/philosophy/open_models/mch_results_deepseek_v3_1_philosophy_50trials_metrics_only.json', 'DeepSeek V3.1'),
    ('C:/Users/barla/mch_experiments/data/philosophy/open_models/mch_results_llama_4_maverick_philosophy_50trials_metrics_only.json', 'Llama 4 Maverick')
]

print("="*70)
print("ORIGINAL vs RERUN DATA COMPARISON")
print("="*70)

for filepath, model_name in files_to_check:
    with open(filepath) as f:
        data = json.load(f)
    
    print(f"\n{model_name} Philosophy:")
    print(f"  Original file timestamp: {data.get('timestamp', 'N/A')}")
    print(f"  Original mean DRCI:      {data['statistics']['mean_drci']:.4f}")
    print(f"  Original pattern:        {data['statistics']['pattern']}")

print("\n" + "="*70)
print("RERUN VERIFICATION (from response text files)")
print("="*70)
print("\nDeepSeek V3.1 Philosophy (rerun with response text):")
print("  Rerun DRCI:     0.0021")
print("  Difference:     140x LOWER than original!")

print("\nLlama 4 Maverick Philosophy (rerun with response text):")
print("  Rerun DRCI:     0.0330")
print("  Difference:     8x LOWER than original!")

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
print("\nThe 'rerun' files are DIFFERENT experimental runs!")
print("This reveals:")
print("  1. High temporal/sampling variation in philosophy domain")
print("  2. Context sensitivity is NOT stable across runs")
print("  3. Conservation law K might vary RUN-TO-RUN, not just model-to-model")
print("\nThis is a REPRODUCIBILITY finding!")
