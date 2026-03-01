import pandas as pd
import json

# Check if we have medical data with response text for these models
medical_path = 'C:/Users/barla/mch_experiments/data/medical/open_models/'

models_to_check = [
    ('mch_results_deepseek_v3_1_medical_50trials.json', 'DeepSeek V3.1'),
    ('mch_results_llama_4_maverick_medical_50trials.json', 'Llama 4 Maverick')
]

print("="*70)
print("CHECKING MEDICAL DOMAIN CONSERVATION LAW")
print("="*70)

for filename, model_name in models_to_check:
    try:
        with open(medical_path + filename) as f:
            data = json.load(f)
        
        # Check if we have the needed metrics already computed
        # We need to load from results/tables if available
        print(f"\n{model_name} (medical):")
        print(f"  File exists: YES")
        print(f"  N trials: {data['n_trials']}")
        print(f"  Response text saved: {data.get('response_text_saved', False)}")
        
    except FileNotFoundError:
        print(f"\n{model_name} (medical): File not found")

print("\n" + "="*70)
print("HYPOTHESIS TEST")
print("="*70)
print("\nIf these models show:")
print("  - Normal K (~0.4) in medical domain")
print("  - Near-zero K (~0.01) in philosophy domain")
print("\nThen: K is MODEL-FAMILY × DOMAIN dependent")
print("      Open models don't engage with philosophy!")
