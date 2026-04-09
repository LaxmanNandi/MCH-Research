import json

with open('C:/Users/barla/mch_experiments/data/philosophy/open_models/mch_results_deepseek_v3_1_philosophy_50trials.json') as f:
    data = json.load(f)

print("Checking 'means' structure in first trial:")
print(json.dumps(data['trials'][0]['means'], indent=2))

print("\n\nChecking 'alignments' structure:")
if 'alignments' in data['trials'][0]:
    aligns = data['trials'][0]['alignments']
    if isinstance(aligns, dict):
        print(f"Keys: {list(aligns.keys())}")
        if 'true' in aligns:
            print(f"\nTrue alignments (first 3): {aligns['true'][:3]}")
        if 'cold' in aligns:
            print(f"Cold alignments (first 3): {aligns['cold'][:3]}")
