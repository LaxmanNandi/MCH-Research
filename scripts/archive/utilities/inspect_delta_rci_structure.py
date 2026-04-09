import json

with open('C:/Users/barla/mch_experiments/data/philosophy/open_models/mch_results_deepseek_v3_1_philosophy_50trials.json') as f:
    data = json.load(f)

print("Structure of delta_rci in first trial:")
print(json.dumps(data['trials'][0]['delta_rci'], indent=2))
