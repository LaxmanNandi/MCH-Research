import json
import os

files = [
    'C:/Users/barla/mch_experiments/data/philosophy/open_models/mch_results_deepseek_v3_1_philosophy_50trials.json',
    'C:/Users/barla/mch_experiments/data/philosophy/open_models/mch_results_deepseek_v3_1_philosophy_50trials_metrics_only.json',
    'C:/Users/barla/mch_experiments/data/philosophy/open_models/mch_results_llama_4_maverick_philosophy_50trials.json',
    'C:/Users/barla/mch_experiments/data/philosophy/open_models/mch_results_llama_4_maverick_philosophy_50trials_metrics_only.json'
]

for filepath in files:
    if os.path.exists(filepath):
        with open(filepath) as f:
            data = json.load(f)
        
        size_mb = os.path.getsize(filepath) / (1024*1024)
        print(f"\n{os.path.basename(filepath)}:")
        print(f"  Size: {size_mb:.1f} MB")
        print(f"  Timestamp: {data.get('timestamp', 'N/A')}")
        print(f"  N trials: {data.get('n_trials', 'N/A')}")
        print(f"  Response text saved: {data.get('response_text_saved', 'N/A')}")
        if 'statistics' in data:
            print(f"  Mean DRCI (stored): {data['statistics'].get('mean_drci', 'N/A')}")
