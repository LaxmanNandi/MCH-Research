import json

# Load DeepSeek with response text
with open('C:/Users/barla/mch_experiments/data/philosophy/open_models/mch_results_deepseek_v3_1_philosophy_50trials.json') as f:
    data = json.load(f)

print("="*70)
print("CHECKING TRIAL STRUCTURE")
print("="*70)

# Check if trials have drci stored
first_trial = data['trials'][0]
print("\nFirst trial keys:", list(first_trial.keys()))

if 'drci' in first_trial:
    print("\nTrial-level DRCI values stored!")
    trial_drcis = [trial['drci'] for trial in data['trials']]
    print(f"Mean of trial DRCIs: {sum(trial_drcis)/len(trial_drcis):.6f}")
    print(f"Stored mean_drci:    {data['statistics']['mean_drci']:.6f}")
    print(f"Match: {abs(sum(trial_drcis)/len(trial_drcis) - data['statistics']['mean_drci']) < 0.0001}")
else:
    print("\nNo trial-level DRCI stored")
    print("Need to check how DRCI is computed")

# Check positions
if 'position_data' in first_trial:
    print("\nPosition data structure:", list(first_trial['position_data'][0].keys()) if first_trial['position_data'] else "Empty")
