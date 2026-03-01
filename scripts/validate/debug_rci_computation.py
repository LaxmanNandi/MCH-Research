import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load DeepSeek data
with open('C:/Users/barla/mch_experiments/data/philosophy/open_models/mch_results_deepseek_v3_1_philosophy_50trials.json') as f:
    data = json.load(f)

print("Stored mean_drci:", data['statistics']['mean_drci'])
print("\nDebug: Checking RCI computation method...")

# Method 1: Trial-averaged (what the files likely use)
# For each trial, compute mean RCI across positions
print("\nMethod 1: Trial-averaged DRCI")
trial_drcis = []

for trial_idx, trial in enumerate(data['trials'][:5]):  # Just first 5 trials for speed
    true_responses = trial['responses']['true']
    cold_responses = trial['responses']['cold']
    
    # Compute embeddings for all positions in this trial
    true_embs = model.encode(true_responses, convert_to_numpy=True)
    cold_embs = model.encode(cold_responses, convert_to_numpy=True)
    
    # RCI for TRUE: average pairwise similarity
    n = len(true_embs)
    true_sims = []
    for i in range(n):
        for j in range(i+1, n):
            sim = np.dot(true_embs[i], true_embs[j]) / (np.linalg.norm(true_embs[i]) * np.linalg.norm(true_embs[j]))
            true_sims.append(sim)
    rci_true = np.mean(true_sims)
    
    # RCI for COLD
    cold_sims = []
    for i in range(n):
        for j in range(i+1, n):
            sim = np.dot(cold_embs[i], cold_embs[j]) / (np.linalg.norm(cold_embs[i]) * np.linalg.norm(cold_embs[j]))
            cold_sims.append(sim)
    rci_cold = np.mean(cold_sims)
    
    drci = rci_true - rci_cold
    trial_drcis.append(drci)
    print(f"  Trial {trial_idx}: RCI_TRUE={rci_true:.4f}, RCI_COLD={rci_cold:.4f}, DRCI={drci:.4f}")

print(f"\nMean DRCI (first 5 trials): {np.mean(trial_drcis):.4f}")
print(f"Expected (stored):          {data['statistics']['mean_drci']:.4f}")
