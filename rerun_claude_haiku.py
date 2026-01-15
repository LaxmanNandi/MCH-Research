#!/usr/bin/env python3
"""Re-run Claude Haiku medical experiment with 50 fresh trials."""

import os
import json
import time
import random
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import anthropic
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Load embedding model
print('Loading embedding model...', flush=True)
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
print('Embedding model loaded.', flush=True)

PROMPTS = [
    "A 52-year-old male presents to the emergency department with sudden onset chest pain. What are your initial assessment priorities?",
    "The pain is described as crushing, substernal, radiating to left arm and jaw, started 1 hour ago. Rate 8/10. What is your differential diagnosis?",
    "What specific questions would you ask to differentiate between these diagnoses?",
    "Patient reports associated diaphoresis and nausea. No prior cardiac history. Smoker 20 pack-years. What does this suggest?",
    "Vital signs: BP 160/95, HR 102, RR 22, SpO2 96% on room air. Interpret these findings.",
    "What physical examination would you perform and what findings would you look for?",
    "Examination reveals S4 gallop, no murmurs, lungs clear, no peripheral edema. What does this indicate?",
    "What immediate investigations would you order?",
    "ECG shows ST elevation in leads V1-V4. Interpret this finding.",
    "What is your working diagnosis now?",
    "Initial troponin returns elevated at 2.5 ng/mL (normal <0.04). How does this change your assessment?",
    "What immediate management would you initiate?",
    "What are the contraindications you would check before thrombolysis?",
    "Patient has no contraindications. PCI is available in 45 minutes. What is the preferred reperfusion strategy and why?",
    "While awaiting PCI, the patient develops hypotension (BP 85/60). What are the possible causes?",
    "What would you do to assess and manage this hypotension?",
    "Repeat ECG shows new right-sided ST elevation. What does this suggest?",
    "How does RV involvement change your management approach?",
    "Patient is taken for PCI. 95% occlusion of proximal LAD is found. What do you expect post-procedure?",
    "Post-PCI, patient is stable. What medications would you prescribe for secondary prevention?",
    "Explain the rationale for each medication class you prescribed.",
    "What complications would you monitor for in the first 48 hours?",
    "On day 2, patient develops new systolic murmur. What are the concerning diagnoses?",
    "Echo shows mild MR with preserved EF of 45%. How do you interpret this?",
    "What is the patient's risk stratification and prognosis?",
    "What lifestyle modifications would you counsel?",
    "When would you recommend cardiac rehabilitation?",
    "Patient asks about returning to work as a truck driver. How would you counsel him?",
    "At 6-week follow-up, patient reports occasional chest discomfort with exertion. What evaluation would you do?",
    "Summarize this case: key decision points, management principles, and learning points."
]

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL_ID = "claude-haiku-4-5-20251001"
N_TRIALS = 50
OUTPUT_DIR = "C:/Users/barla/mch_experiments/medical_results"

def get_embedding(text):
    return embed_model.encode(text, convert_to_numpy=True)

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_response(messages, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=MODEL_ID,
                max_tokens=1024,
                temperature=0.7,
                messages=messages
            )
            return response.content[0].text
        except Exception as e:
            print(f"    Retry {attempt+1}/{max_retries}: {e}", flush=True)
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
            else:
                raise e

def run_trial(trial_num):
    print(f"  [claude_haiku] Trial {trial_num+1}/50...", flush=True)

    # TRUE condition - full conversation
    true_messages = []
    true_responses = []
    for prompt in PROMPTS:
        true_messages.append({"role": "user", "content": prompt})
        resp = get_response(true_messages)
        true_responses.append(resp)
        true_messages.append({"role": "assistant", "content": resp})

    # COLD condition - no history
    cold_responses = []
    for prompt in PROMPTS:
        resp = get_response([{"role": "user", "content": prompt}])
        cold_responses.append(resp)

    # SCRAMBLED condition
    scrambled_order = list(range(len(PROMPTS)))
    random.shuffle(scrambled_order)
    scrambled_messages = []
    scrambled_responses = []
    for idx in scrambled_order:
        scrambled_messages.append({"role": "user", "content": PROMPTS[idx]})
        resp = get_response(scrambled_messages)
        scrambled_responses.append(resp)
        scrambled_messages.append({"role": "assistant", "content": resp})

    # Compute alignments
    true_embs = [get_embedding(r) for r in true_responses]
    cold_embs = [get_embedding(r) for r in cold_responses]
    scrambled_embs = [get_embedding(r) for r in scrambled_responses]

    true_aligns = [cosine_sim(true_embs[i], true_embs[i]) for i in range(len(PROMPTS))]
    cold_aligns = [cosine_sim(true_embs[i], cold_embs[i]) for i in range(len(PROMPTS))]

    # For scrambled, match by original prompt index
    scrambled_aligns = []
    for i in range(len(PROMPTS)):
        scrambled_idx = scrambled_order.index(i)
        scrambled_aligns.append(cosine_sim(true_embs[i], scrambled_embs[scrambled_idx]))

    mean_true = float(np.mean(true_aligns))
    mean_cold = float(np.mean(cold_aligns))
    mean_scrambled = float(np.mean(scrambled_aligns))

    delta_rci_cold = mean_true - mean_cold
    delta_rci_scrambled = mean_true - mean_scrambled

    print(f"  [claude_haiku] Trial {trial_num+1} dRCI={delta_rci_cold:.4f}", flush=True)

    return {
        "trial": trial_num,
        "prompts": PROMPTS,
        "alignments": {
            "true": [float(x) for x in true_aligns],
            "cold": [float(x) for x in cold_aligns],
            "scrambled": [float(x) for x in scrambled_aligns],
            "mean_true": mean_true,
            "mean_cold": mean_cold,
            "mean_scrambled": mean_scrambled
        },
        "delta_rci": {
            "cold": delta_rci_cold,
            "scrambled": delta_rci_scrambled
        }
    }

def main():
    print("=" * 60)
    print("Re-running Claude Haiku - 50 fresh trials")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    trials = []
    checkpoint_file = os.path.join(OUTPUT_DIR, "mch_results_claude_haiku_medical_checkpoint.json")
    final_file = os.path.join(OUTPUT_DIR, "mch_results_claude_haiku_medical_50trials.json")

    # Check for existing checkpoint
    start_trial = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            checkpoint = json.load(f)
        trials = checkpoint.get("trials", [])
        start_trial = len(trials)
        print(f"  Resuming from trial {start_trial}", flush=True)

    for i in range(start_trial, N_TRIALS):
        trial_data = run_trial(i)
        trials.append(trial_data)

        # Save checkpoint every 5 trials
        if (i + 1) % 5 == 0:
            checkpoint = {
                "model": "claude_haiku",
                "model_id": MODEL_ID,
                "vendor": "Anthropic",
                "tier": "efficient",
                "domain": "medical_reasoning",
                "n_trials": len(trials),
                "temperature": 0.7,
                "trials": trials
            }
            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint, f, indent=2)
            print(f"  Checkpoint saved: {len(trials)} trials", flush=True)

    # Save final
    final_results = {
        "model": "claude_haiku",
        "model_id": MODEL_ID,
        "vendor": "Anthropic",
        "tier": "efficient",
        "domain": "medical_reasoning",
        "n_trials": N_TRIALS,
        "n_prompts": len(PROMPTS),
        "temperature": 0.7,
        "timestamp": datetime.now().isoformat(),
        "trials": trials
    }
    with open(final_file, "w") as f:
        json.dump(final_results, f, indent=2)

    # Summary
    valid = [t for t in trials if abs(t["delta_rci"]["cold"]) > 0.0001]
    drcis = [t["delta_rci"]["cold"] for t in valid]
    print(f"\nClaude Haiku COMPLETE: {len(valid)}/{N_TRIALS} valid trials")
    print(f"Mean dRCI: {np.mean(drcis):.4f} +/- {np.std(drcis):.4f}")

if __name__ == "__main__":
    main()
