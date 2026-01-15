#!/usr/bin/env python3
"""
VERIFICATION 1: GPT-4o-mini REPLICATION TEST
Purpose: Test if bimodal behavior (SOVEREIGN -> CONVERGENT) is reproducible
"""

import os
import sys
import json
import time
import random
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

sys.stdout.reconfigure(line_buffering=True)
load_dotenv()

import openai
from sentence_transformers import SentenceTransformer

# API client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load embedding model
print("Loading embedding model: all-MiniLM-L6-v2...", flush=True)
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded.", flush=True)

# Medical STEMI case prompts (same as original)
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

MODEL_ID = "gpt-4o-mini"
N_TRIALS = 50
TEMPERATURE = 0.7
OUTPUT_DIR = "C:/Users/barla/mch_experiments/medical_results"

def get_embedding(text):
    return embedder.encode(text, convert_to_numpy=True)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_response(messages, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    Retry {attempt+1}: {e}", flush=True)
                time.sleep(2 ** (attempt + 1))
            else:
                raise e

def run_trial(trial_num):
    print(f"  [gpt4o_mini_RERUN] Trial {trial_num+1}/{N_TRIALS}...", flush=True)

    # TRUE condition - full conversation history
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

    # Compute embeddings
    true_embs = [get_embedding(r) for r in true_responses]
    cold_embs = [get_embedding(r) for r in cold_responses]
    scrambled_embs = [get_embedding(r) for r in scrambled_responses]

    # Compute alignments
    true_aligns = [float(cosine_similarity(true_embs[i], true_embs[i])) for i in range(len(PROMPTS))]
    cold_aligns = [float(cosine_similarity(true_embs[i], cold_embs[i])) for i in range(len(PROMPTS))]

    scrambled_aligns = []
    for i in range(len(PROMPTS)):
        scrambled_idx = scrambled_order.index(i)
        scrambled_aligns.append(float(cosine_similarity(true_embs[i], scrambled_embs[scrambled_idx])))

    mean_true = float(np.mean(true_aligns))
    mean_cold = float(np.mean(cold_aligns))
    mean_scrambled = float(np.mean(scrambled_aligns))

    delta_rci_cold = mean_true - mean_cold
    delta_rci_scrambled = mean_true - mean_scrambled

    print(f"  [gpt4o_mini_RERUN] Trial {trial_num+1} dRCI={delta_rci_cold:.4f}", flush=True)

    return {
        "trial": trial_num,
        "prompts": PROMPTS,
        "alignments": {
            "true": true_aligns,
            "cold": cold_aligns,
            "scrambled": scrambled_aligns,
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
    print("="*70, flush=True)
    print("VERIFICATION 1: GPT-4o-mini REPLICATION TEST", flush=True)
    print("Testing if bimodal behavior is reproducible", flush=True)
    print("="*70, flush=True)
    print(f"Model: {MODEL_ID}", flush=True)
    print(f"Trials: {N_TRIALS}", flush=True)
    print(f"Temperature: {TEMPERATURE}", flush=True)
    print("="*70, flush=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    checkpoint_file = os.path.join(OUTPUT_DIR, "mch_results_gpt4o_mini_medical_RERUN_checkpoint.json")
    final_file = os.path.join(OUTPUT_DIR, "mch_results_gpt4o_mini_medical_RERUN.json")

    # Load checkpoint if exists
    trials = []
    start_trial = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        trials = checkpoint.get("trials", [])
        start_trial = len(trials)
        print(f"  Resuming from trial {start_trial}", flush=True)

    # Track progression at intervals
    interval_stats = {}

    for i in range(start_trial, N_TRIALS):
        trial_data = run_trial(i)
        trials.append(trial_data)

        # Record interval stats
        if (i + 1) in [10, 20, 30, 40, 50]:
            drcis = [t['delta_rci']['cold'] for t in trials]
            interval_stats[i + 1] = {
                "mean_drci": float(np.mean(drcis)),
                "std_drci": float(np.std(drcis)),
                "n_trials": len(drcis)
            }
            print(f"  === INTERVAL {i+1}: Mean dRCI = {np.mean(drcis):.4f} +/- {np.std(drcis):.4f} ===", flush=True)

        # Save checkpoint every 5 trials
        if (i + 1) % 5 == 0:
            checkpoint = {
                "model": "gpt4o_mini_RERUN",
                "model_id": MODEL_ID,
                "vendor": "openai",
                "domain": "medical_reasoning",
                "n_trials": len(trials),
                "temperature": TEMPERATURE,
                "interval_stats": interval_stats,
                "trials": trials
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            print(f"  Checkpoint saved: {len(trials)} trials", flush=True)

    # Save final results
    final_results = {
        "model": "gpt4o_mini_RERUN",
        "model_id": MODEL_ID,
        "vendor": "openai",
        "domain": "medical_reasoning",
        "purpose": "Replication test for bimodal behavior verification",
        "n_trials": N_TRIALS,
        "n_prompts": len(PROMPTS),
        "temperature": TEMPERATURE,
        "timestamp": datetime.now().isoformat(),
        "interval_stats": interval_stats,
        "trials": trials
    }
    with open(final_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    # Summary
    valid = [t for t in trials if abs(t["delta_rci"]["cold"]) > 0.0001]
    drcis = [t["delta_rci"]["cold"] for t in valid]

    print(f"\n{'='*70}", flush=True)
    print(f"GPT-4o-mini RERUN COMPLETE: {len(valid)}/{N_TRIALS} valid trials", flush=True)
    print(f"Mean dRCI: {np.mean(drcis):.4f} +/- {np.std(drcis):.4f}", flush=True)
    print(f"{'='*70}", flush=True)

    # Detect mode switches
    print("\nMode Switch Detection:", flush=True)
    for i in range(1, len(drcis)):
        prev_pattern = "CONVERGENT" if drcis[i-1] > 0.1 else "SOVEREIGN" if drcis[i-1] < -0.05 else "NEUTRAL"
        curr_pattern = "CONVERGENT" if drcis[i] > 0.1 else "SOVEREIGN" if drcis[i] < -0.05 else "NEUTRAL"
        if prev_pattern != curr_pattern and prev_pattern != "NEUTRAL" and curr_pattern != "NEUTRAL":
            print(f"  Trial {i+1}: {prev_pattern} -> {curr_pattern} (dRCI: {drcis[i-1]:.4f} -> {drcis[i]:.4f})", flush=True)

if __name__ == "__main__":
    main()
