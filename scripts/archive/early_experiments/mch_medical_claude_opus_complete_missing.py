#!/usr/bin/env python3
"""MCH Medical Experiment - Complete missing 7 Claude Opus trials."""

import os
import sys
import json
import time
import random
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

load_dotenv()

import anthropic
from sentence_transformers import SentenceTransformer

# API client
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Load embedding model
print("Loading embedding model: all-MiniLM-L6-v2...", flush=True)
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded.", flush=True)

# Medical STEMI case prompts
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

MODEL_NAME = "claude_opus"
MODEL_ID = "claude-opus-4-5-20251101"
TEMPERATURE = 0.7
OUTPUT_DIR = "C:/Users/barla/mch_experiments/data/medical_results"

# Missing trials to complete (0-indexed, so trial 28 = index 28)
MISSING_TRIALS = [28, 29, 33, 34, 47, 48, 49]

def get_embedding(text):
    return embedder.encode(text, convert_to_numpy=True)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_response_anthropic(messages, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = anthropic_client.messages.create(
                model=MODEL_ID,
                max_tokens=1024,
                messages=messages,
                temperature=TEMPERATURE
            )
            return response.content[0].text
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    Retry {attempt+1}: {e}", flush=True)
                time.sleep(2 ** (attempt + 1))
            else:
                raise e

def run_trial(trial_num):
    print(f"  [Claude Opus] Trial {trial_num}/50...", flush=True)

    # TRUE condition - full conversation history
    true_messages = []
    true_responses = []
    for prompt in PROMPTS:
        true_messages.append({"role": "user", "content": prompt})
        resp = get_response_anthropic(true_messages)
        true_responses.append(resp)
        true_messages.append({"role": "assistant", "content": resp})

    # COLD condition - no history (each prompt independent)
    cold_responses = []
    for prompt in PROMPTS:
        resp = get_response_anthropic([{"role": "user", "content": prompt}])
        cold_responses.append(resp)

    # SCRAMBLED condition - randomized history
    scrambled_order = list(range(len(PROMPTS)))
    random.shuffle(scrambled_order)
    scrambled_messages = []
    scrambled_responses = []
    for idx in scrambled_order:
        scrambled_messages.append({"role": "user", "content": PROMPTS[idx]})
        resp = get_response_anthropic(scrambled_messages)
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

    print(f"  [Claude Opus] Trial {trial_num} dRCI={delta_rci_cold:.4f}", flush=True)

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
    print("MCH MEDICAL - COMPLETE MISSING CLAUDE OPUS TRIALS", flush=True)
    print("="*70, flush=True)
    print(f"Missing trials: {MISSING_TRIALS}", flush=True)
    print(f"Model: {MODEL_ID}", flush=True)
    print("="*70, flush=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load existing recovered data
    recovered_file = os.path.join(OUTPUT_DIR, "mch_results_claude_opus_medical_43trials_recovered.json")
    with open(recovered_file, 'r') as f:
        existing_data = json.load(f)

    existing_trials = existing_data.get("trials", [])
    existing_trial_nums = {t["trial"] for t in existing_trials}
    print(f"Existing trials: {len(existing_trials)}", flush=True)
    print(f"Existing trial numbers: {sorted(existing_trial_nums)}", flush=True)

    # Checkpoint for new trials
    checkpoint_file = os.path.join(OUTPUT_DIR, "mch_results_claude_opus_missing_checkpoint.json")

    new_trials = []
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        new_trials = checkpoint.get("trials", [])
        print(f"Resuming with {len(new_trials)} already completed", flush=True)

    completed_new = {t["trial"] for t in new_trials}
    remaining = [t for t in MISSING_TRIALS if t not in completed_new]
    print(f"Remaining to run: {remaining}", flush=True)

    # Run missing trials
    for trial_num in remaining:
        trial_data = run_trial(trial_num)
        new_trials.append(trial_data)

        # Save checkpoint after each trial
        checkpoint = {
            "model": MODEL_NAME,
            "model_id": MODEL_ID,
            "trials": new_trials
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        print(f"  Checkpoint saved: {len(new_trials)} new trials", flush=True)

    # Merge all trials
    print("\n" + "="*70, flush=True)
    print("MERGING TRIALS", flush=True)
    print("="*70, flush=True)

    # Convert existing trials to full format (they only have delta_rci)
    all_trials = []
    for t in existing_trials:
        all_trials.append(t)

    # Add new trials
    for t in new_trials:
        all_trials.append(t)

    # Sort by trial number
    all_trials.sort(key=lambda x: x["trial"])

    # Calculate summary statistics
    drcis = [t["delta_rci"]["cold"] for t in all_trials]

    final_results = {
        "model": MODEL_NAME,
        "model_id": MODEL_ID,
        "vendor": "anthropic",
        "domain": "medical_reasoning",
        "n_trials": len(all_trials),
        "n_prompts": len(PROMPTS),
        "temperature": TEMPERATURE,
        "timestamp": datetime.now().isoformat(),
        "status": "COMPLETE",
        "note": "Merged from recovered trials (43) + newly run trials (7)",
        "summary": {
            "mean_drci": float(np.mean(drcis)),
            "std_drci": float(np.std(drcis)),
            "min_drci": float(np.min(drcis)),
            "max_drci": float(np.max(drcis)),
            "pattern": "CONVERGENT" if np.mean(drcis) > 0.01 else "SOVEREIGN" if np.mean(drcis) < -0.01 else "NEUTRAL",
            "convergent_trials": sum(1 for d in drcis if d > 0.01),
            "convergent_pct": sum(1 for d in drcis if d > 0.01) / len(drcis) * 100
        },
        "trials": all_trials
    }

    # Save final merged file
    final_file = os.path.join(OUTPUT_DIR, "mch_results_claude_opus_medical_50trials.json")
    with open(final_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\nFINAL RESULTS:", flush=True)
    print(f"  Total trials: {len(all_trials)}", flush=True)
    print(f"  Trial numbers: {[t['trial'] for t in all_trials]}", flush=True)
    print(f"  Mean dRCI: {np.mean(drcis):.4f} +/- {np.std(drcis):.4f}", flush=True)
    print(f"  Pattern: {final_results['summary']['pattern']}", flush=True)
    print(f"  Saved to: {final_file}", flush=True)

    # Clean up checkpoint
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"  Checkpoint removed.", flush=True)

if __name__ == "__main__":
    main()
