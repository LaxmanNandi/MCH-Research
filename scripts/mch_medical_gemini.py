#!/usr/bin/env python3
"""MCH Medical Experiment - Gemini 2.5 Flash and Pro."""

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

import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# API client
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

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

# Models to run
MODELS_TO_RUN = [
    ("gemini_3_pro", "gemini-3-pro-preview", "google"),
]

N_TRIALS = 50
TEMPERATURE = 0.7
OUTPUT_DIR = "C:/Users/barla/mch_experiments/medical_results"

def get_embedding(text):
    return embedder.encode(text, convert_to_numpy=True)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_response_google(model_id, messages, max_retries=5):
    """Get response from Gemini API with conversation history."""
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel(model_id)

            # Convert messages to Gemini format
            gemini_history = []
            for i, m in enumerate(messages[:-1]):  # All except last message
                role = "user" if m["role"] == "user" else "model"
                gemini_history.append({"role": role, "parts": [m["content"]]})

            # Start chat with history
            chat = model.start_chat(history=gemini_history)

            # Send the last message
            response = chat.send_message(
                messages[-1]["content"],
                generation_config=genai.types.GenerationConfig(
                    temperature=TEMPERATURE,
                    max_output_tokens=1024
                )
            )
            return response.text
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    Retry {attempt+1}: {e}", flush=True)
                time.sleep(2 ** (attempt + 1))
            else:
                raise e

def run_trial(model_name, model_id, vendor, trial_num):
    print(f"  [{model_name}] Trial {trial_num+1}/{N_TRIALS}...", flush=True)

    # TRUE condition - full conversation history
    true_messages = []
    true_responses = []
    for prompt in PROMPTS:
        true_messages.append({"role": "user", "content": prompt})
        resp = get_response_google(model_id, true_messages)
        true_responses.append(resp)
        true_messages.append({"role": "assistant", "content": resp})

    # COLD condition - no history (each prompt independent)
    cold_responses = []
    for prompt in PROMPTS:
        resp = get_response_google(model_id, [{"role": "user", "content": prompt}])
        cold_responses.append(resp)

    # SCRAMBLED condition - randomized history
    scrambled_order = list(range(len(PROMPTS)))
    random.shuffle(scrambled_order)
    scrambled_messages = []
    scrambled_responses = []
    for idx in scrambled_order:
        scrambled_messages.append({"role": "user", "content": PROMPTS[idx]})
        resp = get_response_google(model_id, scrambled_messages)
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

    print(f"  [{model_name}] Trial {trial_num+1} dRCI={delta_rci_cold:.4f}", flush=True)

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

def run_model(model_name, model_id, vendor):
    print(f"\n{'='*60}", flush=True)
    print(f"Starting {model_name} ({vendor})", flush=True)
    print(f"{'='*60}", flush=True)

    checkpoint_file = os.path.join(OUTPUT_DIR, f"mch_results_{model_name}_medical_checkpoint.json")
    final_file = os.path.join(OUTPUT_DIR, f"mch_results_{model_name}_medical_50trials.json")

    # Check if already complete
    if os.path.exists(final_file):
        print(f"  {model_name} already complete. Skipping.", flush=True)
        return

    # Load checkpoint if exists
    trials = []
    start_trial = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        trials = checkpoint.get("trials", [])
        start_trial = len(trials)
        print(f"  Resuming from trial {start_trial}", flush=True)

    # Run remaining trials
    for i in range(start_trial, N_TRIALS):
        trial_data = run_trial(model_name, model_id, vendor, i)
        trials.append(trial_data)

        # Save checkpoint every 5 trials
        if (i + 1) % 5 == 0:
            checkpoint = {
                "model": model_name,
                "model_id": model_id,
                "vendor": vendor,
                "domain": "medical_reasoning",
                "n_trials": len(trials),
                "temperature": TEMPERATURE,
                "trials": trials
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            print(f"  Checkpoint saved: {len(trials)} trials", flush=True)

    # Save final results
    final_results = {
        "model": model_name,
        "model_id": model_id,
        "vendor": vendor,
        "domain": "medical_reasoning",
        "n_trials": N_TRIALS,
        "n_prompts": len(PROMPTS),
        "temperature": TEMPERATURE,
        "timestamp": datetime.now().isoformat(),
        "trials": trials
    }
    with open(final_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    # Summary
    valid = [t for t in trials if abs(t["delta_rci"]["cold"]) > 0.0001]
    drcis = [t["delta_rci"]["cold"] for t in valid]
    print(f"\n  {model_name} COMPLETE: {len(valid)}/{N_TRIALS} valid trials", flush=True)
    print(f"  Mean dRCI: {np.mean(drcis):.4f} +/- {np.std(drcis):.4f}", flush=True)

def main():
    print("="*70, flush=True)
    print("MCH MEDICAL REASONING EXPERIMENT - GEMINI 3 PRO", flush=True)
    print("Testing Model Coherence Hypothesis on Clinical Decision Making", flush=True)
    print("="*70, flush=True)
    print(f"Domain: Medical Clinical Reasoning (STEMI Case)", flush=True)
    print(f"Prompts: {len(PROMPTS)}", flush=True)
    print(f"Trials per model: {N_TRIALS}", flush=True)
    print(f"Temperature: {TEMPERATURE}", flush=True)
    print(f"Models: {[m[0] for m in MODELS_TO_RUN]}", flush=True)
    print("="*70, flush=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run each model sequentially
    for model_name, model_id, vendor in MODELS_TO_RUN:
        run_model(model_name, model_id, vendor)

    print("\n" + "="*70, flush=True)
    print("ALL MODELS COMPLETE", flush=True)
    print("="*70, flush=True)

if __name__ == "__main__":
    main()
