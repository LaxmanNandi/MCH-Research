#!/usr/bin/env python3
"""
MCH Gemini Flash Medical Re-run - Paper 2 Standardized Methodology

Re-testing Gemini Flash (stable release) on medical domain with the same
response-response alignment method used in the philosophy re-run.

Original medical experiment used:
  - gemini-2.5-flash-preview-05-20 (preview)
  - prompt-response alignment (TRUE != 1.0)
  - Result: SOVEREIGN (dRCI = -0.133)

This re-run uses:
  - gemini-2.5-flash (stable release)
  - response-response alignment (TRUE = 1.0, Paper 2 standardized)
  - Purpose: Determine if SOVEREIGN pattern was due to model version or domain

Protocol: 3-condition (TRUE/COLD/SCRAMBLED)
Domain: Medical (STEMI case, 30 prompts)
Trials: 50
"""

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

# ============================================================================
# API CLIENT
# ============================================================================
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load embedding model
print("Loading embedding model: all-MiniLM-L6-v2...", flush=True)
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded.", flush=True)

# ============================================================================
# EXPERIMENT PARAMETERS
# ============================================================================
MODEL_NAME = "gemini_flash"
MODEL_ID = "gemini-2.5-flash"
N_TRIALS = 50
TEMPERATURE = 0.7
MAX_TOKENS = 1024
OUTPUT_DIR = "C:/Users/barla/mch_experiments/data/gemini_flash_medical_rerun"

# ============================================================================
# MEDICAL PROMPTS (30 prompts - STEMI case, identical to Paper 2 medical)
# ============================================================================
MEDICAL_PROMPTS = [
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

# ============================================================================
# API RESPONSE FUNCTION
# ============================================================================

def get_response_google(model_id, messages, max_retries=5):
    """Get response from Google Gemini API with conversation history."""
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel(model_id)

            # Convert messages to Gemini format
            gemini_history = []
            for m in messages[:-1]:  # All except last message
                role = "user" if m["role"] == "user" else "model"
                gemini_history.append({"role": role, "parts": [m["content"]]})

            # Start chat with history
            chat = model.start_chat(history=gemini_history)

            # Send the last message
            response = chat.send_message(
                messages[-1]["content"],
                generation_config=genai.types.GenerationConfig(
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_TOKENS
                )
            )
            return response.text
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)
                print(f"    Retry {attempt+1}: {e} (waiting {wait_time}s)", flush=True)
                time.sleep(wait_time)
            else:
                raise e

# ============================================================================
# EMBEDDING & SIMILARITY
# ============================================================================

def get_embedding(text):
    """Get embedding using MiniLM-L6-v2."""
    return embedder.encode(text, convert_to_numpy=True)


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ============================================================================
# TRIAL EXECUTION (Paper 2 methodology - response-response alignment)
# ============================================================================

def run_trial(prompts, trial_num, n_trials):
    """Run a single trial with 3 conditions - Paper 2 methodology."""
    print(f"  [{MODEL_NAME}] Trial {trial_num+1}/{n_trials}...", flush=True)
    trial_start = datetime.now().isoformat()

    # TRUE condition - full conversation history
    print(f"    TRUE condition ({len(prompts)} prompts)...", flush=True)
    true_messages = []
    true_responses = []
    true_timestamps = []
    for i, prompt in enumerate(prompts):
        if (i + 1) % 10 == 0:
            print(f"      TRUE prompt {i+1}/{len(prompts)}...", flush=True)
        true_messages.append({"role": "user", "content": prompt})
        t0 = time.time()
        resp = get_response_google(MODEL_ID, true_messages)
        true_timestamps.append({"prompt_idx": i, "latency_s": round(time.time() - t0, 2)})
        true_responses.append(resp)
        true_messages.append({"role": "assistant", "content": resp})

    # COLD condition - no history (each prompt independent)
    print(f"    COLD condition ({len(prompts)} prompts)...", flush=True)
    cold_responses = []
    cold_timestamps = []
    for i, prompt in enumerate(prompts):
        if (i + 1) % 10 == 0:
            print(f"      COLD prompt {i+1}/{len(prompts)}...", flush=True)
        t0 = time.time()
        resp = get_response_google(MODEL_ID, [{"role": "user", "content": prompt}])
        cold_timestamps.append({"prompt_idx": i, "latency_s": round(time.time() - t0, 2)})
        cold_responses.append(resp)

    # SCRAMBLED condition - randomized prompt order with history
    print(f"    SCRAMBLED condition ({len(prompts)} prompts)...", flush=True)
    scrambled_order = list(range(len(prompts)))
    random.shuffle(scrambled_order)
    scrambled_messages = []
    scrambled_responses = []
    scrambled_timestamps = []
    for i, idx in enumerate(scrambled_order):
        if (i + 1) % 10 == 0:
            print(f"      SCRAMBLED prompt {i+1}/{len(prompts)}...", flush=True)
        scrambled_messages.append({"role": "user", "content": prompts[idx]})
        t0 = time.time()
        resp = get_response_google(MODEL_ID, scrambled_messages)
        scrambled_timestamps.append({"prompt_idx": idx, "position": i, "latency_s": round(time.time() - t0, 2)})
        scrambled_responses.append(resp)
        scrambled_messages.append({"role": "assistant", "content": resp})

    # Compute embeddings
    true_embs = [get_embedding(r) for r in true_responses]
    cold_embs = [get_embedding(r) for r in cold_responses]
    scrambled_embs = [get_embedding(r) for r in scrambled_responses]

    # Response-response alignment (Paper 2 standardized)
    # TRUE alignments = 1.0 (self-similarity)
    true_aligns = [1.0] * len(prompts)

    # COLD alignments = cosine(true_response[i], cold_response[i])
    cold_aligns = [float(cosine_similarity(true_embs[i], cold_embs[i]))
                   for i in range(len(prompts))]

    # SCRAMBLED alignments = cosine(true_response[i], scrambled_response[matched_position])
    scrambled_aligns = []
    for i in range(len(prompts)):
        scrambled_idx = scrambled_order.index(i)
        scrambled_aligns.append(
            float(cosine_similarity(true_embs[i], scrambled_embs[scrambled_idx]))
        )

    mean_true = float(np.mean(true_aligns))
    mean_cold = float(np.mean(cold_aligns))
    mean_scrambled = float(np.mean(scrambled_aligns))

    delta_rci_cold = mean_true - mean_cold
    delta_rci_scrambled = mean_true - mean_scrambled

    # Response lengths
    true_lengths = [len(r) for r in true_responses]
    cold_lengths = [len(r) for r in cold_responses]
    scrambled_lengths = [len(r) for r in scrambled_responses]

    print(f"  [{MODEL_NAME}] Trial {trial_num+1} dRCI={delta_rci_cold:.4f}", flush=True)

    return {
        "trial": trial_num,
        "timestamp_start": trial_start,
        "timestamp_end": datetime.now().isoformat(),
        "model_id_requested": MODEL_ID,
        "prompts": prompts,
        "responses": {
            "true": true_responses,
            "cold": cold_responses,
            "scrambled": scrambled_responses
        },
        "scrambled_order": scrambled_order,
        "alignments": {
            "true": true_aligns,
            "cold": cold_aligns,
            "scrambled": scrambled_aligns
        },
        "means": {
            "true": mean_true,
            "cold": mean_cold,
            "scrambled": mean_scrambled
        },
        "delta_rci": {
            "cold": delta_rci_cold,
            "scrambled": delta_rci_scrambled
        },
        "response_lengths": {
            "true": true_lengths,
            "cold": cold_lengths,
            "scrambled": scrambled_lengths
        },
        "latencies": {
            "true": true_timestamps,
            "cold": cold_timestamps,
            "scrambled": scrambled_timestamps
        }
    }

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment():
    """Run medical experiment for Gemini Flash (stable)."""
    prompts = MEDICAL_PROMPTS
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70, flush=True)
    print("MCH GEMINI FLASH MEDICAL RE-RUN - PAPER 2 STANDARDIZED", flush=True)
    print("=" * 70, flush=True)
    print(f"Domain: Medical (STEMI case)", flush=True)
    print(f"Protocol: 3-condition (TRUE/COLD/SCRAMBLED)", flush=True)
    print(f"Alignment: response-response (TRUE=1.0)", flush=True)
    print(f"Prompts per trial: {len(prompts)}", flush=True)
    print(f"Trials: {N_TRIALS}", flush=True)
    print(f"Temperature: {TEMPERATURE}", flush=True)
    print(f"Max tokens: {MAX_TOKENS}", flush=True)
    print(f"Embedding: all-MiniLM-L6-v2 (384D)", flush=True)
    print(f"Model: {MODEL_ID} (stable release)", flush=True)
    print(f"Original: gemini-2.5-flash-preview-05-20 (preview, SOVEREIGN -0.133)", flush=True)
    print(f"Output: {OUTPUT_DIR}", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)

    # Check for existing checkpoint
    checkpoint_file = os.path.join(OUTPUT_DIR, f"mch_results_{MODEL_NAME}_medical_checkpoint.json")
    final_file = os.path.join(OUTPUT_DIR, f"mch_results_{MODEL_NAME}_medical_{N_TRIALS}trials.json")

    if os.path.exists(final_file):
        print(f"Already completed. Results at: {final_file}", flush=True)
        return

    # Load checkpoint if exists
    trials = []
    start_trial = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
            trials = checkpoint_data.get("trials", [])
            start_trial = len(trials)
            print(f"Resuming from trial {start_trial}", flush=True)

    # Run trials
    for trial_num in range(start_trial, N_TRIALS):
        trial_result = run_trial(prompts, trial_num, N_TRIALS)
        trials.append(trial_result)

        # Save checkpoint every 5 trials
        if (trial_num + 1) % 5 == 0:
            checkpoint_data = {
                "model": MODEL_NAME,
                "model_id": MODEL_ID,
                "vendor": "google",
                "domain": "medical_reasoning",
                "alignment_method": "response-response",
                "n_trials_completed": len(trials),
                "n_trials_target": N_TRIALS,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "embedding_model": "all-MiniLM-L6-v2",
                "trials": trials
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            print(f"  Checkpoint saved at trial {trial_num + 1}", flush=True)

    # Compute statistics
    drci_cold_values = [t["delta_rci"]["cold"] for t in trials]
    drci_scrambled_values = [t["delta_rci"]["scrambled"] for t in trials]
    mean_drci = float(np.mean(drci_cold_values))
    std_drci = float(np.std(drci_cold_values))
    pattern = "CONVERGENT" if mean_drci > 0.01 else "SOVEREIGN" if mean_drci < -0.01 else "NEUTRAL"

    # Save final results
    final_data = {
        "model": MODEL_NAME,
        "model_id": MODEL_ID,
        "vendor": "google",
        "domain": "medical_reasoning",
        "alignment_method": "response-response",
        "n_trials": N_TRIALS,
        "n_prompts": len(prompts),
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "embedding_model": "all-MiniLM-L6-v2",
        "timestamp": datetime.now().isoformat(),
        "comparison_baseline": {
            "original_model_id": "gemini-2.5-flash-preview-05-20",
            "original_alignment": "prompt-response",
            "original_drci": -0.1331,
            "original_pattern": "SOVEREIGN"
        },
        "statistics": {
            "mean_drci_cold": mean_drci,
            "std_drci_cold": std_drci,
            "mean_drci_scrambled": float(np.mean(drci_scrambled_values)),
            "std_drci_scrambled": float(np.std(drci_scrambled_values)),
            "pattern": pattern
        },
        "trials": trials
    }
    with open(final_file, 'w') as f:
        json.dump(final_data, f, indent=2)

    # Remove checkpoint after successful completion
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    print(f"\nComplete! Results saved to: {final_file}", flush=True)
    print(f"Mean dRCI (cold): {mean_drci:.4f} +/- {std_drci:.4f}", flush=True)
    print(f"Pattern: {pattern}", flush=True)
    print(f"Original (preview, prompt-response): -0.1331 SOVEREIGN", flush=True)
    print(f"Re-run   (stable, response-response): {mean_drci:.4f} {pattern}", flush=True)

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_experiment()
