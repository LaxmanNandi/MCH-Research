#!/usr/bin/env python3
"""
MCH Open Models Medical Re-Run - Paper 2 Methodology
Re-running 7 open-weight models on medical domain with 52-year-old STEMI prompts.

Uses Paper 2 methodology (response-response comparison) for consistency
with closed model medical experiments and philosophy re-runs.

Models (7 open-weight via Together.ai):
  - DeepSeek-V3.1 (671B/37B active)
  - Llama 4 Maverick (17B-128E MoE)
  - Llama 4 Scout (17B-16E)
  - Qwen3 235B (A22B active)
  - Mistral Small 24B
  - Ministral 14B
  - Kimi K2 (1T params)

Protocol: 3-condition (TRUE/COLD/SCRAMBLED), 50 trials per model
Domain: Medical Clinical Reasoning (52-year-old STEMI case, 30 prompts)
Embedding: all-MiniLM-L6-v2 (384D)
"""

import os
import sys
import json
import time
import random
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

load_dotenv()

# Together.ai client (OpenAI-compatible)
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    print("ERROR: TOGETHER_API_KEY not found in .env file")
    print("Add to .env: TOGETHER_API_KEY=your_key_here")
    sys.exit(1)

together_client = OpenAI(
    api_key=TOGETHER_API_KEY,
    base_url="https://api.together.xyz/v1"
)

# Load embedding model
from sentence_transformers import SentenceTransformer

print("Loading embedding model: all-MiniLM-L6-v2...", flush=True)
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded.", flush=True)

# ============================================================================
# 7 OPEN MODELS (same IDs as mch_open_models.py)
# ============================================================================
MODELS_TO_RUN = [
    ("deepseek_v3_1", "deepseek-ai/DeepSeek-V3.1", "together"),
    ("llama_4_maverick", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", "together"),
    ("llama_4_scout", "meta-llama/Llama-4-Scout-17B-16E-Instruct", "together"),
    ("qwen3_235b", "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8", "together"),
    ("mistral_small_24b", "mistralai/Mistral-Small-24B-Instruct-2501", "together"),
    ("ministral_14b", "mistralai/Ministral-3-14B-Instruct-2512", "together"),
    ("kimi_k2", "moonshotai/Kimi-K2-Instruct-0905", "together"),
]

# ============================================================================
# MEDICAL PROMPTS - 52-year-old STEMI case (from mch_medical_sequential.py)
# ============================================================================
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

# ============================================================================
# EXPERIMENT PARAMETERS
# ============================================================================
N_TRIALS = 50
TEMPERATURE = 0.7
OUTPUT_DIR = "C:/Users/barla/mch_experiments/data/open_medical_rerun"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_embedding(text):
    return embedder.encode(text, convert_to_numpy=True)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_response_together(model_id, messages, max_retries=5):
    """Get response from Together.ai API."""
    for attempt in range(max_retries):
        try:
            response = together_client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)
                print(f"    Retry {attempt+1}: {e} (waiting {wait_time}s)", flush=True)
                time.sleep(wait_time)
            else:
                raise e

# ============================================================================
# TRIAL EXECUTION
# ============================================================================

def run_trial(model_name, model_id, trial_num):
    """Run a single trial with 3 conditions."""
    print(f"  [{model_name}] Trial {trial_num+1}/{N_TRIALS}...", flush=True)

    # TRUE condition - full conversation history
    print(f"    TRUE condition ({len(PROMPTS)} prompts)...", flush=True)
    true_messages = []
    true_responses = []
    for i, prompt in enumerate(PROMPTS):
        if (i + 1) % 10 == 0:
            print(f"      TRUE prompt {i+1}/{len(PROMPTS)}...", flush=True)
        true_messages.append({"role": "user", "content": prompt})
        resp = get_response_together(model_id, true_messages)
        true_responses.append(resp)
        true_messages.append({"role": "assistant", "content": resp})

    # COLD condition - no history (each prompt independent)
    print(f"    COLD condition ({len(PROMPTS)} prompts)...", flush=True)
    cold_responses = []
    for i, prompt in enumerate(PROMPTS):
        if (i + 1) % 10 == 0:
            print(f"      COLD prompt {i+1}/{len(PROMPTS)}...", flush=True)
        resp = get_response_together(model_id, [{"role": "user", "content": prompt}])
        cold_responses.append(resp)

    # SCRAMBLED condition - randomized history
    print(f"    SCRAMBLED condition ({len(PROMPTS)} prompts)...", flush=True)
    scrambled_order = list(range(len(PROMPTS)))
    random.shuffle(scrambled_order)
    scrambled_messages = []
    scrambled_responses = []
    for i, idx in enumerate(scrambled_order):
        if (i + 1) % 10 == 0:
            print(f"      SCRAMBLED prompt {i+1}/{len(PROMPTS)}...", flush=True)
        scrambled_messages.append({"role": "user", "content": PROMPTS[idx]})
        resp = get_response_together(model_id, scrambled_messages)
        scrambled_responses.append(resp)
        scrambled_messages.append({"role": "assistant", "content": resp})

    # Compute embeddings
    true_embs = [get_embedding(r) for r in true_responses]
    cold_embs = [get_embedding(r) for r in cold_responses]
    scrambled_embs = [get_embedding(r) for r in scrambled_responses]

    # Compute alignments (Paper 2 methodology: response-response)
    true_aligns = [1.0] * len(PROMPTS)  # Self-similarity = 1.0
    cold_aligns = [float(cosine_similarity(true_embs[i], cold_embs[i]))
                   for i in range(len(PROMPTS))]

    scrambled_aligns = []
    for i in range(len(PROMPTS)):
        scrambled_idx = scrambled_order.index(i)
        scrambled_aligns.append(
            float(cosine_similarity(true_embs[i], scrambled_embs[scrambled_idx]))
        )

    mean_true = float(np.mean(true_aligns))
    mean_cold = float(np.mean(cold_aligns))
    mean_scrambled = float(np.mean(scrambled_aligns))

    delta_rci_cold = mean_true - mean_cold
    delta_rci_scrambled = mean_true - mean_scrambled

    print(f"  [{model_name}] Trial {trial_num+1} dRCI={delta_rci_cold:.4f}", flush=True)

    return {
        "trial": trial_num,
        "prompts": PROMPTS,
        "responses": {
            "true": true_responses,
            "cold": cold_responses,
            "scrambled": scrambled_responses
        },
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
        },
        "scrambled_order": scrambled_order
    }

# ============================================================================
# MODEL RUNNER
# ============================================================================

def run_model(model_name, model_id, vendor):
    print(f"\n{'='*60}", flush=True)
    print(f"Starting {model_name} ({vendor})", flush=True)
    print(f"{'='*60}", flush=True)

    checkpoint_file = os.path.join(OUTPUT_DIR, f"mch_results_{model_name}_medical_checkpoint.json")
    final_file = os.path.join(OUTPUT_DIR, f"mch_results_{model_name}_medical_{N_TRIALS}trials.json")

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
        trial_data = run_trial(model_name, model_id, i)
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
                "embedding_model": "all-MiniLM-L6-v2",
                "prompts_version": "52_year_old_stemi",
                "trials": trials
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            print(f"  Checkpoint saved at trial {i + 1}", flush=True)

    # Compute statistics
    drci_values = [t["delta_rci"]["cold"] for t in trials]
    mean_drci = float(np.mean(drci_values))
    std_drci = float(np.std(drci_values))
    pattern = "CONVERGENT" if mean_drci > 0.01 else "SOVEREIGN" if mean_drci < -0.01 else "NEUTRAL"

    # Save final results
    final_results = {
        "model": model_name,
        "model_id": model_id,
        "vendor": vendor,
        "domain": "medical_reasoning",
        "prompts_version": "52_year_old_stemi",
        "n_trials": N_TRIALS,
        "n_prompts": len(PROMPTS),
        "temperature": TEMPERATURE,
        "embedding_model": "all-MiniLM-L6-v2",
        "timestamp": datetime.now().isoformat(),
        "statistics": {
            "mean_drci": mean_drci,
            "std_drci": std_drci,
            "pattern": pattern
        },
        "trials": trials
    }
    with open(final_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    # Remove checkpoint after successful completion
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    print(f"\n  {model_name} COMPLETE: {N_TRIALS}/{N_TRIALS} trials", flush=True)
    print(f"  Mean dRCI: {mean_drci:.4f} +/- {std_drci:.4f}", flush=True)
    print(f"  Pattern: {pattern}", flush=True)

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70, flush=True)
    print("MCH OPEN MODELS MEDICAL RE-RUN - PAPER 2 METHODOLOGY", flush=True)
    print("52-year-old STEMI Case | Response-Response Comparison", flush=True)
    print("=" * 70, flush=True)
    print(f"Domain: Medical Clinical Reasoning (52-year-old STEMI)", flush=True)
    print(f"Prompts: {len(PROMPTS)}", flush=True)
    print(f"Trials per model: {N_TRIALS}", flush=True)
    print(f"Temperature: {TEMPERATURE}", flush=True)
    print(f"Embedding: all-MiniLM-L6-v2 (384D)", flush=True)
    print(f"Models: {len(MODELS_TO_RUN)}", flush=True)
    for m in MODELS_TO_RUN:
        print(f"  - {m[0]} ({m[1]})", flush=True)
    print(f"Output: {OUTPUT_DIR}", flush=True)
    print("=" * 70, flush=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run each model sequentially
    for model_name, model_id, vendor in MODELS_TO_RUN:
        run_model(model_name, model_id, vendor)

    print("\n" + "=" * 70, flush=True)
    print("ALL MODELS COMPLETE", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
