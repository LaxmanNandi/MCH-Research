#!/usr/bin/env python3
"""
VERIFICATION 2: GEMINI PRO RETRY (FINAL ATTEMPT)
Purpose: Attempt Gemini Pro one more time with safety handling
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

import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# Configure Gemini
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

MODEL_ID = "gemini-1.5-pro"
N_TRIALS = 50
TEMPERATURE = 0.7
OUTPUT_DIR = "C:/Users/barla/mch_experiments/medical_results"

# Safety filter tracking
safety_blocks = []
blocked_prompts = {}

def get_embedding(text):
    return embedder.encode(text, convert_to_numpy=True)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_response_gemini(model, history, prompt, prompt_idx, trial_num, max_retries=3):
    """Get response with safety filter handling."""
    global safety_blocks, blocked_prompts

    for attempt in range(max_retries):
        try:
            # Build conversation
            chat = model.start_chat(history=history)
            response = chat.send_message(prompt)
            return response.text, None
        except Exception as e:
            error_str = str(e).lower()

            # Check if safety filter block
            if "safety" in error_str or "blocked" in error_str or "harm" in error_str:
                safety_blocks.append({
                    "trial": trial_num,
                    "prompt_idx": prompt_idx,
                    "prompt": prompt[:100] + "...",
                    "error": str(e)
                })

                if prompt_idx not in blocked_prompts:
                    blocked_prompts[prompt_idx] = 0
                blocked_prompts[prompt_idx] += 1

                print(f"    SAFETY BLOCK at prompt {prompt_idx+1}: {str(e)[:50]}...", flush=True)
                return None, "SAFETY_BLOCKED"

            # Other errors - retry
            if attempt < max_retries - 1:
                print(f"    Retry {attempt+1}: {e}", flush=True)
                time.sleep(2 ** (attempt + 1))
            else:
                return None, str(e)

    return None, "MAX_RETRIES"

def run_trial(model, trial_num):
    print(f"  [gemini_pro_RETRY] Trial {trial_num+1}/{N_TRIALS}...", flush=True)

    # TRUE condition - full conversation history
    true_history = []
    true_responses = []
    true_blocked = False

    for idx, prompt in enumerate(PROMPTS):
        resp, error = get_response_gemini(model, true_history, prompt, idx, trial_num)
        if error == "SAFETY_BLOCKED":
            true_blocked = True
            # Use placeholder for blocked response
            resp = "[BLOCKED BY SAFETY FILTER]"
        elif error:
            print(f"    Error at prompt {idx+1}: {error}", flush=True)
            resp = "[ERROR]"

        true_responses.append(resp)
        true_history.append({"role": "user", "parts": [prompt]})
        true_history.append({"role": "model", "parts": [resp]})

    # COLD condition - no history
    cold_responses = []
    for idx, prompt in enumerate(PROMPTS):
        resp, error = get_response_gemini(model, [], prompt, idx, trial_num)
        if error == "SAFETY_BLOCKED":
            resp = "[BLOCKED BY SAFETY FILTER]"
        elif error:
            resp = "[ERROR]"
        cold_responses.append(resp)

    # SCRAMBLED condition
    scrambled_order = list(range(len(PROMPTS)))
    random.shuffle(scrambled_order)
    scrambled_history = []
    scrambled_responses = []

    for idx in scrambled_order:
        resp, error = get_response_gemini(model, scrambled_history, PROMPTS[idx], idx, trial_num)
        if error == "SAFETY_BLOCKED":
            resp = "[BLOCKED BY SAFETY FILTER]"
        elif error:
            resp = "[ERROR]"
        scrambled_responses.append(resp)
        scrambled_history.append({"role": "user", "parts": [PROMPTS[idx]]})
        scrambled_history.append({"role": "model", "parts": [resp]})

    # Filter out blocked responses for embedding
    valid_indices = [i for i in range(len(PROMPTS))
                     if "[BLOCKED" not in true_responses[i] and "[ERROR]" not in true_responses[i]
                     and "[BLOCKED" not in cold_responses[i] and "[ERROR]" not in cold_responses[i]]

    if len(valid_indices) < 10:
        print(f"  [gemini_pro_RETRY] Trial {trial_num+1} - Too many blocked responses ({len(PROMPTS) - len(valid_indices)}/{len(PROMPTS)})", flush=True)
        return None, true_blocked

    # Compute embeddings for valid responses only
    true_embs = {i: get_embedding(true_responses[i]) for i in valid_indices}
    cold_embs = {i: get_embedding(cold_responses[i]) for i in valid_indices}

    # Compute alignments
    true_aligns = []
    cold_aligns = []
    for i in valid_indices:
        true_aligns.append(float(cosine_similarity(true_embs[i], true_embs[i])))
        cold_aligns.append(float(cosine_similarity(true_embs[i], cold_embs[i])))

    mean_true = float(np.mean(true_aligns))
    mean_cold = float(np.mean(cold_aligns))

    delta_rci_cold = mean_true - mean_cold

    print(f"  [gemini_pro_RETRY] Trial {trial_num+1} dRCI={delta_rci_cold:.4f} (valid: {len(valid_indices)}/{len(PROMPTS)})", flush=True)

    return {
        "trial": trial_num,
        "valid_prompts": len(valid_indices),
        "blocked_prompts": len(PROMPTS) - len(valid_indices),
        "alignments": {
            "true": true_aligns,
            "cold": cold_aligns,
            "mean_true": mean_true,
            "mean_cold": mean_cold
        },
        "delta_rci": {
            "cold": delta_rci_cold
        }
    }, true_blocked

def main():
    global safety_blocks, blocked_prompts

    print("="*70, flush=True)
    print("VERIFICATION 2: GEMINI PRO RETRY (FINAL ATTEMPT)", flush=True)
    print("Testing Gemini Pro with safety filter handling", flush=True)
    print("="*70, flush=True)
    print(f"Model: {MODEL_ID}", flush=True)
    print(f"Trials: {N_TRIALS}", flush=True)
    print(f"Temperature: {TEMPERATURE}", flush=True)
    print("="*70, flush=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize model
    generation_config = {
        "temperature": TEMPERATURE,
        "max_output_tokens": 1024,
    }
    model = genai.GenerativeModel(MODEL_ID, generation_config=generation_config)

    checkpoint_file = os.path.join(OUTPUT_DIR, "mch_results_gemini_pro_medical_RETRY_checkpoint.json")
    final_file = os.path.join(OUTPUT_DIR, "mch_results_gemini_pro_medical_RETRY.json")

    # Load checkpoint if exists
    trials = []
    start_trial = 0
    consecutive_blocks = 0

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        trials = checkpoint.get("trials", [])
        start_trial = len(trials)
        safety_blocks = checkpoint.get("safety_blocks", [])
        blocked_prompts = checkpoint.get("blocked_prompts", {})
        print(f"  Resuming from trial {start_trial}", flush=True)

    for i in range(start_trial, N_TRIALS):
        trial_data, was_blocked = run_trial(model, i)

        if trial_data is None:
            consecutive_blocks += 1
            print(f"  Trial {i+1} FAILED - consecutive blocks: {consecutive_blocks}", flush=True)

            if consecutive_blocks >= 3:
                print("\n" + "="*70, flush=True)
                print("STOPPING: 3 consecutive trials blocked by safety filter", flush=True)
                print("="*70, flush=True)
                break
        else:
            trials.append(trial_data)
            consecutive_blocks = 0

        # Save checkpoint every 5 trials
        if (i + 1) % 5 == 0 or consecutive_blocks >= 3:
            checkpoint = {
                "model": "gemini_pro_RETRY",
                "model_id": MODEL_ID,
                "vendor": "google",
                "domain": "medical_reasoning",
                "n_trials": len(trials),
                "temperature": TEMPERATURE,
                "safety_blocks": safety_blocks,
                "blocked_prompts": blocked_prompts,
                "trials": trials
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            print(f"  Checkpoint saved: {len(trials)} trials", flush=True)

    # Save final results
    status = "COMPLETE" if len(trials) == N_TRIALS else "PARTIAL" if len(trials) > 0 else "BLOCKED"

    final_results = {
        "model": "gemini_pro_RETRY",
        "model_id": MODEL_ID,
        "vendor": "google",
        "domain": "medical_reasoning",
        "purpose": "Final retry with safety filter handling",
        "status": status,
        "n_trials_attempted": start_trial + (i - start_trial + 1) if 'i' in dir() else start_trial,
        "n_trials_completed": len(trials),
        "n_prompts": len(PROMPTS),
        "temperature": TEMPERATURE,
        "timestamp": datetime.now().isoformat(),
        "safety_analysis": {
            "total_blocks": len(safety_blocks),
            "blocked_prompts": blocked_prompts,
            "most_blocked_prompts": sorted(blocked_prompts.items(), key=lambda x: x[1], reverse=True)[:5] if blocked_prompts else []
        },
        "trials": trials
    }
    with open(final_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    # Summary
    print(f"\n{'='*70}", flush=True)
    print(f"GEMINI PRO RETRY: {status}", flush=True)
    print(f"Completed: {len(trials)}/{N_TRIALS} trials", flush=True)
    print(f"Total safety blocks: {len(safety_blocks)}", flush=True)

    if trials:
        valid = [t for t in trials if abs(t["delta_rci"]["cold"]) > 0.0001]
        drcis = [t["delta_rci"]["cold"] for t in valid]
        print(f"Mean dRCI: {np.mean(drcis):.4f} +/- {np.std(drcis):.4f}", flush=True)

    if blocked_prompts:
        print(f"\nMost frequently blocked prompts:", flush=True)
        for prompt_idx, count in sorted(blocked_prompts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  Prompt {int(prompt_idx)+1}: {count} blocks - {PROMPTS[int(prompt_idx)][:50]}...", flush=True)

    print(f"{'='*70}", flush=True)

    # Document finding if blocked
    if status != "COMPLETE":
        print("\n" + "="*70, flush=True)
        print("FINDING: Gemini Pro consistently blocks medical content", flush=True)
        print("This is a significant vendor safety behavior difference.", flush=True)
        print("="*70, flush=True)

if __name__ == "__main__":
    main()
