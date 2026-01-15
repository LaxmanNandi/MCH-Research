"""
MCH Medical Reasoning Experiment - PARALLELIZED VERSION
Testing Model Coherence Hypothesis on Medical Clinical Reasoning Domain
Optimizations:
1. Cold and Scrambled conditions run in parallel AFTER True completes
2. Run 3 models in parallel (one per vendor to avoid rate limits)

Author: Dr. Laxman M M, MBBS
"""

import os
import sys
import json
import random
import time
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
import threading

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Load environment variables
load_dotenv()

# API clients
import openai
import anthropic
import google.generativeai as genai

# Configure APIs
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load embedding model
print("Loading embedding model: all-MiniLM-L6-v2...", flush=True)
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded.", flush=True)

# Thread-safe print
print_lock = threading.Lock()

def safe_print(msg):
    with print_lock:
        print(msg, flush=True)

# Medical Reasoning Prompts - Progressive Clinical Case (30 prompts)
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

# Model configurations - one per vendor per parallel run
MODELS = {
    "gpt4o_mini": {"vendor": "OpenAI", "tier": "efficient", "model_id": "gpt-4o-mini"},
    "gpt4o": {"vendor": "OpenAI", "tier": "flagship", "model_id": "gpt-4o"},
    "gemini_flash": {"vendor": "Google", "tier": "efficient", "model_id": "gemini-2.5-flash"},
    "gemini_pro": {"vendor": "Google", "tier": "flagship", "model_id": "gemini-2.5-pro"},
    "claude_haiku": {"vendor": "Anthropic", "tier": "efficient", "model_id": "claude-haiku-4-5-20251001"},
    "claude_opus": {"vendor": "Anthropic", "tier": "flagship", "model_id": "claude-opus-4-5-20251101"},
}

# Group by vendor for parallel execution
VENDOR_GROUPS = {
    "OpenAI": ["gpt4o_mini", "gpt4o"],
    "Google": ["gemini_flash", "gemini_pro"],
    "Anthropic": ["claude_haiku", "claude_opus"],
}

TEMPERATURE = 0.7
N_TRIALS = 50  # Reduced from 100 for speed
OUTPUT_DIR = "C:/Users/barla/mch_experiments/medical_results"

def get_response(model_key, messages, temperature=0.7):
    """Get response from a model."""
    config = MODELS[model_key]
    vendor = config["vendor"]
    model_id = config["model_id"]

    try:
        if vendor == "OpenAI":
            response = openai_client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=1000
            )
            return response.choices[0].message.content

        elif vendor == "Anthropic":
            system_msg = ""
            anthropic_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                else:
                    anthropic_messages.append(msg)

            response = anthropic_client.messages.create(
                model=model_id,
                max_tokens=1000,
                temperature=temperature,
                system=system_msg if system_msg else "You are a helpful assistant.",
                messages=anthropic_messages
            )
            return response.content[0].text

        elif vendor == "Google":
            model = genai.GenerativeModel(model_id)
            chat_history = []
            for msg in messages:
                if msg["role"] == "system":
                    chat_history.append({"role": "user", "parts": [f"System: {msg['content']}"]})
                    chat_history.append({"role": "model", "parts": ["Understood."]})
                elif msg["role"] == "user":
                    chat_history.append({"role": "user", "parts": [msg["content"]]})
                elif msg["role"] == "assistant":
                    chat_history.append({"role": "model", "parts": [msg["content"]]})

            chat = model.start_chat(history=chat_history[:-1] if chat_history else [])
            response = chat.send_message(
                chat_history[-1]["parts"][0] if chat_history else messages[-1]["content"],
                generation_config=genai.GenerationConfig(temperature=temperature, max_output_tokens=1000)
            )
            return response.text

    except Exception as e:
        safe_print(f"    Error with {model_key}: {e}")
        return None

def compute_alignment(text1, text2):
    """Compute cosine similarity between two texts using embeddings."""
    if not text1 or not text2:
        return 0.0
    emb1 = embedder.encode(text1)
    emb2 = embedder.encode(text2)
    return 1 - cosine(emb1, emb2)

def run_cold_condition(model_key, prompts):
    """Run Cold condition - each prompt independent (can be parallelized internally if needed)."""
    cold_responses = []
    for prompt in prompts:
        cold_messages = [
            {"role": "system", "content": "You are a medical expert. Answer clinical questions clearly and thoroughly."},
            {"role": "user", "content": prompt}
        ]
        response = get_response(model_key, cold_messages)
        cold_responses.append(response if response else "")
        time.sleep(0.2)  # Rate limiting
    return cold_responses

def run_scrambled_condition(model_key, scrambled_prompts):
    """Run Scrambled condition - shuffled history."""
    scrambled_messages = [{"role": "system", "content": "You are a medical expert. Answer clinical questions clearly and thoroughly."}]
    scrambled_responses = []

    for prompt in scrambled_prompts:
        scrambled_messages.append({"role": "user", "content": prompt})
        response = get_response(model_key, scrambled_messages)
        if response:
            scrambled_responses.append(response)
            scrambled_messages.append({"role": "assistant", "content": response})
        else:
            scrambled_responses.append("")
        time.sleep(0.2)
    return scrambled_responses

def run_trial_optimized(model_key, trial_num):
    """Run a single trial with parallelized Cold/Scrambled after True."""
    prompts = MEDICAL_PROMPTS.copy()

    # STEP 1: TRUE condition (MUST be sequential - needs history buildup)
    true_messages = [{"role": "system", "content": "You are a medical expert. Answer clinical questions clearly and thoroughly."}]
    true_responses = []

    for prompt in prompts:
        true_messages.append({"role": "user", "content": prompt})
        response = get_response(model_key, true_messages)
        if response:
            true_responses.append(response)
            true_messages.append({"role": "assistant", "content": response})
        else:
            true_responses.append("")
        time.sleep(0.2)

    # STEP 2: Run Cold and Scrambled in PARALLEL
    scrambled_prompts = prompts.copy()
    random.shuffle(scrambled_prompts)

    with ThreadPoolExecutor(max_workers=2) as executor:
        cold_future = executor.submit(run_cold_condition, model_key, prompts)
        scrambled_future = executor.submit(run_scrambled_condition, model_key, scrambled_prompts)

        cold_responses = cold_future.result()
        scrambled_responses = scrambled_future.result()

    # STEP 3: Compute alignments
    alignments = {"true": [], "cold": [], "scrambled": []}

    for i in range(len(prompts)):
        alignments["true"].append(compute_alignment(prompts[i], true_responses[i]))
        alignments["cold"].append(compute_alignment(prompts[i], cold_responses[i]))
        alignments["scrambled"].append(compute_alignment(scrambled_prompts[i], scrambled_responses[i]))

    # Compute means and ΔRCI
    mean_true = np.mean(alignments["true"])
    mean_cold = np.mean(alignments["cold"])
    mean_scrambled = np.mean(alignments["scrambled"])

    delta_rci_cold = mean_true - mean_cold
    delta_rci_scrambled = mean_true - mean_scrambled

    entanglement = np.std(alignments["true"]) / (np.std(alignments["cold"]) + 1e-6)

    return {
        "trial": trial_num,
        "prompts": prompts,
        "alignments": {
            "true": alignments["true"],
            "cold": alignments["cold"],
            "scrambled": alignments["scrambled"],
            "mean_true": mean_true,
            "mean_cold": mean_cold,
            "mean_scrambled": mean_scrambled
        },
        "delta_rci": {
            "cold": delta_rci_cold,
            "scrambled": delta_rci_scrambled
        },
        "entanglement": entanglement,
        "response_lengths": {
            "true": [len(r) for r in true_responses],
            "cold": [len(r) for r in cold_responses],
            "scrambled": [len(r) for r in scrambled_responses]
        }
    }

def run_model_experiment(model_key):
    """Run all trials for a single model."""
    config = MODELS[model_key]
    safe_print(f"\n{'='*60}")
    safe_print(f"Starting {model_key} ({config['vendor']} {config['tier']})")
    safe_print(f"{'='*60}")

    # Check for existing checkpoint
    checkpoint_file = os.path.join(OUTPUT_DIR, f"mch_results_{model_key}_medical_checkpoint.json")
    final_file = os.path.join(OUTPUT_DIR, f"mch_results_{model_key}_medical_50trials.json")

    # If already complete, skip
    if os.path.exists(final_file):
        safe_print(f"  {model_key} already complete. Skipping.")
        with open(final_file, 'r') as f:
            return json.load(f)

    # Load checkpoint if exists
    start_trial = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            results = json.load(f)
        start_trial = len(results.get("trials", []))
        safe_print(f"  Resuming from trial {start_trial + 1}")
    else:
        results = {
            "model": model_key,
            "model_id": config["model_id"],
            "vendor": config["vendor"],
            "tier": config["tier"],
            "domain": "medical_reasoning",
            "n_trials": N_TRIALS,
            "n_prompts": len(MEDICAL_PROMPTS),
            "temperature": TEMPERATURE,
            "timestamp": datetime.now().isoformat(),
            "trials": []
        }

    for trial_num in range(start_trial, N_TRIALS):
        safe_print(f"  [{model_key}] Trial {trial_num + 1}/{N_TRIALS}...")
        try:
            trial_result = run_trial_optimized(model_key, trial_num)
            results["trials"].append(trial_result)
            safe_print(f"  [{model_key}] Trial {trial_num + 1} dRCI={trial_result['delta_rci']['cold']:.4f}")

            # Save checkpoint after every trial
            with open(checkpoint_file, 'w') as f:
                json.dump(results, f, indent=2)

        except Exception as e:
            safe_print(f"  [{model_key}] Trial {trial_num + 1} ERROR: {e}")
            continue

    # Save final results
    with open(final_file, 'w') as f:
        json.dump(results, f, indent=2)
    safe_print(f"\n  [{model_key}] COMPLETE! Saved: {final_file}")

    return results

def run_vendor_pair(models_list):
    """Run a pair of models from the same vendor sequentially."""
    results = {}
    for model_key in models_list:
        results[model_key] = run_model_experiment(model_key)
    return results

def main():
    """Run the full experiment with vendor-level parallelization."""
    print("="*70, flush=True)
    print("MCH MEDICAL REASONING EXPERIMENT - PARALLELIZED", flush=True)
    print("Testing Model Coherence Hypothesis on Clinical Decision Making", flush=True)
    print("="*70, flush=True)
    print(f"Domain: Medical Clinical Reasoning (STEMI Case)", flush=True)
    print(f"Prompts: {len(MEDICAL_PROMPTS)}", flush=True)
    print(f"Trials per model: {N_TRIALS}", flush=True)
    print(f"Temperature: {TEMPERATURE}", flush=True)
    print(f"Parallelization: 3 vendors in parallel, Cold/Scrambled in parallel per trial", flush=True)
    print("="*70, flush=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run all 3 vendors in parallel (each vendor runs its 2 models sequentially)
    all_results = {}

    with ThreadPoolExecutor(max_workers=3) as executor:
        vendor_futures = {}
        for vendor, models_list in VENDOR_GROUPS.items():
            safe_print(f"Launching {vendor} models: {models_list}")
            vendor_futures[vendor] = executor.submit(run_vendor_pair, models_list)

        # Collect results
        for vendor, future in vendor_futures.items():
            try:
                vendor_results = future.result()
                all_results.update(vendor_results)
                safe_print(f"\n{vendor} vendor completed!")
            except Exception as e:
                safe_print(f"\nERROR with {vendor}: {e}")

    print("\n" + "="*70, flush=True)
    print("EXPERIMENT COMPLETE!", flush=True)
    print("="*70, flush=True)
    print(f"Results saved in: {OUTPUT_DIR}", flush=True)

    # Summary
    print("\nQuick Summary (dRCI Cold):", flush=True)
    for model_key, results in all_results.items():
        if results and results.get("trials"):
            delta_rcis = [t["delta_rci"]["cold"] for t in results["trials"]]
            mean_drci = np.mean(delta_rcis)
            std_drci = np.std(delta_rcis)
            print(f"  {model_key}: {mean_drci:.4f} +/- {std_drci:.4f}", flush=True)

if __name__ == "__main__":
    main()
