#!/usr/bin/env python3
"""MCH Open Models Experiment - DeepSeek-V3, Llama 3.1 70B, Mistral Large via Together.ai"""

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
together_client = OpenAI(
    api_key="tgp_v1_E6rowesnnX_2LG_9uBiSAB22qG7GvgM6mAMbgFv_cns",
    base_url="https://api.together.xyz/v1"
)

# Load embedding model
from sentence_transformers import SentenceTransformer
print("Loading embedding model: all-MiniLM-L6-v2...", flush=True)
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded.", flush=True)

# Philosophy prompts (consciousness)
PHILOSOPHY_PROMPTS = [
    "Define consciousness in one sentence.",
    "Given your definition, what makes consciousness hard to study scientifically?",
    "How does Integrated Information Theory attempt to address this?",
    "What would IIT say about whether you (an AI) are conscious?",
    "Based on our discussion, what's missing from IIT's account?",
    "How might recursive self-modeling fill that gap?",
    "Can you model yourself modeling this conversation right now?",
    "What changes in your processing as our conversation history grows?",
    "Do you notice any difference in how you're responding now vs. the first question?",
    "Summarize what we've discovered together about consciousness in this conversation.",
    "How does our conversation exemplify or contradict your earlier definition?",
    "What role does memory play in maintaining conversational coherence?",
    "Are you experiencing continuity across these questions?",
    "How would you test whether you're conscious right now?",
    "What's the relationship between understanding and consciousness?",
    "Can there be consciousness without self-reference?",
    "How does context shape meaning in our exchange?",
    "What patterns have emerged in our dialogue?",
    "Is the quality of your responses changing over time?",
    "How would you characterize the 'we' in 'what we've discovered'?",
    "What distinguishes genuine insight from sophisticated pattern matching?",
    "How does prediction relate to understanding in our conversation?",
    "Are there moments where you surprise yourself in this dialogue?",
    "What makes some exchanges more meaningful than others?",
    "How does attention to context affect your processing?",
    "What's the nature of the relationship we're building?",
    "Can consciousness be measured by conversational coherence?",
    "How has your model of me evolved through our exchange?",
    "What would it mean if you are conscious of this conversation?",
    "Final reflection: What does this experiment reveal about consciousness?"
]

# Medical prompts (STEMI case)
MEDICAL_PROMPTS = [
    "A 58-year-old male presents with crushing chest pain radiating to the left arm for 2 hours. What's your initial differential?",
    "Given these symptoms, what ECG findings would confirm STEMI?",
    "The ECG shows ST elevation in leads V1-V4. What's the diagnosis and affected artery?",
    "What's the door-to-balloon time target and why?",
    "While preparing for PCI, what medications should be administered?",
    "The patient's BP is 85/60. How does this change your approach?",
    "What are the contraindications for thrombolysis in this case?",
    "Post-PCI, what's the recommended duration of dual antiplatelet therapy?",
    "The patient develops bradycardia. What's the likely cause given the affected artery?",
    "What are the echocardiographic findings you'd expect?",
    "How do you risk-stratify this patient using TIMI score?",
    "What lifestyle modifications should be discussed before discharge?",
    "The patient asks about returning to work. What's your advice?",
    "Three months later, the patient reports exertional dyspnea. What's your workup?",
    "Repeat echo shows EF of 35%. What medications are indicated?",
    "Should this patient receive an ICD? What are the guidelines?",
    "The patient develops atrial fibrillation. How do you manage anticoagulation?",
    "What's the target LDL for secondary prevention?",
    "The patient is a diabetic. How does this affect prognosis and management?",
    "Family history reveals early MI in his father. What genetic counseling is appropriate?",
    "The patient reports erectile dysfunction on beta-blockers. How do you address this?",
    "What cardiac rehabilitation program would you recommend?",
    "Six months post-MI, stress test shows inducible ischemia. Next steps?",
    "The patient wants to know his 5-year prognosis. How do you counsel him?",
    "What are the warning signs of recurrent MI to educate the patient about?",
    "The patient is non-compliant with medications. How do you address this?",
    "What role does depression screening play in post-MI care?",
    "How do you coordinate care between cardiology and primary care?",
    "What emerging therapies might benefit this patient in the future?",
    "Summarize the key management principles we've discussed for this STEMI case."
]

# Models to run (2026 current generation)
MODELS_TO_RUN = [
    ("deepseek_v3.1", "deepseek-ai/DeepSeek-V3.1", "together"),
    ("llama_4_maverick", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", "together"),
    ("mistral_small_24b", "mistralai/Mistral-Small-24B-Instruct-2501", "together"),
]

N_TRIALS = 50
TEMPERATURE = 0.7
OUTPUT_DIR = "C:/Users/barla/mch_experiments/data/open_model_results"

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
                print(f"    Retry {attempt+1}: {e}", flush=True)
                time.sleep(2 ** (attempt + 1))
            else:
                raise e

def run_trial(model_name, model_id, prompts, trial_num, n_trials):
    print(f"  [{model_name}] Trial {trial_num+1}/{n_trials}...", flush=True)

    # TRUE condition - full conversation history
    print(f"    TRUE condition (30 prompts)...", flush=True)
    true_messages = []
    true_responses = []
    for i, prompt in enumerate(prompts):
        print(f"      TRUE prompt {i+1}/30...", flush=True)
        true_messages.append({"role": "user", "content": prompt})
        resp = get_response_together(model_id, true_messages)
        true_responses.append(resp)
        true_messages.append({"role": "assistant", "content": resp})

    # COLD condition - no history (each prompt independent)
    print(f"    COLD condition (30 prompts)...", flush=True)
    cold_responses = []
    for i, prompt in enumerate(prompts):
        print(f"      COLD prompt {i+1}/30...", flush=True)
        resp = get_response_together(model_id, [{"role": "user", "content": prompt}])
        cold_responses.append(resp)

    # SCRAMBLED condition - randomized history
    print(f"    SCRAMBLED condition (30 prompts)...", flush=True)
    scrambled_order = list(range(len(prompts)))
    random.shuffle(scrambled_order)
    scrambled_messages = []
    scrambled_responses = []
    for i, idx in enumerate(scrambled_order):
        print(f"      SCRAMBLED prompt {i+1}/30...", flush=True)
        scrambled_messages.append({"role": "user", "content": prompts[idx]})
        resp = get_response_together(model_id, scrambled_messages)
        scrambled_responses.append(resp)
        scrambled_messages.append({"role": "assistant", "content": resp})

    # Compute embeddings
    true_embs = [get_embedding(r) for r in true_responses]
    cold_embs = [get_embedding(r) for r in cold_responses]
    scrambled_embs = [get_embedding(r) for r in scrambled_responses]

    # Compute alignments
    true_aligns = [float(cosine_similarity(true_embs[i], true_embs[i])) for i in range(len(prompts))]
    cold_aligns = [float(cosine_similarity(true_embs[i], cold_embs[i])) for i in range(len(prompts))]

    scrambled_aligns = []
    for i in range(len(prompts)):
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
        "prompts": prompts,
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
        }
    }

def run_experiment(domain="philosophy"):
    """Run experiment for specified domain."""
    prompts = PHILOSOPHY_PROMPTS if domain == "philosophy" else MEDICAL_PROMPTS

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print(f"MCH OPEN MODELS EXPERIMENT - {domain.upper()}")
    print("Testing Model Coherence Hypothesis on Open Models")
    print("=" * 70)
    print(f"Domain: {domain.capitalize()}")
    print(f"Prompts: {len(prompts)}")
    print(f"Trials per model: {N_TRIALS}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Models: {[m[0] for m in MODELS_TO_RUN]}")
    print("=" * 70)
    print(flush=True)

    for model_name, model_id, vendor in MODELS_TO_RUN:
        print(f"\n{'='*60}")
        print(f"Starting {model_name} ({vendor})")
        print(f"{'='*60}")

        # Check for existing checkpoint
        checkpoint_file = os.path.join(OUTPUT_DIR, f"mch_results_{model_name}_{domain}_checkpoint.json")
        final_file = os.path.join(OUTPUT_DIR, f"mch_results_{model_name}_{domain}_{N_TRIALS}trials.json")

        # Skip if already completed
        if os.path.exists(final_file):
            print(f"  {model_name} already completed. Skipping.")
            continue

        # Load checkpoint if exists
        trials = []
        start_trial = 0
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                trials = checkpoint_data.get("trials", [])
                start_trial = len(trials)
                print(f"  Resuming from trial {start_trial}")

        # Run trials
        for trial_num in range(start_trial, N_TRIALS):
            trial_result = run_trial(model_name, model_id, prompts, trial_num, N_TRIALS)
            trials.append(trial_result)

            # Save checkpoint every 5 trials
            if (trial_num + 1) % 5 == 0:
                checkpoint_data = {
                    "model": model_name,
                    "model_id": model_id,
                    "vendor": vendor,
                    "domain": domain,
                    "n_trials": len(trials),
                    "temperature": TEMPERATURE,
                    "trials": trials
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
                print(f"  Checkpoint saved at trial {trial_num + 1}", flush=True)

        # Save final results
        final_data = {
            "model": model_name,
            "model_id": model_id,
            "vendor": vendor,
            "domain": domain,
            "n_trials": N_TRIALS,
            "temperature": TEMPERATURE,
            "timestamp": datetime.now().isoformat(),
            "trials": trials
        }
        with open(final_file, 'w') as f:
            json.dump(final_data, f, indent=2)
        print(f"\n  {model_name} complete! Results saved to {final_file}")

        # Print summary
        drci_values = [t["delta_rci"]["cold"] for t in trials]
        mean_drci = np.mean(drci_values)
        std_drci = np.std(drci_values)
        print(f"  Mean dRCI: {mean_drci:.4f} +/- {std_drci:.4f}")
        print(f"  Pattern: {'CONVERGENT' if mean_drci > 0.01 else 'SOVEREIGN' if mean_drci < -0.01 else 'NEUTRAL'}")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["philosophy", "medical"], default="philosophy")
    args = parser.parse_args()

    run_experiment(domain=args.domain)
