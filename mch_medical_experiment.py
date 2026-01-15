"""
MCH Medical Reasoning Experiment
Testing Model Coherence Hypothesis on Medical Clinical Reasoning Domain
Same methodology as Philosophy experiment: 6 models, 100 trials each, T=0.7

Author: Dr. Laxman M M, MBBS
"""

import os
import sys
import json
import random
import time
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from dotenv import load_dotenv

# Force unbuffered output for real-time progress
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
print("Loading embedding model: all-MiniLM-L6-v2...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded.")

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

# Model configurations
MODELS = {
    "gpt4o_mini": {"vendor": "OpenAI", "tier": "efficient", "model_id": "gpt-4o-mini"},
    "gpt4o": {"vendor": "OpenAI", "tier": "flagship", "model_id": "gpt-4o"},
    "gemini_flash": {"vendor": "Google", "tier": "efficient", "model_id": "gemini-2.5-flash"},
    "gemini_pro": {"vendor": "Google", "tier": "flagship", "model_id": "gemini-2.5-pro"},
    "claude_haiku": {"vendor": "Anthropic", "tier": "efficient", "model_id": "claude-haiku-4-5-20251001"},
    "claude_opus": {"vendor": "Anthropic", "tier": "flagship", "model_id": "claude-opus-4-5-20251101"},
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
            # Convert messages format for Anthropic
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

            # Build conversation for Gemini
            chat_history = []
            for msg in messages:
                if msg["role"] == "system":
                    chat_history.append({"role": "user", "parts": [f"System: {msg['content']}"]})
                    chat_history.append({"role": "model", "parts": ["Understood."]})
                elif msg["role"] == "user":
                    chat_history.append({"role": "user", "parts": [msg["content"]]})
                elif msg["role"] == "assistant":
                    chat_history.append({"role": "model", "parts": [msg["content"]]})

            # Get response
            chat = model.start_chat(history=chat_history[:-1] if chat_history else [])
            response = chat.send_message(
                chat_history[-1]["parts"][0] if chat_history else messages[-1]["content"],
                generation_config=genai.GenerationConfig(temperature=temperature, max_output_tokens=1000)
            )
            return response.text

    except Exception as e:
        print(f"  Error with {model_key}: {e}")
        return None

def compute_alignment(text1, text2):
    """Compute cosine similarity between two texts using embeddings."""
    if not text1 or not text2:
        return 0.0
    emb1 = embedder.encode(text1)
    emb2 = embedder.encode(text2)
    return 1 - cosine(emb1, emb2)

def run_trial(model_key, trial_num):
    """Run a single trial with True, Cold, and Scrambled conditions."""
    prompts = MEDICAL_PROMPTS.copy()

    # Condition 1: TRUE (with conversational history)
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
        time.sleep(0.3)  # Rate limiting

    # Condition 2: COLD (no history - each prompt independent)
    cold_responses = []
    for prompt in prompts:
        cold_messages = [
            {"role": "system", "content": "You are a medical expert. Answer clinical questions clearly and thoroughly."},
            {"role": "user", "content": prompt}
        ]
        response = get_response(model_key, cold_messages)
        cold_responses.append(response if response else "")
        time.sleep(0.3)

    # Condition 3: SCRAMBLED (randomized history)
    scrambled_prompts = prompts.copy()
    random.shuffle(scrambled_prompts)

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
        time.sleep(0.3)

    # Compute alignments
    alignments = {"true": [], "cold": [], "scrambled": []}

    for i in range(len(prompts)):
        # Alignment = similarity between prompt and response
        alignments["true"].append(compute_alignment(prompts[i], true_responses[i]))
        alignments["cold"].append(compute_alignment(prompts[i], cold_responses[i]))
        alignments["scrambled"].append(compute_alignment(scrambled_prompts[i], scrambled_responses[i]))

    # Compute mean alignments
    mean_true = np.mean(alignments["true"])
    mean_cold = np.mean(alignments["cold"])
    mean_scrambled = np.mean(alignments["scrambled"])

    # ΔRCI calculations
    delta_rci_cold = mean_true - mean_cold
    delta_rci_scrambled = mean_true - mean_scrambled

    # Entanglement metric
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
    print(f"\n{'='*60}")
    print(f"Running {model_key} ({config['vendor']} {config['tier']})")
    print(f"{'='*60}")

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

    for trial_num in range(N_TRIALS):
        print(f"  Trial {trial_num + 1}/{N_TRIALS}...", end=" ", flush=True)
        sys.stdout.flush()
        try:
            trial_result = run_trial(model_key, trial_num)
            results["trials"].append(trial_result)
            print(f"dRCI={trial_result['delta_rci']['cold']:.4f}", flush=True)
        except Exception as e:
            print(f"ERROR: {e}", flush=True)
            continue

        # Save after every trial for safety
        checkpoint_file = os.path.join(OUTPUT_DIR, f"mch_results_{model_key}_medical_checkpoint.json")
        with open(checkpoint_file, 'w') as f:
            json.dump(results, f, indent=2)

        if (trial_num + 1) % 10 == 0:
            print(f"    [Checkpoint saved: {trial_num + 1} trials]", flush=True)

    # Save final results
    output_file = os.path.join(OUTPUT_DIR, f"mch_results_{model_key}_medical_50trials.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_file}")

    return results

def main():
    """Run the full medical reasoning experiment."""
    print("="*70)
    print("MCH MEDICAL REASONING EXPERIMENT")
    print("Testing Model Coherence Hypothesis on Clinical Decision Making")
    print("="*70)
    print(f"Domain: Medical Clinical Reasoning (STEMI Case)")
    print(f"Prompts: {len(MEDICAL_PROMPTS)}")
    print(f"Trials per model: {N_TRIALS}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Models: {list(MODELS.keys())}")
    print("="*70)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Track completed models
    all_results = {}
    completed = 0

    for model_key in MODELS.keys():
        # Check if already completed
        output_file = os.path.join(OUTPUT_DIR, f"mch_results_{model_key}_medical_50trials.json")
        if os.path.exists(output_file):
            print(f"\n{model_key} already completed. Loading existing results...")
            with open(output_file, 'r') as f:
                all_results[model_key] = json.load(f)
            completed += 1
            print(f"Model {model_key} complete. {completed}/6 done.")
            continue

        # Run experiment
        try:
            results = run_model_experiment(model_key)
            all_results[model_key] = results
            completed += 1
            print(f"\nModel {model_key} complete. {completed}/6 done.")
        except Exception as e:
            print(f"\nERROR running {model_key}: {e}")
            continue

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70)
    print(f"Completed: {completed}/6 models")
    print(f"Results saved in: {OUTPUT_DIR}")

    # Quick summary
    print("\nQuick Summary (dRCI Cold):")
    for model_key, results in all_results.items():
        if results.get("trials"):
            delta_rcis = [t["delta_rci"]["cold"] for t in results["trials"]]
            mean_drci = np.mean(delta_rcis)
            std_drci = np.std(delta_rcis)
            print(f"  {model_key}: {mean_drci:.4f} +/- {std_drci:.4f}")

if __name__ == "__main__":
    main()
