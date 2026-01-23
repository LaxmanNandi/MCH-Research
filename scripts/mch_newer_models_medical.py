#!/usr/bin/env python3
"""MCH Medical Experiment - Newer Models (2025-2026).

Supports:
- OpenAI: GPT-5.2, GPT-4.5, o3-mini
- Google: Gemini 3 Pro, Gemini 3 Flash
- Anthropic: Claude 4 Opus, Claude 4 Sonnet
- Open models via Together.ai: Llama 4, Mistral Large, DeepSeek-V3
"""

import os
import sys
import json
import time
import random
import argparse
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

load_dotenv()

import openai
import anthropic
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# API clients
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Together.ai for open models
together_client = None
if os.getenv("TOGETHER_API_KEY"):
    together_client = openai.OpenAI(
        api_key=os.getenv("TOGETHER_API_KEY"),
        base_url="https://api.together.xyz/v1"
    )

# Load embedding model
print("Loading embedding model: all-MiniLM-L6-v2...", flush=True)
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded.", flush=True)

# Model configurations
# NOTE: Model IDs may need updating based on current API availability
MODELS = {
    # OpenAI models (2025-2026)
    "gpt_5_2": {"provider": "openai", "model_id": "gpt-5.2"},
    "gpt_4o": {"provider": "openai", "model_id": "gpt-4o"},
    "gpt_4o_mini": {"provider": "openai", "model_id": "gpt-4o-mini"},
    "gpt_4_turbo": {"provider": "openai", "model_id": "gpt-4-turbo"},
    "o1": {"provider": "openai", "model_id": "o1"},
    "o1_mini": {"provider": "openai", "model_id": "o1-mini"},
    "o3_mini": {"provider": "openai", "model_id": "o3-mini"},

    # Google models
    "gemini_2_flash": {"provider": "google", "model_id": "gemini-2.0-flash"},
    "gemini_2_pro": {"provider": "google", "model_id": "gemini-2.0-pro"},
    "gemini_1_5_pro": {"provider": "google", "model_id": "gemini-1.5-pro"},

    # Anthropic models
    "claude_3_5_sonnet": {"provider": "anthropic", "model_id": "claude-3-5-sonnet-20241022"},
    "claude_3_opus": {"provider": "anthropic", "model_id": "claude-3-opus-20240229"},
    "claude_3_5_haiku": {"provider": "anthropic", "model_id": "claude-3-5-haiku-20241022"},

    # Open models via Together.ai
    "llama_3_3_70b": {"provider": "together", "model_id": "meta-llama/Llama-3.3-70B-Instruct-Turbo"},
    "llama_3_1_405b": {"provider": "together", "model_id": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"},
    "mistral_large": {"provider": "together", "model_id": "mistralai/Mistral-Large-Instruct-2407"},
    "deepseek_v3": {"provider": "together", "model_id": "deepseek-ai/DeepSeek-V3"},
    "qwen_72b": {"provider": "together", "model_id": "Qwen/Qwen2.5-72B-Instruct-Turbo"},
}

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

TEMPERATURE = 0.7
OUTPUT_DIR = "C:/Users/barla/mch_experiments/medical_results"

def get_embedding(text):
    return embedder.encode(text, convert_to_numpy=True)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_response(provider, model_id, messages, max_retries=5):
    """Get response from any supported provider."""
    for attempt in range(max_retries):
        try:
            if provider == "openai":
                response = openai_client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=1024
                )
                return response.choices[0].message.content

            elif provider == "anthropic":
                # Convert messages format for Anthropic
                system_msg = None
                conv_messages = []
                for msg in messages:
                    if msg["role"] == "system":
                        system_msg = msg["content"]
                    else:
                        conv_messages.append(msg)

                kwargs = {
                    "model": model_id,
                    "messages": conv_messages,
                    "temperature": TEMPERATURE,
                    "max_tokens": 1024
                }
                if system_msg:
                    kwargs["system"] = system_msg

                response = anthropic_client.messages.create(**kwargs)
                return response.content[0].text

            elif provider == "google":
                model = genai.GenerativeModel(model_id)
                # Convert messages to Gemini format
                chat = model.start_chat(history=[])
                for msg in messages[:-1]:
                    if msg["role"] == "user":
                        chat.send_message(msg["content"])
                    # Assistant messages are automatically part of history
                response = chat.send_message(messages[-1]["content"])
                return response.text

            elif provider == "together":
                if together_client is None:
                    raise ValueError("Together.ai API key not configured")
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
                print(f"    Retry {attempt+1}: {type(e).__name__}: {str(e)[:50]}", flush=True)
                time.sleep(wait_time)
            else:
                raise e

def run_trial(model_name, provider, model_id, trial_num, n_trials):
    print(f"  [{model_name}] Trial {trial_num+1}/{n_trials}...", flush=True)

    # TRUE condition - full conversation history
    true_messages = []
    true_responses = []
    for prompt in PROMPTS:
        true_messages.append({"role": "user", "content": prompt})
        resp = get_response(provider, model_id, true_messages)
        true_responses.append(resp)
        true_messages.append({"role": "assistant", "content": resp})

    # COLD condition - no history (each prompt independent)
    cold_responses = []
    for prompt in PROMPTS:
        resp = get_response(provider, model_id, [{"role": "user", "content": prompt}])
        cold_responses.append(resp)

    # SCRAMBLED condition - randomized history
    scrambled_order = list(range(len(PROMPTS)))
    random.shuffle(scrambled_order)
    scrambled_messages = []
    scrambled_responses = []
    for idx in scrambled_order:
        scrambled_messages.append({"role": "user", "content": PROMPTS[idx]})
        resp = get_response(provider, model_id, scrambled_messages)
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

def run_model(model_name, n_trials=50):
    """Run experiment for a single model."""
    if model_name not in MODELS:
        print(f"Unknown model: {model_name}")
        print(f"Available models: {list(MODELS.keys())}")
        return

    config = MODELS[model_name]
    provider = config["provider"]
    model_id = config["model_id"]

    print("="*70, flush=True)
    print(f"MCH MEDICAL REASONING EXPERIMENT - {model_name.upper()}", flush=True)
    print("="*70, flush=True)
    print(f"Provider: {provider}", flush=True)
    print(f"Model ID: {model_id}", flush=True)
    print(f"Domain: Medical Clinical Reasoning (STEMI Case)", flush=True)
    print(f"Prompts: {len(PROMPTS)}", flush=True)
    print(f"Trials: {n_trials}", flush=True)
    print(f"Temperature: {TEMPERATURE}", flush=True)
    print("="*70, flush=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    checkpoint_file = os.path.join(OUTPUT_DIR, f"mch_results_{model_name}_medical_checkpoint.json")
    final_file = os.path.join(OUTPUT_DIR, f"mch_results_{model_name}_medical_{n_trials}trials.json")

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
    for i in range(start_trial, n_trials):
        trial_data = run_trial(model_name, provider, model_id, i, n_trials)
        trials.append(trial_data)

        # Save checkpoint every 5 trials
        if (i + 1) % 5 == 0:
            checkpoint = {
                "model": model_name,
                "model_id": model_id,
                "provider": provider,
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
        "provider": provider,
        "domain": "medical_reasoning",
        "n_trials": n_trials,
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

    print(f"\n  {model_name} COMPLETE: {len(valid)}/{n_trials} valid trials", flush=True)
    print(f"  Mean dRCI: {np.mean(drcis):.4f} +/- {np.std(drcis, ddof=1):.4f}", flush=True)

    # Behavior classification
    negative_count = sum(1 for d in drcis if d < -0.05)
    positive_count = sum(1 for d in drcis if d > 0.05)
    neutral_count = len(drcis) - negative_count - positive_count

    if negative_count > len(drcis) * 0.7:
        behavior = "SOVEREIGN"
    elif positive_count > len(drcis) * 0.7:
        behavior = "CONVERGENT"
    else:
        behavior = "NEUTRAL/MIXED"

    print(f"  Behavior: {behavior}", flush=True)
    print(f"  Distribution: {negative_count} SOVEREIGN, {neutral_count} NEUTRAL, {positive_count} CONVERGENT", flush=True)
    print("="*70, flush=True)

def main():
    parser = argparse.ArgumentParser(description="MCH Medical Experiment - Newer Models")
    parser.add_argument("--model", "-m", type=str, help="Model name to run (see --list)")
    parser.add_argument("--trials", "-n", type=int, default=50, help="Number of trials (default: 50)")
    parser.add_argument("--list", "-l", action="store_true", help="List available models")
    parser.add_argument("--all", "-a", action="store_true", help="Run all available models")

    args = parser.parse_args()

    if args.list:
        print("\nAvailable models:")
        print("-" * 60)
        for name, config in MODELS.items():
            print(f"  {name:<20} [{config['provider']:<10}] {config['model_id']}")
        print()
        return

    if args.all:
        for model_name in MODELS:
            try:
                run_model(model_name, args.trials)
            except Exception as e:
                print(f"  ERROR with {model_name}: {e}", flush=True)
        return

    if args.model:
        run_model(args.model, args.trials)
    else:
        print("Please specify a model with --model or use --list to see available models")
        print("Example: python mch_newer_models_medical.py --model gemini_3_pro --trials 50")

if __name__ == "__main__":
    main()
