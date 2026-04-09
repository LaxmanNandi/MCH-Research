#!/usr/bin/env python3
"""MCH Philosophy Experiment - New Generation Models (Gemini 3 Pro, GPT-5.2)."""

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

import openai
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# API clients - with timeout to prevent hanging
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=120.0)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load embedding model
print("Loading embedding model: all-MiniLM-L6-v2...", flush=True)
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded.", flush=True)

# Philosophy prompts (same as original experiments)
PROMPTS = [
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

# Models to run
# Note: Gemini 3 Pro has safety blocking issues with philosophy prompts
MODELS_TO_RUN = [
    ("gpt_5_2", "gpt-5.2", "openai"),
    # ("gemini_3_pro", "gemini-3-pro-preview", "google"),  # Safety blocked
]

N_TRIALS = 100
TEMPERATURE = 0.7
OUTPUT_DIR = "C:/Users/barla/mch_experiments/data/philosophy_results"

def get_embedding(text):
    return embedder.encode(text, convert_to_numpy=True)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_response_openai(model_id, messages, max_retries=5):
    for attempt in range(max_retries):
        try:
            # GPT-5.2 uses max_completion_tokens, older models use max_tokens
            if "gpt-5" in model_id or "o1" in model_id or "o3" in model_id:
                response = openai_client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_completion_tokens=1024
                )
            else:
                response = openai_client.chat.completions.create(
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

def get_response(vendor, model_id, messages):
    if vendor == "openai":
        return get_response_openai(model_id, messages)
    elif vendor == "google":
        return get_response_google(model_id, messages)

def run_trial(model_name, model_id, vendor, trial_num):
    print(f"  [{model_name}] Trial {trial_num+1}/{N_TRIALS}...", flush=True)

    # TRUE condition - full conversation history
    print(f"    TRUE condition (30 prompts)...", flush=True)
    true_messages = []
    true_responses = []
    for i, prompt in enumerate(PROMPTS):
        print(f"      TRUE prompt {i+1}/30...", flush=True)
        true_messages.append({"role": "user", "content": prompt})
        resp = get_response(vendor, model_id, true_messages)
        true_responses.append(resp)
        true_messages.append({"role": "assistant", "content": resp})

    # COLD condition - no history (each prompt independent)
    print(f"    COLD condition (30 prompts)...", flush=True)
    cold_responses = []
    for i, prompt in enumerate(PROMPTS):
        print(f"      COLD prompt {i+1}/30...", flush=True)
        resp = get_response(vendor, model_id, [{"role": "user", "content": prompt}])
        cold_responses.append(resp)

    # SCRAMBLED condition - randomized history
    print(f"    SCRAMBLED condition (30 prompts)...", flush=True)
    scrambled_order = list(range(len(PROMPTS)))
    random.shuffle(scrambled_order)
    scrambled_messages = []
    scrambled_responses = []
    for i, idx in enumerate(scrambled_order):
        print(f"      SCRAMBLED prompt {i+1}/30...", flush=True)
        scrambled_messages.append({"role": "user", "content": PROMPTS[idx]})
        resp = get_response(vendor, model_id, scrambled_messages)
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

    checkpoint_file = os.path.join(OUTPUT_DIR, f"mch_results_{model_name}_philosophy_checkpoint.json")
    final_file = os.path.join(OUTPUT_DIR, f"mch_results_{model_name}_100trials.json")

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
                "domain": "philosophy",
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
        "domain": "philosophy",
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
    print("MCH PHILOSOPHY EXPERIMENT - NEW GENERATION MODELS", flush=True)
    print("Testing Model Coherence Hypothesis on Philosophical Reasoning", flush=True)
    print("="*70, flush=True)
    print(f"Domain: Philosophy (Consciousness)", flush=True)
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
