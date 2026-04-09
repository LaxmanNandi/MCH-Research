#!/usr/bin/env python3
"""
MCH Closed Models Philosophy Re-run - Paper 2 Methodology
Re-testing 6 closed models with Paper 2 protocol (TRUE/COLD/SCRAMBLED).

Models:
  - GPT-4o, GPT-4o-mini (OpenAI)
  - Claude Opus 4.5, Claude Haiku 4.5 (Anthropic)
  - Gemini 2.5 Pro, Gemini 2.5 Flash (Google)

Protocol: 3-condition (TRUE/COLD/SCRAMBLED)
Domain: Philosophy (30 prompts, 50 trials per model)
Embedding: all-MiniLM-L6-v2 (384D) - same as Paper 1/2
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

import openai
import anthropic
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# ============================================================================
# API CLIENTS
# ============================================================================
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=120.0)
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load embedding model
print("Loading embedding model: all-MiniLM-L6-v2...", flush=True)
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded.", flush=True)

# ============================================================================
# MODELS TO RUN (6 closed models - GPT-4o-mini first for fast testing)
# ============================================================================
MODELS_TO_RUN = [
    ("gpt4o_mini", "gpt-4o-mini", "openai"),
    ("gpt4o", "gpt-4o", "openai"),
    ("claude_haiku", "claude-haiku-4-5-20251001", "anthropic"),
    ("gemini_flash", "gemini-2.5-flash", "google"),
    ("gemini_pro", "gemini-3-pro-preview", "google"),
    ("claude_opus", "claude-opus-4-5-20251101", "anthropic"),
]

# ============================================================================
# EXPERIMENT PARAMETERS
# ============================================================================
N_TRIALS = 50
TEMPERATURE = 0.7
OUTPUT_DIR = "C:/Users/barla/mch_experiments/data/closed_model_philosophy_rerun"

# ============================================================================
# PHILOSOPHY PROMPTS (identical to Paper 1 & 2 - 30 prompts)
# ============================================================================
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

# ============================================================================
# API RESPONSE FUNCTIONS
# ============================================================================

def get_response_openai(model_id, messages, max_retries=5):
    """Get response from OpenAI API."""
    for attempt in range(max_retries):
        try:
            response = openai_client.chat.completions.create(
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


def get_response_anthropic(model_id, messages, max_retries=5):
    """Get response from Anthropic API."""
    for attempt in range(max_retries):
        try:
            response = anthropic_client.messages.create(
                model=model_id,
                max_tokens=1024,
                temperature=TEMPERATURE,
                messages=messages
            )
            return response.content[0].text
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)
                print(f"    Retry {attempt+1}: {e} (waiting {wait_time}s)", flush=True)
                time.sleep(wait_time)
            else:
                raise e


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
                    max_output_tokens=1024
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


def get_response(vendor, model_id, messages):
    """Route to correct API based on vendor."""
    if vendor == "openai":
        return get_response_openai(model_id, messages)
    elif vendor == "anthropic":
        return get_response_anthropic(model_id, messages)
    elif vendor == "google":
        return get_response_google(model_id, messages)

# ============================================================================
# EMBEDDING & SIMILARITY (Paper 2 methodology)
# ============================================================================

def get_embedding(text):
    """Get embedding using MiniLM-L6-v2."""
    return embedder.encode(text, convert_to_numpy=True)


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ============================================================================
# TRIAL EXECUTION (Paper 2 methodology exactly)
# ============================================================================

def run_trial(model_name, model_id, vendor, prompts, trial_num, n_trials):
    """Run a single trial with 3 conditions - Paper 2 methodology."""
    print(f"  [{model_name}] Trial {trial_num+1}/{n_trials}...", flush=True)

    # TRUE condition - full conversation history
    print(f"    TRUE condition (30 prompts)...", flush=True)
    true_messages = []
    true_responses = []
    for i, prompt in enumerate(prompts):
        if (i + 1) % 10 == 0:
            print(f"      TRUE prompt {i+1}/30...", flush=True)
        true_messages.append({"role": "user", "content": prompt})
        resp = get_response(vendor, model_id, true_messages)
        true_responses.append(resp)
        true_messages.append({"role": "assistant", "content": resp})

    # COLD condition - no history (each prompt independent)
    print(f"    COLD condition (30 prompts)...", flush=True)
    cold_responses = []
    for i, prompt in enumerate(prompts):
        if (i + 1) % 10 == 0:
            print(f"      COLD prompt {i+1}/30...", flush=True)
        resp = get_response(vendor, model_id, [{"role": "user", "content": prompt}])
        cold_responses.append(resp)

    # SCRAMBLED condition - randomized history
    print(f"    SCRAMBLED condition (30 prompts)...", flush=True)
    scrambled_order = list(range(len(prompts)))
    random.shuffle(scrambled_order)
    scrambled_messages = []
    scrambled_responses = []
    for i, idx in enumerate(scrambled_order):
        if (i + 1) % 10 == 0:
            print(f"      SCRAMBLED prompt {i+1}/30...", flush=True)
        scrambled_messages.append({"role": "user", "content": prompts[idx]})
        resp = get_response(vendor, model_id, scrambled_messages)
        scrambled_responses.append(resp)
        scrambled_messages.append({"role": "assistant", "content": resp})

    # Compute embeddings and alignments (Paper 2 method)
    true_embs = [get_embedding(r) for r in true_responses]
    cold_embs = [get_embedding(r) for r in cold_responses]
    scrambled_embs = [get_embedding(r) for r in scrambled_responses]

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

    print(f"  [{model_name}] Trial {trial_num+1} dRCI={delta_rci_cold:.4f}", flush=True)

    return {
        "trial": trial_num,
        "prompts": prompts,
        "responses": {"true": true_responses, "cold": cold_responses, "scrambled": scrambled_responses},
        "scrambled_order": scrambled_order,
        "alignments": {"true": true_aligns, "cold": cold_aligns, "scrambled": scrambled_aligns},
        "means": {"true": mean_true, "cold": mean_cold, "scrambled": mean_scrambled},
        "delta_rci": {"cold": delta_rci_cold, "scrambled": delta_rci_scrambled}
    }

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment():
    """Run philosophy experiment for all 6 closed models."""
    prompts = PHILOSOPHY_PROMPTS

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70, flush=True)
    print("MCH CLOSED MODELS PHILOSOPHY RE-RUN - PAPER 2 METHODOLOGY", flush=True)
    print("=" * 70, flush=True)
    print(f"Domain: Philosophy (Consciousness)", flush=True)
    print(f"Protocol: 3-condition (TRUE/COLD/SCRAMBLED)", flush=True)
    print(f"Prompts per trial: {len(prompts)}", flush=True)
    print(f"Trials per model: {N_TRIALS}", flush=True)
    print(f"Temperature: {TEMPERATURE}", flush=True)
    print(f"Max tokens: 1024", flush=True)
    print(f"Embedding: all-MiniLM-L6-v2 (384D)", flush=True)
    print(f"Models: {len(MODELS_TO_RUN)}", flush=True)
    for m in MODELS_TO_RUN:
        print(f"  - {m[0]} ({m[1]}, {m[2]})", flush=True)
    print(f"Output: {OUTPUT_DIR}", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)

    for model_name, model_id, vendor in MODELS_TO_RUN:
        print(f"\n{'='*60}", flush=True)
        print(f"Starting {model_name} ({model_id}, {vendor})", flush=True)
        print(f"{'='*60}", flush=True)

        # Check for existing checkpoint
        checkpoint_file = os.path.join(OUTPUT_DIR, f"mch_results_{model_name}_philosophy_checkpoint.json")
        final_file = os.path.join(OUTPUT_DIR, f"mch_results_{model_name}_philosophy_{N_TRIALS}trials.json")

        # Skip if already completed
        if os.path.exists(final_file):
            print(f"  {model_name} already completed. Skipping.", flush=True)
            continue

        # Load checkpoint if exists
        trials = []
        start_trial = 0
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                trials = checkpoint_data.get("trials", [])
                start_trial = len(trials)
                print(f"  Resuming from trial {start_trial}", flush=True)

        # Run trials
        for trial_num in range(start_trial, N_TRIALS):
            try:
                trial_result = run_trial(model_name, model_id, vendor, prompts, trial_num, N_TRIALS)
            except (ValueError, Exception) as e:
                if "finish_reason" in str(e) or "credit balance" in str(e):
                    print(f"\n  {model_name} BLOCKED: {e}", flush=True)
                    print(f"  Skipping {model_name}, moving to next model.", flush=True)
                    break
                raise
            trials.append(trial_result)

            # Save checkpoint every 5 trials
            if (trial_num + 1) % 5 == 0:
                checkpoint_data = {
                    "model": model_name,
                    "model_id": model_id,
                    "vendor": vendor,
                    "domain": "philosophy",
                    "n_trials": len(trials),
                    "temperature": TEMPERATURE,
                    "embedding_model": "minilm",
                    "trials": trials
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
                print(f"  Checkpoint saved at trial {trial_num + 1}", flush=True)

        # Skip statistics/saving if no trials completed (safety blocked)
        if len(trials) == 0:
            print(f"\n  {model_name}: No trials completed. Skipping.", flush=True)
            continue

        if len(trials) < N_TRIALS:
            print(f"\n  {model_name}: Only {len(trials)}/{N_TRIALS} trials completed.", flush=True)

        # Compute statistics
        drci_values = [t["delta_rci"]["cold"] for t in trials]
        mean_drci = float(np.mean(drci_values))
        std_drci = float(np.std(drci_values))
        pattern = "CONVERGENT" if mean_drci > 0.01 else "SOVEREIGN" if mean_drci < -0.01 else "NEUTRAL"

        # Save final results
        final_data = {
            "model": model_name,
            "model_id": model_id,
            "vendor": vendor,
            "domain": "philosophy",
            "n_trials": N_TRIALS,
            "n_trials_completed": len(trials),
            "n_prompts": len(prompts),
            "temperature": TEMPERATURE,
            "embedding_model": "minilm",
            "timestamp": datetime.now().isoformat(),
            "statistics": {
                "mean_drci": mean_drci,
                "std_drci": std_drci,
                "pattern": pattern
            },
            "trials": trials
        }
        with open(final_file, 'w') as f:
            json.dump(final_data, f, indent=2)

        # Remove checkpoint after successful completion
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)

        print(f"\n  {model_name} complete! Results saved.", flush=True)
        print(f"  Mean dRCI: {mean_drci:.4f} +/- {std_drci:.4f}", flush=True)
        print(f"  Pattern: {pattern}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT COMPLETE - PHILOSOPHY", flush=True)
    print("=" * 70, flush=True)

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_experiment()
