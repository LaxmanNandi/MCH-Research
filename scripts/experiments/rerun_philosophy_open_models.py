#!/usr/bin/env python3
"""
MCH Philosophy Open Models Re-run — WITH Response Text Saving
=============================================================
Re-runs 7 open-weight models on philosophy domain via Together.ai,
this time saving full response text for entanglement analysis (Paper 4).

Original run (mch_open_models.py) used single-embedding mode which
discarded response text. This script preserves it.

Models (7 open-weight via Together.ai):
  - DeepSeek-V3.1 (671B/37B active)
  - Llama 4 Maverick (17B-128E MoE)
  - Llama 4 Scout (17B-16E)
  - Qwen3 235B (22B active)
  - Mistral Small 24B
  - Ministral 14B
  - Kimi K2 (1T params, MoE)

Protocol: 3-condition (TRUE/COLD/SCRAMBLED), 50 trials, 30 prompts
Embedding: all-MiniLM-L6-v2 (384D) — same as Papers 1-4
Output: data/philosophy/open_models/ (overwrites metrics-only files)

Estimated cost: ~$50-100 in Together.ai credits
Estimated time: ~6-12 hours (7 models × 50 trials × 90 API calls each)
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
# MODELS TO RUN (7 open-weight models — same IDs as original run)
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
# EXPERIMENT PARAMETERS
# ============================================================================
N_TRIALS = 50
TEMPERATURE = 0.7
OUTPUT_DIR = "C:/Users/barla/mch_experiments/data/philosophy/open_models"

# ============================================================================
# PHILOSOPHY PROMPTS (identical to Papers 1-4 — 30 prompts)
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
# HELPER FUNCTIONS
# ============================================================================

def get_embedding(text):
    """Get embedding using MiniLM-L6-v2."""
    return embedder.encode(text, convert_to_numpy=True)


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
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
# TRIAL EXECUTION (Paper 2 methodology — WITH response text saving)
# ============================================================================

def run_trial(model_name, model_id, prompts, trial_num, n_trials):
    """Run a single trial with 3 conditions. Saves full response text."""
    print(f"  [{model_name}] Trial {trial_num+1}/{n_trials}...", flush=True)

    # TRUE condition — full conversation history
    print(f"    TRUE condition (30 prompts)...", flush=True)
    true_messages = []
    true_responses = []
    for i, prompt in enumerate(prompts):
        if (i + 1) % 10 == 0:
            print(f"      TRUE prompt {i+1}/30...", flush=True)
        true_messages.append({"role": "user", "content": prompt})
        resp = get_response_together(model_id, true_messages)
        true_responses.append(resp)
        true_messages.append({"role": "assistant", "content": resp})

    # COLD condition — no history (each prompt independent)
    print(f"    COLD condition (30 prompts)...", flush=True)
    cold_responses = []
    for i, prompt in enumerate(prompts):
        if (i + 1) % 10 == 0:
            print(f"      COLD prompt {i+1}/30...", flush=True)
        resp = get_response_together(model_id, [{"role": "user", "content": prompt}])
        cold_responses.append(resp)

    # SCRAMBLED condition — randomized history
    print(f"    SCRAMBLED condition (30 prompts)...", flush=True)
    scrambled_order = list(range(len(prompts)))
    random.shuffle(scrambled_order)
    scrambled_messages = []
    scrambled_responses = []
    for i, idx in enumerate(scrambled_order):
        if (i + 1) % 10 == 0:
            print(f"      SCRAMBLED prompt {i+1}/30...", flush=True)
        scrambled_messages.append({"role": "user", "content": prompts[idx]})
        resp = get_response_together(model_id, scrambled_messages)
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
        }
    }

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment():
    """Run philosophy experiment for all 7 open models with response saving."""
    prompts = PHILOSOPHY_PROMPTS

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70, flush=True)
    print("MCH PHILOSOPHY OPEN MODELS RE-RUN — WITH RESPONSE TEXT", flush=True)
    print("=" * 70, flush=True)
    print(f"Purpose: Save response text for Paper 3/4 entanglement analysis", flush=True)
    print(f"Domain: Philosophy (Consciousness)", flush=True)
    print(f"Protocol: 3-condition (TRUE/COLD/SCRAMBLED)", flush=True)
    print(f"Prompts per trial: {len(prompts)}", flush=True)
    print(f"Trials per model: {N_TRIALS}", flush=True)
    print(f"Temperature: {TEMPERATURE}", flush=True)
    print(f"Max tokens: 1024", flush=True)
    print(f"Embedding: all-MiniLM-L6-v2 (384D)", flush=True)
    print(f"Models: {len(MODELS_TO_RUN)}", flush=True)
    for m in MODELS_TO_RUN:
        print(f"  - {m[0]} ({m[1]})", flush=True)
    print(f"Output: {OUTPUT_DIR}", flush=True)
    print(f"Response saving: ENABLED", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)

    for model_name, model_id, vendor in MODELS_TO_RUN:
        print(f"\n{'='*60}", flush=True)
        print(f"Starting {model_name} ({model_id})", flush=True)
        print(f"{'='*60}", flush=True)

        # File paths
        checkpoint_file = os.path.join(OUTPUT_DIR, f"mch_results_{model_name}_philosophy_rerun_checkpoint.json")
        final_file = os.path.join(OUTPUT_DIR, f"mch_results_{model_name}_philosophy_{N_TRIALS}trials.json")

        # Check if already completed WITH responses
        if os.path.exists(final_file):
            with open(final_file, 'r') as f:
                existing = json.load(f)
            # Check if existing file already has response text
            if existing.get("trials") and "responses" in existing["trials"][0]:
                print(f"  {model_name} already has response text. Skipping.", flush=True)
                continue
            else:
                print(f"  {model_name} exists but WITHOUT response text. Re-running.", flush=True)
                # Rename old file as backup
                backup_file = final_file.replace(".json", "_metrics_only.json")
                os.rename(final_file, backup_file)
                print(f"  Old file backed up to: {os.path.basename(backup_file)}", flush=True)

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
                trial_result = run_trial(model_name, model_id, prompts, trial_num, N_TRIALS)
            except Exception as e:
                print(f"\n  {model_name} ERROR at trial {trial_num}: {e}", flush=True)
                print(f"  Saving checkpoint and moving to next model.", flush=True)
                break
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
                    "response_text_saved": True,
                    "trials": trials
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
                print(f"  Checkpoint saved at trial {trial_num + 1}", flush=True)

        # Skip if no trials completed
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
            "response_text_saved": True,
            "rerun_reason": "Original run did not save response text; needed for Paper 3/4 entanglement analysis",
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

        print(f"\n  {model_name} complete! Results saved with response text.", flush=True)
        print(f"  Mean dRCI: {mean_drci:.4f} +/- {std_drci:.4f}", flush=True)
        print(f"  Pattern: {pattern}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("PHILOSOPHY OPEN MODELS RE-RUN COMPLETE", flush=True)
    print("=" * 70, flush=True)
    print("Next steps:", flush=True)
    print("  1. Run validate_entanglement.py to recompute VRI with new data", flush=True)
    print("  2. Update Paper 3 model counts (4 → 11 philosophy models)", flush=True)
    print("  3. Update Paper 4 model counts (11 → 18 model-domain runs)", flush=True)
    print("=" * 70, flush=True)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="MCH Philosophy Open Models Re-run with Response Text Saving"
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Run only a specific model (e.g., 'deepseek_v3_1')")
    args = parser.parse_args()

    if args.model:
        # Filter to single model
        matching = [m for m in MODELS_TO_RUN if m[0] == args.model]
        if not matching:
            print(f"ERROR: Model '{args.model}' not found. Available:")
            for m in MODELS_TO_RUN:
                print(f"  {m[0]}")
            sys.exit(1)
        MODELS_TO_RUN = matching
        print(f"Running single model: {args.model}", flush=True)

    run_experiment()
