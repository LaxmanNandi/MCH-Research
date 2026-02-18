#!/usr/bin/env python3
"""
Paper 7 Pilot — Context Utilization Depth (CUD)

Measures the minimum conversation history (K) a model needs to achieve
near-TRUE context sensitivity at a given position.

METHODOLOGY (consistent with MCH Papers 1-6):
  RCI = cosine_sim(response_TRUE, response_COLD)  [response-to-response]
  ΔRCI = 1.0 - RCI

For CUD, we introduce TRUNCATED(K):
  RCI_TRUNCATED(K) = cosine_sim(response_TRUNCATED(K), response_COLD)
  ΔRCI_TRUNCATED(K) = 1.0 - RCI_TRUNCATED(K)

  CUD = min K where ΔRCI_TRUNCATED(K) >= 0.90 * ΔRCI_TRUE

As K increases, response_TRUNCATED diverges from response_COLD (gains context),
so ΔRCI_TRUNCATED rises toward ΔRCI_TRUE.
"""

import os
import sys
import json
import time
import random
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load API keys from environment or .env file
def load_env_key(key_name):
    val = os.getenv(key_name)
    if not val:
        env_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith(f"{key_name}="):
                        val = line.strip().split("=", 1)[1]
    return val

TOGETHER_API_KEY = load_env_key("TOGETHER_API_KEY")
GOOGLE_API_KEY = load_env_key("GOOGLE_API_KEY")

# Together.ai client
BASE_URL = "https://api.together.xyz/v1"

# Models: each entry is (model_id, vendor)
MODELS = {
    "deepseek_v3_1": ("deepseek-ai/DeepSeek-V3.1", "together"),
    "gemini_flash": ("gemini-2.5-flash", "google"),
    "llama_4_maverick": ("meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", "together"),
    "qwen3_235b": ("Qwen/Qwen3-235B-A22B-Instruct-2507-FP8", "together"),
}

N_TRIALS = 50
TEMPERATURE = 0.7
MAX_TOKENS = 1024

# Positions to test — K values adjusted per position
# Medical P30: full history = 29 message pairs, so K goes up to 29
# Philosophy P15: full history = 14 message pairs, so K goes up to 14
POSITIONS = {
    "medical": {
        "position": 30,
        "k_values": [1, 5, 10, 15, 20, 29],  # K=29 = full history = TRUE equivalent
    },
    "philosophy": {
        "position": 15,
        "k_values": [1, 3, 5, 7, 10, 14],    # K=14 = full history = TRUE equivalent
    }
}

# ============================================================================
# PROMPTS (identical to Papers 1-6)
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

DOMAIN_PROMPTS = {
    "medical": MEDICAL_PROMPTS,
    "philosophy": PHILOSOPHY_PROMPTS,
}

# ============================================================================
# API CLIENTS
# ============================================================================

# Together.ai client (OpenAI-compatible)
together_client = None
if TOGETHER_API_KEY:
    together_client = OpenAI(base_url=BASE_URL, api_key=TOGETHER_API_KEY, timeout=120.0)

# Google Gemini client
genai = None
if GOOGLE_API_KEY:
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_API_KEY)

def call_api_together(model_id, messages, max_retries=5):
    """Call Together.ai API."""
    for attempt in range(max_retries):
        try:
            response = together_client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)
                print(f"    Retry {attempt+1}: {e} (waiting {wait_time}s)", flush=True)
                time.sleep(wait_time)
            else:
                print(f"    FAILED after {max_retries} retries: {e}", flush=True)
                return None

def call_api_google(model_id, messages, max_retries=5):
    """Call Google Gemini API with conversation history."""
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel(model_id)
            # Convert messages to Gemini format
            gemini_history = []
            for m in messages[:-1]:
                role = "user" if m["role"] == "user" else "model"
                gemini_history.append({"role": role, "parts": [m["content"]]})
            chat = model.start_chat(history=gemini_history)
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
                print(f"    FAILED after {max_retries} retries: {e}", flush=True)
                return None

def call_api(model_id, messages, vendor="together", max_retries=5):
    """Dispatch to the appropriate API."""
    if vendor == "google":
        return call_api_google(model_id, messages, max_retries)
    else:
        return call_api_together(model_id, messages, max_retries)

# ============================================================================
# EMBEDDING & SIMILARITY (matches Papers 1-6 exactly)
# ============================================================================

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    """Get 384D embedding using all-MiniLM-L6-v2."""
    return embedder.encode(text, convert_to_numpy=True)

def cosine_similarity(a, b):
    """Cosine similarity between two embedding vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ============================================================================
# CONVERSATION GENERATION
# ============================================================================

def generate_true_conversation(model_id, prompts, up_to_position, vendor="together"):
    """
    Generate a full TRUE conversation up to the target position.
    Returns (responses, full_history) where history is the message list.
    """
    messages = []
    responses = []

    for i in range(up_to_position):
        messages.append({"role": "user", "content": prompts[i]})
        resp = call_api(model_id, messages, vendor=vendor)
        if resp is None:
            return None, None
        responses.append(resp)
        messages.append({"role": "assistant", "content": resp})
        time.sleep(0.3)

    return responses, messages

def truncate_history(full_history, K):
    """
    Truncate conversation history to last K message pairs.
    full_history is a list of alternating user/assistant messages.
    K message pairs = K*2 individual messages.
    """
    if K * 2 >= len(full_history):
        return list(full_history)  # Full history, no truncation
    return list(full_history[-(K * 2):])

# ============================================================================
# SINGLE TRIAL
# ============================================================================

def run_trial(model_id, prompts, position, k_values, trial_num, vendor="together"):
    """
    Run one trial for a given position.

    For each trial:
    1. Generate fresh TRUE conversation up to position (MCH protocol: new conversation per trial)
    2. Get response_TRUE at target position (with full history)
    3. Get response_COLD at target position (no history) — ONCE per trial
    4. For each K: get response_TRUNCATED(K) at target position (with last K pairs)
    5. Compute RCI and ΔRCI using response-to-response cosine similarity
    """
    target_prompt = prompts[position - 1]  # 0-indexed

    # Step 1 & 2: Generate TRUE conversation and get TRUE response at target position
    # The conversation includes responses at positions 1..position, so response at position is responses[-1]
    true_responses, full_history = generate_true_conversation(model_id, prompts, position, vendor=vendor)
    if true_responses is None:
        print(f"      Trial {trial_num}: TRUE conversation failed, skipping", flush=True)
        return None

    response_true = true_responses[-1]  # Response at target position with full context

    # History BEFORE the target position (for truncation)
    # full_history includes the target position's prompt+response, so exclude last 2
    history_before_target = full_history[:-2]  # Messages before the target prompt

    # Step 3: Get COLD response — ONCE per trial (not per K)
    response_cold = call_api(model_id, [{"role": "user", "content": target_prompt}], vendor=vendor)
    if response_cold is None:
        print(f"      Trial {trial_num}: COLD response failed, skipping", flush=True)
        return None
    time.sleep(0.3)

    # Compute embeddings for TRUE and COLD
    emb_true = get_embedding(response_true)
    emb_cold = get_embedding(response_cold)

    # MCH RCI: cosine_sim(response_TRUE, response_COLD)
    rci_true_cold = cosine_similarity(emb_true, emb_cold)
    drci_true = 1.0 - rci_true_cold  # ΔRCI_TRUE for this trial

    # Step 4: For each K, get TRUNCATED(K) response
    k_results = {}
    for K in k_values:
        truncated_hist = truncate_history(history_before_target, K)
        messages_with_prompt = truncated_hist + [{"role": "user", "content": target_prompt}]
        response_trunc = call_api(model_id, messages_with_prompt, vendor=vendor)

        if response_trunc is None:
            print(f"      Trial {trial_num}, K={K}: TRUNCATED response failed", flush=True)
            k_results[K] = None
            continue

        time.sleep(0.3)

        # Step 5: Compute RCI_TRUNCATED(K) = cosine_sim(response_TRUNCATED, response_COLD)
        emb_trunc = get_embedding(response_trunc)
        rci_trunc_cold = cosine_similarity(emb_trunc, emb_cold)
        drci_trunc = 1.0 - rci_trunc_cold

        # Also compute similarity of TRUNCATED to TRUE (diagnostic)
        sim_trunc_true = cosine_similarity(emb_trunc, emb_true)

        k_results[K] = {
            "rci_trunc_cold": rci_trunc_cold,
            "drci_truncated": drci_trunc,
            "sim_trunc_true": sim_trunc_true,
            "response_truncated": response_trunc
        }

    return {
        "trial": trial_num,
        "rci_true_cold": rci_true_cold,
        "drci_true": drci_true,
        "k_results": k_results,
        "response_true": response_true,
        "response_cold": response_cold
    }

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_pilot(run_only=None):
    """Run the full CUD pilot experiment. Optionally run only a specific model."""

    # Filter models if running single model
    models_to_run = MODELS
    if run_only:
        if run_only not in MODELS:
            print(f"ERROR: Model '{run_only}' not found. Available: {list(MODELS.keys())}")
            sys.exit(1)
        models_to_run = {run_only: MODELS[run_only]}

    print("=" * 70)
    print("PAPER 7 PILOT — CONTEXT UTILIZATION DEPTH (CUD)")
    print("=" * 70)
    print(f"Models: {list(models_to_run.keys())}")
    print(f"Trials per condition: {N_TRIALS}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Embedding: all-MiniLM-L6-v2 (384D)")
    print(f"RCI method: response-to-response cosine similarity (MCH Papers 1-6)")
    print("=" * 70)

    # Estimate total API calls
    total_calls = 0
    for domain, config in POSITIONS.items():
        pos = config["position"]
        n_k = len(config["k_values"])
        calls_per_trial = pos + 1 + n_k
        total_calls += len(models_to_run) * N_TRIALS * calls_per_trial
    print(f"Estimated API calls: {total_calls}")
    print()

    for model_name, (model_id, vendor) in models_to_run.items():
        # Validate API key
        if vendor == "together" and not TOGETHER_API_KEY:
            print(f"  SKIPPING {model_name}: TOGETHER_API_KEY not set")
            continue
        if vendor == "google" and not GOOGLE_API_KEY:
            print(f"  SKIPPING {model_name}: GOOGLE_API_KEY not set")
            continue

        print(f"\n{'=' * 60}")
        print(f"MODEL: {model_name} ({model_id})")
        print(f"{'=' * 60}")

        for domain, config in POSITIONS.items():
            position = config["position"]
            k_values = config["k_values"]
            prompts = DOMAIN_PROMPTS[domain]

            print(f"\n  Domain: {domain}, Position: P{position}")
            print(f"  K values: {k_values}")
            print(f"  History length: {position - 1} message pairs")

            # Output file for this model-domain
            output_file = os.path.join(SCRIPT_DIR, f"results/raw/{model_name}_{domain}_P{position}.json")
            checkpoint_file = os.path.join(SCRIPT_DIR, f"results/raw/{model_name}_{domain}_P{position}_checkpoint.json")

            # Resume from checkpoint if exists
            all_trials = []
            start_trial = 0
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                all_trials = checkpoint.get("trials", [])
                start_trial = len(all_trials)
                print(f"  Resuming from trial {start_trial}")

            for trial in range(start_trial, N_TRIALS):
                print(f"\n    Trial {trial + 1}/{N_TRIALS}...", flush=True)

                result = run_trial(model_id, prompts, position, k_values, trial, vendor=vendor)

                if result is not None:
                    all_trials.append(result)
                    drci_true = result["drci_true"]
                    k_summary = []
                    for K in k_values:
                        kr = result["k_results"].get(K)
                        if kr:
                            k_summary.append(f"K={K}:{kr['drci_truncated']:.3f}")
                    print(f"    dRCI_TRUE={drci_true:.4f} | {' '.join(k_summary)}", flush=True)

                # Checkpoint every 5 trials
                if (trial + 1) % 5 == 0:
                    save_checkpoint(all_trials, model_name, domain, position, checkpoint_file)
                    print(f"    Checkpoint saved at trial {trial + 1}", flush=True)

            # Save final results (without response text for smaller file)
            save_final(all_trials, model_name, domain, position, output_file)
            print(f"\n  Saved: {output_file} ({len(all_trials)} trials)")

    print(f"\n{'=' * 70}")
    print("PILOT COMPLETE")
    print(f"{'=' * 70}")

def save_checkpoint(trials, model_name, domain, position, filepath):
    """Save checkpoint with full data including responses."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data = {
        "model": model_name,
        "domain": domain,
        "position": position,
        "n_trials": len(trials),
        "trials": trials
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

def save_final(trials, model_name, domain, position, filepath):
    """Save final results (strip response text to reduce file size)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Strip response text for compact storage
    compact_trials = []
    for t in trials:
        ct = {
            "trial": t["trial"],
            "rci_true_cold": t["rci_true_cold"],
            "drci_true": t["drci_true"],
            "k_results": {}
        }
        for K, kr in t["k_results"].items():
            if kr is not None:
                ct["k_results"][K] = {
                    "rci_trunc_cold": kr["rci_trunc_cold"],
                    "drci_truncated": kr["drci_truncated"],
                    "sim_trunc_true": kr["sim_trunc_true"]
                }
        compact_trials.append(ct)

    data = {
        "model": model_name,
        "domain": domain,
        "position": position,
        "n_trials": len(compact_trials),
        "methodology": {
            "rci": "cosine_sim(response_TRUNCATED, response_COLD)",
            "drci": "1.0 - RCI",
            "embedding": "all-MiniLM-L6-v2 (384D)",
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
        },
        "trials": compact_trials
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

# ============================================================================

if __name__ == "__main__":
    # Usage: python run_pilot.py [model_name]
    # e.g., python run_pilot.py gemini_flash
    model_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_pilot(run_only=model_arg)
