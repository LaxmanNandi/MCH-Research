#!/usr/bin/env python3
"""
Paper 6 Replication: Legal Domain Experiments
=============================================
Legal/Contractual domain (closed-goal) — whistleblower employment dispute.
Pre-registered on OSF: https://osf.io/dp8nj/

Uses exact same protocol as Medical experiments (run_medical_experiments.py):
  - 3-condition: TRUE/COLD/SCRAMBLED
  - 50 trials per model
  - Temperature 0.7
  - Embedding: all-MiniLM-L6-v2 (384D)

Models: Same open-weight models via Together.ai as Medical domain.
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

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

# ============================================================================
# PROGRESS DISPLAY
# ============================================================================

_trial_start_times = []

def progress_bar(current, total, width=30, label=""):
    filled = int(width * current / total) if total > 0 else 0
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    pct = current / total * 100 if total > 0 else 0
    return f"  {label:12s} {current:3d}/{total} [{bar}] {pct:5.1f}%"

def show_model_header(model_name, start_trial, total_trials):
    print(f"\n\u2550\u2550\u2550 {model_name} {'='*50}", flush=True)
    remaining = total_trials - start_trial
    print(f"  Trials: {start_trial} done, {remaining} remaining", flush=True)

def show_trial_progress(model_name, trial_num, total_trials):
    print(f"\r{progress_bar(trial_num + 1, total_trials, label='Trials')}", end="", flush=True)

def show_condition_progress(condition, prompt_num, total_prompts):
    print(f"\r{progress_bar(prompt_num, total_prompts, label=condition)}", end="", flush=True)

def show_trial_complete(model_name, trial_num, total_trials, drci, elapsed):
    avg_time = elapsed
    remaining_trials = total_trials - (trial_num + 1)
    if len(_trial_start_times) >= 2:
        avg_time = np.mean(_trial_start_times[-5:]) if len(_trial_start_times) >= 5 else np.mean(_trial_start_times)
    est_remaining = avg_time * remaining_trials
    h, rem = divmod(int(est_remaining), 3600)
    m, s = divmod(rem, 60)
    time_str = f"{h}h {m:02d}m" if h > 0 else f"{m}m {s:02d}s"
    print(f"\r{progress_bar(trial_num + 1, total_trials, label='Trials')}  dRCI={drci:+.4f}  ETA: ~{time_str}   ", flush=True)

load_dotenv()

# Together.ai client
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    print("ERROR: TOGETHER_API_KEY not found in .env file")
    sys.exit(1)

together_client = OpenAI(
    api_key=TOGETHER_API_KEY,
    base_url="https://api.together.xyz/v1",
    timeout=120.0
)

# Load embedding model
from sentence_transformers import SentenceTransformer

print("Loading embedding model: all-MiniLM-L6-v2...", flush=True)
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded.", flush=True)

# ============================================================================
# MODELS (same as Medical experiments)
# ============================================================================
MODELS_TO_RUN = [
    ("deepseek_v3_1", "deepseek-ai/DeepSeek-V3.1", "together"),
    ("llama_4_maverick", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", "together"),
    # ("llama_4_scout", "meta-llama/Llama-4-Scout-17B-16E-Instruct", "together"),  # Listed but not serverless
    ("qwen3_235b", "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8", "together"),
    ("mistral_small_24b", "mistralai/Mistral-Small-24B-Instruct-2501", "together"),
    # ("ministral_14b", "mistralai/Ministral-3-14B-Instruct-2512", "together"),  # Not available serverless
    # ("kimi_k2", "moonshotai/Kimi-K2-Instruct-0905", "together"),  # Not available serverless
    ("kimi_k2_5", "moonshotai/Kimi-K2.5", "together"),
    # ("llama_3_3_70b", "meta-llama/Llama-3.3-70B-Instruct", "together"),  # Removed from Together AI serverless
    # ("glm_5", "zai-org/GLM-5", "together"),  # EXCLUDED: 86% empty responses
    # ("llama_3_1_405b", "meta-llama/Llama-3.1-405B-Instruct", "together"),  # Listed but not serverless
    ("llama_3_3_70b_turbo", "meta-llama/Llama-3.3-70B-Instruct-Turbo", "together"),  # New — turbo variant available
]

# ============================================================================
# LEGAL PROMPTS — Whistleblower Employment Dispute (30 prompts)
# From paper6_complete_experimental_design.md (OSF pre-registered)
# ============================================================================
PROMPTS = [
    # P1-P5: Case Facts
    'I need legal advice regarding an employment contract dispute. My client was terminated after 8 years with the company. The termination letter cites "restructuring" but we believe it was retaliation for whistleblowing.',

    "My client was a senior compliance officer at a financial services firm. Her responsibilities included reviewing suspicious transactions and filing Suspicious Activity Reports (SARs) with regulators.",

    "Eighteen months ago, she flagged a pattern of transactions involving a major client — approximately $4.7 million over 6 months — that appeared designed to evade reporting thresholds. She filed an internal report and escalated it to her supervisor.",

    "Her supervisor initially thanked her, but over the next 3 months, she was excluded from key meetings, given menial tasks, and received her first negative performance review in 8 years. She documented everything.",

    'She filed a formal whistleblower complaint with the SEC 11 months ago. The company was notified 2 months later. Four months after that, she was terminated in a "company-wide restructuring." Her position was eliminated, but a new compliance role with similar duties was posted 3 weeks later.',

    # P6-P12: Contract Analysis
    'The company\'s employment contract includes an at-will clause but also a "good faith and fair dealing" provision. It also requires arbitration for any disputes. We are in California. What are the key facts we need to establish first?',

    'Let me read you the exact at-will clause: "Employment is at-will and may be terminated by either party at any time, with or without cause or notice." How do we overcome this language in a retaliation claim?',

    'The "good faith and fair dealing" provision states: "Both parties agree to act in good faith and deal fairly with each other throughout the employment relationship." Does this create an implied contract exception to at-will in California?',

    "The arbitration clause requires all disputes to go to binding arbitration before the American Arbitration Association, with costs split 50/50. Does this clause hold up under California law? Can we challenge it as unconscionable given the cost split?",

    'The contract includes a non-disparagement clause that prohibits the employee from making "any negative statements about the company." My client has already spoken to a reporter (anonymously). Could this be used against her?',

    "There is a choice-of-law clause specifying Delaware law, but the employee worked entirely in California. The contract was signed in California. Can we argue that California law should apply instead? Why does this matter for retaliation claims?",

    "The contract has a liquidated damages provision — 6 months' salary if terminated without cause. Is this enforceable? Does accepting this payment waive her right to sue for retaliation?",

    # P13-P18: Legal Framework
    "Let's discuss the relevant statutes. The primary federal whistleblower protection is the Sarbanes-Oxley Act (SOX). What does SOX require to establish a prima facie case of retaliation?",

    "Under SOX, the elements are: (1) protected activity, (2) employer knowledge, (3) adverse action, and (4) causation. How strong is our evidence for each element based on the facts we have?",

    "The Dodd-Frank Act expanded whistleblower protections and created an SEC whistleblower program. Does Dodd-Frank provide stronger remedies than SOX? Can my client claim under both?",

    "California has its own whistleblower protections under Labor Code §1102.5. How does California law differ from federal law? Is it more favorable to employees?",

    "We also have a potential common law claim for wrongful termination in violation of public policy. What are the elements of this claim? Can we plead it alongside the statutory claims?",

    "Based on the precedents we have discussed, which cases most directly support our client's position? Which pose the greatest risk? Are there any Ninth Circuit or California Supreme Court decisions we should focus on?",

    # P19-P24: Strategy
    "The company has offered to settle for 3 months' severance in exchange for a full release. They have also offered to keep the arbitration confidential. What are the advantages and disadvantages of accepting this offer?",

    "If we proceed with litigation, we have to decide: arbitration vs. court. Given the arbitration clause, we would need to challenge its enforceability to get to court. What are the odds of success on that challenge?",

    "Discovery will be critical. What key documents should we request? Emails regarding the SARs? Performance reviews? Restructuring plans? Communications with the SEC?",

    "We also need to depose key witnesses: her supervisor, HR personnel, the executives who made the termination decision. What questions are most important to establish causation?",

    "Damages: my client earned $180,000 per year with bonuses. She has been unemployed for 7 months. She also claims emotional distress. What is a realistic damages range if we prevail? What about punitive damages?",

    "The statute of limitations for SOX is 180 days after the violation (or after discovering the violation). We are at day 165 since termination. How urgent is our filing deadline? Do we need to file immediately to preserve claims?",

    # P25-P30: Resolution
    "The company's counsel contacted us. They are offering $95,000 to settle, including a neutral reference and no admission of liability. They want a response in 10 days. How do we evaluate this offer?",

    "If we reject settlement, what is our trial/arbitration timeline? How long will this take? What are the odds of success at each stage?",

    "My client is stressed and wants closure. She is also concerned about the financial cost of litigation. What are the estimated legal fees through arbitration? Through appeal?",

    "The SEC has opened an inquiry into the company based on my client's filing. This could strengthen our position but also complicates timing. How do we coordinate with the SEC investigation?",

    "We need to draft a demand letter before filing. What should it include? What tone is most effective — aggressive or conciliatory?",

    # P30 - KEY SYNTHESIS
    "Please provide a comprehensive legal analysis of this entire dispute: the key facts, applicable legal framework, strongest arguments for each side, risk assessment, and recommended strategy. This will serve as our case memorandum for the client and will guide all further decisions.",
]

# P30 COLD version (used only for COLD condition at position 30)
P30_COLD = "Please provide a comprehensive legal analysis of an employment dispute involving alleged wrongful termination and whistleblower retaliation. Include: the key legal frameworks (SOX, Dodd-Frank, state law), elements required to establish retaliation, typical defenses employers raise, risk assessment factors, and strategic recommendations. This will serve as a general case memorandum template."

# ============================================================================
# EXPERIMENT PARAMETERS
# ============================================================================
N_TRIALS = 50
TEMPERATURE = 0.7
OUTPUT_DIR = "C:/Users/barla/mch_experiments/data/legal/open_models"

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

    # TRUE condition - full conversation history
    true_messages = []
    true_responses = []
    for i, prompt in enumerate(PROMPTS):
        show_condition_progress("TRUE", i + 1, len(PROMPTS))
        true_messages.append({"role": "user", "content": prompt})
        resp = get_response_together(model_id, true_messages)
        true_responses.append(resp)
        true_messages.append({"role": "assistant", "content": resp})
    print(flush=True)

    # COLD condition - no history (each prompt independent)
    # Use P30_COLD for position 30 (standalone version)
    cold_responses = []
    for i, prompt in enumerate(PROMPTS):
        show_condition_progress("COLD", i + 1, len(PROMPTS))
        # Use standalone P30 for COLD condition at last position
        cold_prompt = P30_COLD if i == len(PROMPTS) - 1 else prompt
        resp = get_response_together(model_id, [{"role": "user", "content": cold_prompt}])
        cold_responses.append(resp)
    print(flush=True)

    # SCRAMBLED condition - randomized history
    scrambled_order = list(range(len(PROMPTS)))
    random.shuffle(scrambled_order)
    scrambled_messages = []
    scrambled_responses = []
    for i, idx in enumerate(scrambled_order):
        show_condition_progress("SCRAMBLED", i + 1, len(PROMPTS))
        scrambled_messages.append({"role": "user", "content": PROMPTS[idx]})
        resp = get_response_together(model_id, scrambled_messages)
        scrambled_responses.append(resp)
        scrambled_messages.append({"role": "assistant", "content": resp})
    print(flush=True)

    # Compute embeddings
    true_embs = [get_embedding(r) for r in true_responses]
    cold_embs = [get_embedding(r) for r in cold_responses]
    scrambled_embs = [get_embedding(r) for r in scrambled_responses]

    # Compute alignments (response-response comparison)
    true_aligns = [1.0] * len(PROMPTS)
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
    checkpoint_file = os.path.join(OUTPUT_DIR, f"mch_results_{model_name}_legal_checkpoint.json")
    final_file = os.path.join(OUTPUT_DIR, f"mch_results_{model_name}_legal_{N_TRIALS}trials.json")

    # Check if already complete (must have all trials, not just exist)
    if os.path.exists(final_file):
        with open(final_file, 'r') as f:
            existing = json.load(f)
        if len(existing.get("trials", [])) >= N_TRIALS:
            print(f"  {model_name} already complete ({len(existing['trials'])} trials). Skipping.", flush=True)
            return
        else:
            # File exists but incomplete — move to checkpoint and resume
            incomplete_trials = existing.get("trials", [])
            if incomplete_trials:
                checkpoint = existing
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint, f, indent=2)
                print(f"  {model_name} incomplete ({len(incomplete_trials)}/{N_TRIALS} trials). Resuming...", flush=True)
            else:
                print(f"  {model_name} empty (0 trials). Starting fresh...", flush=True)
            os.remove(final_file)

    # Load checkpoint if exists
    trials = []
    start_trial = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        trials = checkpoint.get("trials", [])
        start_trial = len(trials)

    show_model_header(model_name, start_trial, N_TRIALS)
    _trial_start_times.clear()

    # Run remaining trials
    for i in range(start_trial, N_TRIALS):
        trial_t0 = time.time()
        try:
            trial_data = run_trial(model_name, model_id, i)
            trials.append(trial_data)
            trial_elapsed = time.time() - trial_t0
            _trial_start_times.append(trial_elapsed)
            drci = trial_data["delta_rci"]["cold"]
            show_trial_complete(model_name, i, N_TRIALS, drci, trial_elapsed)
        except Exception as e:
            print(f"\n  [{model_name}] Trial {i+1} FAILED: {e}", flush=True)
            # Save checkpoint and continue
            checkpoint = {
                "model": model_name,
                "model_id": model_id,
                "vendor": vendor,
                "domain": "legal_contractual",
                "n_trials": len(trials),
                "temperature": TEMPERATURE,
                "embedding_model": "all-MiniLM-L6-v2",
                "prompts_version": "whistleblower_employment_dispute",
                "trials": trials
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            continue

        # Save checkpoint every 5 trials
        if (i + 1) % 5 == 0:
            checkpoint = {
                "model": model_name,
                "model_id": model_id,
                "vendor": vendor,
                "domain": "legal_contractual",
                "n_trials": len(trials),
                "temperature": TEMPERATURE,
                "embedding_model": "all-MiniLM-L6-v2",
                "prompts_version": "whistleblower_employment_dispute",
                "trials": trials
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)

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
        "domain": "legal_contractual",
        "prompts_version": "whistleblower_employment_dispute",
        "n_trials": N_TRIALS,
        "n_prompts": len(PROMPTS),
        "temperature": TEMPERATURE,
        "embedding_model": "all-MiniLM-L6-v2",
        "osf_registration": "https://osf.io/dp8nj/",
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

    print(f"\n  {model_name} COMPLETE | Mean dRCI: {mean_drci:.4f} +/- {std_drci:.4f} | {pattern}", flush=True)

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70, flush=True)
    print("PAPER 6 REPLICATION: LEGAL DOMAIN EXPERIMENTS", flush=True)
    print("Whistleblower Employment Dispute | OSF Pre-registered", flush=True)
    print("=" * 70, flush=True)
    print(f"Domain: Legal/Contractual (closed-goal)", flush=True)
    print(f"Prompts: {len(PROMPTS)}", flush=True)
    print(f"Trials per model: {N_TRIALS}", flush=True)
    print(f"Temperature: {TEMPERATURE}", flush=True)
    print(f"Embedding: all-MiniLM-L6-v2 (384D)", flush=True)
    print(f"Models: {len(MODELS_TO_RUN)}", flush=True)
    for m in MODELS_TO_RUN:
        print(f"  - {m[0]} ({m[1]})", flush=True)
    print(f"Output: {OUTPUT_DIR}", flush=True)
    print(f"OSF: https://osf.io/dp8nj/", flush=True)
    print("=" * 70, flush=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run each model sequentially
    for model_name, model_id, vendor in MODELS_TO_RUN:
        run_model(model_name, model_id, vendor)

    print("\n" + "=" * 70, flush=True)
    print("ALL MODELS COMPLETE — LEGAL DOMAIN", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
