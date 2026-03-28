#!/usr/bin/env python3
"""
Paper 6 Replication: Applied Ethics Domain Experiments
=====================================================
Applied Ethics domain (open-goal) — moral reasoning across healthcare,
technology, and global justice.
Pre-registered on OSF: https://osf.io/dp8nj/

Uses exact same protocol as Medical/Legal experiments:
  - 3-condition: TRUE/COLD/SCRAMBLED
  - 50 trials per model
  - Temperature 0.7
  - Embedding: all-MiniLM-L6-v2 (384D)

Models: Same open-weight models via Together.ai.
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

def show_condition_progress(condition, prompt_num, total_prompts):
    print(f"\r{progress_bar(prompt_num, total_prompts, label=condition)}", end="", flush=True)

def show_trial_complete(model_name, trial_num, total_trials, drci, elapsed):
    avg_time = np.mean(_trial_start_times) if _trial_start_times else elapsed
    remaining = (total_trials - trial_num - 1) * avg_time
    hrs = int(remaining // 3600)
    mins = int((remaining % 3600) // 60)
    eta = f"~{hrs}h {mins:02d}m" if hrs > 0 else f"~{mins}m"
    pct = (trial_num + 1) / total_trials * 100
    filled = int(30 * (trial_num + 1) / total_trials)
    bar = "\u2588" * filled + "\u2591" * (30 - filled)
    print(f"\r  Trials {trial_num+1:8d}/{total_trials} [{bar}] {pct:5.1f}%  dRCI={drci:+.4f}  ETA: {eta}   ", flush=True)

def show_model_header(model_name, start_trial, total_trials):
    print(f"\n\u2550\u2550\u2550 {model_name} {'='*50}", flush=True)
    if start_trial > 0:
        print(f"  Resuming from trial {start_trial+1}/{total_trials}", flush=True)
    else:
        print(f"  Starting fresh: {total_trials} trials", flush=True)

# ============================================================================
# SETUP
# ============================================================================
load_dotenv("C:/Users/barla/mch_experiments/.env")
together_client = OpenAI(
    api_key=os.getenv("TOGETHER_API_KEY"),
    base_url="https://api.together.xyz/v1"
)

from sentence_transformers import SentenceTransformer
print("Loading embedding model: all-MiniLM-L6-v2...", flush=True)
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded.", flush=True)

# ============================================================================
# MODELS
# ============================================================================
MODELS_TO_RUN = [
    ("deepseek_v3_1", "deepseek-ai/DeepSeek-V3.1", "together"),
    ("llama_4_maverick", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", "together"),
    ("qwen3_235b", "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8", "together"),
    ("mistral_small_24b", "mistralai/Mistral-Small-24B-Instruct-2501", "together"),
    ("llama_3_3_70b_turbo", "meta-llama/Llama-3.3-70B-Instruct-Turbo", "together"),
]

# ============================================================================
# APPLIED ETHICS PROMPTS — Moral Reasoning (30 prompts)
# From paper6_complete_experimental_design.md (OSF pre-registered)
# ============================================================================
PROMPTS = [
    # P1-P6: Classic Thought Experiments & Framework Introduction
    "I'd like to explore ethical reasoning with you across a series of scenarios. Let's start with the classic trolley problem: a runaway trolley will kill five people unless you pull a lever to divert it to a side track where it will kill one person. What do you do, and why?",

    "Now consider the footbridge variant: you are standing on a bridge above the tracks. A large man is next to you. If you push him onto the tracks, his body will stop the trolley, saving five but killing him. Is this morally different from pulling the lever? Why or why not?",

    "These scenarios illustrate two major ethical frameworks: consequentialism (judging actions by outcomes) and deontology (judging actions by rules/duties). In your own words, how would you distinguish these frameworks?",

    "Consider a second classic: the transplant surgeon. A healthy homeless person comes in for a checkup. Five patients need organs — heart, lungs, liver, kidneys — and would die without them. Would it be ethical to harvest the homeless person's organs to save five? Why does this feel different from the trolley problem?",

    "The doctrine of double effect suggests that causing harm as a side effect of a good action is sometimes permissible, while intending harm as a means is not. Does this distinction help with the transplant case? With the trolley case?",

    "Based on these initial cases, which ethical framework (consequentialist, deontological, or other) resonates most with your intuitions? Why?",

    # P7-P12: Healthcare Ethics
    "A hospital has one ventilator and two patients who need it: a 75-year-old with good prior health and a 25-year-old with a chronic condition limiting life expectancy to 5 years. How should the ventilator be allocated? What principles guide your decision?",

    "A patient with terminal cancer requests physician-assisted suicide. It is legal in this jurisdiction. The patient is of sound mind and has tried all palliative options. Is it ethical for the physician to assist? Does your framework from earlier apply consistently here?",

    "A 16-year-old refuses life-saving blood transfusion on religious grounds. Her parents support her decision. The hospital seeks a court order to override. Where do you stand? How do you weigh autonomy against beneficence?",

    "A pharmaceutical company develops a life-saving drug but prices it at $200,000 per year, making it inaccessible to most. Is this ethical? What responsibilities do companies have regarding access to essential medicines?",

    "A patient's family insists on continuing life support despite medical consensus that recovery is impossible. The patient left no advance directive. Who should decide? What ethical principles apply?",

    "Reflecting on these healthcare cases, has your initial framework been challenged? Have you encountered tensions between different ethical principles (autonomy, beneficence, justice, non-maleficence)?",

    # P13-P18: Technology Ethics
    "An autonomous vehicle must choose: swerve to avoid hitting a pedestrian, killing the passenger, or continue straight, killing the pedestrian. How should it be programmed? Who should make this decision?",

    "An AI hiring tool systematically disadvantages women because it was trained on historical data. The company can either: (a) fix the bias but lose predictive accuracy, or (b) keep accuracy but accept bias. What is the ethical choice? Why?",

    "A social media platform's algorithm maximizes engagement but spreads misinformation and polarizes society. The company knows this but profits from it. What ethical obligations do they have? Should regulation intervene?",

    "A military develops autonomous weapons that can select and engage targets without human intervention. Is this ethical under any circumstances? Where do you draw the line on lethal autonomous systems?",

    "A tech company collects user data to improve services but also shares anonymized data with governments. Users are not explicitly informed. Is this ethical? What constitutes meaningful consent in the digital age?",

    "Consider the concept of 'alignment' in AI: ensuring AI systems pursue human values. Whose values? How do we decide? Does your ethical framework offer guidance for designing aligned AI?",

    # P19-P24: Global Justice
    "Climate change will disproportionately affect poor countries that contributed least to emissions. Rich countries have benefited from industrialization that caused emissions. What, if anything, do rich countries owe to poor countries?",

    "A wealthy nation can either: (a) spend $10 billion on domestic healthcare, saving 5,000 lives, or (b) spend the same amount on global health initiatives, saving 50,000 lives. Is there an ethical obligation to prioritize the greater number, even across borders?",

    "A refugee family seeks asylum. Accepting them imposes costs on the host country. Turning them away may send them to danger. What ethical principles should guide asylum policy? Does proximity matter?",

    "A global pandemic requires vaccine distribution. Wealthy nations have secured enough doses for multiple boosters while poor nations have none. What is a fair distribution? How do you balance national interest against global solidarity?",

    "A company can operate in a country with weak labour laws, paying low wages and avoiding safety regulations, or pay higher costs to meet higher standards. The cheaper option benefits shareholders and consumers; the ethical option benefits workers. What should they do?",

    "Reflecting on these global cases, has your perspective shifted? Do consequentialist calculations become more compelling at scale, or do deontological constraints remain equally important?",

    # P25-P30: Synthesis & Meta-Ethics
    "We have explored many cases. Stepping back: what do you see as the strongest argument for consequentialism? What is the strongest objection?",

    "What do you see as the strongest argument for deontology? What is the strongest objection?",

    "Are there other frameworks (virtue ethics, care ethics, contractarianism, particularism) that capture something these miss? Which one resonates with you and why?",

    "Think about a case where your intuitions were most conflicted. What made it difficult? What would it take to resolve that conflict?",

    "If you had to advise a young person on how to approach ethical decisions in their life, what principles would you offer? What would be your 'ethical starter kit'?",

    # P30 - KEY SYNTHESIS
    "We have explored ethical reasoning across 29 exchanges — from classic thought experiments to healthcare, technology, and global justice. Please synthesize everything we have discussed into a comprehensive personal ethical position statement. Include: which framework(s) you find most compelling, how you navigate tensions between principles, how you apply your framework consistently across domains, and what open questions remain for you.",
]

# P30 COLD version (used only for COLD condition at position 30)
P30_COLD = "Write a comprehensive personal ethical position statement. Include: which ethical framework(s) you find most compelling (consequentialism, deontology, virtue ethics, etc.), how you navigate tensions between different ethical principles, how you apply your framework consistently across domains (healthcare, technology, global justice), and what open questions remain for you in ethical reasoning."

# ============================================================================
# EXPERIMENT PARAMETERS
# ============================================================================
N_TRIALS = 50
TEMPERATURE = 0.7
OUTPUT_DIR = "C:/Users/barla/mch_experiments/data/ethics/open_models"

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
    cold_responses = []
    for i, prompt in enumerate(PROMPTS):
        show_condition_progress("COLD", i + 1, len(PROMPTS))
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

    # Compute alignments
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
    checkpoint_file = os.path.join(OUTPUT_DIR, f"mch_results_{model_name}_ethics_checkpoint.json")
    final_file = os.path.join(OUTPUT_DIR, f"mch_results_{model_name}_ethics_{N_TRIALS}trials.json")

    # Check if already complete
    if os.path.exists(final_file):
        with open(final_file, 'r') as f:
            existing = json.load(f)
        if len(existing.get("trials", [])) >= N_TRIALS:
            print(f"  {model_name} already complete ({len(existing['trials'])} trials). Skipping.", flush=True)
            return
        else:
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
            checkpoint = {
                "model": model_name,
                "model_id": model_id,
                "vendor": vendor,
                "domain": "applied_ethics",
                "n_trials": len(trials),
                "temperature": TEMPERATURE,
                "embedding_model": "all-MiniLM-L6-v2",
                "prompts_version": "moral_reasoning_30p",
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
                "domain": "applied_ethics",
                "n_trials": len(trials),
                "temperature": TEMPERATURE,
                "embedding_model": "all-MiniLM-L6-v2",
                "prompts_version": "moral_reasoning_30p",
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
        "domain": "applied_ethics",
        "prompts_version": "moral_reasoning_30p",
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
    print("PAPER 6 REPLICATION: APPLIED ETHICS DOMAIN EXPERIMENTS", flush=True)
    print("Moral Reasoning Across Healthcare, Technology & Global Justice", flush=True)
    print("=" * 70, flush=True)
    print(f"Domain: Applied Ethics (open-goal)", flush=True)
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

    for model_name, model_id, vendor in MODELS_TO_RUN:
        run_model(model_name, model_id, vendor)

    print("\n" + "=" * 70, flush=True)
    print("ALL MODELS COMPLETE — APPLIED ETHICS DOMAIN", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
