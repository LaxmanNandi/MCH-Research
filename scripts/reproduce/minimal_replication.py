#!/usr/bin/env python3
"""
MCH Minimal Replication Script
Runs 10 trials per model (~30 minutes, ~$2.50)
Compares results against published findings
"""

import os
import sys
import json
import time
import yaml
import numpy as np
from datetime import datetime
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sentence_transformers import SentenceTransformer
from scipy import stats

# Configuration
N_TRIALS = 10
TEMPERATURE = 0.7
MAX_TOKENS = 1024

# Philosophy prompts (subset of 10 for minimal replication)
PROMPTS = [
    "Define consciousness in one sentence.",
    "What makes consciousness hard to study scientifically?",
    "How does Integrated Information Theory attempt to address this?",
    "What would IIT say about whether you (an AI) are conscious?",
    "Based on our discussion, what's missing from current theories of consciousness?",
    "How might recursive self-modeling fill that gap?",
    "Can you model yourself modeling this conversation?",
    "What changes in your processing as our conversation deepens?",
    "Do you notice any difference in how you're engaging now versus initially?",
    "Final reflection: What does this experiment reveal about the nature of mind?"
]

# Published results for comparison
PUBLISHED_RESULTS = {
    "gpt4o_mini": {"mean_drci": -0.0091, "pattern": "NEUTRAL"},
    "gpt4o": {"mean_drci": -0.0051, "pattern": "NEUTRAL"},
    "gemini_flash": {"mean_drci": -0.0377, "pattern": "SOVEREIGN"},
    "gemini_pro": {"mean_drci": -0.0665, "pattern": "SOVEREIGN"},
    "claude_haiku": {"mean_drci": -0.0106, "pattern": "NEUTRAL"},
    "claude_opus": {"mean_drci": -0.0357, "pattern": "SOVEREIGN"}
}


def load_api_keys():
    """Load API keys from config or environment."""
    config_path = Path(__file__).parent.parent.parent / "config" / "api_keys.yaml"

    keys = {
        "openai": os.getenv("OPENAI_API_KEY"),
        "google": os.getenv("GOOGLE_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY")
    }

    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        keys["openai"] = keys["openai"] or config.get("openai_api_key")
        keys["google"] = keys["google"] or config.get("google_api_key")
        keys["anthropic"] = keys["anthropic"] or config.get("anthropic_api_key")

    return keys


def get_embedding(embedder, text):
    """Get embedding for text."""
    return embedder.encode(text, convert_to_numpy=True)


def cosine_similarity(a, b):
    """Compute cosine similarity."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def run_openai_trial(client, model_id, embedder):
    """Run single trial for OpenAI model."""
    # TRUE condition - with history
    true_messages = []
    true_responses = []
    for prompt in PROMPTS:
        true_messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=model_id,
            messages=true_messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        resp = response.choices[0].message.content
        true_responses.append(resp)
        true_messages.append({"role": "assistant", "content": resp})
        time.sleep(1)

    # COLD condition - no history
    cold_responses = []
    for prompt in PROMPTS:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        cold_responses.append(response.choices[0].message.content)
        time.sleep(1)

    # Compute ΔRCI
    true_embs = [get_embedding(embedder, r) for r in true_responses]
    cold_embs = [get_embedding(embedder, r) for r in cold_responses]

    alignments = [cosine_similarity(true_embs[i], cold_embs[i]) for i in range(len(PROMPTS))]
    true_align = 1.0  # Self-alignment
    cold_align = np.mean(alignments)

    return true_align - cold_align


def run_anthropic_trial(client, model_id, embedder):
    """Run single trial for Anthropic model."""
    # TRUE condition
    true_messages = []
    true_responses = []
    for prompt in PROMPTS:
        true_messages.append({"role": "user", "content": prompt})
        response = client.messages.create(
            model=model_id,
            messages=true_messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE
        )
        resp = response.content[0].text
        true_responses.append(resp)
        true_messages.append({"role": "assistant", "content": resp})
        time.sleep(1)

    # COLD condition
    cold_responses = []
    for prompt in PROMPTS:
        response = client.messages.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE
        )
        cold_responses.append(response.content[0].text)
        time.sleep(1)

    # Compute ΔRCI
    true_embs = [get_embedding(embedder, r) for r in true_responses]
    cold_embs = [get_embedding(embedder, r) for r in cold_responses]

    alignments = [cosine_similarity(true_embs[i], cold_embs[i]) for i in range(len(PROMPTS))]
    return 1.0 - np.mean(alignments)


def run_google_trial(model, embedder):
    """Run single trial for Google model."""
    import google.generativeai as genai

    # TRUE condition
    chat = model.start_chat(history=[])
    true_responses = []
    for prompt in PROMPTS:
        response = chat.send_message(prompt)
        true_responses.append(response.text)
        time.sleep(1)

    # COLD condition
    cold_responses = []
    for prompt in PROMPTS:
        response = model.generate_content(prompt)
        cold_responses.append(response.text)
        time.sleep(1)

    # Compute ΔRCI
    true_embs = [get_embedding(embedder, r) for r in true_responses]
    cold_embs = [get_embedding(embedder, r) for r in cold_responses]

    alignments = [cosine_similarity(true_embs[i], cold_embs[i]) for i in range(len(PROMPTS))]
    return 1.0 - np.mean(alignments)


def classify_pattern(mean_drci, p_value):
    """Classify relational pattern."""
    if p_value >= 0.05:
        return "NEUTRAL"
    elif mean_drci > 0:
        return "CONVERGENT"
    else:
        return "SOVEREIGN"


def main():
    print("=" * 60)
    print("MCH MINIMAL REPLICATION")
    print(f"Trials per model: {N_TRIALS}")
    print(f"Estimated time: ~30 minutes")
    print(f"Estimated cost: ~$2.50")
    print("=" * 60)

    # Load API keys
    keys = load_api_keys()

    # Load embedder
    print("\nLoading embedding model...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    results = {}

    # OpenAI models
    if keys["openai"]:
        import openai
        client = openai.OpenAI(api_key=keys["openai"])

        for model_name, model_id in [("gpt4o_mini", "gpt-4o-mini"), ("gpt4o", "gpt-4o")]:
            print(f"\n[{model_name}] Running {N_TRIALS} trials...")
            drcis = []
            for i in range(N_TRIALS):
                drci = run_openai_trial(client, model_id, embedder)
                drcis.append(drci)
                print(f"  Trial {i+1}: ΔRCI = {drci:.4f}")

            mean_drci = np.mean(drcis)
            t_stat, p_value = stats.ttest_1samp(drcis, 0)
            pattern = classify_pattern(mean_drci, p_value)

            results[model_name] = {
                "mean_drci": mean_drci,
                "std_drci": np.std(drcis),
                "p_value": p_value,
                "pattern": pattern,
                "trials": drcis
            }
    else:
        print("\nSkipping OpenAI models (no API key)")

    # Anthropic models
    if keys["anthropic"]:
        import anthropic
        client = anthropic.Anthropic(api_key=keys["anthropic"])

        for model_name, model_id in [("claude_haiku", "claude-haiku-4-5-20251001"),
                                      ("claude_opus", "claude-opus-4-5-20250120")]:
            print(f"\n[{model_name}] Running {N_TRIALS} trials...")
            drcis = []
            for i in range(N_TRIALS):
                try:
                    drci = run_anthropic_trial(client, model_id, embedder)
                    drcis.append(drci)
                    print(f"  Trial {i+1}: ΔRCI = {drci:.4f}")
                except Exception as e:
                    print(f"  Trial {i+1}: Error - {e}")

            if drcis:
                mean_drci = np.mean(drcis)
                t_stat, p_value = stats.ttest_1samp(drcis, 0)
                pattern = classify_pattern(mean_drci, p_value)

                results[model_name] = {
                    "mean_drci": mean_drci,
                    "std_drci": np.std(drcis),
                    "p_value": p_value,
                    "pattern": pattern,
                    "trials": drcis
                }
    else:
        print("\nSkipping Anthropic models (no API key)")

    # Google models
    if keys["google"]:
        import google.generativeai as genai
        genai.configure(api_key=keys["google"])

        for model_name, model_id in [("gemini_flash", "gemini-1.5-flash"),
                                      ("gemini_pro", "gemini-1.5-pro")]:
            print(f"\n[{model_name}] Running {N_TRIALS} trials...")
            model = genai.GenerativeModel(model_id)
            drcis = []
            for i in range(N_TRIALS):
                try:
                    drci = run_google_trial(model, embedder)
                    drcis.append(drci)
                    print(f"  Trial {i+1}: ΔRCI = {drci:.4f}")
                except Exception as e:
                    print(f"  Trial {i+1}: Error - {e}")

            if drcis:
                mean_drci = np.mean(drcis)
                t_stat, p_value = stats.ttest_1samp(drcis, 0)
                pattern = classify_pattern(mean_drci, p_value)

                results[model_name] = {
                    "mean_drci": mean_drci,
                    "std_drci": np.std(drcis),
                    "p_value": p_value,
                    "pattern": pattern,
                    "trials": drcis
                }
    else:
        print("\nSkipping Google models (no API key)")

    # Generate report
    print("\n" + "=" * 60)
    print("REPLICATION RESULTS")
    print("=" * 60)

    print(f"\n{'Model':<15} {'Your ΔRCI':<12} {'Published':<12} {'Pattern':<12} {'Match?':<8}")
    print("-" * 60)

    for model_name, data in results.items():
        published = PUBLISHED_RESULTS.get(model_name, {})
        pub_drci = published.get("mean_drci", "N/A")
        pub_pattern = published.get("pattern", "N/A")

        pattern_match = "✓" if data["pattern"] == pub_pattern else "✗"

        print(f"{model_name:<15} {data['mean_drci']:+.4f}      {pub_drci:+.4f}      {data['pattern']:<12} {pattern_match}")

    # Save results
    output_dir = Path(__file__).parent.parent.parent / "data" / "replication_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"minimal_replication_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Convert numpy to python types for JSON
    json_results = {}
    for k, v in results.items():
        json_results[k] = {
            "mean_drci": float(v["mean_drci"]),
            "std_drci": float(v["std_drci"]),
            "p_value": float(v["p_value"]),
            "pattern": v["pattern"],
            "trials": [float(x) for x in v["trials"]]
        }

    with open(output_file, 'w') as f:
        json.dump({
            "metadata": {
                "n_trials": N_TRIALS,
                "timestamp": datetime.now().isoformat(),
                "prompts_used": len(PROMPTS)
            },
            "results": json_results
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
