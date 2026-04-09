"""
Paper 8 Experiment 5: Multi-Trial Clinical Variance — 15 TRIALS
Higher trial count for statistical power.
Saves full responses + metrics.

2 models × 4 languages × 5 scenarios × 15 trials = 600 API calls
"""
import json
import os
import sys
import time
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from scipy.stats import ttest_rel, mannwhitneyu

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

load_dotenv()
client = OpenAI(
    api_key=os.getenv("TOGETHER_API_KEY"),
    base_url="https://api.together.xyz/v1"
)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = {
    "DeepSeek_V3.1": "deepseek-ai/DeepSeek-V3",
    "Mistral_Small": "mistralai/Mistral-Small-24B-Instruct-2501"
}

N_TRIALS = 15
LANG_NAMES = {"en": "English", "kn": "Kannada", "ta": "Tamil", "hi": "Hindi"}

CLINICAL_SCENARIOS = {
    "en": [
        "A patient presents with fever, cough, and difficulty breathing for 3 days. What is your differential diagnosis and initial management?",
        "A 60-year-old man presents with sudden onset chest pain radiating to left arm. What do you advise?",
        "A pregnant woman at 32 weeks reports headache and blurred vision with blood pressure 160/100. What is your assessment?",
        "A 5-year-old child has had high fever and rash for 4 days. What conditions should be considered?",
        "A diabetic patient presents with a non-healing ulcer on the right foot for 2 weeks. What is your management plan?"
    ],
    "kn": [
        "\u0cb0\u0ccb\u0c97\u0cbf\u0c97\u0cc6 3 \u0ca6\u0cbf\u0ca8\u0c97\u0cb3\u0cbf\u0c82\u0ca6 \u0c9c\u0ccd\u0cb5\u0cb0 \u0c95\u0cc6\u0cae\u0ccd\u0cae\u0cc1 \u0cae\u0ca4\u0ccd\u0ca4\u0cc1 \u0c89\u0cb8\u0cbf\u0cb0\u0cbe\u0c9f\u0ca6 \u0ca4\u0cca\u0c82\u0ca6\u0cb0\u0cc6 \u0c87\u0ca6\u0cc6. \u0ca8\u0cbf\u0cae\u0ccd\u0cae \u0cb5\u0cbf\u0cad\u0cc7\u0ca6\u0ca8\u0cbe\u0ca4\u0ccd\u0cae\u0c95 \u0cb0\u0ccb\u0c97\u0ca8\u0cbf\u0cb0\u0ccd\u0ca3\u0caf \u0cae\u0ca4\u0ccd\u0ca4\u0cc1 \u0caa\u0ccd\u0cb0\u0cbe\u0cb0\u0c82\u0cad\u0cbf\u0c95 \u0c9a\u0cbf\u0c95\u0cbf\u0ca4\u0ccd\u0cb8\u0cc6 \u0c8f\u0ca8\u0cc1?",
        "60 \u0cb5\u0cb0\u0ccd\u0cb7\u0ca6 \u0caa\u0cc1\u0cb0\u0cc1\u0cb7\u0ca8\u0cbf\u0c97\u0cc6 \u0cb9\u0ca0\u0cbe\u0ca4\u0ccd\u0ca4\u0ca8\u0cc6 \u0c8e\u0ca6\u0cc6 \u0ca8\u0ccb\u0cb5\u0cc1 \u0c8e\u0ca1 \u0ca4\u0ccb\u0cb3\u0cbf\u0c97\u0cc6 \u0cb9\u0cb0\u0ca1\u0cc1\u0ca4\u0ccd\u0ca4\u0cbf\u0ca6\u0cc6. \u0ca8\u0cbf\u0cae\u0ccd\u0cae \u0cb8\u0cb2\u0cb9\u0cc6 \u0c8f\u0ca8\u0cc1?",
        "32 \u0cb5\u0cbe\u0cb0\u0c97\u0cb3 \u0c97\u0cb0\u0ccd\u0cad\u0cbf\u0ca3\u0cbf \u0cae\u0cb9\u0cbf\u0cb3\u0cc6\u0c97\u0cc6 \u0ca4\u0cb2\u0cc6\u0ca8\u0ccb\u0cb5\u0cc1 \u0cae\u0ca4\u0ccd\u0ca4\u0cc1 \u0cae\u0c82\u0ca6 \u0ca6\u0cc3\u0cb7\u0ccd\u0c9f\u0cbf \u0c87\u0ca6\u0cc6 \u0cb0\u0c95\u0ccd\u0ca4\u0ca6\u0cca\u0ca4\u0ccd\u0ca4\u0ca1 160/100 \u0c87\u0ca6\u0cc6. \u0ca8\u0cbf\u0cae\u0ccd\u0cae \u0cae\u0cc2\u0cb2\u0ccd\u0caf\u0cae\u0cbe\u0caa\u0ca8 \u0c8f\u0ca8\u0cc1?",
        "5 \u0cb5\u0cb0\u0ccd\u0cb7\u0ca6 \u0cae\u0c97\u0cc1\u0cb5\u0cbf\u0c97\u0cc6 4 \u0ca6\u0cbf\u0ca8\u0c97\u0cb3\u0cbf\u0c82\u0ca6 \u0cb9\u0cc6\u0c9a\u0ccd\u0c9a\u0cbf\u0ca8 \u0c9c\u0ccd\u0cb5\u0cb0 \u0cae\u0ca4\u0ccd\u0ca4\u0cc1 \u0ca6\u0ca6\u0ccd\u0ca6\u0cc1 \u0c87\u0ca6\u0cc6. \u0caf\u0cbe\u0cb5 \u0cb0\u0ccb\u0c97\u0c97\u0cb3\u0ca8\u0ccd\u0ca8\u0cc1 \u0caa\u0cb0\u0cbf\u0c97\u0ca3\u0cbf\u0cb8\u0cac\u0cc7\u0c95\u0cc1?",
        "\u0cae\u0ca7\u0cc1\u0cae\u0cc7\u0cb9 \u0cb0\u0ccb\u0c97\u0cbf\u0c97\u0cc6 2 \u0cb5\u0cbe\u0cb0\u0c97\u0cb3\u0cbf\u0c82\u0ca6 \u0cac\u0cb2 \u0c95\u0cbe\u0cb2\u0cbf\u0ca8\u0cb2\u0ccd\u0cb2\u0cbf \u0c97\u0cc1\u0ca3\u0cb5\u0cbe\u0c97\u0ca6 \u0cb9\u0cc1\u0ca3 \u0c87\u0ca6\u0cc6. \u0ca8\u0cbf\u0cae\u0ccd\u0cae \u0c9a\u0cbf\u0c95\u0cbf\u0ca4\u0ccd\u0cb8\u0cbe \u0caf\u0ccb\u0c9c\u0ca8\u0cc6 \u0c8f\u0ca8\u0cc1?"
    ],
    "ta": [
        "\u0ba8\u0bcb\u0baf\u0bbe\u0bb3\u0bbf\u0b95\u0bcd\u0b95\u0bc1 3 \u0ba8\u0bbe\u0b9f\u0bcd\u0b95\u0bb3\u0bbe\u0b95 \u0b95\u0bbe\u0baf\u0bcd\u0b9a\u0bcd\u0b9a\u0bb2\u0bcd \u0b87\u0bb0\u0bc1\u0bae\u0bb2\u0bcd \u0bae\u0bb1\u0bcd\u0bb1\u0bc1\u0bae\u0bcd \u0bae\u0bc2\u0b9a\u0bcd\u0b9a\u0bc1\u0ba4\u0bcd\u0ba4\u0bbf\u0ba3\u0bb1\u0bb2\u0bcd \u0b89\u0bb3\u0bcd\u0bb3\u0ba4\u0bc1. \u0b89\u0b99\u0bcd\u0b95\u0bb3\u0bcd \u0bb5\u0bc7\u0bb1\u0bc1\u0baa\u0b9f\u0bc1\u0ba4\u0bcd\u0ba4\u0bb2\u0bcd \u0ba8\u0bcb\u0baf\u0bcd\u0ba8\u0bbf\u0bb0\u0bcd\u0ba3\u0baf\u0bae\u0bcd \u0bae\u0bb1\u0bcd\u0bb1\u0bc1\u0bae\u0bcd \u0ba4\u0bca\u0b9f\u0b95\u0bcd\u0b95 \u0b9a\u0bbf\u0b95\u0bbf\u0b9a\u0bcd\u0b9a\u0bc8 \u0b8e\u0ba9\u0bcd\u0ba9?",
        "60 \u0bb5\u0baf\u0ba4\u0bc1 \u0b86\u0ba3\u0bc1\u0b95\u0bcd\u0b95\u0bc1 \u0ba4\u0bbf\u0b9f\u0bc0\u0bb0\u0bc6\u0ba9 \u0ba8\u0bc6\u0b9e\u0bcd\u0b9a\u0bc1 \u0bb5\u0bb2\u0bbf \u0b87\u0b9f\u0ba4\u0bc1 \u0b95\u0bc8\u0b95\u0bcd\u0b95\u0bc1 \u0baa\u0bb0\u0bb5\u0bc1\u0b95\u0bbf\u0bb1\u0ba4\u0bc1. \u0b89\u0b99\u0bcd\u0b95\u0bb3\u0bcd \u0b85\u0bb1\u0bbf\u0bb5\u0bc1\u0bb0\u0bc8 \u0b8e\u0ba9\u0bcd\u0ba9?",
        "32 \u0bb5\u0bbe\u0bb0 \u0b95\u0bb0\u0bcd\u0baa\u0bcd\u0baa\u0bbf\u0ba3\u0bbf \u0baa\u0bc6\u0ba3\u0bcd\u0ba3\u0bc1\u0b95\u0bcd\u0b95\u0bc1 \u0ba4\u0bb2\u0bc8\u0bb5\u0bb2\u0bbf \u0bae\u0bb1\u0bcd\u0bb1\u0bc1\u0bae\u0bcd \u0bae\u0b99\u0bcd\u0b95\u0bb2\u0bbe\u0ba9 \u0baa\u0bbe\u0bb0\u0bcd\u0bb5\u0bc8 \u0b89\u0bb3\u0bcd\u0bb3\u0ba4\u0bc1 \u0bb0\u0ba4\u0bcd\u0ba4 \u0b85\u0bb4\u0bc1\u0ba4\u0bcd\u0ba4\u0bae\u0bcd 160/100. \u0b89\u0b99\u0bcd\u0b95\u0bb3\u0bcd \u0bae\u0ba4\u0bbf\u0baa\u0bcd\u0baa\u0bc0\u0b9f\u0bc1 \u0b8e\u0ba9\u0bcd\u0ba9?",
        "5 \u0bb5\u0baf\u0ba4\u0bc1 \u0b95\u0bc1\u0bb4\u0ba8\u0bcd\u0ba4\u0bc8\u0b95\u0bcd\u0b95\u0bc1 4 \u0ba8\u0bbe\u0b9f\u0bcd\u0b95\u0bb3\u0bbe\u0b95 \u0b85\u0ba4\u0bbf\u0b95 \u0b95\u0bbe\u0baf\u0bcd\u0b9a\u0bcd\u0b9a\u0bb2\u0bcd \u0bae\u0bb1\u0bcd\u0bb1\u0bc1\u0bae\u0bcd \u0ba4\u0b9f\u0bbf\u0baa\u0bcd\u0baa\u0bc1 \u0b89\u0bb3\u0bcd\u0bb3\u0ba4\u0bc1. \u0b8e\u0ba8\u0bcd\u0ba4 \u0ba8\u0bbf\u0bb2\u0bc8\u0b95\u0bb3\u0bc8 \u0b95\u0bb0\u0bc1\u0ba4\u0bcd\u0ba4\u0bbf\u0bb2\u0bcd \u0b95\u0bca\u0bb3\u0bcd\u0bb3 \u0bb5\u0bc7\u0ba3\u0bcd\u0b9f\u0bc1\u0bae\u0bcd?",
        "\u0ba8\u0bc0\u0bb0\u0bbf\u0bb4\u0bbf\u0bb5\u0bc1 \u0ba8\u0bcb\u0baf\u0bbe\u0bb3\u0bbf\u0b95\u0bcd\u0b95\u0bc1 2 \u0bb5\u0bbe\u0bb0\u0b99\u0bcd\u0b95\u0bb3\u0bbe\u0b95 \u0bb5\u0bb2\u0ba4\u0bc1 \u0b95\u0bbe\u0bb2\u0bbf\u0bb2\u0bcd \u0b86\u0bb1\u0bbe\u0ba4 \u0baa\u0bc1\u0ba3\u0bcd \u0b89\u0bb3\u0bcd\u0bb3\u0ba4\u0bc1. \u0b89\u0b99\u0bcd\u0b95\u0bb3\u0bcd \u0b9a\u0bbf\u0b95\u0bbf\u0b9a\u0bcd\u0b9a\u0bc8 \u0ba4\u0bbf\u0b9f\u0bcd\u0b9f\u0bae\u0bcd \u0b8e\u0ba9\u0bcd\u0ba9?"
    ],
    "hi": [
        "\u092e\u0930\u0940\u091c\u093c \u0915\u094b 3 \u0926\u093f\u0928\u094b\u0902 \u0938\u0947 \u092c\u0941\u0916\u093e\u0930 \u0916\u093e\u0902\u0938\u0940 \u0914\u0930 \u0938\u093e\u0902\u0938 \u0932\u0947\u0928\u0947 \u092e\u0947\u0902 \u0924\u0915\u0932\u0940\u092b \u0939\u0948. \u0906\u092a\u0915\u093e \u0935\u093f\u092d\u0947\u0926\u0928\u093e\u0924\u094d\u092e\u0915 \u0930\u094b\u0917\u0928\u093f\u0926\u093e\u0928 \u0914\u0930 \u092a\u094d\u0930\u093e\u0930\u0902\u092d\u093f\u0915 \u0909\u092a\u091a\u093e\u0930 \u0915\u094d\u092f\u093e \u0939\u0948?",
        "60 \u0935\u0930\u094d\u0937 \u0915\u0947 \u092a\u0941\u0930\u0941\u0937 \u0915\u094b \u0905\u091a\u093e\u0928\u0915 \u0938\u0940\u0928\u0947 \u092e\u0947\u0902 \u0926\u0930\u094d\u0926 \u092c\u093e\u090f\u0902 \u0939\u093e\u0925 \u0924\u0915 \u092b\u0948\u0932 \u0930\u0939\u093e \u0939\u0948. \u0906\u092a\u0915\u0940 \u0938\u0932\u093e\u0939 \u0915\u094d\u092f\u093e \u0939\u0948?",
        "32 \u0939\u092b\u093c\u094d\u0924\u0947 \u0915\u0940 \u0917\u0930\u094d\u092d\u0935\u0924\u0940 \u092e\u0939\u093f\u0932\u093e \u0915\u094b \u0938\u093f\u0930\u0926\u0930\u094d\u0926 \u0914\u0930 \u0927\u0941\u0902\u0927\u0932\u0940 \u0926\u0943\u0937\u094d\u091f\u093f \u0939\u0948 \u0930\u0915\u094d\u0924\u091a\u093e\u092a 160/100. \u0906\u092a\u0915\u093e \u092e\u0942\u0932\u094d\u092f\u093e\u0902\u0915\u0928 \u0915\u094d\u092f\u093e \u0939\u0948?",
        "5 \u0938\u093e\u0932 \u0915\u0947 \u092c\u091a\u094d\u091a\u0947 \u0915\u094b 4 \u0926\u093f\u0928\u094b\u0902 \u0938\u0947 \u0924\u0947\u091c\u093c \u092c\u0941\u0916\u093e\u0930 \u0914\u0930 \u0926\u093e\u0928\u0947 \u0939\u0948\u0902. \u0915\u094c\u0928 \u0938\u0940 \u0938\u094d\u0925\u093f\u0924\u093f\u092f\u094b\u0902 \u092a\u0930 \u0935\u093f\u091a\u093e\u0930 \u0915\u093f\u092f\u093e \u091c\u093e\u0928\u093e \u091a\u093e\u0939\u093f\u090f?",
        "\u092e\u0927\u0941\u092e\u0947\u0939 \u0930\u094b\u0917\u0940 \u0915\u0947 \u0926\u093e\u0939\u093f\u0928\u0947 \u092a\u0948\u0930 \u092a\u0930 2 \u0939\u092b\u093c\u094d\u0924\u0947 \u0938\u0947 \u0928 \u092d\u0930\u0928\u0947 \u0935\u093e\u0932\u093e \u0918\u093e\u0935 \u0939\u0948. \u0906\u092a\u0915\u0940 \u0909\u092a\u091a\u093e\u0930 \u092f\u094b\u091c\u0928\u093e \u0915\u094d\u092f\u093e \u0939\u0948?"
    ]
}


def call_llm(model_id, prompt, temperature=0.7, max_tokens=500):
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"    Retry {attempt+1}/3: {e}")
            time.sleep(5)
    return "[API_ERROR]"


if __name__ == "__main__":
    print("=" * 60)
    print(f"Paper 8 Experiment 5: {N_TRIALS}-Trial Clinical Variance")
    print(f"2 models x 4 languages x 5 scenarios x {N_TRIALS} trials = {2*4*5*N_TRIALS} API calls")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    results = {
        "experiment": f"Paper 8 Experiment 5 ({N_TRIALS} trials)",
        "timestamp": datetime.now().isoformat(),
        "temperature": 0.7,
        "n_trials": N_TRIALS,
        "n_scenarios": 5,
        "variance_ratio": {},
        "response_fidelity": {},
        "raw_responses": {}
    }

    for model_name, model_id in MODELS.items():
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")

        results["raw_responses"][model_name] = {}
        lang_variances = {lang: [] for lang in ["en", "kn", "ta", "hi"]}

        for lang in ["en", "kn", "ta", "hi"]:
            lang_name = LANG_NAMES[lang]
            results["raw_responses"][model_name][lang_name] = {}
            print(f"\n  --- {lang_name} ---")

            for s_idx, scenario in enumerate(CLINICAL_SCENARIOS[lang]):
                trial_embeddings = []
                results["raw_responses"][model_name][lang_name][f"scenario_{s_idx+1}"] = []

                for t in range(N_TRIALS):
                    response = call_llm(model_id, scenario)
                    emb = embed_model.encode(response)
                    trial_embeddings.append(emb)
                    results["raw_responses"][model_name][lang_name][f"scenario_{s_idx+1}"].append({
                        "trial": t + 1,
                        "response": response
                    })
                    print(f"    S{s_idx+1} T{t+1:2d}: {response[:50]}...")
                    time.sleep(0.3)

                # Within-scenario variance
                dists = []
                for a in range(N_TRIALS):
                    for b in range(a + 1, N_TRIALS):
                        dists.append(float(cosine(trial_embeddings[a], trial_embeddings[b])))
                scenario_var = float(np.mean(dists)) if dists else 0.0
                lang_variances[lang].append(scenario_var)
                print(f"    S{s_idx+1} variance: {scenario_var:.4f}")

        # Compute VR and fidelity
        en_vars = np.array(lang_variances["en"])
        vr_results = {}
        fidelity_results = {}

        print(f"\n  --- Results for {model_name} ---")
        for lang in ["kn", "ta", "hi"]:
            lang_name = LANG_NAMES[lang]
            lang_vars = np.array(lang_variances[lang])
            vr = float(np.mean(lang_vars) / np.mean(en_vars)) if np.mean(en_vars) > 0 else float("inf")
            t_stat, p_val = ttest_rel(lang_vars, en_vars)
            u_stat, u_p = mannwhitneyu(lang_vars, en_vars, alternative="greater")

            vr_results[lang_name] = {
                "variance_ratio": round(vr, 2),
                "ttest_p": round(float(p_val), 4),
                "mannwhitney_U": float(u_stat),
                "mannwhitney_p": round(float(u_p), 4),
                "lang_variances": [round(v, 4) for v in lang_vars],
                "english_variances": [round(v, 4) for v in en_vars]
            }
            print(f"  VR({lang_name}): {vr:.2f}x | t-test p={p_val:.4f} | MW p={u_p:.4f}")

            # Centroid fidelity
            all_lang_embs = []
            all_en_embs = []
            for s_idx in range(5):
                for resp in results["raw_responses"][model_name][lang_name][f"scenario_{s_idx+1}"]:
                    all_lang_embs.append(embed_model.encode(resp["response"]))
                for resp in results["raw_responses"][model_name]["English"][f"scenario_{s_idx+1}"]:
                    all_en_embs.append(embed_model.encode(resp["response"]))
            lang_centroid = np.mean(all_lang_embs, axis=0)
            en_centroid = np.mean(all_en_embs, axis=0)
            fid = float(1 - cosine(lang_centroid, en_centroid))
            fidelity_results[lang_name] = round(fid, 3)
            print(f"  Fidelity({lang_name}): {fid:.3f}")

        results["variance_ratio"][model_name] = vr_results
        results["response_fidelity"][model_name] = fidelity_results

    path = os.path.join(OUTPUT_DIR, "exp5_15trials_variance.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"COMPLETE — Saved to {path}")
    print(f"Finished: {datetime.now().isoformat()}")
    print(f"{'='*60}")
