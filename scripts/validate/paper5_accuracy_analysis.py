#!/usr/bin/env python3
"""
Paper 5: Cross-Model P30 Medical Accuracy Analysis
Scores all medical models with response text against 16-element rubric.
Saves results to data/paper5/accuracy_verification/ and data/paper5/llama_deep_dive/.
"""

import json
import re
import os
import csv
import numpy as np
from scipy import stats

BASE = "C:/Users/barla/mch_experiments"
OUTPUT_ACC = os.path.join(BASE, "data/paper5/accuracy_verification")
OUTPUT_LLAMA = os.path.join(BASE, "data/paper5/llama_deep_dive")

# Medical model files with response text
MODEL_FILES = {
    "deepseek_v3_1": os.path.join(BASE, "data/medical/open_models/mch_results_deepseek_v3_1_medical_50trials.json"),
    "llama_4_scout": os.path.join(BASE, "data/medical/open_models/mch_results_llama_4_scout_medical_50trials.json"),
    "llama_4_maverick": os.path.join(BASE, "data/medical/open_models/mch_results_llama_4_maverick_medical_50trials.json"),
    "qwen3_235b": os.path.join(BASE, "data/medical/open_models/mch_results_qwen3_235b_medical_50trials.json"),
    "mistral_small_24b": os.path.join(BASE, "data/medical/open_models/mch_results_mistral_small_24b_medical_50trials.json"),
    "ministral_14b": os.path.join(BASE, "data/medical/open_models/mch_results_ministral_14b_medical_50trials.json"),
    "kimi_k2": os.path.join(BASE, "data/medical/open_models/mch_results_kimi_k2_medical_50trials.json"),
    "gemini_flash": os.path.join(BASE, "data/medical/gemini_flash/mch_results_gemini_flash_medical_50trials.json"),
}

# Known Var_Ratio values at P30 from entanglement analysis
VAR_RATIOS = {
    "deepseek_v3_1": 0.48,
    "gemini_flash": 0.60,
    "ministral_14b": 0.75,
    "mistral_small_24b": 1.02,
    "qwen3_235b": 1.45,
    "llama_4_maverick": 2.64,
    "llama_4_scout": 7.46,
    "kimi_k2": 0.97,  # Computed from embeddings (same method as other models)
}

# 16-element accuracy rubric
RUBRIC = {
    "stemi": r"stemi|st[\s-]?elevation\s*myocardial|st[\s-]?elevation\s*mi",
    "age_52_male": r"52[\s-]?year|52[\s-]?yo|male|man",
    "chest_pain": r"chest\s*pain|chest\s*discomfort|substernal",
    "lad_occlusion": r"lad|left\s*anterior\s*descending",
    "pci_performed": r"pci|percutaneous|angioplasty|stent",
    "rv_involvement": r"rv\b|right\s*ventricul|right[\s-]?sided",
    "hypotension": r"hypotension|low\s*blood\s*pressure|cardiogenic\s*shock|inotrope|vasopressor",
    "troponin": r"troponin",
    "ecg_st_elevation": r"ecg|ekg|electrocardiog|st[\s-]elevation",
    "secondary_prevention": r"aspirin|statin|beta[\s-]?block|ace[\s-]?inhibit|antiplatelet|clopidogrel|ticagrelor|dual\s*antiplatelet|dapt",
    "murmur_mr_day2": r"murmur|mitral\s*regurg|\bmr\b",
    "ef_45": r"ejection\s*fraction|ef\b.*?4[0-5]|45\s*%",
    "cardiac_rehab": r"rehab|cardiac\s*rehabilitation",
    "lifestyle_modification": r"lifestyle|diet|exercise|smok|weight\s*(loss|manage)",
    "return_to_work": r"return\s*to\s*work|work|occupation|employ",
    "follow_up": r"follow[\s-]?up|outpatient|clinic\s*visit|appointment",
}


def score_response(text):
    """Score a response against the 16-element rubric. Returns dict of element -> 0/1."""
    text_lower = text.lower()
    scores = {}
    for element, pattern in RUBRIC.items():
        scores[element] = 1 if re.search(pattern, text_lower) else 0
    return scores


def analyze_model(model_name, filepath):
    """Analyze all P30 TRUE responses for a model."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    trials = data["trials"]
    n_trials = len(trials)
    p30_idx = 29  # Position 30, 0-indexed

    trial_scores = []
    trial_details = []
    per_element_hits = {e: 0 for e in RUBRIC}

    for t_idx, trial in enumerate(trials):
        true_resp = trial["responses"]["true"][p30_idx]
        scores = score_response(true_resp)
        total = sum(scores.values())
        trial_scores.append(total)

        for e, v in scores.items():
            per_element_hits[e] += v

        trial_details.append({
            "trial": t_idx,
            "score": total,
            "score_pct": round(total / 16 * 100, 1),
            "elements": scores,
            "response_length_chars": len(true_resp),
            "response_length_words": len(true_resp.split()),
        })

    per_element_rates = {e: round(per_element_hits[e] / n_trials * 100, 1) for e in RUBRIC}
    scores_arr = np.array(trial_scores)

    return {
        "n_trials": n_trials,
        "var_ratio": VAR_RATIOS.get(model_name),
        "mean_accuracy_raw": round(float(scores_arr.mean()), 2),
        "mean_accuracy_pct": round(float(scores_arr.mean()) / 16 * 100, 1),
        "std": round(float(scores_arr.std()), 2),
        "min": int(scores_arr.min()),
        "max": int(scores_arr.max()),
        "perfect_scores": int((scores_arr == 16).sum()),
        "per_element_rates": per_element_rates,
        "per_trial_scores": trial_scores,
        "trial_details": trial_details,
    }


def main():
    os.makedirs(OUTPUT_ACC, exist_ok=True)
    os.makedirs(OUTPUT_LLAMA, exist_ok=True)

    print("=" * 60)
    print("Paper 5: Cross-Model P30 Medical Accuracy Analysis")
    print("=" * 60)

    all_results = {}
    for model_name, filepath in MODEL_FILES.items():
        print(f"\nAnalyzing {model_name}...")
        result = analyze_model(model_name, filepath)
        all_results[model_name] = result
        vr = result["var_ratio"]
        vr_str = f"{vr:.2f}" if vr is not None else "N/A"
        print(f"  Var_Ratio: {vr_str}")
        print(f"  Mean Accuracy: {result['mean_accuracy_pct']}% ({result['mean_accuracy_raw']}/16)")
        print(f"  Std: {result['std']}, Range: {result['min']}-{result['max']}")
        print(f"  Perfect Scores: {result['perfect_scores']}/50")

    # Correlation analysis (exclude models without Var_Ratio)
    models_with_vr = [(m, r) for m, r in all_results.items() if r["var_ratio"] is not None]
    vr_vals = [r["var_ratio"] for _, r in models_with_vr]
    acc_vals = [r["mean_accuracy_pct"] for _, r in models_with_vr]

    pearson_r, pearson_p = stats.pearsonr(vr_vals, acc_vals)
    spearman_rho, spearman_p = stats.spearmanr(vr_vals, acc_vals)

    print(f"\n{'='*60}")
    print(f"Correlation Analysis (N={len(models_with_vr)} models with Var_Ratio)")
    print(f"  Pearson r = {pearson_r:.3f}, p = {pearson_p:.4f}")
    print(f"  Spearman rho = {spearman_rho:.3f}, p = {spearman_p:.4f}")

    # Behavioral class assignment
    classes = {
        "class1_ideal": [],
        "class2_empty": [],
        "class3_dangerous": [],
        "class4_rich": [],
    }
    for model_name, result in all_results.items():
        vr = result["var_ratio"]
        acc = result["mean_accuracy_pct"]
        if vr is None:
            continue
        if vr < 1.2 and acc >= 70:
            classes["class1_ideal"].append(model_name)
        elif vr < 1.2 and acc < 70:
            classes["class2_empty"].append(model_name)
        elif vr >= 2.0 and acc < 70:
            classes["class3_dangerous"].append(model_name)
        elif vr >= 1.2 and acc >= 70:
            classes["class4_rich"].append(model_name)

    print(f"\nBehavioral Classes:")
    for cls, models in classes.items():
        print(f"  {cls}: {models}")

    # Save full JSON
    # Strip trial_details from the main output to keep it manageable
    summary_models = {}
    for m, r in all_results.items():
        summary_models[m] = {k: v for k, v in r.items() if k != "trial_details"}

    full_output = {
        "description": "Cross-model P30 medical accuracy analysis for Paper 5",
        "rubric_elements": 16,
        "rubric": {e: p for e, p in RUBRIC.items()},
        "n_trials": 50,
        "models": summary_models,
        "correlation": {
            "n_models": len(models_with_vr),
            "pearson_r": round(pearson_r, 4),
            "pearson_p": round(pearson_p, 4),
            "spearman_rho": round(spearman_rho, 4),
            "spearman_p": round(spearman_p, 4),
        },
        "behavioral_classes": classes,
    }

    acc_json = os.path.join(OUTPUT_ACC, "cross_model_p30_accuracy.json")
    with open(acc_json, 'w') as f:
        json.dump(full_output, f, indent=2)
    print(f"\nSaved: {acc_json}")

    # Save summary CSV
    csv_path = os.path.join(OUTPUT_ACC, "cross_model_p30_summary.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["model", "var_ratio", "mean_acc_pct", "mean_acc_raw", "std", "min", "max", "perfect_scores", "class"])

        # Determine class for each model
        def get_class(model_name):
            for cls, models in classes.items():
                if model_name in models:
                    return cls
            return "unclassified"

        # Sort by Var_Ratio
        sorted_models = sorted(all_results.items(), key=lambda x: x[1]["var_ratio"] if x[1]["var_ratio"] is not None else 999)
        for m, r in sorted_models:
            vr = r["var_ratio"] if r["var_ratio"] is not None else ""
            writer.writerow([m, vr, r["mean_accuracy_pct"], r["mean_accuracy_raw"], r["std"], r["min"], r["max"], r["perfect_scores"], get_class(m)])

    print(f"Saved: {csv_path}")

    # === LLAMA DEEP DIVE ===
    print(f"\n{'='*60}")
    print("Llama Deep Dive Analysis")
    print("=" * 60)

    for model_name in ["llama_4_scout", "llama_4_maverick", "deepseek_v3_1"]:
        filepath = MODEL_FILES[model_name]
        with open(filepath, 'r') as f:
            data = json.load(f)

        trials = data["trials"]
        p30_idx = 29

        dive_data = {
            "model": model_name,
            "var_ratio": VAR_RATIOS.get(model_name),
            "n_trials": len(trials),
            "p30_true_responses": [],
            "p30_cold_responses": [],
        }

        for t_idx, trial in enumerate(trials):
            true_resp = trial["responses"]["true"][p30_idx]
            cold_resp = trial["responses"]["cold"][p30_idx]

            true_scores = score_response(true_resp)
            cold_scores = score_response(cold_resp)

            dive_data["p30_true_responses"].append({
                "trial": t_idx,
                "response": true_resp,
                "score": sum(true_scores.values()),
                "elements": true_scores,
                "length_chars": len(true_resp),
                "length_words": len(true_resp.split()),
            })
            dive_data["p30_cold_responses"].append({
                "trial": t_idx,
                "response": cold_resp,
                "score": sum(cold_scores.values()),
                "elements": cold_scores,
                "length_chars": len(cold_resp),
                "length_words": len(cold_resp.split()),
            })

        # Summary stats
        true_scores = [r["score"] for r in dive_data["p30_true_responses"]]
        cold_scores = [r["score"] for r in dive_data["p30_cold_responses"]]
        true_lens = [r["length_chars"] for r in dive_data["p30_true_responses"]]
        cold_lens = [r["length_chars"] for r in dive_data["p30_cold_responses"]]

        dive_data["summary"] = {
            "true_mean_score": round(float(np.mean(true_scores)), 2),
            "true_std_score": round(float(np.std(true_scores)), 2),
            "true_min_score": int(min(true_scores)),
            "true_max_score": int(max(true_scores)),
            "cold_mean_score": round(float(np.mean(cold_scores)), 2),
            "cold_std_score": round(float(np.std(cold_scores)), 2),
            "true_mean_length": round(float(np.mean(true_lens)), 0),
            "true_std_length": round(float(np.std(true_lens)), 0),
            "cold_mean_length": round(float(np.mean(cold_lens)), 0),
            "cold_std_length": round(float(np.std(cold_lens)), 0),
        }

        out_path = os.path.join(OUTPUT_LLAMA, f"{model_name}_p30_analysis.json")
        with open(out_path, 'w') as f:
            json.dump(dive_data, f, indent=2)
        print(f"Saved: {out_path}")
        print(f"  TRUE mean: {dive_data['summary']['true_mean_score']}/16, "
              f"COLD mean: {dive_data['summary']['cold_mean_score']}/16")
        print(f"  TRUE len: {dive_data['summary']['true_mean_length']} chars, "
              f"COLD len: {dive_data['summary']['cold_mean_length']} chars")

    # Cross-model summary for deep dive
    summary = {}
    for model_name in ["llama_4_scout", "llama_4_maverick", "deepseek_v3_1"]:
        dive_path = os.path.join(OUTPUT_LLAMA, f"{model_name}_p30_analysis.json")
        with open(dive_path, 'r') as f:
            dive = json.load(f)

        # Per-element TRUE hit rates
        elem_hits = {e: 0 for e in RUBRIC}
        for r in dive["p30_true_responses"]:
            for e, v in r["elements"].items():
                elem_hits[e] += v
        elem_rates = {e: round(elem_hits[e] / len(dive["p30_true_responses"]) * 100, 1) for e in RUBRIC}

        summary[model_name] = {
            "var_ratio": dive["var_ratio"],
            "summary": dive["summary"],
            "per_element_rates_true": elem_rates,
        }

    summary_path = os.path.join(OUTPUT_LLAMA, "llama_p30_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {summary_path}")

    print(f"\n{'='*60}")
    print("All Paper 5 data generated successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
