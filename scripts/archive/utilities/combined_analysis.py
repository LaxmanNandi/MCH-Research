#!/usr/bin/env python3
"""Combined Analysis of MCH Experiments - Philosophy + Medical"""

import json
import numpy as np
from pathlib import Path

# Philosophy results
PHILOSOPHY_FILES = {
    "GPT-4o": "C:/Users/barla/mch_experiments/data/philosophy_results/mch_results_gpt4o_100trials.json",
    "GPT-4o-mini": "C:/Users/barla/mch_experiments/data/philosophy_results/mch_results_gpt4o_mini_n100_merged.json",
    "GPT-5.2": "C:/Users/barla/mch_experiments/data/philosophy_results/mch_results_gpt_5_2_100trials.json",
    "Claude Opus": "C:/Users/barla/mch_experiments/data/philosophy_results/mch_results_claude_opus_100trials.json",
    "Claude Haiku": "C:/Users/barla/mch_experiments/data/philosophy_results/mch_results_claude_haiku_100trials.json",
    "Gemini 2.5 Pro": "C:/Users/barla/mch_experiments/data/philosophy_results/mch_results_gemini_pro_100trials.json",
    "Gemini 2.5 Flash": "C:/Users/barla/mch_experiments/data/philosophy_results/mch_results_gemini_flash_100trials.json",
}

# Medical results
MEDICAL_FILES = {
    "GPT-4o": "C:/Users/barla/mch_experiments/data/medical_results/mch_results_gpt4o_medical_50trials.json",
    "GPT-4o-mini": "C:/Users/barla/mch_experiments/data/medical_results/mch_results_gpt4o_mini_rerun_medical_50trials.json",
    "GPT-5.2": "C:/Users/barla/mch_experiments/data/medical_results/mch_results_gpt_5_2_medical_50trials.json",
    "Claude Haiku": "C:/Users/barla/mch_experiments/data/medical_results/mch_results_claude_haiku_medical_50trials.json",
    "Gemini 2.5 Flash": "C:/Users/barla/mch_experiments/data/medical_results/mch_results_gemini_flash_medical_50trials.json",
}

def classify_pattern(mean_drci):
    if mean_drci > 0.01:
        return "CONVERGENT"
    elif mean_drci < -0.01:
        return "SOVEREIGN"
    else:
        return "NEUTRAL"

def analyze_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        trials = data.get('trials', [])
        if not trials:
            return None

        # Extract dRCI values - handle different data formats
        drci_values = []
        for t in trials:
            # Format 1: delta_rci.cold (new format)
            if 'delta_rci' in t and isinstance(t['delta_rci'], dict):
                drci_values.append(t['delta_rci'].get('cold', 0))
            # Format 2: controls.cold.delta_rci (old format)
            elif 'controls' in t and 'cold' in t['controls']:
                drci_values.append(t['controls']['cold'].get('delta_rci', 0))
            # Format 3: metrics.delta_rci
            elif 'metrics' in t and 'delta_rci' in t['metrics']:
                drci_values.append(t['metrics']['delta_rci'])

        if not drci_values:
            return None

        mean_drci = np.mean(drci_values)
        std_drci = np.std(drci_values)
        n_trials = len(drci_values)

        # Count patterns
        convergent = sum(1 for d in drci_values if d > 0.01)
        sovereign = sum(1 for d in drci_values if d < -0.01)
        neutral = n_trials - convergent - sovereign

        return {
            'n_trials': n_trials,
            'mean_drci': mean_drci,
            'std_drci': std_drci,
            'pattern': classify_pattern(mean_drci),
            'convergent_pct': convergent / n_trials * 100,
            'sovereign_pct': sovereign / n_trials * 100,
            'neutral_pct': neutral / n_trials * 100
        }
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

print("=" * 80)
print("MCH COMBINED ANALYSIS - PHILOSOPHY + MEDICAL")
print("=" * 80)

# Philosophy Analysis
print("\n" + "=" * 80)
print("PHILOSOPHY DOMAIN (Consciousness)")
print("=" * 80)
print(f"{'Model':<20} {'Trials':>7} {'Mean dRCI':>12} {'Std':>8} {'Pattern':>12} {'Conv%':>7}")
print("-" * 80)

philosophy_results = {}
for model, filepath in PHILOSOPHY_FILES.items():
    result = analyze_file(filepath)
    if result:
        philosophy_results[model] = result
        print(f"{model:<20} {result['n_trials']:>7} {result['mean_drci']:>12.4f} {result['std_drci']:>8.4f} {result['pattern']:>12} {result['convergent_pct']:>6.1f}%")

# Medical Analysis
print("\n" + "=" * 80)
print("MEDICAL DOMAIN (STEMI Case)")
print("=" * 80)
print(f"{'Model':<20} {'Trials':>7} {'Mean dRCI':>12} {'Std':>8} {'Pattern':>12} {'Conv%':>7}")
print("-" * 80)

medical_results = {}
for model, filepath in MEDICAL_FILES.items():
    result = analyze_file(filepath)
    if result:
        medical_results[model] = result
        print(f"{model:<20} {result['n_trials']:>7} {result['mean_drci']:>12.4f} {result['std_drci']:>8.4f} {result['pattern']:>12} {result['convergent_pct']:>6.1f}%")

# Cross-domain comparison
print("\n" + "=" * 80)
print("CROSS-DOMAIN COMPARISON")
print("=" * 80)
print(f"{'Model':<20} {'Phil dRCI':>12} {'Med dRCI':>12} {'Diff':>10} {'Same Pattern?':>14}")
print("-" * 80)

common_models = set(philosophy_results.keys()) & set(medical_results.keys())
for model in sorted(common_models):
    phil = philosophy_results[model]
    med = medical_results[model]
    diff = phil['mean_drci'] - med['mean_drci']
    same = "YES" if phil['pattern'] == med['pattern'] else "NO"
    print(f"{model:<20} {phil['mean_drci']:>12.4f} {med['mean_drci']:>12.4f} {diff:>10.4f} {same:>14}")

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

phil_drci = [r['mean_drci'] for r in philosophy_results.values()]
med_drci = [r['mean_drci'] for r in medical_results.values()]

print(f"\nPhilosophy Domain:")
print(f"  Models tested: {len(philosophy_results)}")
print(f"  Mean dRCI across models: {np.mean(phil_drci):.4f} +/- {np.std(phil_drci):.4f}")
print(f"  Range: {min(phil_drci):.4f} to {max(phil_drci):.4f}")
print(f"  All CONVERGENT: {all(r['pattern'] == 'CONVERGENT' for r in philosophy_results.values())}")

print(f"\nMedical Domain:")
print(f"  Models tested: {len(medical_results)}")
print(f"  Mean dRCI across models: {np.mean(med_drci):.4f} +/- {np.std(med_drci):.4f}")
print(f"  Range: {min(med_drci):.4f} to {max(med_drci):.4f}")
print(f"  All CONVERGENT: {all(r['pattern'] == 'CONVERGENT' for r in medical_results.values())}")

# Models NOT tested in medical
print("\n" + "=" * 80)
print("COVERAGE GAPS")
print("=" * 80)
phil_only = set(philosophy_results.keys()) - set(medical_results.keys())
med_only = set(medical_results.keys()) - set(philosophy_results.keys())

if phil_only:
    print(f"Philosophy only (not in Medical): {', '.join(phil_only)}")
if med_only:
    print(f"Medical only (not in Philosophy): {', '.join(med_only)}")

print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)
print("""
1. DOMAIN-DEPENDENT BEHAVIOR: Models show dramatically different patterns between domains:
   - Philosophy: Mixed patterns (GPT-5.2 CONVERGENT, others NEUTRAL/SOVEREIGN)
   - Medical: Strong CONVERGENT for all GPT/Claude models (100%)

2. GPT-5.2 UNIQUE: Only model showing strong CONVERGENT in BOTH domains:
   - Philosophy: dRCI = 0.3101 +/- 0.0142 (100 trials, 100% CONVERGENT)
   - Medical: dRCI = 0.3786 +/- 0.0210 (50 trials, 100% CONVERGENT)

3. MEDICAL > PHILOSOPHY: Same models show higher dRCI in medical domain
   - GPT-4o: Phil -0.0051 -> Med +0.2993 (NEUTRAL -> CONVERGENT)
   - GPT-4o-mini: Phil -0.0091 -> Med +0.3189 (NEUTRAL -> CONVERGENT)
   - Claude Haiku: Phil -0.0106 -> Med +0.3400 (SOVEREIGN -> CONVERGENT)

4. GEMINI ANOMALY:
   - Gemini 2.5 Flash: SOVEREIGN in BOTH domains (unique negative pattern)
   - Gemini 2.5 Pro: Works Philosophy (SOVEREIGN), BLOCKED on Medical
   - Gemini 3 Pro: BLOCKED on both domains (novel finding)

5. INTERPRETATION:
   - Medical domain requires stronger context coherence (clinical reasoning)
   - Philosophy allows more "free-form" responses (less context-dependent)
   - GPT-5.2 shows strongest recursive coherence across all conditions
""")
