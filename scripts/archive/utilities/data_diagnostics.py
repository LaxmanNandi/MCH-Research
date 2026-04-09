#!/usr/bin/env python3
"""Data diagnostics for all MCH model results."""

import json
import numpy as np
from pathlib import Path

print('='*80)
print('MCH DATA DIAGNOSTICS - ALL MODELS')
print('='*80)

# Philosophy files
PHILOSOPHY_FILES = {
    'GPT-4o': 'data/philosophy_results/mch_results_gpt4o_100trials.json',
    'GPT-4o-mini': 'data/philosophy_results/mch_results_gpt4o_mini_n100_merged.json',
    'GPT-5.2': 'data/philosophy_results/mch_results_gpt_5_2_100trials.json',
    'Claude Opus': 'data/philosophy_results/mch_results_claude_opus_100trials.json',
    'Claude Haiku': 'data/philosophy_results/mch_results_claude_haiku_100trials.json',
    'Gemini 2.5 Pro': 'data/philosophy_results/mch_results_gemini_pro_100trials.json',
    'Gemini 2.5 Flash': 'data/philosophy_results/mch_results_gemini_flash_100trials.json',
}

# Medical files
MEDICAL_FILES = {
    'GPT-4o': 'data/medical_results/mch_results_gpt4o_medical_50trials.json',
    'GPT-4o-mini': 'data/medical_results/mch_results_gpt4o_mini_rerun_medical_50trials.json',
    'GPT-5.2': 'data/medical_results/mch_results_gpt_5_2_medical_50trials.json',
    'Claude Haiku': 'data/medical_results/mch_results_claude_haiku_medical_50trials.json',
    'Claude Opus': 'data/medical_results/mch_results_claude_opus_medical_50trials.json',
    'Gemini 2.5 Flash': 'data/medical_results/mch_results_gemini_flash_medical_50trials.json',
}

def analyze_file(filepath, model_name, domain):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Get model_id
        model_id = data.get('model_id', data.get('model', 'N/A'))

        trials = data.get('trials', [])
        n_trials = len(trials)

        # Extract dRCI values - handle different formats
        drci_values = []
        for t in trials:
            if 'delta_rci' in t and isinstance(t['delta_rci'], dict):
                drci_values.append(t['delta_rci'].get('cold', 0))
            elif 'controls' in t and 'cold' in t['controls']:
                drci_values.append(t['controls']['cold'].get('delta_rci', 0))
            elif 'metrics' in t and 'delta_rci' in t['metrics']:
                drci_values.append(t['metrics']['delta_rci'])

        if not drci_values:
            return {'status': 'NO_DATA', 'file': filepath}

        mean_drci = np.mean(drci_values)
        std_drci = np.std(drci_values)

        # Pattern classification
        if mean_drci > 0.01:
            pattern = 'CONVERGENT'
        elif mean_drci < -0.01:
            pattern = 'SOVEREIGN'
        else:
            pattern = 'NEUTRAL'

        convergent_count = sum(1 for d in drci_values if d > 0.01)

        return {
            'status': 'OK',
            'model_id': model_id,
            'n_trials': n_trials,
            'n_drci': len(drci_values),
            'mean_drci': mean_drci,
            'std_drci': std_drci,
            'min_drci': np.min(drci_values),
            'max_drci': np.max(drci_values),
            'pattern': pattern,
            'convergent_pct': convergent_count / len(drci_values) * 100,
            'file': filepath
        }
    except FileNotFoundError:
        return {'status': 'FILE_NOT_FOUND', 'file': filepath}
    except Exception as e:
        return {'status': f'ERROR: {e}', 'file': filepath}

# Analyze Philosophy domain
print('\n' + '='*80)
print('PHILOSOPHY DOMAIN (Expected: 100 trials each)')
print('='*80)
print(f"{'Model':<18} {'Trials':>7} {'Mean dRCI':>10} {'Std':>8} {'Pattern':>12} {'Conv%':>7} {'Model ID':<30}")
print('-'*100)

phil_results = {}
for model, filepath in PHILOSOPHY_FILES.items():
    result = analyze_file(filepath, model, 'Philosophy')
    phil_results[model] = result
    if result['status'] == 'OK':
        print(f"{model:<18} {result['n_trials']:>7} {result['mean_drci']:>10.4f} {result['std_drci']:>8.4f} {result['pattern']:>12} {result['convergent_pct']:>6.1f}% {result['model_id']:<30}")
    else:
        print(f"{model:<18} {result['status']}")

# Analyze Medical domain
print('\n' + '='*80)
print('MEDICAL DOMAIN (Expected: 50 trials each)')
print('='*80)
print(f"{'Model':<18} {'Trials':>7} {'Mean dRCI':>10} {'Std':>8} {'Pattern':>12} {'Conv%':>7} {'Model ID':<30}")
print('-'*100)

med_results = {}
for model, filepath in MEDICAL_FILES.items():
    result = analyze_file(filepath, model, 'Medical')
    med_results[model] = result
    if result['status'] == 'OK':
        print(f"{model:<18} {result['n_trials']:>7} {result['mean_drci']:>10.4f} {result['std_drci']:>8.4f} {result['pattern']:>12} {result['convergent_pct']:>6.1f}% {result['model_id']:<30}")
    else:
        print(f"{model:<18} {result['status']}")

# Summary
print('\n' + '='*80)
print('SUMMARY')
print('='*80)

phil_ok = sum(1 for r in phil_results.values() if r['status'] == 'OK')
med_ok = sum(1 for r in med_results.values() if r['status'] == 'OK')
total_phil_trials = sum(r['n_trials'] for r in phil_results.values() if r['status'] == 'OK')
total_med_trials = sum(r['n_trials'] for r in med_results.values() if r['status'] == 'OK')

print(f'Philosophy: {phil_ok}/{len(PHILOSOPHY_FILES)} models OK, {total_phil_trials} total trials')
print(f'Medical: {med_ok}/{len(MEDICAL_FILES)} models OK, {total_med_trials} total trials')
print(f'GRAND TOTAL: {total_phil_trials + total_med_trials} trials')

# Verification against paper figures
print('\n' + '='*80)
print('VERIFICATION vs PAPER FIGURES')
print('='*80)

paper_phil = {
    'GPT-4o-mini': -0.0091,
    'GPT-4o': -0.0051,
    'GPT-5.2': 0.3101,
    'Gemini 2.5 Flash': -0.0377,
    'Gemini 2.5 Pro': -0.0665,
    'Claude Haiku': -0.0106,
    'Claude Opus': -0.0357,
}

paper_med = {
    'GPT-4o': 0.2993,
    'GPT-4o-mini': 0.3189,
    'GPT-5.2': 0.3786,
    'Claude Haiku': 0.3400,
    'Claude Opus': 0.3470,  # placeholder - actual is 0.3384
    'Gemini 2.5 Flash': -0.1331,
}

print('\nPhilosophy Domain:')
print(f"{'Model':<20} {'Paper':>10} {'Actual':>10} {'Diff':>10} {'Match':>8}")
print('-'*60)
for model, expected in paper_phil.items():
    if model in phil_results and phil_results[model]['status'] == 'OK':
        actual = phil_results[model]['mean_drci']
        diff = actual - expected
        match = 'YES' if abs(diff) < 0.001 else 'CLOSE' if abs(diff) < 0.01 else 'CHECK'
        print(f"{model:<20} {expected:>10.4f} {actual:>10.4f} {diff:>+10.4f} {match:>8}")

print('\nMedical Domain:')
print(f"{'Model':<20} {'Paper':>10} {'Actual':>10} {'Diff':>10} {'Match':>8}")
print('-'*60)
for model, expected in paper_med.items():
    if model in med_results and med_results[model]['status'] == 'OK':
        actual = med_results[model]['mean_drci']
        diff = actual - expected
        match = 'YES' if abs(diff) < 0.001 else 'CLOSE' if abs(diff) < 0.01 else 'CHECK'
        print(f"{model:<20} {expected:>10.4f} {actual:>10.4f} {diff:>+10.4f} {match:>8}")

print('\n' + '='*80)
print('MODEL IDs VERIFICATION')
print('='*80)
print('\nPhilosophy:')
for model, result in phil_results.items():
    if result['status'] == 'OK':
        print(f"  {model}: {result['model_id']}")

print('\nMedical:')
for model, result in med_results.items():
    if result['status'] == 'OK':
        print(f"  {model}: {result['model_id']}")
