#!/usr/bin/env python3
"""Check which models have response text saved"""

import json
from pathlib import Path

BASE_DIR = Path("C:/Users/barla/mch_experiments")

models_to_check = {
    'Phil Closed': [
        'data/philosophy/closed_models/mch_results_gpt4o_philosophy_50trials.json',
        'data/philosophy/closed_models/mch_results_gpt4o_mini_philosophy_50trials.json',
        'data/philosophy/closed_models/mch_results_claude_haiku_philosophy_50trials.json',
        'data/philosophy/closed_models/mch_results_gemini_flash_philosophy_50trials.json',
    ],
    'Phil Open': [
        'data/philosophy/open_models/mch_results_deepseek_v3_1_philosophy_50trials.json',
        'data/philosophy/open_models/mch_results_kimi_k2_philosophy_50trials.json',
        'data/philosophy/open_models/mch_results_llama_4_maverick_philosophy_50trials.json',
        'data/philosophy/open_models/mch_results_llama_4_scout_philosophy_50trials.json',
        'data/philosophy/open_models/mch_results_ministral_14b_philosophy_50trials.json',
        'data/philosophy/open_models/mch_results_mistral_small_24b_philosophy_50trials.json',
        'data/philosophy/open_models/mch_results_qwen3_235b_philosophy_50trials.json',
    ],
    'Med Closed': [
        'data/medical/closed_models/mch_results_gpt4o_medical_50trials.json',
        'data/medical/closed_models/mch_results_gpt4o_mini_medical_50trials.json',
        'data/medical/closed_models/mch_results_gpt_5_2_medical_50trials.json',
        'data/medical/closed_models/mch_results_claude_haiku_medical_50trials.json',
        'data/medical/closed_models/mch_results_gemini_flash_medical_50trials.json',
    ],
    'Med Open': [
        'data/medical/open_models/mch_results_deepseek_v3_1_medical_50trials.json',
        'data/medical/open_models/mch_results_llama_4_maverick_medical_50trials.json',
        'data/medical/open_models/mch_results_llama_4_scout_medical_50trials.json',
        'data/medical/open_models/mch_results_ministral_14b_medical_50trials.json',
        'data/medical/open_models/mch_results_mistral_small_24b_medical_50trials.json',
        'data/medical/open_models/mch_results_qwen3_235b_medical_50trials.json',
    ]
}

print("Response Text Availability Check")
print("="*80)

has_text_count = 0
no_text_count = 0

for category, paths in models_to_check.items():
    print(f'\n{category}:')
    for path in paths:
        full_path = BASE_DIR / path
        model_name = path.split('/')[-1].replace('mch_results_', '').replace('_50trials.json', '')

        if full_path.exists():
            with open(full_path) as f:
                data = json.load(f)
            has_responses = 'responses' in data['trials'][0]

            if has_responses:
                print(f'  {model_name:35s} [HAS TEXT]')
                has_text_count += 1
            else:
                print(f'  {model_name:35s} [NO TEXT]')
                no_text_count += 1
        else:
            print(f'  {model_name:35s} [NOT FOUND]')

print("\n" + "="*80)
print(f"Summary: {has_text_count} models WITH text, {no_text_count} models WITHOUT text")
print("="*80)
