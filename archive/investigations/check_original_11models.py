#!/usr/bin/env python3
"""
Check correlation using only the original 11 models from Paper 4
"""

import pandas as pd
import numpy as np
from scipy import stats

# Load full dataset
df = pd.read_csv("c:/Users/barla/mch_experiments/results/tables/entanglement_position_data.csv")

print("="*80)
print("CHECKING ORIGINAL 11 MODELS VS EXPANDED 22 MODELS")
print("="*80)

# Original 11 models from Paper 4
original_11 = [
    'DeepSeek V3.1 (Med)',
    'Llama 4 Maverick (Med)',
    'Llama 4 Scout (Med)',
    'Mistral Small 24B (Med)',
    'Ministral 14B (Med)',
    'Qwen3 235B (Med)',
    'Gemini Flash (Med)',
    'GPT-4o (Phil)',
    'GPT-4o-mini (Phil)',
    'Claude Haiku (Phil)',
    'Gemini Flash (Phil)',
]

# Filter to original 11
df_original = df[df['model'].isin(original_11)]

print(f"\nOriginal 11 models:")
print(f"  Models: {df_original['model'].nunique()}")
print(f"  Data points: {len(df_original)} (expected 330)")

# Compute correlation for original 11
r_orig, p_orig = stats.pearsonr(df_original['drci'], df_original['mi_proxy'])

print(f"\nOriginal 11 models:")
print(f"  r = {r_orig:.4f}, p = {p_orig:.2e}")
print(f"  r^2 = {r_orig**2:.4f}")

# Compute correlation for all 22
r_all, p_all = stats.pearsonr(df['drci'], df['mi_proxy'])

print(f"\nAll 22 models:")
print(f"  r = {r_all:.4f}, p = {p_all:.2e}")
print(f"  r^2 = {r_all**2:.4f}")

print(f"\nChange from expansion:")
print(f"  Delta_r = {r_all - r_orig:+.4f}")
print(f"  r_orig = {r_orig:.3f}, r_expanded = {r_all:.3f}")

# Check individual model correlations for original 11
print(f"\nPer-model correlations (original 11):")
for model in original_11:
    model_df = df[df['model'] == model]
    r_m, p_m = stats.pearsonr(model_df['drci'], model_df['mi_proxy'])
    sig = '***' if p_m < 0.001 else '**' if p_m < 0.01 else '*' if p_m < 0.05 else 'ns'
    print(f"  {model:30s} r={r_m:+.3f} ({sig})")

print()
print("="*80)
