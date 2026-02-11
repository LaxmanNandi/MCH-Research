#!/usr/bin/env python3
"""
Recompute correlation from historical CSV data (commit 68b8adf)
to verify the r=0.76 claim
"""

import pandas as pd
import numpy as np
from scipy import stats
import subprocess

# Extract historical CSV from git
print("Extracting historical entanglement data from commit 68b8adf...")
subprocess.run([
    'git', 'show', '68b8adf:analysis/entanglement_position_data.csv'
], cwd='c:/Users/barla/mch_experiments',
   stdout=open('c:/Users/barla/mch_experiments/scripts/validate/historical_entanglement.csv', 'w'),
   check=True)

# Load historical data
df = pd.read_csv('c:/Users/barla/mch_experiments/scripts/validate/historical_entanglement.csv')

print(f"\nHistorical data (commit 68b8adf):")
print(f"  Rows: {len(df)}")
print(f"  Columns: {list(df.columns)}")
print(f"  Unique models: {df['model'].nunique()}")
print(f"  Models: {sorted(df['model'].unique())}")

# Compute correlation
r, p = stats.pearsonr(df['drci'], df['mutual_info_proxy'])

print(f"\nHistorical correlation (should match r=0.76 claim):")
print(f"  r = {r:.4f}, p = {p:.2e}")
print(f"  r^2 = {r**2:.4f}")

# Check by domain
phil_df = df[df['model'].str.contains('Phil')]
med_df = df[df['model'].str.contains('Med')]

r_phil, p_phil = stats.pearsonr(phil_df['drci'], phil_df['mutual_info_proxy'])
r_med, p_med = stats.pearsonr(med_df['drci'], med_df['mutual_info_proxy'])

print(f"\nBy domain:")
print(f"  Philosophy: r = {r_phil:.4f}, p = {p_phil:.2e} (n={len(phil_df)})")
print(f"  Medical:    r = {r_med:.4f}, p = {p_med:.2e} (n={len(med_df)})")

# Check per-model correlations
print(f"\nPer-model correlations:")
for model in sorted(df['model'].unique()):
    model_df = df[df['model'] == model]
    r_m, p_m = stats.pearsonr(model_df['drci'], model_df['mutual_info_proxy'])
    sig = '***' if p_m < 0.001 else '**' if p_m < 0.01 else '*' if p_m < 0.05 else 'ns'
    print(f"  {model:30s} r={r_m:+.3f} ({sig})")

# Check distribution of values
print(f"\nData distribution:")
print(f"  DRCI:      mean={df['drci'].mean():.4f}, std={df['drci'].std():.4f}")
print(f"  MI_Proxy:  mean={df['mutual_info_proxy'].mean():.4f}, std={df['mutual_info_proxy'].std():.4f}")
print(f"  Var_Ratio: mean={df['var_ratio'].mean():.4f}, std={df['var_ratio'].std():.4f}")
