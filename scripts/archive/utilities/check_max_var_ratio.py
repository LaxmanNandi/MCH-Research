import pandas as pd

df = pd.read_csv('C:/Users/barla/mch_experiments/scripts/validate/paper4_rerun_verification.csv')

print("="*70)
print("MAX VAR_RATIO BY MODEL (Philosophy Rerun Data)")
print("="*70)

for model in df['model'].unique():
    model_data = df[df['model'] == model]
    max_row = model_data.loc[model_data['var_ratio'].idxmax()]
    
    print(f"\n{model} ({int(max_row['n_trials'])} trials):")
    print(f"  Max Var_Ratio: {max_row['var_ratio']:.4f}")
    print(f"  At position:   {int(max_row['position'])}")
    print(f"  Mean Var_Ratio: {model_data['var_ratio'].mean():.4f}")

print("\n" + "="*70)
print("NOTE")
print("="*70)
print("\nPaper 4's Llama safety anomaly (Var_Ratio = 7.46) was in MEDICAL domain.")
print("This analysis is PHILOSOPHY domain - different data!")
print("\nLlama is actually STABLE in philosophy (max 2.92).")
print("The safety issue is medical-specific.")
