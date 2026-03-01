import pandas as pd

# Load rerun verification data
df = pd.read_csv('C:/Users/barla/mch_experiments/scripts/validate/paper4_rerun_verification.csv')

# Calculate conservation product for each model
model_products = df.groupby('model').apply(
    lambda x: pd.Series({
        'mean_delta_rci': x['delta_rci'].mean(),
        'mean_var_ratio': x['var_ratio'].mean(),
        'product_K': x['delta_rci'].mean() * x['var_ratio'].mean(),
        'n_positions': len(x)
    })
)

print("="*70)
print("CONSERVATION CONSTRAINT TEST (Philosophy Rerun Models)")
print("="*70)
print("\nPaper 6 prediction: Philosophy K ~ 0.301")
print("\nRerun models:")
print(model_products.round(4))

print(f"\nMean K (rerun): {model_products['product_K'].mean():.4f}")
print(f"SD K (rerun):   {model_products['product_K'].std():.4f}")
print(f"CV K (rerun):   {model_products['product_K'].std() / abs(model_products['product_K'].mean()):.4f}")

print("\nPaper 6 (original philosophy): K=0.301, CV=0.166")
print(f"Status: {'CONSISTENT' if abs(model_products['product_K'].mean() - 0.301) < 0.15 else 'DIFFERENT'}")
