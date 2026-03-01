import pandas as pd

# Load rerun verification data
df = pd.read_csv('C:/Users/barla/mch_experiments/scripts/validate/paper4_rerun_verification.csv')

# Filter to only complete models (50 trials)
complete_models = df[df['model'].isin(['DeepSeek V3.1', 'Llama 4 Maverick'])]

# Calculate conservation product for each model
model_products = complete_models.groupby('model').apply(
    lambda x: pd.Series({
        'mean_delta_rci': x['delta_rci'].mean(),
        'mean_var_ratio': x['var_ratio'].mean(),
        'product_K': x['delta_rci'].mean() * x['var_ratio'].mean(),
        'n_trials': x['n_trials'].iloc[0],
        'n_positions': len(x)
    })
)

print("="*70)
print("CONSERVATION CONSTRAINT TEST (Complete Models Only - 50 trials)")
print("="*70)
print("\nPaper 6 prediction: Philosophy K ~ 0.301")
print("\nComplete rerun models:")
print(model_products.round(4))

mean_K = model_products['product_K'].mean()
sd_K = model_products['product_K'].std()
cv_K = sd_K / abs(mean_K) if mean_K != 0 else float('inf')

print(f"\nMean K (complete): {mean_K:.4f}")
print(f"SD K (complete):   {sd_K:.4f}")
print(f"CV K (complete):   {cv_K:.4f}")

print("\n" + "="*70)
print("COMPARISON")
print("="*70)
print(f"Paper 6 (original philosophy):  K=0.301, CV=0.166")
print(f"Rerun (complete models):        K={mean_K:.4f}, CV={cv_K:.4f}")
print(f"\nDifference: {abs(mean_K - 0.301):.4f} ({abs(mean_K - 0.301)/0.301*100:.1f}% of Paper 6 K)")

# Check position-level K for each model
print("\n" + "="*70)
print("POSITION-LEVEL CONSERVATION PRODUCT")
print("="*70)
for model_name in ['DeepSeek V3.1', 'Llama 4 Maverick']:
    model_data = complete_models[complete_models['model'] == model_name]
    model_data['K_position'] = model_data['delta_rci'] * model_data['var_ratio']
    
    print(f"\n{model_name}:")
    print(f"  Mean K across positions: {model_data['K_position'].mean():.4f}")
    print(f"  SD K:                    {model_data['K_position'].std():.4f}")
    print(f"  Min K:                   {model_data['K_position'].min():.4f}")
    print(f"  Max K:                   {model_data['K_position'].max():.4f}")
