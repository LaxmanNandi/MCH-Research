#!/usr/bin/env python3
"""
Type 2 Scaling Law Validation
Check if existing medical position data fits the predicted log-scaled curve
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress
from scipy.optimize import curve_fit

print("="*80)
print("TYPE 2 SCALING LAW VALIDATION")
print("="*80)

# Load existing position-level data
df = pd.read_csv('c:/Users/barla/mch_experiments/analysis/position_drci_data.csv')

# Filter for medical domain
medical_df = df[df['domain'] == 'medical'].copy()

print(f"\nMedical models found: {medical_df['model'].unique()}")

# Target positions for Type 2 analysis
target_positions = [5, 10, 15, 20, 25, 30]

# Predicted values from the scaling law: dRCI_Type2 = 0.36 * log(n) - 0.43
# where n = position - 1 (number of prior exchanges)
def predict_type2(position):
    n = position - 1
    if n <= 0:
        return 0
    return 0.36 * np.log(n) - 0.43

predicted_values = [predict_type2(p) for p in target_positions]

print("\n" + "="*80)
print("PREDICTED VALUES (Type 2 Scaling Law)")
print("="*80)

print("\n| Position | n (prior) | Predicted dRCI |")
print("|----------|-----------|----------------|")
for pos, pred in zip(target_positions, predicted_values):
    n = pos - 1
    print(f"| P{pos:2d}      | {n:2d}        | {pred:+.4f}         |")

# ============================================================================
# EXTRACT ACTUAL VALUES FROM MEDICAL DATA
# ============================================================================

print("\n" + "="*80)
print("ACTUAL VALUES (From Existing Data)")
print("="*80)

# Compute mean dRCI across all medical models at each position
actual_values = []
actual_std = []

for pos in target_positions:
    pos_data = medical_df[medical_df['position'] == pos]['mean_drci_cold']
    if len(pos_data) > 0:
        actual_values.append(pos_data.mean())
        actual_std.append(pos_data.std())
    else:
        actual_values.append(np.nan)
        actual_std.append(np.nan)

print("\n| Position | Actual dRCI | Std Dev |")
print("|----------|-------------|---------|")
for pos, actual, std in zip(target_positions, actual_values, actual_std):
    if not np.isnan(actual):
        print(f"| P{pos:2d}      | {actual:+.4f}      | {std:.4f}  |")
    else:
        print(f"| P{pos:2d}      | N/A         | N/A     |")

# ============================================================================
# COMPARE PREDICTED VS ACTUAL
# ============================================================================

print("\n" + "="*80)
print("PREDICTED VS ACTUAL")
print("="*80)

print("\n| Position | Predicted | Actual   | Difference | % Error |")
print("|----------|-----------|----------|------------|---------|")

differences = []
for pos, pred, actual in zip(target_positions, predicted_values, actual_values):
    if not np.isnan(actual):
        diff = actual - pred
        pct_error = abs(diff / actual) * 100 if actual != 0 else 0
        differences.append(diff)
        print(f"| P{pos:2d}      | {pred:+.4f}    | {actual:+.4f}   | {diff:+.4f}     | {pct_error:5.1f}%   |")
    else:
        print(f"| P{pos:2d}      | {pred:+.4f}    | N/A      | N/A        | N/A     |")

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("STATISTICAL FIT ANALYSIS")
print("="*80)

# Remove NaN values
valid_idx = ~np.isnan(actual_values)
pred_valid = np.array(predicted_values)[valid_idx]
actual_valid = np.array(actual_values)[valid_idx]
pos_valid = np.array(target_positions)[valid_idx]

if len(actual_valid) >= 3:
    # Pearson correlation
    r, p_val = pearsonr(pred_valid, actual_valid)
    r_squared = r**2

    # Linear regression (actual ~ predicted)
    slope, intercept, _, _, _ = linregress(pred_valid, actual_valid)

    print(f"\nCorrelation Analysis:")
    print(f"  Pearson r       = {r:+.4f}")
    print(f"  R-squared (R^2) = {r_squared:.4f}")
    print(f"  p-value         = {p_val:.4e}")

    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    print(f"  Significance    = {sig}")

    print(f"\nLinear Fit (Actual ~ Predicted):")
    print(f"  Slope           = {slope:.4f}")
    print(f"  Intercept       = {intercept:+.4f}")
    print(f"  Actual = {slope:.4f} * Predicted {intercept:+.4f}")

    # Mean Absolute Error
    mae = np.mean(np.abs(actual_valid - pred_valid))
    print(f"\nMean Absolute Error:")
    print(f"  MAE             = {mae:.4f}")

    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((actual_valid - pred_valid)**2))
    print(f"  RMSE            = {rmse:.4f}")

    # FIT ALTERNATIVE LOG MODEL TO DATA
    print("\n" + "="*80)
    print("ALTERNATIVE FIT: Log Model to Actual Data")
    print("="*80)

    # Define log function
    def log_func(pos, alpha, beta):
        n = pos - 1
        return alpha * np.log(n) + beta

    # Fit to actual data
    try:
        # Use positions 1-30 from medical data
        all_positions = medical_df['position'].unique()
        all_positions.sort()

        all_actual = []
        for pos in all_positions:
            pos_data = medical_df[medical_df['position'] == pos]['mean_drci_cold']
            if len(pos_data) > 0:
                all_actual.append(pos_data.mean())
            else:
                all_actual.append(np.nan)

        valid_all = ~np.isnan(all_actual)
        pos_all_valid = all_positions[valid_all]
        actual_all_valid = np.array(all_actual)[valid_all]

        # Filter out position 1 (log(0) is undefined)
        pos_all_valid = pos_all_valid[pos_all_valid > 1]
        actual_all_valid = actual_all_valid[len(actual_all_valid) - len(pos_all_valid):]

        # Fit
        popt, pcov = curve_fit(log_func, pos_all_valid, actual_all_valid, p0=[0.36, -0.43])
        alpha_fit, beta_fit = popt

        print(f"\nFitted Log Model:")
        print(f"  dRCI(P) = {alpha_fit:.4f} * log(P-1) {beta_fit:+.4f}")
        print(f"\nComparison to Predicted Model:")
        print(f"  Predicted: dRCI(P) = 0.3600 * log(P-1) - 0.4300")
        print(f"  Fitted:    dRCI(P) = {alpha_fit:.4f} * log(P-1) {beta_fit:+.4f}")
        print(f"\nParameter Differences:")
        print(f"  Alpha: {alpha_fit - 0.36:+.4f} (fitted - predicted)")
        print(f"  Beta:  {beta_fit - (-0.43):+.4f} (fitted - predicted)")

        # R^2 for fitted model
        predicted_fit = log_func(pos_all_valid, alpha_fit, beta_fit)
        ss_res = np.sum((actual_all_valid - predicted_fit)**2)
        ss_tot = np.sum((actual_all_valid - np.mean(actual_all_valid))**2)
        r_squared_fit = 1 - (ss_res / ss_tot)

        print(f"\nFitted Model R^2:")
        print(f"  R^2 = {r_squared_fit:.4f}")

    except Exception as e:
        print(f"\nCould not fit alternative log model: {e}")

else:
    print("\nInsufficient data for statistical analysis (need at least 3 points)")

# ============================================================================
# GENERATE PLOT
# ============================================================================

print("\n" + "="*80)
print("GENERATING PLOTS")
print("="*80)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Predicted vs Actual scatter
ax = axes[0]
if len(actual_valid) >= 2:
    ax.scatter(pred_valid, actual_valid, s=100, alpha=0.7, c='steelblue', edgecolors='black', linewidth=1.5)

    # Add labels
    for pos, pred, actual in zip(pos_valid, pred_valid, actual_valid):
        ax.annotate(f'P{int(pos)}', (pred, actual), fontsize=9, ha='center', va='bottom', alpha=0.7)

    # Diagonal line (perfect fit)
    lim_min = min(pred_valid.min(), actual_valid.min()) - 0.05
    lim_max = max(pred_valid.max(), actual_valid.max()) + 0.05
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.5, linewidth=2, label='Perfect Fit')

    # Best fit line
    if len(actual_valid) >= 2:
        fit_x = np.linspace(lim_min, lim_max, 100)
        fit_y = slope * fit_x + intercept
        ax.plot(fit_x, fit_y, 'r-', alpha=0.7, linewidth=2, label=f'Best Fit (R^2={r_squared:.3f})')

    ax.set_xlabel('Predicted dRCI (Scaling Law)', fontsize=12)
    ax.set_ylabel('Actual dRCI (Medical Data)', fontsize=12)
    ax.set_title(f'Type 2 Scaling Law Validation\nR^2 = {r_squared:.4f}, p = {p_val:.4e}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

# Plot 2: Position curves
ax = axes[1]

# Plot all medical models
for model in medical_df['model'].unique():
    model_data = medical_df[medical_df['model'] == model]
    ax.plot(model_data['position'], model_data['mean_drci_cold'],
            marker='o', markersize=4, alpha=0.5, linewidth=1, label=model)

# Plot predicted curve
if 'pos_all_valid' in locals() and 'popt' in locals():
    pred_curve = log_func(pos_all_valid, alpha_fit, beta_fit)
    ax.plot(pos_all_valid, pred_curve, 'r-', linewidth=3, alpha=0.8, label=f'Fitted Log Model (R^2={r_squared_fit:.3f})')

# Highlight target positions
ax.scatter(pos_valid, actual_valid, s=150, c='red', marker='X', zorder=5,
           edgecolors='black', linewidth=1.5, label='Type 2 Test Positions')

ax.set_xlabel('Position in Conversation', fontsize=12)
ax.set_ylabel('dRCI (Mean across models)', fontsize=12)
ax.set_title('Medical Domain: Position Curves + Log Fit', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 31)

plt.tight_layout()
output_path = "c:/Users/barla/mch_experiments/analysis/type2_scaling_validation.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"[OK] Plot saved: {output_path}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

results_df = pd.DataFrame({
    'position': pos_valid,
    'predicted_drci': pred_valid,
    'actual_drci': actual_valid,
    'difference': actual_valid - pred_valid,
    'percent_error': np.abs((actual_valid - pred_valid) / actual_valid) * 100
})

results_df.to_csv('c:/Users/barla/mch_experiments/analysis/type2_scaling_validation.csv', index=False)
print("[OK] Results saved: analysis/type2_scaling_validation.csv")

# ============================================================================
# CONCLUSION
# ============================================================================

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if 'r_squared' in locals():
    if r_squared >= 0.85:
        print("\n[CONFIRMED] Type 2 Scaling Law holds with R^2 >= 0.85!")
        print(f"  R^2 = {r_squared:.4f}")
        print(f"  Correlation r = {r:+.4f}")
        print(f"  p-value = {p_val:.4e} {sig}")
        print("\n  The existing data VALIDATES the log-scaled dependency.")
        print("  No additional experiments needed to confirm the scaling law.")
    elif r_squared >= 0.70:
        print("\n[STRONG FIT] Type 2 Scaling Law shows strong support (R^2 >= 0.70)")
        print(f"  R^2 = {r_squared:.4f}")
        print(f"  The model explains {r_squared*100:.1f}% of variance.")
        print("\n  Additional positions (P5, P15, P25) would strengthen validation.")
    else:
        print("\n[MODERATE FIT] Type 2 Scaling Law shows moderate fit")
        print(f"  R^2 = {r_squared:.4f}")
        print("\n  The log-scaled model may need refinement.")
        print("  Consider testing additional positions or alternative functions.")

if 'alpha_fit' in locals():
    print(f"\n[FITTED MODEL] Best-fit log curve:")
    print(f"  dRCI(P) = {alpha_fit:.4f} * log(P-1) {beta_fit:+.4f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
