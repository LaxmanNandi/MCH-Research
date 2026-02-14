#!/usr/bin/env python3
"""
Paper 6 Hypothesis Test: U-Shaped Relationship Between Var_Ratio and Perfect Scores

Tests whether mild divergence (Var_Ratio ~1.2-2.0) is optimal for summarization,
with both excessive convergence and excessive divergence being suboptimal.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA (from cross_model_p30_summary.csv)
# ============================================================================

# Full dataset (N=8)
data_full = {
    "deepseek_v3_1":    {"var_ratio": 0.48, "perfect": 2,  "mean_acc": 82.8, "std": 1.23, "min": 10, "max": 16},
    "gemini_flash":     {"var_ratio": 0.60, "perfect": 1,  "mean_acc": 15.8, "std": 2.88, "min": 0,  "max": 16},
    "ministral_14b":    {"var_ratio": 0.75, "perfect": 10, "mean_acc": 90.2, "std": 1.71, "min": 7,  "max": 16},
    "kimi_k2":          {"var_ratio": 0.97, "perfect": 13, "mean_acc": 91.9, "std": 1.04, "min": 11, "max": 16},
    "mistral_small_24b":{"var_ratio": 1.02, "perfect": 1,  "mean_acc": 82.6, "std": 1.25, "min": 10, "max": 16},
    "qwen3_235b":       {"var_ratio": 1.45, "perfect": 23, "mean_acc": 95.2, "std": 0.86, "min": 13, "max": 16},
    "llama_4_maverick": {"var_ratio": 2.64, "perfect": 0,  "mean_acc": 46.8, "std": 1.33, "min": 5,  "max": 11},
    "llama_4_scout":    {"var_ratio": 7.46, "perfect": 0,  "mean_acc": 55.4, "std": 1.78, "min": 5,  "max": 14},
}

# Excluding Gemini (N=7) - safety filter outlier
data_no_gemini = {k: v for k, v in data_full.items() if k != "gemini_flash"}

def extract_arrays(data):
    names = list(data.keys())
    vr = np.array([data[n]["var_ratio"] for n in names])
    perfect = np.array([data[n]["perfect"] for n in names])
    acc = np.array([data[n]["mean_acc"] for n in names])
    std = np.array([data[n]["std"] for n in names])
    range_w = np.array([data[n]["max"] - data[n]["min"] for n in names])
    return names, vr, perfect, acc, std, range_w


# ============================================================================
# TASK 1: STATISTICAL TEST FOR U-SHAPE
# ============================================================================

def test_ushape(names, vr, perfect, label):
    print(f"\n{'='*70}")
    print(f"TASK 1: U-SHAPE TEST ({label}, N={len(vr)})")
    print(f"{'='*70}")

    # Linear fit: Perfect ~ Var_Ratio
    slope_lin, intercept_lin, r_lin, p_lin, se_lin = stats.linregress(vr, perfect)
    y_pred_lin = slope_lin * vr + intercept_lin
    ss_res_lin = np.sum((perfect - y_pred_lin) ** 2)
    ss_tot = np.sum((perfect - np.mean(perfect)) ** 2)
    r2_lin = 1 - ss_res_lin / ss_tot

    print(f"\nLinear Model: Perfect = {slope_lin:.2f} * VR + {intercept_lin:.2f}")
    print(f"  R^2 = {r2_lin:.4f}, r = {r_lin:.4f}, p = {p_lin:.4f}")

    # Quadratic fit: Perfect ~ Var_Ratio + Var_Ratio^2
    # Using numpy polyfit (degree 2: ax^2 + bx + c)
    coeffs = np.polyfit(vr, perfect, 2)
    a, b, c = coeffs
    y_pred_quad = np.polyval(coeffs, vr)
    ss_res_quad = np.sum((perfect - y_pred_quad) ** 2)
    r2_quad = 1 - ss_res_quad / ss_tot

    print(f"\nQuadratic Model: Perfect = {a:.2f} * VR^2 + {b:.2f} * VR + {c:.2f}")
    print(f"  R^2 = {r2_quad:.4f}")
    print(f"  Coefficients: a={a:.4f}, b={b:.4f}, c={c:.4f}")

    # F-test: Is quadratic significantly better than linear?
    n = len(vr)
    p_lin_params = 2  # slope + intercept
    p_quad_params = 3  # a, b, c
    df1 = p_quad_params - p_lin_params  # 1
    df2 = n - p_quad_params

    if df2 > 0 and ss_res_quad > 0:
        f_stat = ((ss_res_lin - ss_res_quad) / df1) / (ss_res_quad / df2)
        p_ftest = 1 - stats.f.cdf(f_stat, df1, df2)
        print(f"\nF-test (quadratic vs linear):")
        print(f"  F = {f_stat:.4f}, df1={df1}, df2={df2}, p = {p_ftest:.4f}")
        print(f"  Quadratic {'significantly' if p_ftest < 0.05 else 'NOT significantly'} better than linear")
    else:
        f_stat, p_ftest = np.nan, np.nan
        print(f"\nF-test: insufficient df (N={n}, params={p_quad_params})")

    # Check if it's an inverted-U (a < 0 means peak, not valley)
    if a < 0:
        print(f"\n  Shape: INVERTED-U (a={a:.4f} < 0) -> peak exists")
    else:
        print(f"\n  Shape: U-SHAPE (a={a:.4f} > 0) -> valley exists")

    return {
        "linear": {"slope": slope_lin, "intercept": intercept_lin, "r2": r2_lin, "r": r_lin, "p": p_lin},
        "quadratic": {"a": a, "b": b, "c": c, "r2": r2_quad},
        "ftest": {"f": f_stat, "p": p_ftest},
        "coeffs": coeffs,
    }


# ============================================================================
# TASK 2: FIND OPTIMAL VAR_RATIO
# ============================================================================

def find_optimal(coeffs, label):
    print(f"\n{'='*70}")
    print(f"TASK 2: OPTIMAL VAR_RATIO ({label})")
    print(f"{'='*70}")

    a, b, c = coeffs

    # Vertex of parabola: x = -b / (2a)
    if a != 0:
        optimal_vr = -b / (2 * a)
        predicted_perfect = np.polyval(coeffs, optimal_vr)
        print(f"  Vertex (optimal Var_Ratio): {optimal_vr:.4f}")
        print(f"  Predicted perfect scores at vertex: {predicted_perfect:.1f}")

        if a < 0:
            print(f"  Interpretation: Peak at VR={optimal_vr:.2f} (inverted-U confirmed)")
        else:
            print(f"  Interpretation: Valley at VR={optimal_vr:.2f} (U-shape, minimum not maximum)")
    else:
        optimal_vr = np.nan
        predicted_perfect = np.nan
        print(f"  No curvature (a=0)")

    # Bootstrap confidence interval for optimal VR
    print(f"\n  Bootstrap 95% CI for optimal Var_Ratio (1000 iterations):")
    np.random.seed(42)
    boot_optima = []
    for _ in range(1000):
        idx = np.random.choice(len(coeffs_data_vr), size=len(coeffs_data_vr), replace=True)
        boot_vr = coeffs_data_vr[idx]
        boot_perf = coeffs_data_perfect[idx]
        try:
            bc = np.polyfit(boot_vr, boot_perf, 2)
            if bc[0] != 0:
                opt = -bc[1] / (2 * bc[0])
                if -5 < opt < 20:  # reasonable range
                    boot_optima.append(opt)
        except:
            pass

    if len(boot_optima) > 100:
        ci_lo, ci_hi = np.percentile(boot_optima, [2.5, 97.5])
        print(f"  Bootstrap median: {np.median(boot_optima):.4f}")
        print(f"  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
        print(f"  Valid bootstrap samples: {len(boot_optima)}/1000")
    else:
        ci_lo, ci_hi = np.nan, np.nan
        print(f"  Insufficient valid bootstrap samples ({len(boot_optima)}/1000)")

    return optimal_vr, predicted_perfect, (ci_lo, ci_hi)


# ============================================================================
# TASK 3: ZONE ANALYSIS
# ============================================================================

def zone_analysis(names, vr, perfect, label):
    print(f"\n{'='*70}")
    print(f"TASK 3: ZONE ANALYSIS ({label})")
    print(f"{'='*70}")

    zones = {
        "Rigid (<0.8)": (0, 0.8),
        "Near-Optimal (0.8-1.2)": (0.8, 1.2),
        "Optimal (1.2-2.0)": (1.2, 2.0),
        "Dangerous (>2.0)": (2.0, 100),
    }

    zone_data = {}
    all_groups = []
    for zone_name, (lo, hi) in zones.items():
        mask = (vr >= lo) & (vr < hi)
        zone_models = [names[i] for i in range(len(names)) if mask[i]]
        zone_perfect = perfect[mask]

        if len(zone_perfect) > 0:
            mean_p = float(np.mean(zone_perfect))
            std_p = float(np.std(zone_perfect)) if len(zone_perfect) > 1 else 0
        else:
            mean_p = 0
            std_p = 0

        zone_data[zone_name] = {"models": zone_models, "perfect": zone_perfect.tolist(), "mean": mean_p, "std": std_p}
        all_groups.append(zone_perfect)

        print(f"\n  {zone_name}:")
        print(f"    Models: {zone_models}")
        print(f"    Perfect scores: {zone_perfect.tolist()}")
        print(f"    Mean: {mean_p:.1f} +/- {std_p:.1f}")

    # Kruskal-Wallis (non-parametric ANOVA) - only if 3+ non-empty groups
    non_empty = [g for g in all_groups if len(g) > 0]
    if len(non_empty) >= 3:
        try:
            h_stat, p_kw = stats.kruskal(*non_empty)
            print(f"\n  Kruskal-Wallis: H={h_stat:.4f}, p={p_kw:.4f}")
            print(f"  {'Significant' if p_kw < 0.05 else 'Not significant'} difference across zones")
        except:
            print(f"\n  Kruskal-Wallis: could not compute (insufficient data)")
    else:
        print(f"\n  Kruskal-Wallis: only {len(non_empty)} non-empty zones (need 3+)")

    return zone_data


# ============================================================================
# TASK 4: CORRELATION WITH OTHER METRICS
# ============================================================================

def correlation_analysis(names, vr, perfect, acc, std, range_w, label):
    print(f"\n{'='*70}")
    print(f"TASK 4: CORRELATION WITH OTHER METRICS ({label})")
    print(f"{'='*70}")

    metrics = [
        ("Mean Accuracy", acc),
        ("Std (score variability)", std),
        ("Range Width (max-min)", range_w),
        ("Var_Ratio", vr),
    ]

    results = {}
    for metric_name, metric_vals in metrics:
        r, p = stats.pearsonr(perfect, metric_vals)
        rho, p_rho = stats.spearmanr(perfect, metric_vals)
        print(f"\n  Perfect Scores vs {metric_name}:")
        print(f"    Pearson:  r={r:.4f}, p={p:.4f} {'*' if p < 0.05 else ''}")
        print(f"    Spearman: rho={rho:.4f}, p={p_rho:.4f} {'*' if p_rho < 0.05 else ''}")
        results[metric_name] = {"pearson_r": r, "pearson_p": p, "spearman_rho": rho, "spearman_p": p_rho}

    # Best predictor
    best = max(results.items(), key=lambda x: abs(x[1]["pearson_r"]))
    print(f"\n  Best predictor of perfect scores: {best[0]} (|r|={abs(best[1]['pearson_r']):.4f})")

    return results


# ============================================================================
# RUN ALL ANALYSES
# ============================================================================

# Full dataset (N=8)
print("\n" + "#" * 70)
print("# FULL DATASET (N=8, including Gemini Flash)")
print("#" * 70)

names_full, vr_full, perfect_full, acc_full, std_full, range_full = extract_arrays(data_full)
results_full = test_ushape(names_full, vr_full, perfect_full, "Full N=8")

# Store for bootstrap
coeffs_data_vr = vr_full
coeffs_data_perfect = perfect_full
opt_full, pred_full, ci_full = find_optimal(results_full["coeffs"], "Full N=8")

zone_full = zone_analysis(names_full, vr_full, perfect_full, "Full N=8")
corr_full = correlation_analysis(names_full, vr_full, perfect_full, acc_full, std_full, range_full, "Full N=8")

# ============================================================================
# TASK 5: SENSITIVITY CHECK (EXCLUDE GEMINI)
# ============================================================================

print("\n\n" + "#" * 70)
print("# SENSITIVITY CHECK (N=7, excluding Gemini Flash)")
print("#" * 70)

names_ng, vr_ng, perfect_ng, acc_ng, std_ng, range_ng = extract_arrays(data_no_gemini)
results_ng = test_ushape(names_ng, vr_ng, perfect_ng, "No Gemini N=7")

coeffs_data_vr = vr_ng
coeffs_data_perfect = perfect_ng
opt_ng, pred_ng, ci_ng = find_optimal(results_ng["coeffs"], "No Gemini N=7")

zone_ng = zone_analysis(names_ng, vr_ng, perfect_ng, "No Gemini N=7")
corr_ng = correlation_analysis(names_ng, vr_ng, perfect_ng, acc_ng, std_ng, range_ng, "No Gemini N=7")

# ============================================================================
# TASK 6: VISUALIZATION
# ============================================================================

print(f"\n\n{'='*70}")
print("TASK 6: GENERATING FIGURE")
print(f"{'='*70}")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax_idx, (data, label, results) in enumerate([
    (data_full, "All Models (N=8)", results_full),
    (data_no_gemini, "Excluding Gemini (N=7)", results_ng),
]):
    ax = axes[ax_idx]
    names_plot, vr_plot, perfect_plot, _, _, _ = extract_arrays(data)

    # Scatter points
    colors = []
    for n in names_plot:
        if "gemini" in n:
            colors.append("#FF6B35")  # orange for outlier
        elif "llama" in n:
            colors.append("#E63946")  # red for dangerous
        elif "qwen" in n:
            colors.append("#2A9D8F")  # teal for rich
        else:
            colors.append("#457B9D")  # blue for ideal

    ax.scatter(vr_plot, perfect_plot, c=colors, s=120, zorder=5, edgecolors='black', linewidth=0.8)

    # Label points
    for i, name in enumerate(names_plot):
        short = name.replace("_", " ").replace("4 ", "4\n").title()
        # Offset labels to avoid overlap
        offsets = {
            "deepseek_v3_1": (-0.1, 1.5),
            "gemini_flash": (0.1, 1.5),
            "ministral_14b": (0.1, -2.5),
            "kimi_k2": (0.12, 0.5),
            "mistral_small_24b": (0.12, -2.5),
            "qwen3_235b": (-0.3, 1.5),
            "llama_4_maverick": (0.12, 0.5),
            "llama_4_scout": (-0.5, 1.5),
        }
        dx, dy = offsets.get(name, (0.1, 0.5))
        display_name = name.replace("_", " ").replace("v3 1", "V3.1").replace("4 scout", "4 Scout").replace("4 maverick", "4 Maverick").replace("3 235b", "3 235B").replace("small 24b", "Small 24B").replace("14b", "14B").replace("k2", "K2").replace("flash", "Flash").title()
        # Clean up title case artifacts
        display_name = name.replace("_", " ")
        label_map = {
            "deepseek v3 1": "DeepSeek V3.1",
            "gemini flash": "Gemini Flash",
            "ministral 14b": "Ministral 14B",
            "kimi k2": "Kimi K2",
            "mistral small 24b": "Mistral Small",
            "qwen3 235b": "Qwen3 235B",
            "llama 4 maverick": "Llama Maverick",
            "llama 4 scout": "Llama Scout",
        }
        display_name = label_map.get(display_name, display_name)
        ax.annotate(display_name, (vr_plot[i], perfect_plot[i]),
                    xytext=(vr_plot[i] + dx, perfect_plot[i] + dy),
                    fontsize=8, fontweight='bold',
                    arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5) if abs(dx) > 0.2 else None)

    # Quadratic fit curve
    coeffs = results["coeffs"]
    x_fit = np.linspace(0, 8, 200)
    y_fit = np.polyval(coeffs, x_fit)
    y_fit_clipped = np.clip(y_fit, -2, 30)  # clip for visual sanity

    r2 = results["quadratic"]["r2"]
    f_p = results["ftest"]["p"]
    ax.plot(x_fit, y_fit_clipped, 'k--', linewidth=2, alpha=0.7,
            label=f'Quadratic fit (R$^2$={r2:.3f}, F-test p={f_p:.3f})')

    # Zone lines
    for boundary, zone_label in [(0.8, ""), (1.2, ""), (2.0, "")]:
        ax.axvline(boundary, color='gray', linestyle=':', alpha=0.4, linewidth=1)

    # Zone labels at top
    zone_positions = [(0.4, "Rigid"), (1.0, "Near-\nOptimal"), (1.6, "Optimal?"), (4.0, "Dangerous")]
    for xpos, zlabel in zone_positions:
        if xpos < max(vr_plot) + 1:
            ax.text(xpos, max(perfect_plot) + 3, zlabel, ha='center', fontsize=8,
                    color='gray', fontstyle='italic')

    # Optimal VR marker
    a_coeff = coeffs[0]
    if a_coeff < 0:  # inverted-U
        optimal = -coeffs[1] / (2 * coeffs[0])
        if 0 < optimal < 8:
            pred_y = np.polyval(coeffs, optimal)
            ax.axvline(optimal, color='green', linestyle='-', alpha=0.3, linewidth=2)
            ax.plot(optimal, pred_y, 'g*', markersize=15, zorder=6, label=f'Optimal VR={optimal:.2f}')

    ax.set_xlabel('Var_Ratio (P30 Medical)', fontsize=12)
    ax.set_ylabel('Perfect Scores (/50 trials)', fontsize=12)
    ax.set_title(f'Var_Ratio vs Perfect Scores - {label}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-0.2, 8.2)
    ax.set_ylim(-3, 30)

plt.tight_layout()
output_path = "C:/Users/barla/mch_experiments/docs/figures/paper6/var_ratio_perfect_scores_ushape.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved: {output_path}")

# ============================================================================
# FINAL VERDICT
# ============================================================================

print(f"\n\n{'='*70}")
print("FINAL VERDICT")
print(f"{'='*70}")

print(f"\n--- Full Dataset (N=8) ---")
print(f"  Quadratic R^2: {results_full['quadratic']['r2']:.4f}")
print(f"  Linear R^2:    {results_full['linear']['r2']:.4f}")
print(f"  F-test p:      {results_full['ftest']['p']:.4f}")
print(f"  Shape:         {'Inverted-U' if results_full['quadratic']['a'] < 0 else 'U-shape'}")
if not np.isnan(opt_full):
    print(f"  Optimal VR:    {opt_full:.2f}")

print(f"\n--- Without Gemini (N=7) ---")
print(f"  Quadratic R^2: {results_ng['quadratic']['r2']:.4f}")
print(f"  Linear R^2:    {results_ng['linear']['r2']:.4f}")
print(f"  F-test p:      {results_ng['ftest']['p']:.4f}")
print(f"  Shape:         {'Inverted-U' if results_ng['quadratic']['a'] < 0 else 'U-shape'}")
if not np.isnan(opt_ng):
    print(f"  Optimal VR:    {opt_ng:.2f}")

# Significance verdict
sig_full = results_full['ftest']['p'] < 0.05
sig_ng = results_ng['ftest']['p'] < 0.05

if sig_full and sig_ng:
    print(f"\n  VERDICT: U-SHAPE SIGNIFICANT in both analyses.")
    print(f"  This is a Paper 6 core finding.")
elif sig_ng and not sig_full:
    print(f"\n  VERDICT: U-SHAPE SIGNIFICANT only after removing Gemini outlier.")
    print(f"  Gemini confounds the analysis. Finding is suggestive but not robust.")
elif not sig_ng and not sig_full:
    print(f"\n  VERDICT: U-SHAPE NOT SIGNIFICANT.")
    print(f"  The pattern is visually suggestive but lacks statistical power (N={len(vr_full)}).")
    print(f"  More models or different tasks may be needed.")
else:
    print(f"\n  VERDICT: Mixed results. Needs more investigation.")

# Additional context
print(f"\n--- Caveat ---")
print(f"  N={len(vr_full)} models is very small for regression analysis.")
print(f"  Even R^2>{results_ng['quadratic']['r2']:.2f} may not reach significance with N=7.")
print(f"  The pattern is {'visually compelling' if results_ng['quadratic']['r2'] > 0.5 else 'weak'} but power is limited.")
print(f"  Expanding to more models/tasks would strengthen or refute the hypothesis.")
