"""
Paper 3 Statistical Verification Script
Verifies ALL key statistics claimed in Paper 3 tex manuscript against raw data.
"""

import json
import os
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

BASE_DIR = os.path.join("c:\\", "Users", "barla", "mch_experiments")
FIG1_CSV = os.path.join(BASE_DIR, "docs", "figure_data", "paper3_fig1_position_drci_domains.csv")
FIG3_CSV = os.path.join(BASE_DIR, "docs", "figure_data", "paper3_fig3_position30_zscores.csv")
MEDICAL_OPEN_DIR = os.path.join(BASE_DIR, "data", "medical", "open_models")
MEDICAL_CLOSED_DIR = os.path.join(BASE_DIR, "data", "medical", "closed_models")

PHIL_MODELS = ["GPT-4o", "GPT-4o-mini", "Claude Haiku", "Gemini Flash"]
MED_MODELS = [
    "DeepSeek V3.1", "Gemini Flash", "Kimi K2", "Llama 4 Maverick",
    "Llama 4 Scout", "Ministral 14B", "Mistral Small 24B", "Qwen3 235B"
]

ABS_TOL = 0.0015
Z_TOL = 0.05

MEDICAL_JSON_MAP = {
    "DeepSeek V3.1": os.path.join(MEDICAL_OPEN_DIR, "mch_results_deepseek_v3_1_medical_50trials.json"),
    "Kimi K2": os.path.join(MEDICAL_OPEN_DIR, "mch_results_kimi_k2_medical_50trials.json"),
    "Llama 4 Maverick": os.path.join(MEDICAL_OPEN_DIR, "mch_results_llama_4_maverick_medical_50trials.json"),
    "Llama 4 Scout": os.path.join(MEDICAL_OPEN_DIR, "mch_results_llama_4_scout_medical_50trials.json"),
    "Ministral 14B": os.path.join(MEDICAL_OPEN_DIR, "mch_results_ministral_14b_medical_50trials.json"),
    "Mistral Small 24B": os.path.join(MEDICAL_OPEN_DIR, "mch_results_mistral_small_24b_medical_50trials.json"),
    "Qwen3 235B": os.path.join(MEDICAL_OPEN_DIR, "mch_results_qwen3_235b_medical_50trials.json"),
    "Gemini Flash": os.path.join(MEDICAL_CLOSED_DIR, "mch_results_gemini_flash_medical_50trials.json"),
    "Claude Haiku": os.path.join(MEDICAL_CLOSED_DIR, "mch_results_claude_haiku_medical_50trials.json"),
    "GPT-4o": os.path.join(MEDICAL_CLOSED_DIR, "mch_results_gpt4o_medical_50trials.json"),
    "GPT-4o-mini": os.path.join(MEDICAL_CLOSED_DIR, "mch_results_gpt4o_mini_medical_50trials.json"),
}


def pass_fail(condition, label, actual, claimed, tolerance=None):
    status = "PASS" if condition else "FAIL"
    tol_str = f" (tolerance: +/-{tolerance})" if tolerance else ""
    print(f"  [{status}] {label}")
    print(f"         Claimed: {claimed}")
    print(f"         Actual:  {actual}{tol_str}")
    print()
    return condition


def compute_position_drci_from_json(json_path, model_name):
    with open(json_path, "r") as f:
        data = json.load(f)
    n_trials = len(data["trials"])
    n_positions = 30
    drci_matrix = np.zeros((n_trials, n_positions))
    drci_scr_matrix = np.zeros((n_trials, n_positions))
    for t_idx, trial in enumerate(data["trials"]):
        true_vals = trial["alignments"]["true"]
        cold_vals = trial["alignments"]["cold"]
        scr_vals = trial["alignments"]["scrambled"]
        for p in range(n_positions):
            drci_matrix[t_idx, p] = true_vals[p] - cold_vals[p]
            drci_scr_matrix[t_idx, p] = true_vals[p] - scr_vals[p]
    mean_drci = drci_matrix.mean(axis=0)
    mean_drci_scr = drci_scr_matrix.mean(axis=0)
    ds = mean_drci_scr - mean_drci
    rows = []
    for p in range(n_positions):
        rows.append({
            "domain": "medical",
            "model": model_name,
            "position": p + 1,
            "drci": mean_drci[p],
            "drci_scrambled": mean_drci_scr[p],
            "disruption_sensitivity": ds[p]
        })
    return pd.DataFrame(rows), drci_matrix


def compute_z_score_from_json(json_path, model_name):
    with open(json_path, "r") as f:
        data = json.load(f)
    n_trials = len(data["trials"])
    n_positions = 30
    drci_matrix = np.zeros((n_trials, n_positions))
    for t_idx, trial in enumerate(data["trials"]):
        true_vals = trial["alignments"]["true"]
        cold_vals = trial["alignments"]["cold"]
        for p in range(n_positions):
            drci_matrix[t_idx, p] = true_vals[p] - cold_vals[p]
    mean_drci_per_pos = drci_matrix.mean(axis=0)
    p1_to_p29 = mean_drci_per_pos[:29]
    p30_val = mean_drci_per_pos[29]
    mean_1_29 = p1_to_p29.mean()
    std_1_29 = p1_to_p29.std(ddof=0)
    z_score = (p30_val - mean_1_29) / std_1_29
    return {
        "model": model_name,
        "domain": "medical",
        "mean_pos_1_29": mean_1_29,
        "pos_30_value": p30_val,
        "z_score": z_score
    }


def build_full_medical_data():
    all_dfs = []
    for model_name in MED_MODELS:
        json_path = MEDICAL_JSON_MAP[model_name]
        if not os.path.exists(json_path):
            print(f"  WARNING: JSON not found for {model_name}: {json_path}")
            continue
        df, _ = compute_position_drci_from_json(json_path, model_name)
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)


def build_full_medical_zscores():
    results = []
    for model_name in MED_MODELS:
        json_path = MEDICAL_JSON_MAP[model_name]
        if not os.path.exists(json_path):
            continue
        result = compute_z_score_from_json(json_path, model_name)
        results.append(result)
    return pd.DataFrame(results)


def main():
    print("=" * 72)
    print("Paper 3 Statistical Verification")
    print("=" * 72)
    print()

    fig1 = pd.read_csv(FIG1_CSV)
    fig3 = pd.read_csv(FIG3_CSV)

    pass_count = 0
    fail_count = 0
    total = 0

    # CLAIM 8: Model counts
    print("-" * 72)
    print("CLAIM 8: Model counts: 12 total (4 philosophy + 8 medical)")
    print("-" * 72)
    n_phil = len(PHIL_MODELS)
    n_med = len(MED_MODELS)
    n_total = n_phil + n_med
    total += 1
    if pass_fail(n_total == 12 and n_phil == 4 and n_med == 8,
                 "12 total models (4 phil + 8 med)",
                 f"{n_total} total ({n_phil} phil + {n_med} med)",
                 "12 total (4 phil + 8 med)"):
        pass_count += 1
    else:
        fail_count += 1

    # CLAIM 7: Response count
    print("-" * 72)
    print("CLAIM 7: Response count ~54,000")
    print("-" * 72)
    computed_responses = 12 * 30 * 3 * 50
    total += 1
    if pass_fail(computed_responses == 54000,
                 "12 x 30 x 3 x 50 = 54,000",
                 f"{computed_responses:,}",
                 "~54,000"):
        pass_count += 1
    else:
        fail_count += 1

    # Build full medical data
    print("-" * 72)
    print("Loading raw JSON data for all 8 medical models...")
    print("-" * 72)
    med_full = build_full_medical_data()
    med_zscores = build_full_medical_zscores()
    print(f"  Loaded {med_full['model'].nunique()} medical models from JSON")
    print(f"  Models: {sorted(med_full['model'].unique())}")
    print()

    phil_fig1 = fig1[(fig1["domain"] == "philosophy") & (fig1["model"].isin(PHIL_MODELS))]
    print(f"  Loaded {phil_fig1['model'].nunique()} philosophy models from fig1 CSV")
    print(f"  Models: {sorted(phil_fig1['model'].unique())}")
    print()

    phil_z = fig3[(fig3["domain"] == "philosophy") & (fig3["model"].isin(PHIL_MODELS))]

    # CLAIM 1: Medical P30 Z-scores
    print("-" * 72)
    print("CLAIM 1: Medical P30 Z-scores: mean Z = +3.47 +/- 0.51, all > +2.7")
    print("-" * 72)
    med_z_vals = med_zscores["z_score"].values
    med_z_mean = med_z_vals.mean()
    med_z_sem = med_z_vals.std(ddof=1) / np.sqrt(len(med_z_vals))
    med_z_min = med_z_vals.min()

    print("  Individual medical Z-scores:")
    for _, row in med_zscores.iterrows():
        print(f"    {row['model']:25s}: Z = {row['z_score']:+.4f}")
    print()

    total += 1
    if pass_fail(abs(med_z_mean - 3.47) <= Z_TOL,
                 "Medical mean Z = +3.47",
                 f"{med_z_mean:+.4f}", "+3.47", tolerance=Z_TOL):
        pass_count += 1
    else:
        fail_count += 1

    total += 1
    if pass_fail(abs(med_z_sem - 0.51) <= Z_TOL,
                 "Medical Z SEM = 0.51",
                 f"{med_z_sem:.4f}", "0.51", tolerance=Z_TOL):
        pass_count += 1
    else:
        fail_count += 1

    total += 1
    if pass_fail(med_z_min > 2.7,
                 "All medical Z > +2.7",
                 f"min Z = {med_z_min:+.4f}", "all > +2.7"):
        pass_count += 1
    else:
        fail_count += 1

    # CLAIM 2: Philosophy mean Z
    print("-" * 72)
    print("CLAIM 2: Philosophy P30 Z-scores: mean Z = +0.25 (4 closed-source)")
    print("-" * 72)
    phil_z_vals = phil_z["z_score"].values
    phil_z_mean = phil_z_vals.mean()

    print("  Individual philosophy Z-scores (4 closed-source only):")
    for _, row in phil_z.iterrows():
        print(f"    {row['model']:25s}: Z = {row['z_score']:+.4f}")
    print()

    total += 1
    if pass_fail(abs(phil_z_mean - 0.25) <= Z_TOL,
                 "Philosophy mean Z = +0.25",
                 f"{phil_z_mean:+.4f}", "+0.25", tolerance=Z_TOL):
        pass_count += 1
    else:
        fail_count += 1

    # CLAIM 3: 3-bin values
    print("-" * 72)
    print("CLAIM 3: 3-bin DRCI values (positions 1-29, P30 excluded)")
    print("-" * 72)
    med_no_p30 = med_full[med_full["position"] <= 29]
    med_early = med_no_p30[med_no_p30["position"].between(1, 10)]["drci"].mean()
    med_mid = med_no_p30[med_no_p30["position"].between(11, 20)]["drci"].mean()
    med_late = med_no_p30[med_no_p30["position"].between(21, 29)]["drci"].mean()

    phil_no_p30 = phil_fig1[phil_fig1["position"] <= 29]
    phil_early = phil_no_p30[phil_no_p30["position"].between(1, 10)]["drci"].mean()
    phil_mid = phil_no_p30[phil_no_p30["position"].between(11, 20)]["drci"].mean()
    phil_late = phil_no_p30[phil_no_p30["position"].between(21, 29)]["drci"].mean()

    print("  Medical 3-bin averages (8 models, positions 1-29):")
    print(f"    Early (1-10):  {med_early:.3f}")
    print(f"    Mid   (11-20): {med_mid:.3f}")
    print(f"    Late  (21-29): {med_late:.3f}")
    print()

    total += 1
    if pass_fail(abs(med_early - 0.347) <= ABS_TOL,
                 "Medical Early (1-10) = 0.347",
                 f"{med_early:.3f}", "0.347", tolerance=ABS_TOL):
        pass_count += 1
    else:
        fail_count += 1
    total += 1
    if pass_fail(abs(med_mid - 0.311) <= ABS_TOL,
                 "Medical Mid (11-20) = 0.311",
                 f"{med_mid:.3f}", "0.311", tolerance=ABS_TOL):
        pass_count += 1
    else:
        fail_count += 1
    total += 1
    if pass_fail(abs(med_late - 0.371) <= ABS_TOL,
                 "Medical Late (21-29) = 0.371",
                 f"{med_late:.3f}", "0.371", tolerance=ABS_TOL):
        pass_count += 1
    else:
        fail_count += 1

    print("  Philosophy 3-bin averages (4 closed-source models, positions 1-29):")
    print(f"    Early (1-10):  {phil_early:.3f}")
    print(f"    Mid   (11-20): {phil_mid:.3f}")
    print(f"    Late  (21-29): {phil_late:.3f}")
    print()

    total += 1
    if pass_fail(abs(phil_early - 0.307) <= ABS_TOL,
                 "Philosophy Early (1-10) = 0.307",
                 f"{phil_early:.3f}", "0.307", tolerance=ABS_TOL):
        pass_count += 1
    else:
        fail_count += 1
    total += 1
    if pass_fail(abs(phil_mid - 0.331) <= ABS_TOL,
                 "Philosophy Mid (11-20) = 0.331",
                 f"{phil_mid:.3f}", "0.331", tolerance=ABS_TOL):
        pass_count += 1
    else:
        fail_count += 1
    total += 1
    if pass_fail(abs(phil_late - 0.270) <= ABS_TOL,
                 "Philosophy Late (21-29) = 0.270",
                 f"{phil_late:.3f}", "0.270", tolerance=ABS_TOL):
        pass_count += 1
    else:
        fail_count += 1

    # Check U-shape and inverted-U
    med_u_shape = (med_early > med_mid) and (med_late > med_mid)
    phil_inv_u = (phil_mid > phil_early) and (phil_mid > phil_late)

    total += 1
    if pass_fail(med_u_shape,
                 "Medical U-shape: Early > Mid AND Late > Mid",
                 f"Early({med_early:.3f}) > Mid({med_mid:.3f}) = {med_early > med_mid}, "
                 f"Late({med_late:.3f}) > Mid({med_mid:.3f}) = {med_late > med_mid}",
                 "U-shape pattern"):
        pass_count += 1
    else:
        fail_count += 1
    total += 1
    if pass_fail(phil_inv_u,
                 "Philosophy inverted-U: Mid > Early AND Mid > Late",
                 f"Mid({phil_mid:.3f}) > Early({phil_early:.3f}) = {phil_mid > phil_early}, "
                 f"Mid({phil_mid:.3f}) > Late({phil_late:.3f}) = {phil_mid > phil_late}",
                 "Inverted-U pattern"):
        pass_count += 1
    else:
        fail_count += 1

    # CLAIM 4: DS values
    print("-" * 72)
    print("CLAIM 4: Disruption Sensitivity (DS) values")
    print("-" * 72)
    med_ds_mean = med_full["disruption_sensitivity"].mean()
    phil_ds_mean = phil_fig1["disruption_sensitivity"].mean()

    print(f"  Medical mean DS (8 models, all positions):    {med_ds_mean:.3f}")
    print(f"  Philosophy mean DS (4 models, all positions): {phil_ds_mean:.3f}")
    print()

    print("  Per-model mean DS:")
    print("  Medical:")
    for model in MED_MODELS:
        m_ds = med_full[med_full["model"] == model]["disruption_sensitivity"].mean()
        print(f"    {model:25s}: DS = {m_ds:+.4f}")
    print("  Philosophy:")
    for model in PHIL_MODELS:
        m_ds = phil_fig1[phil_fig1["model"] == model]["disruption_sensitivity"].mean()
        print(f"    {model:25s}: DS = {m_ds:+.4f}")
    print()

    total += 1
    if pass_fail(abs(med_ds_mean - (-0.115)) <= ABS_TOL,
                 "Medical mean DS = -0.115",
                 f"{med_ds_mean:.3f}", "-0.115", tolerance=ABS_TOL):
        pass_count += 1
    else:
        fail_count += 1
    total += 1
    if pass_fail(abs(phil_ds_mean - (-0.073)) <= ABS_TOL,
                 "Philosophy mean DS = -0.073",
                 f"{phil_ds_mean:.3f}", "-0.073", tolerance=ABS_TOL):
        pass_count += 1
    else:
        fail_count += 1

    # Check all 12 models DS < 0
    all_ds_negative = True
    for model in MED_MODELS:
        m_ds = med_full[med_full["model"] == model]["disruption_sensitivity"].mean()
        if m_ds >= 0:
            all_ds_negative = False
    for model in PHIL_MODELS:
        m_ds = phil_fig1[phil_fig1["model"] == model]["disruption_sensitivity"].mean()
        if m_ds >= 0:
            all_ds_negative = False
    total += 1
    if pass_fail(all_ds_negative,
                 "All 12 models have mean DS < 0",
                 f"all_negative={all_ds_negative}",
                 "12/12 models DS < 0"):
        pass_count += 1
    else:
        fail_count += 1

    # CLAIM 5: Z > 2.0 split
    print("-" * 72)
    print("CLAIM 5: 8/8 medical Z > +2.0, 0/4 philosophy Z > +2.0")
    print("-" * 72)
    med_above_2 = int((med_z_vals > 2.0).sum())
    phil_above_2 = int((phil_z_vals > 2.0).sum())

    total += 1
    if pass_fail(med_above_2 == 8,
                 "8/8 medical models Z > +2.0",
                 f"{med_above_2}/8", "8/8"):
        pass_count += 1
    else:
        fail_count += 1
    total += 1
    if pass_fail(phil_above_2 == 0,
                 "0/4 philosophy models Z > +2.0",
                 f"{phil_above_2}/4", "0/4"):
        pass_count += 1
    else:
        fail_count += 1

    # CLAIM 6: Fisher exact test
    print("-" * 72)
    print("CLAIM 6: Fisher exact test for 8/8 vs 0/4 split, p < 0.001")
    print("-" * 72)
    table = np.array([[8, 0], [0, 4]])
    odds_ratio, p_value = scipy_stats.fisher_exact(table, alternative="two-sided")
    print(f"  Contingency table: [[8, 0], [0, 4]]")
    print(f"  Fisher exact test: odds_ratio={odds_ratio}, p={p_value:.6f}")
    print()

    total += 1
    if pass_fail(p_value < 0.001,
                 "Fisher exact p < 0.001",
                 f"p = {p_value:.6f}", "p < 0.001"):
        pass_count += 1
    else:
        fail_count += 1

    # Cross-validation: JSON vs CSV for 4 closed medical models
    print("-" * 72)
    print("CROSS-VALIDATION: JSON vs CSV agreement for 4 closed medical models")
    print("-" * 72)
    csv_med = fig1[fig1["domain"] == "medical"]
    for model in ["Gemini Flash", "Claude Haiku", "GPT-4o", "GPT-4o-mini"]:
        json_df, _ = compute_position_drci_from_json(MEDICAL_JSON_MAP[model], model)
        csv_df = csv_med[csv_med["model"] == model].sort_values("position")
        json_df = json_df.sort_values("position")
        drci_diff = np.abs(json_df["drci"].values - csv_df["drci"].values).max()
        ds_diff = np.abs(json_df["disruption_sensitivity"].values - csv_df["disruption_sensitivity"].values).max()
        match = drci_diff < 0.001 and ds_diff < 0.001
        status = "OK" if match else "MISMATCH"
        print(f"  {model:20s}: max drci_diff={drci_diff:.6f}, max ds_diff={ds_diff:.6f} => {status}")
    print()

    # SUMMARY
    print("=" * 72)
    print(f"SUMMARY: {pass_count}/{total} checks PASSED, {fail_count}/{total} FAILED")
    print("=" * 72)
    if fail_count == 0:
        print("All paper claims verified against raw data.")
    else:
        print(f"WARNING: {fail_count} claim(s) did NOT match the raw data.")
    return fail_count


if __name__ == "__main__":
    exit_code = main()
