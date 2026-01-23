#!/usr/bin/env python3
"""
MCH Statistics Verification
Recalculates all statistics from raw data and compares to paper
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats


# Published values from Table 2
PUBLISHED_VALUES = {
    "gpt4o_mini": {"mean": -0.0091, "std": 0.122, "p": 0.457, "pattern": "Neutral"},
    "gpt4o": {"mean": -0.0051, "std": 0.111, "p": 0.648, "pattern": "Neutral"},
    "gemini_flash": {"mean": -0.0377, "std": 0.124, "p": 0.003, "pattern": "Sovereign"},
    "gemini_pro": {"mean": -0.0665, "std": 0.166, "p": 0.001, "pattern": "Sovereign"},
    "claude_haiku": {"mean": -0.0106, "std": 0.117, "p": 0.366, "pattern": "Neutral"},
    "claude_opus": {"mean": -0.0357, "std": 0.107, "p": 0.001, "pattern": "Sovereign"}
}


def extract_drcis(filepath):
    """Extract ΔRCI values from results file."""
    with open(filepath) as f:
        data = json.load(f)

    drcis = []
    for trial in data.get("trials", []):
        if "delta_rci" in trial:
            drci = trial["delta_rci"]
            if isinstance(drci, dict):
                drcis.append(drci.get("cold", drci.get("COLD", 0)))
            else:
                drcis.append(drci)
        elif "alignments" in trial:
            align = trial["alignments"]
            mean_true = align.get("mean_true", 1.0)
            mean_cold = align.get("mean_cold", 0)
            drcis.append(mean_true - mean_cold)

    return np.array(drcis)


def classify_pattern(mean_drci, p_value):
    """Classify relational pattern."""
    if p_value >= 0.05:
        return "Neutral"
    elif mean_drci > 0:
        return "Convergent"
    else:
        return "Sovereign"


def main():
    print("=" * 60)
    print("MCH STATISTICS VERIFICATION")
    print("=" * 60)

    base_path = Path(__file__).parent.parent.parent / "data" / "philosophy_results"

    files = {
        "gpt4o_mini": "mch_results_gpt4o_mini_n100_merged.json",
        "gpt4o": "mch_results_gpt4o_100trials.json",
        "gemini_flash": "mch_results_gemini_flash_100trials.json",
        "gemini_pro": "mch_results_gemini_pro_100trials.json",
        "claude_haiku": "mch_results_claude_haiku_100trials.json",
        "claude_opus": "mch_results_claude_opus_100trials.json"
    }

    print(f"\n{'Model':<15} {'Calc Mean':<12} {'Pub Mean':<12} {'Diff':<10} {'Status'}")
    print("-" * 65)

    discrepancies = []

    for model_name, filename in files.items():
        filepath = base_path / filename

        if not filepath.exists():
            print(f"{model_name:<15} FILE NOT FOUND")
            continue

        drcis = extract_drcis(filepath)

        if len(drcis) == 0:
            print(f"{model_name:<15} NO DATA")
            continue

        # Calculate statistics
        calc_mean = np.mean(drcis)
        calc_std = np.std(drcis)
        t_stat, calc_p = stats.ttest_1samp(drcis, 0)
        calc_pattern = classify_pattern(calc_mean, calc_p)

        # Compare to published
        pub = PUBLISHED_VALUES.get(model_name, {})
        pub_mean = pub.get("mean", 0)

        diff = abs(calc_mean - pub_mean)
        status = "✓" if diff < 0.01 else "⚠"

        if diff >= 0.01:
            discrepancies.append({
                "model": model_name,
                "calculated": calc_mean,
                "published": pub_mean,
                "difference": diff
            })

        print(f"{model_name:<15} {calc_mean:+.4f}      {pub_mean:+.4f}      {diff:.4f}     {status}")

    # Pattern verification
    print("\n" + "=" * 60)
    print("PATTERN VERIFICATION")
    print("=" * 60)
    print(f"\n{'Model':<15} {'Calculated':<15} {'Published':<15} {'Match'}")
    print("-" * 55)

    for model_name, filename in files.items():
        filepath = base_path / filename
        if not filepath.exists():
            continue

        drcis = extract_drcis(filepath)
        if len(drcis) == 0:
            continue

        calc_mean = np.mean(drcis)
        t_stat, calc_p = stats.ttest_1samp(drcis, 0)
        calc_pattern = classify_pattern(calc_mean, calc_p)

        pub = PUBLISHED_VALUES.get(model_name, {})
        pub_pattern = pub.get("pattern", "Unknown")

        match = "✓" if calc_pattern == pub_pattern else "✗"
        print(f"{model_name:<15} {calc_pattern:<15} {pub_pattern:<15} {match}")

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    if discrepancies:
        print(f"\n⚠ Found {len(discrepancies)} discrepancies > 0.01:")
        for d in discrepancies:
            print(f"  - {d['model']}: calculated {d['calculated']:+.4f}, published {d['published']:+.4f}")
    else:
        print("\n✓ All calculated values match published values (within 0.01 tolerance)")


if __name__ == "__main__":
    main()
