#!/usr/bin/env python3
"""
MCH Data Integrity Checker
Verifies all data files are complete and consistent
"""

import os
import json
from pathlib import Path


def check_json_schema(filepath, expected_trials):
    """Check JSON file has correct schema."""
    issues = []

    try:
        with open(filepath) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return [f"Invalid JSON: {e}"]

    # Check required fields
    if "trials" not in data:
        issues.append("Missing 'trials' key")
        return issues

    trials = data["trials"]

    if len(trials) < expected_trials:
        issues.append(f"Incomplete: {len(trials)}/{expected_trials} trials")

    # Check trial structure
    for i, trial in enumerate(trials[:5]):  # Sample check
        if "delta_rci" not in trial and "alignments" not in trial:
            issues.append(f"Trial {i}: Missing delta_rci or alignments")

    return issues


def main():
    print("=" * 60)
    print("MCH DATA INTEGRITY CHECK")
    print("=" * 60)

    base_path = Path(__file__).parent.parent.parent

    # Philosophy results (100 trials each)
    philosophy_dir = base_path / "data" / "philosophy_results"
    expected_philosophy_files = [
        ("mch_results_gpt4o_mini_n100_merged.json", 100),
        ("mch_results_gpt4o_100trials.json", 100),
        ("mch_results_gemini_flash_100trials.json", 100),
        ("mch_results_gemini_pro_100trials.json", 100),
        ("mch_results_claude_haiku_100trials.json", 100),
        ("mch_results_claude_opus_100trials.json", 100),
    ]

    print("\n[Philosophy Domain - 100 trials each]")
    print("-" * 60)

    for filename, expected in expected_philosophy_files:
        filepath = philosophy_dir / filename
        if not filepath.exists():
            print(f"  ✗ {filename}: FILE NOT FOUND")
        else:
            issues = check_json_schema(filepath, expected)
            if issues:
                print(f"  ⚠ {filename}: {', '.join(issues)}")
            else:
                with open(filepath) as f:
                    data = json.load(f)
                n_trials = len(data.get("trials", []))
                print(f"  ✓ {filename}: {n_trials} trials")

    # Medical results (50 trials each)
    medical_dir = base_path / "data" / "medical_results"
    expected_medical_files = [
        ("mch_results_gpt4o_mini_medical_50trials.json", 50),
        ("mch_results_gpt4o_medical_50trials.json", 50),
        ("mch_results_gemini_flash_medical_50trials.json", 50),
    ]

    print("\n[Medical Domain - 50 trials each]")
    print("-" * 60)

    for filename, expected in expected_medical_files:
        filepath = medical_dir / filename
        if not filepath.exists():
            print(f"  ✗ {filename}: FILE NOT FOUND")
        else:
            issues = check_json_schema(filepath, expected)
            if issues:
                print(f"  ⚠ {filename}: {', '.join(issues)}")
            else:
                with open(filepath) as f:
                    data = json.load(f)
                n_trials = len(data.get("trials", []))
                print(f"  ✓ {filename}: {n_trials} trials")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_philosophy = 0
    for filename, expected in expected_philosophy_files:
        filepath = philosophy_dir / filename
        if filepath.exists():
            with open(filepath) as f:
                data = json.load(f)
            total_philosophy += len(data.get("trials", []))

    total_medical = 0
    for filename, expected in expected_medical_files:
        filepath = medical_dir / filename
        if filepath.exists():
            with open(filepath) as f:
                data = json.load(f)
            total_medical += len(data.get("trials", []))

    print(f"Philosophy trials: {total_philosophy}/600")
    print(f"Medical trials: {total_medical}/300")
    print(f"Total trials: {total_philosophy + total_medical}")


if __name__ == "__main__":
    main()
