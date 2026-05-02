#!/usr/bin/env python3
"""
Paper 6 — Unified K Computation Pipeline
=========================================

Reads paper6_manuscript_data.json (the authoritative post-audit data source)
and computes K = ΔRCI × Var_Ratio for all 24 model-domain runs in one pass.

No fallbacks. No mixed sources. No "if X then Y else Z" logic.
Single input → single computation → single output CSV.

This script addresses Codex audit finding #3 ("Mixed-Pipeline Concern") by
providing a clean, reproducible recomputation that confirms the K values
reported in the manuscript.

Inputs:
  data/paper6/paper6_manuscript_data.json

Outputs:
  data/paper6/conservation_K_unified.csv  (24 rows, K for each run)
  Console summary with domain CVs

Run from repository root:
  python scripts/paper6/paper6_unified_K_pipeline.py
"""

import json
import sys
from pathlib import Path
from statistics import mean, stdev

# Ensure Δ etc. render on Windows consoles
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Resolve repo root from this script's location (no hardcoded paths)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

INPUT_JSON = REPO_ROOT / "data" / "paper6" / "paper6_manuscript_data.json"
OUTPUT_CSV = REPO_ROOT / "data" / "paper6" / "conservation_K_unified.csv"

# Models excluded from K analysis (documented in METHODOLOGY_AUDIT.md §5)
EXCLUDED = {
    ("kimi_k2_5", "Legal"): "21% empty responses, COLD refusal pattern",
    ("glm_5", "Legal"): "86% empty responses across all conditions",
}


def load_authoritative_data() -> dict:
    """Load the post-audit unified data source."""
    if not INPUT_JSON.exists():
        sys.exit(f"ERROR: input not found: {INPUT_JSON}")
    with open(INPUT_JSON, encoding="utf-8") as f:
        return json.load(f)


def compute_K_unified(data: dict) -> list[dict]:
    """Compute K for every run with one formula. No fallbacks."""
    rows = []
    for domain, runs in data["model_runs"].items():
        for run in runs:
            model = run["model"]
            key = (model, domain)
            excluded_reason = EXCLUDED.get(key)

            drci = run["drci"]
            var_ratio = run["var_ratio"]
            K_recomputed = drci * var_ratio

            # Sanity check: recomputed K must match stored K
            K_stored = run.get("K")
            mismatch = None
            if K_stored is not None and abs(K_recomputed - K_stored) > 1e-3:
                mismatch = f"stored={K_stored:.6f} vs recomputed={K_recomputed:.6f}"

            rows.append({
                "model": model,
                "domain": domain,
                "drci": round(drci, 6),
                "var_ratio": round(var_ratio, 6),
                "K": round(K_recomputed, 6),
                "n_trials": run.get("n_trials"),
                "embedding": run.get("embedding"),
                "excluded": "yes" if excluded_reason else "no",
                "exclusion_reason": excluded_reason or "",
                "mismatch_with_stored_K": mismatch or "",
            })
    return rows


def domain_summary(rows: list[dict]) -> dict:
    """Compute mean K and CV per domain, excluding flagged runs."""
    by_domain = {}
    for r in rows:
        if r["excluded"] == "yes":
            continue
        by_domain.setdefault(r["domain"], []).append(r["K"])

    summary = {}
    for domain, ks in by_domain.items():
        m = mean(ks)
        s = stdev(ks) if len(ks) > 1 else 0.0
        cv = s / m if m else 0.0
        summary[domain] = {
            "N": len(ks),
            "mean_K": round(m, 4),
            "std_K": round(s, 4),
            "CV": round(cv, 4),
            "min_K": round(min(ks), 4),
            "max_K": round(max(ks), 4),
        }
    return summary


def write_csv(rows: list[dict], path: Path) -> None:
    """Write the unified K table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    lines = [",".join(fields)]
    for r in rows:
        lines.append(",".join(str(r[f]) for f in fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    data = load_authoritative_data()
    rows = compute_K_unified(data)
    write_csv(rows, OUTPUT_CSV)

    # Console output
    print(f"\nUnified K computation across {len(rows)} model-domain runs")
    print(f"Source: {INPUT_JSON.name}")
    print(f"Output: {OUTPUT_CSV.relative_to(REPO_ROOT)}")
    print()

    # Per-row table
    print(f"{'Model':<22} {'Domain':<12} {'ΔRCI':>8} {'Var_Ratio':>10} {'K':>8} {'Status':>10}")
    print("-" * 75)
    for r in rows:
        status = "EXCLUDED" if r["excluded"] == "yes" else ""
        if r["mismatch_with_stored_K"]:
            status = "MISMATCH"
        print(
            f"{r['model']:<22} {r['domain']:<12} "
            f"{r['drci']:>8.4f} {r['var_ratio']:>10.4f} {r['K']:>8.4f} {status:>10}"
        )

    # Domain summary
    print()
    print("Domain summary (excluded models removed):")
    print(f"{'Domain':<12} {'N':>3} {'Mean K':>9} {'Std K':>9} {'CV':>7}")
    print("-" * 45)
    for domain, s in domain_summary(rows).items():
        print(
            f"{domain:<12} {s['N']:>3} {s['mean_K']:>9.4f} "
            f"{s['std_K']:>9.4f} {s['CV']:>7.4f}"
        )

    # Mismatch check
    mismatches = [r for r in rows if r["mismatch_with_stored_K"]]
    if mismatches:
        print(f"\nWARNING: {len(mismatches)} stored K value(s) disagree with recomputed K:")
        for r in mismatches:
            print(f"  {r['model']} ({r['domain']}): {r['mismatch_with_stored_K']}")
        return 1

    print("\nAll stored K values match recomputed values within tolerance (1e-3).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
