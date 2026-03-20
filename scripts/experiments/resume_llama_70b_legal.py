#!/usr/bin/env python3
"""
Resume Llama 3.3 70B Turbo legal experiment from 14/50 trials.
Runs ONLY this model — wrapper around run_legal_experiments.py logic.
"""

import sys
sys.path.insert(0, "C:/Users/barla/mch_experiments/scripts/experiments")

# Override MODELS_TO_RUN before importing main logic
import run_legal_experiments as leg

# Only run Llama 3.3 70B Turbo
leg.MODELS_TO_RUN = [
    ("llama_3_3_70b_turbo", "meta-llama/Llama-3.3-70B-Instruct-Turbo", "together"),
]

if __name__ == "__main__":
    print("=" * 70)
    print("RESUMING: Llama 3.3 70B Turbo — Legal Domain (14/50 → 50/50)")
    print("=" * 70)
    leg.main()
