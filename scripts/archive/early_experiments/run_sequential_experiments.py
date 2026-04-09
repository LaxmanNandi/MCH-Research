#!/usr/bin/env python3
"""Sequential experiment runner - runs philosophy after medical completes."""

import subprocess
import sys

print("="*70, flush=True)
print("MCH SEQUENTIAL EXPERIMENT RUNNER", flush=True)
print("="*70, flush=True)
print("Step 1: GPT-4o-mini Medical Re-run (will resume from checkpoint)", flush=True)
print("Step 2: GPT-5.2 Philosophy (100 trials)", flush=True)
print("="*70, flush=True)

# Step 1: Run medical re-run (will resume from checkpoint)
print("\n>>> Starting GPT-4o-mini Medical Re-run...\n", flush=True)
result1 = subprocess.run(
    [sys.executable, "C:/Users/barla/mch_experiments/scripts/mch_medical_gpt4o_mini_rerun.py"],
    capture_output=False
)

if result1.returncode != 0:
    print(f"\n!!! Medical re-run failed with code {result1.returncode}", flush=True)
else:
    print("\n>>> Medical re-run COMPLETE\n", flush=True)

# Step 2: Run philosophy experiment
print("\n>>> Starting GPT-5.2 Philosophy Experiment...\n", flush=True)
result2 = subprocess.run(
    [sys.executable, "C:/Users/barla/mch_experiments/scripts/mch_philosophy_newgen.py"],
    capture_output=False
)

if result2.returncode != 0:
    print(f"\n!!! Philosophy experiment failed with code {result2.returncode}", flush=True)
else:
    print("\n>>> Philosophy experiment COMPLETE\n", flush=True)

print("\n" + "="*70, flush=True)
print("ALL SEQUENTIAL EXPERIMENTS FINISHED", flush=True)
print("="*70, flush=True)
