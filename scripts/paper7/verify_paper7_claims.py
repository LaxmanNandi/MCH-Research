#!/usr/bin/env python3
"""Verify Paper 7 content-order decomposition claims against raw data."""

import json
import numpy as np
from scipy import stats
import sys

sys.stdout.reconfigure(encoding='utf-8')

medical_files = [
    ('data/medical/closed_models/mch_results_claude_haiku_medical_50trials.json', 'claude_haiku'),
    ('data/medical/closed_models/mch_results_gemini_flash_medical_50trials.json', 'gemini_flash'),
    ('data/medical/closed_models/mch_results_gpt4o_medical_50trials.json', 'gpt4o'),
    ('data/medical/closed_models/mch_results_gpt4o_mini_rerun_medical_50trials.json', 'gpt4o_mini'),
    ('data/medical/closed_models/mch_results_gpt_5_2_medical_50trials.json', 'gpt_5_2'),
    ('data/medical/open_models/mch_results_deepseek_v3_1_medical_50trials.json', 'deepseek_v3_1'),
    ('data/medical/open_models/mch_results_kimi_k2_medical_50trials.json', 'kimi_k2'),
    ('data/medical/open_models/mch_results_llama_4_maverick_medical_50trials.json', 'llama_4_maverick'),
    ('data/medical/open_models/mch_results_llama_4_scout_medical_50trials.json', 'llama_4_scout'),
    ('data/medical/open_models/mch_results_ministral_14b_medical_50trials.json', 'ministral_14b'),
    ('data/medical/open_models/mch_results_mistral_small_24b_medical_50trials.json', 'mistral_small_24b'),
    ('data/medical/open_models/mch_results_qwen3_235b_medical_50trials.json', 'qwen3_235b'),
]

philosophy_files = [
    ('data/philosophy/closed_models/mch_results_claude_haiku_philosophy_50trials.json', 'claude_haiku'),
    ('data/philosophy/closed_models/mch_results_gemini_flash_philosophy_50trials.json', 'gemini_flash'),
    ('data/philosophy/closed_models/mch_results_gpt4o_mini_philosophy_50trials.json', 'gpt4o_mini'),
    ('data/philosophy/closed_models/mch_results_gpt4o_philosophy_50trials.json', 'gpt4o'),
    ('data/philosophy/closed_models/mch_results_gpt_5_2_philosophy_50trials.json', 'gpt_5_2'),
    ('data/philosophy/open_models/mch_results_deepseek_v3_1_philosophy_50trials.json', 'deepseek_v3_1'),
    ('data/philosophy/open_models/mch_results_kimi_k2_philosophy_50trials.json', 'kimi_k2'),
    ('data/philosophy/open_models/mch_results_llama_4_maverick_philosophy_50trials.json', 'llama_4_maverick'),
    ('data/philosophy/open_models/mch_results_llama_4_scout_philosophy_50trials_metrics_only.json', 'llama_4_scout'),
    ('data/philosophy/open_models/mch_results_ministral_14b_philosophy_50trials.json', 'ministral_14b'),
    ('data/philosophy/open_models/mch_results_mistral_small_24b_philosophy_50trials_metrics_only.json', 'mistral_small_24b'),
    ('data/philosophy/open_models/mch_results_qwen3_235b_philosophy_50trials_metrics_only.json', 'qwen3_235b'),
]


def load_data(filepath):
    return json.load(open(filepath))['trials']


###############################################################################
# CLAIMS 1-3: Content-Order Decomposition
###############################################################################
print("=" * 80)
print("CLAIMS 1-3: CONTENT-ORDER DECOMPOSITION")
print("=" * 80)

med_fracs = []
phil_fracs = []

print("\n--- MEDICAL (N=12) ---")
for fp, label in medical_files:
    trials = load_data(fp)
    last = len(trials[0]['alignments']['true']) - 1
    ces = [t['alignments']['scrambled'][last] - t['alignments']['cold'][last] for t in trials]
    drcis = [t['alignments']['true'][last] - t['alignments']['cold'][last] for t in trials]
    cf = np.mean(ces) / np.mean(drcis) * 100
    med_fracs.append(cf)
    print(f"  {label:25s}: CF = {cf:.1f}%")

print(f"\n  Mean: {np.mean(med_fracs):.1f}% +/- {np.std(med_fracs, ddof=1):.1f}%")

print("\n--- PHILOSOPHY (N=12) ---")
for fp, label in philosophy_files:
    trials = load_data(fp)
    last = len(trials[0]['alignments']['true']) - 1
    ces = [t['alignments']['scrambled'][last] - t['alignments']['cold'][last] for t in trials]
    drcis = [t['alignments']['true'][last] - t['alignments']['cold'][last] for t in trials]
    cf = np.mean(ces) / np.mean(drcis) * 100
    phil_fracs.append(cf)
    print(f"  {label:25s}: CF = {cf:.1f}%")

print(f"\n  Mean: {np.mean(phil_fracs):.1f}% +/- {np.std(phil_fracs, ddof=1):.1f}%")

# Stats
u, p = stats.mannwhitneyu(med_fracs, phil_fracs, alternative='two-sided')
pooled = np.sqrt((np.std(med_fracs, ddof=1)**2 + np.std(phil_fracs, ddof=1)**2) / 2)
d = (np.mean(med_fracs) - np.mean(phil_fracs)) / pooled

print(f"\n  Mann-Whitney U={u:.1f}, p={p:.6f}, Cohen's d={d:.2f}")

print("\n--- VERIFICATION ---")
print(f"  Claim 1: Medical  59.8% +/- 9.0%  => Found {np.mean(med_fracs):.1f}% +/- {np.std(med_fracs, ddof=1):.1f}%")
c1 = abs(np.mean(med_fracs) - 59.8) < 0.5
print(f"    Mean: {'VERIFIED' if c1 else 'MISMATCH'}  (diff={np.mean(med_fracs)-59.8:+.1f})")
c1s = abs(np.std(med_fracs, ddof=1) - 9.0) < 1.0
print(f"    SD:   {'VERIFIED' if c1s else 'MISMATCH'}  (found {np.std(med_fracs, ddof=1):.1f} vs claimed 9.0)")

print(f"  Claim 2: Philosophy 39.8% +/- 12.3% => Found {np.mean(phil_fracs):.1f}% +/- {np.std(phil_fracs, ddof=1):.1f}%")
c2 = abs(np.mean(phil_fracs) - 39.8) < 1.0
print(f"    Mean: {'VERIFIED' if c2 else 'MISMATCH'}  (diff={np.mean(phil_fracs)-39.8:+.1f})")
c2s = abs(np.std(phil_fracs, ddof=1) - 12.3) < 1.0
print(f"    SD:   {'VERIFIED' if c2s else 'MISMATCH'}  (found {np.std(phil_fracs, ddof=1):.1f} vs claimed 12.3)")

print(f"  Claim 3: U=133.0, p=0.000239, d=1.85 => Found U={u:.1f}, p={p:.6f}, d={d:.2f}")
c3u = abs(u - 133.0) < 1.0
c3p = abs(p - 0.000239) < 0.0005
c3d = abs(d - 1.85) < 0.1
print(f"    U:    {'VERIFIED' if c3u else 'MISMATCH'}")
print(f"    p:    {'VERIFIED' if c3p else 'MISMATCH'}")
print(f"    d:    {'VERIFIED' if c3d else 'MISMATCH'}")


###############################################################################
# CLAIMS 4-5: P30 z-scores
###############################################################################
print("\n" + "=" * 80)
print("CLAIMS 4-5: P30 SPIKE Z-SCORES (Medical only)")
print("=" * 80)

med_tc_z = []
med_sc_z = []

for fp, label in medical_files:
    trials = load_data(fp)
    n_pos = len(trials[0]['alignments']['true'])

    tc_by_pos = []
    sc_by_pos = []
    for pos in range(n_pos):
        tc = [t['alignments']['true'][pos] - t['alignments']['cold'][pos] for t in trials]
        sc = [t['alignments']['scrambled'][pos] - t['alignments']['cold'][pos] for t in trials]
        tc_by_pos.append(np.mean(tc))
        sc_by_pos.append(np.mean(sc))

    tc_arr = np.array(tc_by_pos)
    sc_arr = np.array(sc_by_pos)

    tc_z = (tc_arr[-1] - np.mean(tc_arr)) / np.std(tc_arr) if np.std(tc_arr) > 0 else 0
    sc_z = (sc_arr[-1] - np.mean(sc_arr)) / np.std(sc_arr) if np.std(sc_arr) > 0 else 0

    med_tc_z.append((label, tc_z))
    med_sc_z.append((label, sc_z))
    print(f"  {label:25s}: TRUE-COLD z={tc_z:+.2f}  SCRAMBLED-COLD z={sc_z:+.2f}")

tc_vals = [z for _, z in med_tc_z]
sc_vals = [z for _, z in med_sc_z]

print(f"\n  TRUE-COLD range:      {min(tc_vals):+.2f} to {max(tc_vals):+.2f}")
print(f"  SCRAMBLED-COLD range: {min(sc_vals):+.2f} to {max(sc_vals):+.2f}")

print(f"\n  Claim 4: TRUE-COLD z = +2.00 to +4.17")
print(f"  Found:   TRUE-COLD z = {min(tc_vals):+.2f} to {max(tc_vals):+.2f}")
c4 = abs(min(tc_vals) - 2.00) < 0.2 and abs(max(tc_vals) - 4.17) < 0.2
print(f"  Status:  {'VERIFIED' if c4 else 'MISMATCH'}")

print(f"\n  Claim 5: SCRAMBLED-COLD z = +0.97 to +4.75")
print(f"  Found:   SCRAMBLED-COLD z = {min(sc_vals):+.2f} to {max(sc_vals):+.2f}")
c5 = abs(min(sc_vals) - 0.97) < 0.2 and abs(max(sc_vals) - 4.75) < 0.2
print(f"  Status:  {'VERIFIED' if c5 else 'MISMATCH'}")


###############################################################################
# CLAIM 7: Legal domain Maverick hierarchy
###############################################################################
print("\n" + "=" * 80)
print("CLAIM 7: LEGAL DOMAIN MAVERICK TRUE > SCRAMBLED > COLD")
print("=" * 80)

legal_fp = 'data/legal/open_models/mch_results_llama_4_maverick_legal_50trials.json'
try:
    legal_data = json.load(open(legal_fp))
    trials = legal_data['trials']
    n_trials = len(trials)
    n_pos = len(trials[0]['alignments']['true'])

    # Check at last position for each trial
    hierarchy_count = 0
    violations = []

    for i, t in enumerate(trials):
        last = n_pos - 1
        true_val = t['alignments']['true'][last]
        cold_val = t['alignments']['cold'][last]
        scr_val = t['alignments']['scrambled'][last]

        if true_val > scr_val > cold_val:
            hierarchy_count += 1
        else:
            violations.append((i, true_val, scr_val, cold_val))

    print(f"  Total trials: {n_trials}")
    print(f"  Trials with TRUE > SCRAMBLED > COLD at P{n_pos}: {hierarchy_count}/{n_trials}")

    # Also check across all positions (mean alignment across positions)
    hierarchy_mean = 0
    for t in trials:
        true_mean = np.mean(t['alignments']['true'])
        cold_mean = np.mean(t['alignments']['cold'])
        scr_mean = np.mean(t['alignments']['scrambled'])
        if true_mean > scr_mean > cold_mean:
            hierarchy_mean += 1

    print(f"  Trials with TRUE > SCRAMBLED > COLD (mean across all positions): {hierarchy_mean}/{n_trials}")

    # Check claim: 45/45 trials
    print(f"\n  Claim: 45/45 trials TRUE > SCRAMBLED > COLD")
    print(f"  Found: {hierarchy_count}/{n_trials} at P{n_pos}, {hierarchy_mean}/{n_trials} by mean")

    if violations:
        print(f"  Violations at P{n_pos}:")
        for idx, tv, sv, cv in violations[:5]:
            print(f"    Trial {idx}: true={tv:.4f} scram={sv:.4f} cold={cv:.4f}")

except FileNotFoundError:
    print(f"  File not found: {legal_fp}")
except Exception as e:
    print(f"  Error: {e}")


###############################################################################
# CLAIM 6: Exploration Arc (requires response text + embedding)
###############################################################################
print("\n" + "=" * 80)
print("CLAIM 6: EXPLORATION ARC")
print("=" * 80)
print("  Exploration Arc requires embedding response texts and computing pairwise")
print("  inter-trial diversity. This requires the sentence_transformers model.")
print("  Computing for files with available response text...")

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Medical files with response text
    med_arc_files = [
        ('data/medical/closed_models/mch_results_gemini_flash_medical_50trials.json', 'gemini_flash'),
        ('data/medical/open_models/mch_results_deepseek_v3_1_medical_50trials.json', 'deepseek_v3_1'),
        ('data/medical/open_models/mch_results_kimi_k2_medical_50trials.json', 'kimi_k2'),
        ('data/medical/open_models/mch_results_llama_4_maverick_medical_50trials.json', 'llama_4_maverick'),
        ('data/medical/open_models/mch_results_llama_4_scout_medical_50trials.json', 'llama_4_scout'),
        ('data/medical/open_models/mch_results_ministral_14b_medical_50trials.json', 'ministral_14b'),
        ('data/medical/open_models/mch_results_mistral_small_24b_medical_50trials.json', 'mistral_small_24b'),
        ('data/medical/open_models/mch_results_qwen3_235b_medical_50trials.json', 'qwen3_235b'),
    ]

    phil_arc_files = [
        ('data/philosophy/closed_models/mch_results_claude_haiku_philosophy_50trials.json', 'claude_haiku'),
        ('data/philosophy/closed_models/mch_results_gemini_flash_philosophy_50trials.json', 'gemini_flash'),
        ('data/philosophy/closed_models/mch_results_gpt4o_mini_philosophy_50trials.json', 'gpt4o_mini'),
        ('data/philosophy/closed_models/mch_results_gpt4o_philosophy_50trials.json', 'gpt4o'),
        ('data/philosophy/open_models/mch_results_deepseek_v3_1_philosophy_50trials.json', 'deepseek_v3_1'),
        ('data/philosophy/open_models/mch_results_llama_4_maverick_philosophy_50trials.json', 'llama_4_maverick'),
    ]

    def compute_exploration_arc(filepath, label):
        """Compute Exploration Arc = Late diversity / Early diversity for TRUE responses."""
        data = json.load(open(filepath))
        trials = data['trials']
        n_pos = len(trials[0]['responses']['true'])
        n_trials = len(trials)

        # Early = P1-P5 (indices 0-4), Late = last 5 positions
        early_range = range(0, min(5, n_pos))
        late_range = range(max(0, n_pos - 5), n_pos)

        def pairwise_diversity_at_positions(pos_range):
            """Mean pairwise (1 - cosine_sim) across trials at given positions."""
            all_div = []
            for pos in pos_range:
                # Get all trial responses at this position
                texts = [trials[t]['responses']['true'][pos] for t in range(n_trials)]
                embs = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
                sim_matrix = cosine_similarity(embs)
                # Extract upper triangle (pairwise)
                n = len(texts)
                for i in range(n):
                    for j in range(i+1, n):
                        all_div.append(1.0 - sim_matrix[i][j])
            return np.mean(all_div)

        early_div = pairwise_diversity_at_positions(early_range)
        late_div = pairwise_diversity_at_positions(late_range)
        arc = late_div / early_div if early_div > 0 else float('nan')

        return arc, early_div, late_div

    med_arcs = []
    phil_arcs = []

    print("\n--- MEDICAL ---")
    for fp, label in med_arc_files:
        arc, early, late = compute_exploration_arc(fp, label)
        med_arcs.append(arc)
        print(f"  {label:25s}: Arc={arc:.2f} (early={early:.4f}, late={late:.4f})")

    print(f"\n  Medical Mean Arc: {np.mean(med_arcs):.2f}")

    print("\n--- PHILOSOPHY ---")
    for fp, label in phil_arc_files:
        arc, early, late = compute_exploration_arc(fp, label)
        phil_arcs.append(arc)
        print(f"  {label:25s}: Arc={arc:.2f} (early={early:.4f}, late={late:.4f})")

    print(f"\n  Philosophy Mean Arc: {np.mean(phil_arcs):.2f}")

    print(f"\n  Claim 6: Medical ~1.06, Philosophy ~2.03")
    print(f"  Found:   Medical {np.mean(med_arcs):.2f}, Philosophy {np.mean(phil_arcs):.2f}")
    c6m = abs(np.mean(med_arcs) - 1.06) < 0.15
    c6p = abs(np.mean(phil_arcs) - 2.03) < 0.30
    print(f"  Medical:    {'VERIFIED' if c6m else 'MISMATCH'}")
    print(f"  Philosophy: {'VERIFIED' if c6p else 'MISMATCH'}")

except ImportError:
    print("  sentence_transformers not available. Cannot compute Exploration Arc.")
except Exception as e:
    print(f"  Error computing Exploration Arc: {e}")
    import traceback
    traceback.print_exc()


print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
