#!/usr/bin/env python3
"""
MCH EEG Pilot: Does RCI hold for biological neural networks?
=============================================================
Tests the core MCH hypothesis on human EEG data:
  - Wake (high coherence) = TRUE condition analog
  - REM (content present, order disrupted) = SCRAMBLED analog
  - Deep Sleep (minimal coherence) = COLD condition analog

Hypothesis: RCI(Wake) > RCI(REM) > RCI(Deep Sleep)
            = same hierarchy as TRUE > SCRAMBLED > COLD in LLMs

Dataset: Sleep-EDF (PhysioNet) - 2 subjects, freely available via MNE
Metric: EEG spectral coherence across channels as RCI analog
"""

import numpy as np
import mne
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path

# Import shared path helper (no hardcoded absolute paths)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.paths import data_dir  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')

print("=" * 70)
print("MCH EEG PILOT: Brain Coherence Across Sleep Stages")
print("Hypothesis: Wake > REM > Deep Sleep (mirrors TRUE > SCRAM > COLD)")
print("=" * 70)
print()

# ============================================================================
# DOWNLOAD SLEEP-EDF DATA (20 subjects via MNE)
# ============================================================================
print("Downloading Sleep-EDF data (20 subjects)...", flush=True)
from mne.datasets import sleep_physionet

# Download 20 subjects (age group)
paths = sleep_physionet.age.fetch_data(subjects=list(range(20)), recording=[1])
print(f"Downloaded {len(paths)} recordings.\n")

# ============================================================================
# PROCESS EACH SUBJECT
# ============================================================================

all_results = []

for subj_idx, (psg_file, hyp_file) in enumerate(paths):
    print(f"--- Subject {subj_idx} ---")

    # Load raw EEG
    raw = mne.io.read_raw_edf(psg_file, preload=True, verbose=False)
    annots = mne.read_annotations(hyp_file)
    raw.set_annotations(annots, verbose=False)

    # Get events from annotations
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    print(f"  Channels: {raw.ch_names}")
    print(f"  Sampling rate: {raw.info['sfreq']} Hz")
    print(f"  Sleep stages found: {list(event_id.keys())}")

    # Map sleep stages to our conditions
    # Sleep-EDF annotations: 'Sleep stage W', 'Sleep stage 1', 'Sleep stage 2',
    #                        'Sleep stage 3', 'Sleep stage 4', 'Sleep stage R'
    stage_map = {}
    for key in event_id:
        if 'W' in key:
            stage_map['Wake'] = event_id[key]
        elif '3' in key or '4' in key:
            stage_map['Deep'] = event_id[key]
        elif 'R' in key:
            stage_map['REM'] = event_id[key]
        elif '1' in key:
            stage_map['Light'] = event_id[key]
        elif '2' in key:
            stage_map['N2'] = event_id[key]

    print(f"  Stage mapping: {stage_map}")

    # Pick EEG channels only
    eeg_channels = [ch for ch in raw.ch_names if 'EEG' in ch.upper() or 'Fpz' in ch or 'Pz' in ch]
    if not eeg_channels:
        # Try first two channels
        eeg_channels = raw.ch_names[:2]

    print(f"  Using channels: {eeg_channels}")

    # Create epochs per sleep stage (30-second windows = standard sleep scoring)
    epoch_duration = 30.0  # seconds

    # Compute coherence-like metric for each stage
    stage_coherences = {}

    for stage_name, stage_id in stage_map.items():
        # Get events for this stage
        stage_events = events[events[:, 2] == stage_id]

        if len(stage_events) < 5:
            print(f"  {stage_name}: only {len(stage_events)} epochs, skipping")
            continue

        # Create epochs
        try:
            epochs = mne.Epochs(raw, stage_events, tmin=0, tmax=epoch_duration - 1/raw.info['sfreq'],
                              baseline=None, preload=True, verbose=False,
                              picks=eeg_channels)

            if len(epochs) < 5:
                print(f"  {stage_name}: only {len(epochs)} valid epochs, skipping")
                continue

            data = epochs.get_data()  # (n_epochs, n_channels, n_times)
            n_epochs, n_channels, n_times = data.shape

            # ================================================================
            # RCI ANALOG: Multi-metric coherence measure
            # ================================================================

            epoch_coherences = []

            for ep in range(n_epochs):
                epoch_data = data[ep]  # (n_channels, n_times)

                # Metric 1: Inter-channel correlation (analog of embedding similarity)
                if n_channels >= 2:
                    corr_matrix = np.corrcoef(epoch_data)
                    # Upper triangle correlations (excluding diagonal)
                    upper_tri = corr_matrix[np.triu_indices(n_channels, k=1)]
                    mean_corr = np.mean(np.abs(upper_tri))
                else:
                    # Single channel: autocorrelation at lag
                    signal = epoch_data[0]
                    half = len(signal) // 2
                    mean_corr = np.abs(np.corrcoef(signal[:half], signal[half:])[0, 1])

                # Metric 2: Spectral entropy (lower = more organized = higher coherence)
                from scipy.signal import welch
                freqs_list = []
                for ch in range(n_channels):
                    freqs, psd = welch(epoch_data[ch], fs=raw.info['sfreq'], nperseg=min(256, n_times))
                    # Normalize PSD to probability distribution
                    psd_norm = psd / np.sum(psd)
                    # Shannon entropy
                    entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
                    freqs_list.append(entropy)
                mean_entropy = np.mean(freqs_list)
                max_entropy = np.log2(len(freqs))  # maximum possible entropy
                normalized_entropy = mean_entropy / max_entropy  # 0 to 1

                # Metric 3: Signal variance stability across channels
                if n_channels >= 2:
                    channel_vars = np.var(epoch_data, axis=1)
                    var_stability = 1.0 - (np.std(channel_vars) / (np.mean(channel_vars) + 1e-10))
                    var_stability = max(0, var_stability)
                else:
                    var_stability = 0.5

                # Combined RCI analog (weighted)
                # Higher correlation + lower entropy + stable variance = higher coherence
                rci = (0.4 * mean_corr) + (0.3 * (1 - normalized_entropy)) + (0.3 * var_stability)
                epoch_coherences.append(rci)

            mean_rci = np.mean(epoch_coherences)
            std_rci = np.std(epoch_coherences)
            var_rci = np.var(epoch_coherences)

            stage_coherences[stage_name] = {
                'mean_rci': mean_rci,
                'std_rci': std_rci,
                'var_rci': var_rci,
                'n_epochs': len(epochs),
                'values': epoch_coherences
            }

            print(f"  {stage_name:6s}: RCI = {mean_rci:.4f} +/- {std_rci:.4f} (N={len(epochs)} epochs)")

        except Exception as e:
            print(f"  {stage_name}: Error - {e}")

    all_results.append({
        'subject': subj_idx,
        'stages': stage_coherences
    })
    print()

# ============================================================================
# ANALYSIS: Does the hierarchy hold?
# ============================================================================
print("=" * 70)
print("RESULTS: Brain Coherence Hierarchy")
print("=" * 70)
print()

from scipy.stats import mannwhitneyu, ttest_ind

for result in all_results:
    subj = result['subject']
    stages = result['stages']

    print(f"Subject {subj}:")

    # Print all stages sorted by RCI
    sorted_stages = sorted(stages.items(), key=lambda x: x[1]['mean_rci'], reverse=True)
    for name, data in sorted_stages:
        bar = '#' * int(data['mean_rci'] * 50)
        print(f"  {name:6s}: RCI = {data['mean_rci']:.4f} +/- {data['std_rci']:.4f} | {bar}")

    # Test key hypothesis: Wake > Deep Sleep
    if 'Wake' in stages and 'Deep' in stages:
        wake_vals = stages['Wake']['values']
        deep_vals = stages['Deep']['values']
        u_stat, p_val = mannwhitneyu(wake_vals, deep_vals, alternative='greater')
        effect = (np.mean(wake_vals) - np.mean(deep_vals)) / np.sqrt((np.var(wake_vals) + np.var(deep_vals)) / 2)
        print(f"\n  Wake > Deep Sleep: U={u_stat:.0f}, p={p_val:.6f}, Cohen's d={effect:.3f}")
        print(f"  {'*** CONFIRMED ***' if p_val < 0.05 else 'Not significant'}")

    # Test: Wake > REM
    if 'Wake' in stages and 'REM' in stages:
        wake_vals = stages['Wake']['values']
        rem_vals = stages['REM']['values']
        u_stat, p_val = mannwhitneyu(wake_vals, rem_vals, alternative='greater')
        effect = (np.mean(wake_vals) - np.mean(rem_vals)) / np.sqrt((np.var(wake_vals) + np.var(rem_vals)) / 2)
        print(f"  Wake > REM: U={u_stat:.0f}, p={p_val:.6f}, Cohen's d={effect:.3f}")
        print(f"  {'*** CONFIRMED ***' if p_val < 0.05 else 'Not significant'}")

    # Test: REM > Deep Sleep
    if 'REM' in stages and 'Deep' in stages:
        rem_vals = stages['REM']['values']
        deep_vals = stages['Deep']['values']
        u_stat, p_val = mannwhitneyu(rem_vals, deep_vals, alternative='greater')
        effect = (np.mean(rem_vals) - np.mean(deep_vals)) / np.sqrt((np.var(rem_vals) + np.var(deep_vals)) / 2)
        print(f"  REM > Deep Sleep: U={u_stat:.0f}, p={p_val:.6f}, Cohen's d={effect:.3f}")
        print(f"  {'*** CONFIRMED ***' if p_val < 0.05 else 'Not significant'}")

    # Full hierarchy test
    if 'Wake' in stages and 'REM' in stages and 'Deep' in stages:
        hierarchy_holds = (stages['Wake']['mean_rci'] > stages['REM']['mean_rci'] > stages['Deep']['mean_rci'])
        print(f"\n  HIERARCHY (Wake > REM > Deep): {'HOLDS' if hierarchy_holds else 'VIOLATED'}")

    # Variance ratio analog
    if 'Wake' in stages and 'Deep' in stages:
        vr = stages['Wake']['var_rci'] / stages['Deep']['var_rci'] if stages['Deep']['var_rci'] > 0 else float('inf')
        drci = stages['Wake']['mean_rci'] - stages['Deep']['mean_rci']
        K = drci * vr
        print(f"\n  DELTA_RCI (Wake - Deep): {drci:.4f}")
        print(f"  Var_Ratio (Wake/Deep):   {vr:.4f}")
        print(f"  K = dRCI x VR:           {K:.4f}")

    print()

# ============================================================================
# CROSS-SUBJECT COMPARISON
# ============================================================================
print("=" * 70)
print("CROSS-SUBJECT: Does pattern replicate?")
print("=" * 70)
print()

for stage_name in ['Wake', 'REM', 'Deep', 'Light', 'N2']:
    values = []
    for result in all_results:
        if stage_name in result['stages']:
            values.append(result['stages'][stage_name]['mean_rci'])
    if values:
        print(f"  {stage_name:6s}: {' | '.join(f'{v:.4f}' for v in values)} (mean={np.mean(values):.4f})")

print()
print("=" * 70)
print("MCH EEG PILOT COMPLETE")
print("=" * 70)
print()
print("Mapping to LLM framework:")
print("  Wake    = TRUE condition     (full context, coherent)")
print("  REM     = SCRAMBLED condition (content present, order disrupted)")
print("  Deep    = COLD condition     (minimal coherence)")
print("  Light/N2 = intermediate states")
print()
print("If Wake > REM > Deep holds: the MCH hierarchy generalizes to biological neural networks.")

# Save results
import json
output = {
    'experiment': 'MCH EEG Pilot — Brain Coherence Across Sleep Stages',
    'hypothesis': 'RCI(Wake) > RCI(REM) > RCI(Deep) mirrors TRUE > SCRAMBLED > COLD',
    'dataset': 'Sleep-EDF (PhysioNet) via MNE',
    'n_subjects': len(all_results),
    'rci_components': {
        'inter_channel_correlation': 0.4,
        'spectral_organization': 0.3,
        'variance_stability': 0.3
    },
    'results': []
}

for result in all_results:
    subj_data = {'subject': result['subject'], 'stages': {}}
    for stage, data in result['stages'].items():
        subj_data['stages'][stage] = {
            'mean_rci': float(data['mean_rci']),
            'std_rci': float(data['std_rci']),
            'var_rci': float(data['var_rci']),
            'n_epochs': data['n_epochs']
        }
    output['results'].append(subj_data)

eeg_out = data_dir() / "eeg_pilot" / "sleep_rci_pilot.json"
eeg_out.parent.mkdir(parents=True, exist_ok=True)
with open(eeg_out, 'w') as f:
    json.dump(output, f, indent=2)

print("Results saved to data/eeg_pilot/sleep_rci_pilot.json")
