#!/usr/bin/env python3
"""Paper 6 Conservation Law — Full Re-Verification"""
import json, csv, sys, os
import numpy as np
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')
BASE = "C:/Users/barla/mch_experiments"

print("=" * 70)
print("PAPER 6 CONSERVATION LAW — FULL RE-VERIFICATION")
print("=" * 70)
print()

# ============================================================
# STEP 1: Load conservation product CSV
# ============================================================
print("STEP 1: Loading computed data")
print("-" * 40)
data = []
with open(f"{BASE}/data/paper6/conservation_product_test.csv", encoding='utf-8') as f:
    for row in csv.DictReader(f):
        row['drci'] = float(row['drci'])
        row['var_ratio'] = float(row['var_ratio'])
        row['product'] = float(row['product'])
        data.append(row)

print(f"  Total runs in CSV: {len(data)}")
print()

# ============================================================
# STEP 2: Count unique models, domains, vendors
# ============================================================
print("STEP 2: Model/Domain/Vendor count")
print("-" * 40)

vendor_map = {
    'deepseek_v3_1': 'DeepSeek',
    'gemini_flash': 'Google',
    'kimi_k2': 'Moonshot',
    'llama_4_maverick': 'Meta',
    'llama_4_scout': 'Meta',
    'ministral_14b': 'Mistral',
    'mistral_small_24b': 'Mistral',
    'qwen3_235b': 'Alibaba',
    'claude_haiku': 'Anthropic',
    'gpt4o': 'OpenAI',
    'gpt4o_mini': 'OpenAI',
}

unique_models = set(d['model'] for d in data)
unique_domains = set(d['domain'] for d in data)
unique_vendors = set(vendor_map.get(d['model'], '?') for d in data)

med_models = set(d['model'] for d in data if d['domain'] == 'Medical')
phil_models = set(d['model'] for d in data if d['domain'] == 'Philosophy')
cross_domain = med_models & phil_models

print(f"  Unique models:  {len(unique_models)}")
for m in sorted(unique_models):
    doms = [d['domain'] for d in data if d['model'] == m]
    print(f"    {m:<25} [{', '.join(doms)}] — {vendor_map.get(m, '?')}")
print(f"  Unique domains: {len(unique_domains)} ({sorted(unique_domains)})")
print(f"  Unique vendors: {len(unique_vendors)} ({sorted(unique_vendors)})")
print(f"  Medical runs: {len([d for d in data if d['domain'] == 'Medical'])}")
print(f"  Philosophy runs: {len([d for d in data if d['domain'] == 'Philosophy'])}")
print(f"  Cross-domain models: {len(cross_domain)} ({sorted(cross_domain)})")
print(f"  Medical-only:   {sorted(med_models - phil_models)}")
print(f"  Philosophy-only: {sorted(phil_models - med_models)}")
print()

# ============================================================
# STEP 3: Re-verify product computation
# ============================================================
print("STEP 3: Re-verify Product = dRCI x Var_Ratio")
print("-" * 40)

all_match = True
for d in data:
    recomputed = d['drci'] * d['var_ratio']
    diff = abs(recomputed - d['product'])
    if diff > 0.0001:
        print(f"  MISMATCH: {d['model']} {d['domain']}: computed={recomputed:.6f} stored={d['product']:.6f}")
        all_match = False
print(f"  All products match recomputation: {all_match}")
print()

# ============================================================
# STEP 4: Cross-check dRCI against source JSON files
# ============================================================
print("STEP 4: Cross-check dRCI against source JSON files")
print("-" * 40)

json_dirs = {
    'Medical': [
        f"{BASE}/data/medical/closed_models",
        f"{BASE}/data/medical/open_models",
        f"{BASE}/data/medical/gemini_flash",
    ],
    'Philosophy': [
        f"{BASE}/data/philosophy/closed_models",
        f"{BASE}/data/philosophy/open_models",
    ],
}

for d in data:
    model = d['model']
    domain = d['domain']
    found = False

    for dir_path in json_dirs.get(domain, []):
        if not os.path.exists(dir_path):
            continue
        for fname in os.listdir(dir_path):
            if not fname.endswith('.json'):
                continue
            if 'checkpoint' in fname or 'metrics_only' in fname or 'BACKUP' in fname:
                continue
            if model not in fname:
                continue

            fpath = os.path.join(dir_path, fname)
            try:
                with open(fpath, encoding='utf-8') as fh:
                    jdata = json.load(fh)
                trials = jdata.get('trials', [])
                if not trials:
                    continue

                drcis = []
                for t in trials:
                    dr = t.get('delta_rci', t.get('drci', None))
                    if isinstance(dr, dict):
                        drcis.append(dr.get('cold', None))
                    elif isinstance(dr, (int, float)):
                        drcis.append(dr)

                drcis = [x for x in drcis if x is not None]
                if drcis:
                    json_drci = np.mean(drcis)
                    diff = abs(json_drci - d['drci'])
                    status = "OK" if diff < 0.02 else f"DIFF={diff:.4f}"
                    print(f"  {model:<25} {domain:<12} CSV={d['drci']:.4f}  JSON={json_drci:.4f}  {status}")
                    found = True
                    break
            except Exception as e:
                pass

    if not found:
        print(f"  {model:<25} {domain:<12} CSV={d['drci']:.4f}  JSON=NOT FOUND (computed from embeddings)")

print()

# ============================================================
# STEP 5: Recompute all statistics
# ============================================================
print("STEP 5: Recompute conservation law statistics")
print("-" * 40)

med = [d for d in data if d['domain'] == 'Medical']
phil = [d for d in data if d['domain'] == 'Philosophy']

mp = np.array([d['product'] for d in med])
pp = np.array([d['product'] for d in phil])
ap = np.array([d['product'] for d in data])

for label, prods in [("Medical", mp), ("Philosophy", pp), ("Overall", ap)]:
    mean = np.mean(prods)
    sd = np.std(prods, ddof=1)
    cv = sd / mean
    ci = stats.t.ppf(0.975, len(prods) - 1) * sd / np.sqrt(len(prods))
    print(f"  {label:>12}: N={len(prods)}, K={mean:.4f}, SD={sd:.4f}, CV={cv:.4f}, 95%CI=[{mean - ci:.4f}, {mean + ci:.4f}]")

u, mw_p = stats.mannwhitneyu(mp, pp, alternative='two-sided')
t_stat, tt_p = stats.ttest_ind(mp, pp, equal_var=False)
pooled_sd = np.sqrt((np.std(mp, ddof=1) ** 2 + np.std(pp, ddof=1) ** 2) / 2)
d_cohen = (np.mean(mp) - np.mean(pp)) / pooled_sd

print(f"\n  Mann-Whitney U = {u:.1f}, p = {mw_p:.4f}")
print(f"  Welch t = {t_stat:.4f}, p = {tt_p:.4f}")
print(f"  Cohen d = {d_cohen:.4f}")
print()

# ============================================================
# STEP 6: Conservation law verdict
# ============================================================
print("STEP 6: Conservation law verdict")
print("-" * 40)

cv_med = np.std(mp, ddof=1) / np.mean(mp)
cv_phil = np.std(pp, ddof=1) / np.mean(pp)
cv_all = np.std(ap, ddof=1) / np.mean(ap)

for label, cv in [("Medical", cv_med), ("Philosophy", cv_phil), ("Overall", cv_all)]:
    if cv < 0.20:
        v = "CONSERVATION HOLDS (CV < 0.20)"
    elif cv < 0.30:
        v = "WEAK CONSERVATION (0.20 < CV < 0.30)"
    else:
        v = "NO CONSERVATION (CV > 0.30)"
    print(f"  {label:>12}: CV = {cv:.4f} => {v}")

print()
print("=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)

# Final summary
print()
print("EXACT NUMBERS FOR PAPER 6:")
print(f"  Total model-domain runs: {len(data)}")
print(f"  Unique models: {len(unique_models)}")
print(f"  Unique vendors: {len(unique_vendors)}")
print(f"  Domains: {len(unique_domains)}")
print(f"  Medical: N={len(med)}, K={np.mean(mp):.3f}, SD={np.std(mp, ddof=1):.3f}, CV={cv_med:.3f}")
print(f"  Philosophy: N={len(phil)}, K={np.mean(pp):.3f}, SD={np.std(pp, ddof=1):.3f}, CV={cv_phil:.3f}")
print(f"  Domain difference: U={u:.0f}, p={mw_p:.4f}, d={d_cohen:.2f}")

# Check what paper currently says
print()
print("PAPER DRAFT CLAIMS TO VERIFY:")
print("  '14 model-domain configurations' => ACTUAL:", len(data))
print("  '11 architectures' => ACTUAL:", len(unique_models))
print("  '8 vendors' => ACTUAL:", len(unique_vendors))
print("  'K(Medical) = 0.429' => ACTUAL:", f"{np.mean(mp):.3f}")
print("  'K(Philosophy) = 0.301' => ACTUAL:", f"{np.mean(pp):.3f}")
print("  'CV = 0.170 (Med)' => ACTUAL:", f"{cv_med:.3f}")
print("  'CV = 0.167 (Phil)' => ACTUAL:", f"{cv_phil:.3f}")
print("  'U = 46, p = 0.003' => ACTUAL:", f"U={u:.0f}, p={mw_p:.4f}")
print("  'Cohen d = 2.06' => ACTUAL:", f"d={d_cohen:.2f}")
