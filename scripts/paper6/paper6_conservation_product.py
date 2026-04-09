#!/usr/bin/env python3
"""
Paper 6: Simpler Conservation Law Test
=======================================
Tests whether Product = ΔRCI × Var_Ratio is approximately constant.
No MI estimation needed.
"""
import pandas as pd
import numpy as np
import json
import sys
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')
BASE = Path("C:/Users/barla/mch_experiments")

# =============================================================
# COMBINE ALL SOURCES
# =============================================================

# Source 1: Paper 6 data (14 runs, embedding-based VR and dRCI)
with open(BASE / "data/paper6/conservation_law_verification/conservation_law_results.json") as f:
    p6 = json.load(f)

# Source 2: Entanglement position data (12 runs, embedding-based VR)
ent = pd.read_csv(BASE / "results/tables/entanglement_position_data.csv")
ent_vr = ent.groupby('model').agg(
    var_ratio=('var_ratio', 'mean'),
    drci_pos=('drci', 'mean')
).reset_index()

# Source 3: Trial-level dRCI (24 runs)
trial = pd.read_csv(BASE / "results/tables/trial_level_drci.csv")
trial_drci = trial.groupby(['model', 'domain'])['delta_rci'].mean().reset_index()
trial_drci.columns = ['model_short', 'domain', 'drci_trial']

# Map entanglement model names to short names + domain
name_map = {
    'GPT-4o-mini (Phil)': ('gpt4o_mini', 'Philosophy'),
    'GPT-4o (Phil)': ('gpt4o', 'Philosophy'),
    'Claude Haiku (Phil)': ('claude_haiku', 'Philosophy'),
    'Gemini Flash (Phil)': ('gemini_flash', 'Philosophy'),
    'Gemini Flash (Med)': ('gemini_flash', 'Medical'),
    'DeepSeek V3.1 (Med)': ('deepseek_v3_1', 'Medical'),
    'Llama 4 Maverick (Med)': ('llama_4_maverick', 'Medical'),
    'Llama 4 Scout (Med)': ('llama_4_scout', 'Medical'),
    'Qwen3 235B (Med)': ('qwen3_235b', 'Medical'),
    'Mistral Small 24B (Med)': ('mistral_small_24b', 'Medical'),
    'Ministral 14B (Med)': ('ministral_14b', 'Medical'),
    'Kimi K2 (Med)': ('kimi_k2', 'Medical'),
}

results = []
seen = set()

# Priority 1: Paper 6 data (freshest, embedding-based VR and dRCI)
for r in p6:
    key = f"{r['model']}_{r['domain']}"
    if key not in seen:
        seen.add(key)
        results.append({
            'model': r['model'],
            'domain': r['domain'],
            'drci': r['drci'],
            'var_ratio': r['var_ratio'],
            'source': 'Paper6_embedding'
        })

# Priority 2: Entanglement data (for any missing runs)
for _, row in ent_vr.iterrows():
    if row['model'] in name_map:
        short, domain = name_map[row['model']]
        key = f"{short}_{domain}"
        if key not in seen:
            mask = (trial_drci['model_short'] == short) & (trial_drci['domain'] == domain.lower())
            drci_rows = trial_drci[mask]
            if len(drci_rows) > 0:
                drci = drci_rows['drci_trial'].values[0]
            else:
                drci = 1.0 - row['drci_pos']
            seen.add(key)
            results.append({
                'model': short,
                'domain': domain,
                'drci': drci,
                'var_ratio': row['var_ratio'],
                'source': 'Entanglement_CSV'
            })

# Known correction: Gemini Flash Medical dRCI was -0.133 -> +0.427
for r in results:
    if r['model'] == 'gemini_flash' and r['domain'] == 'Medical' and r['drci'] < 0:
        r['drci'] = 0.4270

# Compute product
for r in results:
    r['product'] = r['drci'] * r['var_ratio']

results.sort(key=lambda x: (x['domain'], -x['product']))

# =============================================================
# DISPLAY TABLE
# =============================================================
print("CONSERVATION PRODUCT TEST: ΔRCI × Var_Ratio")
print("=" * 85)
print(f"{'Model':<25} {'Domain':<12} {'ΔRCI':>8} {'Var_Ratio':>10} {'Product':>10}  Source")
print("-" * 85)
for r in results:
    print(f"{r['model']:<25} {r['domain']:<12} {r['drci']:>8.4f} {r['var_ratio']:>10.4f} {r['product']:>10.4f}  {r['source']}")

products = np.array([r['product'] for r in results])
med_prods = np.array([r['product'] for r in results if r['domain'] == 'Medical'])
phil_prods = np.array([r['product'] for r in results if r['domain'] == 'Philosophy'])

print(f"\n{'=' * 60}")
print(f"ALL RUNS (N={len(results)})")
print(f"{'=' * 60}")
print(f"  Mean Product:    {np.mean(products):.4f}")
print(f"  Std Product:     {np.std(products):.4f}")
print(f"  CV:              {np.std(products)/np.mean(products):.4f}")
print(f"  Min:             {np.min(products):.4f}")
print(f"  Max:             {np.max(products):.4f}")
print(f"  Range:           {np.max(products) - np.min(products):.4f}")
print(f"  Median:          {np.median(products):.4f}")

for label, prods in [("Medical", med_prods), ("Philosophy", phil_prods)]:
    if len(prods) > 1:
        cv = np.std(prods) / np.mean(prods)
        print(f"\n  --- {label} (N={len(prods)}) ---")
        print(f"  Mean Product:    {np.mean(prods):.4f}")
        print(f"  Std Product:     {np.std(prods):.4f}")
        print(f"  CV:              {cv:.4f}")
        print(f"  Min:             {np.min(prods):.4f}")
        print(f"  Max:             {np.max(prods):.4f}")

# Domain comparison
t, p = stats.ttest_ind(med_prods, phil_prods)
print(f"\n  Domain comparison (t-test):")
print(f"  Medical mean:    {np.mean(med_prods):.4f}")
print(f"  Philosophy mean: {np.mean(phil_prods):.4f}")
print(f"  t = {t:.4f}, p = {p:.4f}")

# Verdict
cv_all = np.std(products) / np.mean(products)
cv_med = np.std(med_prods) / np.mean(med_prods)
cv_phil = np.std(phil_prods) / np.mean(phil_prods)

print(f"\n{'=' * 60}")
print("VERDICT")
print(f"{'=' * 60}")

for label, cv in [("Overall", cv_all), ("Medical", cv_med), ("Philosophy", cv_phil)]:
    if cv < 0.2:
        v = "CONSERVATION LAW HOLDS"
    elif cv < 0.3:
        v = "WEAK CONSERVATION"
    else:
        v = "NO CONSERVATION"
    print(f"  {label:>12} CV = {cv:.3f}  =>  {v}")

# =============================================================
# SAVE CSV
# =============================================================
csv_path = BASE / "data" / "paper6" / "conservation_product_test.csv"
csv_path.parent.mkdir(parents=True, exist_ok=True)
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['model', 'domain', 'drci', 'var_ratio', 'product', 'source'])
    w.writeheader()
    for r in results:
        w.writerow({k: r[k] for k in ['model', 'domain', 'drci', 'var_ratio', 'product', 'source']})
print(f"\nSaved: {csv_path}")

# =============================================================
# FIGURE
# =============================================================
fig, ax = plt.subplots(figsize=(14, 7))

colors = {'Medical': '#e74c3c', 'Philosophy': '#3498db'}
x_positions = range(len(results))

for i, r in enumerate(results):
    ax.bar(i, r['product'], color=colors[r['domain']],
           edgecolor='black', linewidth=0.5, alpha=0.85)

# Mean lines
ax.axhline(y=np.mean(products), color='black', linestyle='--', linewidth=2,
           label=f"Overall Mean = {np.mean(products):.3f} (CV={cv_all:.3f})")
ax.axhline(y=np.mean(med_prods), color='#e74c3c', linestyle=':', linewidth=1.5, alpha=0.7,
           label=f"Medical Mean = {np.mean(med_prods):.3f} (CV={cv_med:.3f})")
ax.axhline(y=np.mean(phil_prods), color='#3498db', linestyle=':', linewidth=1.5, alpha=0.7,
           label=f"Philosophy Mean = {np.mean(phil_prods):.3f} (CV={cv_phil:.3f})")

# Labels
model_labels = []
for r in results:
    name = r['model'].replace('_', ' ').title()
    if len(name) > 15:
        name = name[:14] + '.'
    model_labels.append(f"{name}\n({r['domain'][:4]})")

ax.set_xticks(x_positions)
ax.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel(r'Product = $\Delta$RCI $\times$ Var_Ratio', fontsize=12)
ax.set_title(
    f"Conservation Law Test: $\\Delta$RCI $\\times$ Var_Ratio Across {len(results)} Model-Domain Runs\n"
    f"Overall CV = {cv_all:.3f} | Medical CV = {cv_med:.3f} | Philosophy CV = {cv_phil:.3f}",
    fontsize=13
)
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max(products) * 1.15)

# Verdict annotation
if cv_all < 0.2:
    verdict_text = "CONSERVATION LAW HOLDS"
    box_color = '#d5f5e3'
elif cv_med < 0.2 and cv_phil < 0.2:
    verdict_text = "DOMAIN-SPECIFIC CONSERVATION"
    box_color = '#fef9e7'
elif cv_all < 0.3:
    verdict_text = "WEAK CONSERVATION"
    box_color = '#fef9e7'
else:
    verdict_text = "NO CONSERVATION"
    box_color = '#fadbd8'

ax.text(0.02, 0.95, f"Verdict: {verdict_text}",
        transform=ax.transAxes, fontsize=14, fontweight='bold', va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor=box_color, edgecolor='gray'))

plt.tight_layout()
fig_path = BASE / "docs" / "figures" / "paper6" / "fig_conservation_product.png"
fig.savefig(fig_path, dpi=200)
plt.close()
print(f"Saved: {fig_path}")
