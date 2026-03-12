"""Verify K decomposition claims from Paper 7."""
import csv
import numpy as np
from scipy import stats

# Read Paper 6 conservation data
p6 = {}
with open('c:/Users/barla/mch_experiments/data/paper6/conservation_product_test.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['model'] and row['domain']:
            key = (row['model'].strip(), row['domain'].strip())
            p6[key] = {
                'drci': float(row['drci']),
                'var_ratio': float(row['var_ratio']),
                'product': float(row['product'])
            }

# Read variance decomposition data
vd = {}
with open('c:/Users/barla/mch_experiments/paper7_submission/data/variance_decomposition/var_ratio_scrambled_decomposition_results.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        model = row.get('Model', '').strip()
        domain = row.get('Domain', '').strip()
        if model and domain and model != 'Domain Summaries' and model != '':
            try:
                key = (model, domain)
                vd[key] = {
                    'vr_total': float(row['VR_Total']),
                    'vr_content': float(row['VR_Content']),
                    'vr_order': float(row['VR_Order'])
                }
            except (ValueError, KeyError):
                pass

print('=== Paper 6 keys ===')
for k in sorted(p6.keys()):
    print(f'  {k}')
print()
print('=== Variance Decomposition keys ===')
for k in sorted(vd.keys()):
    print(f'  {k}')
print()

# Match and compute
med_k_total, med_k_content, med_k_order = [], [], []
phil_k_total, phil_k_content, phil_k_order = [], [], []
med_models, phil_models = [], []

print('=== Per-model K decomposition ===')
header = f'{"Model":<22} {"Domain":<12} {"DRCI":>8} {"VR_Tot":>8} {"VR_Cont":>8} {"VR_Ord":>8} {"K_total":>8} {"K_content":>8} {"K_order":>8}'
print(header)
print('-' * 100)

for key in sorted(p6.keys()):
    if key in vd:
        drci = p6[key]['drci']
        vr_t = vd[key]['vr_total']
        vr_c = vd[key]['vr_content']
        vr_o = vd[key]['vr_order']
        k_t = drci * vr_t
        k_c = drci * vr_c
        k_o = drci * vr_o
        print(f'{key[0]:<22} {key[1]:<12} {drci:>8.4f} {vr_t:>8.4f} {vr_c:>8.4f} {vr_o:>8.4f} {k_t:>8.4f} {k_c:>8.4f} {k_o:>8.4f}')
        if key[1] == 'Medical':
            med_k_total.append(k_t)
            med_k_content.append(k_c)
            med_k_order.append(k_o)
            med_models.append(key[0])
        else:
            phil_k_total.append(k_t)
            phil_k_content.append(k_c)
            phil_k_order.append(k_o)
            phil_models.append(key[0])
    else:
        print(f'  NO MATCH for {key}')

# Unmatched from VD
for key in sorted(vd.keys()):
    if key not in p6:
        print(f'  In VD but not Paper 6: {key}')

print()
print(f'Matched: {len(med_models)} Medical models, {len(phil_models)} Philosophy models')
print(f'Medical models: {med_models}')
print(f'Philosophy models: {phil_models}')

print()
print('=== Domain summaries ===')
for label, med, phil in [('K_total', med_k_total, phil_k_total),
                          ('K_content', med_k_content, phil_k_content),
                          ('K_order', med_k_order, phil_k_order)]:
    med_arr = np.array(med)
    phil_arr = np.array(phil)
    med_mean = np.mean(med_arr)
    med_std = np.std(med_arr, ddof=1)
    med_cv = med_std / med_mean if med_mean != 0 else 0
    phil_mean = np.mean(phil_arr)
    phil_std = np.std(phil_arr, ddof=1)
    phil_cv = phil_std / phil_mean if phil_mean != 0 else 0
    u_stat, p_val = stats.mannwhitneyu(med_arr, phil_arr, alternative='two-sided')
    print(f'{label}:')
    print(f'  Medical:    mean={med_mean:.4f}, std={med_std:.4f}, CV={med_cv:.4f} (N={len(med)})')
    print(f'  Philosophy: mean={phil_mean:.4f}, std={phil_std:.4f}, CV={phil_cv:.4f} (N={len(phil)})')
    print(f'  Mann-Whitney U={u_stat:.1f}, p={p_val:.6f}')
    print()

print('=' * 60)
print('=== CLAIM VERIFICATION ===')
print('=' * 60)

med_kt_mean = np.mean(med_k_total)
phil_kt_mean = np.mean(phil_k_total)
med_cv_kt = np.std(np.array(med_k_total), ddof=1) / med_kt_mean
phil_cv_kt = np.std(np.array(phil_k_total), ddof=1) / phil_kt_mean

u_kt, p_kt = stats.mannwhitneyu(np.array(med_k_total), np.array(phil_k_total), alternative='two-sided')
u_kc, p_kc = stats.mannwhitneyu(np.array(med_k_content), np.array(phil_k_content), alternative='two-sided')
u_ko, p_ko = stats.mannwhitneyu(np.array(med_k_order), np.array(phil_k_order), alternative='two-sided')

print()
claims = [
    ('Medical K_total=0.437', f'{med_kt_mean:.3f}', abs(med_kt_mean - 0.437) < 0.005),
    ('Medical K_total CV=0.09', f'{med_cv_kt:.2f}', abs(med_cv_kt - 0.09) < 0.005),
    ('Philosophy K_total=0.120', f'{phil_kt_mean:.3f}', abs(phil_kt_mean - 0.120) < 0.005),
    ('Philosophy K_total CV=0.58', f'{phil_cv_kt:.2f}', abs(phil_cv_kt - 0.58) < 0.005),
    (f'K_total U=56, p=0.0003', f'U={u_kt:.0f}, p={p_kt:.4f}', abs(u_kt - 56) < 0.5 and abs(p_kt - 0.0003) < 0.001),
    (f'K_content U=48, p=0.021', f'U={u_kc:.0f}, p={p_kc:.4f}', abs(u_kc - 48) < 0.5 and abs(p_kc - 0.021) < 0.01),
    (f'K_order U=56, p=0.0003', f'U={u_ko:.0f}, p={p_ko:.4f}', abs(u_ko - 56) < 0.5 and abs(p_ko - 0.0003) < 0.001),
]

for claim_text, computed, match in claims:
    status = "CONFIRMED" if match else "MISMATCH"
    print(f'  Claim: {claim_text:<35} Computed: {computed:<20} [{status}]')

# Additional: check VR decomposition relationship
print()
print('=== VR decomposition check ===')
print('VR_Total should equal VR_Content * VR_Order (multiplicative decomposition)')
for key in sorted(vd.keys()):
    vr_t = vd[key]['vr_total']
    vr_c = vd[key]['vr_content']
    vr_o = vd[key]['vr_order']
    product = vr_c * vr_o
    ratio = product / vr_t if vr_t != 0 else 0
    print(f'  {key[0]:<22} {key[1]:<12} VR_Tot={vr_t:.4f} VR_C*VR_O={product:.4f} ratio={ratio:.4f}')
