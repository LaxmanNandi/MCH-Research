import json
import os
import statistics

# All files to process
files = [
    # Philosophy rerun (closed models)
    ('C:/Users/barla/mch_experiments/data/closed_model_philosophy_rerun/mch_results_gpt4o_mini_philosophy_50trials.json', 'philosophy'),
    ('C:/Users/barla/mch_experiments/data/closed_model_philosophy_rerun/mch_results_gpt4o_philosophy_50trials.json', 'philosophy'),
    ('C:/Users/barla/mch_experiments/data/closed_model_philosophy_rerun/mch_results_claude_haiku_philosophy_checkpoint.json', 'philosophy'),
    # Medical results (closed models)
    ('C:/Users/barla/mch_experiments/data/medical_results/mch_results_gpt4o_mini_medical_50trials.json', 'medical'),
    ('C:/Users/barla/mch_experiments/data/medical_results/mch_results_gpt4o_medical_50trials.json', 'medical'),
    ('C:/Users/barla/mch_experiments/data/medical_results/mch_results_claude_haiku_medical_50trials.json', 'medical'),
    ('C:/Users/barla/mch_experiments/data/medical_results/mch_results_gemini_flash_medical_50trials.json', 'medical'),
    ('C:/Users/barla/mch_experiments/data/medical_results/gemini_pro_safety_blocked.json', 'medical'),
    ('C:/Users/barla/mch_experiments/data/medical_results/mch_results_gpt_5_2_medical_50trials.json', 'medical'),
    ('C:/Users/barla/mch_experiments/data/medical_results/mch_results_gpt4o_mini_rerun_medical_50trials.json', 'medical'),
    ('C:/Users/barla/mch_experiments/data/medical_results/mch_results_claude_opus_medical_43trials_recovered.json', 'medical'),
    ('C:/Users/barla/mch_experiments/data/medical_results/mch_results_claude_opus_medical_50trials.json', 'medical'),
    # Open model results (philosophy)
    ('C:/Users/barla/mch_experiments/data/open_model_results/mch_results_deepseek_v3_1_philosophy_50trials.json', 'philosophy'),
    ('C:/Users/barla/mch_experiments/data/open_model_results/mch_results_llama_4_maverick_philosophy_50trials.json', 'philosophy'),
    ('C:/Users/barla/mch_experiments/data/open_model_results/mch_results_llama_4_scout_philosophy_50trials.json', 'philosophy'),
    ('C:/Users/barla/mch_experiments/data/open_model_results/mch_results_qwen3_235b_philosophy_50trials.json', 'philosophy'),
    ('C:/Users/barla/mch_experiments/data/open_model_results/mch_results_mistral_small_24b_philosophy_50trials.json', 'philosophy'),
    ('C:/Users/barla/mch_experiments/data/open_model_results/mch_results_ministral_14b_philosophy_50trials.json', 'philosophy'),
    ('C:/Users/barla/mch_experiments/data/open_model_results/mch_results_kimi_k2_philosophy_50trials.json', 'philosophy'),
]

results = []

for fpath, domain_label in files:
    fname = os.path.basename(fpath)

    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        results.append({
            'file': fname,
            'model': 'ERROR',
            'model_id': '',
            'domain': domain_label,
            'n_trials': 0,
            'error': str(e)
        })
        continue

    # Check if this is the blocked file
    if 'model_attempts' in data:
        results.append({
            'file': fname,
            'model': 'gemini_pro (2.5+3)',
            'model_id': 'gemini-2.5-pro / gemini-3-pro',
            'domain': domain_label,
            'n_trials': 0,
            'status': 'BLOCKED',
            'mean_true': None,
            'mean_cold': None,
            'mean_scrambled': None,
            'delta_true_cold': None,
            'delta_true_scrambled': None,
            'ordering_holds': None,
            'note': 'Safety filter blocked - no trial data'
        })
        continue

    model_name = data.get('model', 'unknown')
    model_id = data.get('model_id', 'unknown')
    domain = data.get('domain', domain_label)
    n_trials_declared = data.get('n_trials', 0)
    status = data.get('status', 'COMPLETE')

    trials = data.get('trials', [])

    # Check for recovered format (delta_rci only, no raw alignments)
    if trials and 'delta_rci' in trials[0] and 'alignments' not in trials[0]:
        cold_drcis = []
        for t in trials:
            dr = t.get('delta_rci', {})
            c = dr.get('cold')
            if c is not None:
                cold_drcis.append(c)

        mean_drci = data.get('summary', {}).get('mean_drci', None)

        results.append({
            'file': fname,
            'model': model_name,
            'model_id': model_id,
            'domain': domain,
            'n_trials': len(cold_drcis),
            'status': status,
            'mean_true': None,
            'mean_cold': None,
            'mean_scrambled': None,
            'delta_true_cold': None,
            'delta_true_scrambled': None,
            'ordering_holds': None,
            'mean_drci_cold': statistics.mean(cold_drcis) if cold_drcis else None,
            'note': f'RECOVERED format - only dRCI values (mean_drci={mean_drci})'
        })
        continue

    # Standard format: trials[i].alignments.true[], .cold[], .scrambled[]
    all_true_means = []
    all_cold_means = []
    all_scrambled_means = []

    has_scrambled = False

    for t in trials:
        aligns = t.get('alignments', {})

        true_vals = aligns.get('true', [])
        cold_vals = aligns.get('cold', [])
        scrambled_vals = aligns.get('scrambled', [])

        if true_vals:
            all_true_means.append(statistics.mean(true_vals))
        if cold_vals:
            all_cold_means.append(statistics.mean(cold_vals))
        if scrambled_vals:
            has_scrambled = True
            all_scrambled_means.append(statistics.mean(scrambled_vals))

    mean_true = statistics.mean(all_true_means) if all_true_means else None
    mean_cold = statistics.mean(all_cold_means) if all_cold_means else None
    mean_scrambled = statistics.mean(all_scrambled_means) if all_scrambled_means else None

    delta_true_cold = (mean_true - mean_cold) if (mean_true is not None and mean_cold is not None) else None
    delta_true_scrambled = (mean_true - mean_scrambled) if (mean_true is not None and mean_scrambled is not None) else None

    # Check ordering: TRUE > SCRAMBLED > COLD
    if mean_true is not None and mean_scrambled is not None and mean_cold is not None:
        ordering_holds = mean_true > mean_scrambled > mean_cold
    else:
        ordering_holds = None

    results.append({
        'file': fname,
        'model': model_name,
        'model_id': model_id,
        'domain': domain,
        'n_trials': len(all_true_means),
        'status': status,
        'has_scrambled': has_scrambled,
        'mean_true': mean_true,
        'mean_cold': mean_cold,
        'mean_scrambled': mean_scrambled,
        'delta_true_cold': delta_true_cold,
        'delta_true_scrambled': delta_true_scrambled,
        'ordering_holds': ordering_holds,
        'note': ''
    })


def fmt(val, width=10):
    if val is None:
        return 'N/A'.rjust(width)
    return f"{val:.6f}".rjust(width)

def fmt_short(val, width=8):
    if val is None:
        return 'N/A'.rjust(width)
    return f"{val:.4f}".rjust(width)


# ===== MAIN TABLE =====
print()
print('=' * 155)
print('  MCH EXPERIMENT RESULTS: TRUE vs SCRAMBLED vs COLD MEAN ALIGNMENT VALUES (across all trials)')
print('=' * 155)
print()

hdr = (f"{'#':>2}  {'Model':<26} {'Domain':<10} {'N':>4}  "
       f"{'Mean TRUE':>10}  {'Mean SCRM':>10}  {'Mean COLD':>10}  "
       f"{'D(T-C)':>8}  {'D(T-S)':>8}  {'T>S>C':>5}  {'Notes':<30}")
print(hdr)
print('-' * 155)

for i, r in enumerate(results):
    model = r.get('model', '?')
    domain = r.get('domain', '?')
    n = r.get('n_trials', 0)

    mt = fmt(r.get('mean_true'))
    ms = fmt(r.get('mean_scrambled'))
    mc = fmt(r.get('mean_cold'))
    dtc = fmt_short(r.get('delta_true_cold'))
    dts = fmt_short(r.get('delta_true_scrambled'))

    oh = r.get('ordering_holds')
    if oh is True:
        order = '  YES'
    elif oh is False:
        order = '   NO'
    else:
        order = '  N/A'

    note = r.get('note', '') or r.get('error', '') or ''
    if r.get('status') and r.get('status') != 'COMPLETE':
        note = f"[{r['status']}] {note}"
    note = note[:45]

    print(f"{i+1:>2}  {model:<26} {domain:<10} {n:>4}  "
          f"{mt}  {ms}  {mc}  "
          f"{dtc}  {dts}  {order}  {note}")


# ===== DETAILED FILE LISTING =====
print()
print()
print('=' * 100)
print('  DETAILED FILE INVENTORY')
print('=' * 100)

for i, (fpath, _) in enumerate(files):
    fname = os.path.basename(fpath)
    dirn = os.path.basename(os.path.dirname(fpath))
    r = results[i]

    print(f"\n  File {i+1}: {fname}")
    print(f"    Directory : {dirn}/")
    print(f"    Model     : {r.get('model','?')} ({r.get('model_id','?')})")
    print(f"    Domain    : {r.get('domain','?')}")
    print(f"    Trials    : {r.get('n_trials','?')}")

    mt = r.get('mean_true')
    mc = r.get('mean_cold')
    ms = r.get('mean_scrambled')

    print(f"    Mean TRUE      : {mt:.6f}" if mt is not None else "    Mean TRUE      : N/A")
    print(f"    Mean SCRAMBLED : {ms:.6f}" if ms is not None else "    Mean SCRAMBLED : N/A")
    print(f"    Mean COLD      : {mc:.6f}" if mc is not None else "    Mean COLD      : N/A")

    dtc = r.get('delta_true_cold')
    dts = r.get('delta_true_scrambled')
    print(f"    Delta T-COLD   : {dtc:.6f}" if dtc is not None else "    Delta T-COLD   : N/A")
    print(f"    Delta T-SCRM   : {dts:.6f}" if dts is not None else "    Delta T-SCRM   : N/A")

    oh = r.get('ordering_holds')
    if oh is True:
        print("    T>S>C ordering : YES")
    elif oh is False:
        print("    T>S>C ordering : NO")
    else:
        print("    T>S>C ordering : N/A")

    note = r.get('note', '')
    if note:
        print(f"    Notes     : {note}")


# ===== SUMMARY STATISTICS =====
print()
print()
print('=' * 100)
print('  SUMMARY STATISTICS BY DOMAIN')
print('=' * 100)

# Philosophy
phil = [r for r in results if r.get('domain') == 'philosophy' and r.get('mean_true') is not None]
print(f"\n  PHILOSOPHY DOMAIN ({len(phil)} models with full alignment data):")
print(f"  {'Model':<26} {'Mean TRUE':>10}  {'Mean SCRM':>10}  {'Mean COLD':>10}  {'D(T-C)':>8}  {'D(T-S)':>8}  {'T>S>C':>5}")
print(f"  {'-'*90}")
for r in phil:
    mt = f"{r['mean_true']:.6f}"
    mc = f"{r['mean_cold']:.6f}"
    ms = f"{r['mean_scrambled']:.6f}" if r['mean_scrambled'] is not None else 'N/A'
    dtc = f"{r['delta_true_cold']:.4f}" if r['delta_true_cold'] is not None else 'N/A'
    dts = f"{r['delta_true_scrambled']:.4f}" if r['delta_true_scrambled'] is not None else 'N/A'
    oh = 'YES' if r.get('ordering_holds') else ('NO' if r.get('ordering_holds') is False else 'N/A')
    print(f"  {r['model']:<26} {mt:>10}  {ms:>10}  {mc:>10}  {dtc:>8}  {dts:>8}  {oh:>5}")

if phil:
    avg_true = statistics.mean([r['mean_true'] for r in phil])
    avg_cold = statistics.mean([r['mean_cold'] for r in phil])
    scrm_vals = [r['mean_scrambled'] for r in phil if r['mean_scrambled'] is not None]
    avg_scrm = statistics.mean(scrm_vals) if scrm_vals else None
    print(f"  {'-'*90}")
    ms_avg = f"{avg_scrm:.6f}" if avg_scrm is not None else 'N/A'
    print(f"  {'AVERAGE':<26} {avg_true:>10.6f}  {ms_avg:>10}  {avg_cold:>10.6f}")

# Medical
med = [r for r in results if r.get('domain') in ('medical_reasoning', 'medical') and r.get('mean_true') is not None]
print(f"\n  MEDICAL DOMAIN ({len(med)} models with full alignment data):")
print(f"  {'Model':<26} {'Mean TRUE':>10}  {'Mean SCRM':>10}  {'Mean COLD':>10}  {'D(T-C)':>8}  {'D(T-S)':>8}  {'T>S>C':>5}")
print(f"  {'-'*90}")
for r in med:
    mt = f"{r['mean_true']:.6f}"
    mc = f"{r['mean_cold']:.6f}"
    ms = f"{r['mean_scrambled']:.6f}" if r['mean_scrambled'] is not None else 'N/A'
    dtc = f"{r['delta_true_cold']:.4f}" if r['delta_true_cold'] is not None else 'N/A'
    dts = f"{r['delta_true_scrambled']:.4f}" if r['delta_true_scrambled'] is not None else 'N/A'
    oh = 'YES' if r.get('ordering_holds') else ('NO' if r.get('ordering_holds') is False else 'N/A')
    print(f"  {r['model']:<26} {mt:>10}  {ms:>10}  {mc:>10}  {dtc:>8}  {dts:>8}  {oh:>5}")

if med:
    avg_true = statistics.mean([r['mean_true'] for r in med])
    avg_cold = statistics.mean([r['mean_cold'] for r in med])
    scrm_vals = [r['mean_scrambled'] for r in med if r['mean_scrambled'] is not None]
    avg_scrm = statistics.mean(scrm_vals) if scrm_vals else None
    print(f"  {'-'*90}")
    ms_avg = f"{avg_scrm:.6f}" if avg_scrm is not None else 'N/A'
    print(f"  {'AVERAGE':<26} {avg_true:>10.6f}  {ms_avg:>10}  {avg_cold:>10.6f}")

# Special cases
special = [r for r in results if r.get('mean_true') is None]
if special:
    print(f"\n  SPECIAL CASES ({len(special)} files without full alignment data):")
    for r in special:
        note = r.get('note', '') or r.get('error', '') or r.get('status', '')
        drci = r.get('mean_drci_cold')
        drci_str = f", mean_dRCI_cold={drci:.4f}" if drci is not None else ""
        print(f"    {r['model']:<26} [{r['domain']}] {note}{drci_str}")

# Overall ordering stats
print(f"\n  ORDERING ANALYSIS (TRUE > SCRAMBLED > COLD):")
all_with_order = [r for r in results if r.get('ordering_holds') is not None]
holds = sum(1 for r in all_with_order if r['ordering_holds'])
fails = sum(1 for r in all_with_order if not r['ordering_holds'])
print(f"    Models where T>S>C holds: {holds}/{len(all_with_order)}")
print(f"    Models where T>S>C fails: {fails}/{len(all_with_order)}")
if all_with_order:
    for r in all_with_order:
        symbol = 'PASS' if r['ordering_holds'] else 'FAIL'
        print(f"      [{symbol}] {r['model']:<26} ({r['domain']}) "
              f"T={r['mean_true']:.4f} S={r['mean_scrambled']:.4f} C={r['mean_cold']:.4f}")

print()
print('=' * 100)
