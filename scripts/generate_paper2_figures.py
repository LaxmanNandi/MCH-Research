"""
Paper 2 Figure Generation: Cross-Domain AI Behavior Framework
Generates all 6 figures for the standardized cross-domain study.

Usage: python scripts/generate_paper2_figures.py
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from pathlib import Path

# ============================================================
# Configuration
# ============================================================
BASE_DIR = Path("C:/Users/barla/mch_experiments")
DATA_DIR = BASE_DIR / "data"
FIG_DIR = BASE_DIR / "papers" / "paper2_standardized" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Publication style
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})

# Model registry: (file_key, display_name, vendor, type)
MODELS = {
    'philosophy': {
        'closed': [
            ('gpt4o', 'GPT-4o', 'OpenAI'),
            ('gpt4o_mini', 'GPT-4o-mini', 'OpenAI'),
            ('gpt_5_2', 'GPT-5.2', 'OpenAI'),
            ('claude_haiku', 'Claude Haiku', 'Anthropic'),
            ('gemini_flash', 'Gemini Flash', 'Google'),
        ],
        'open': [
            ('deepseek_v3_1', 'DeepSeek V3.1', 'DeepSeek'),
            ('kimi_k2', 'Kimi K2', 'Moonshot'),
            ('llama_4_maverick', 'Llama 4 Maverick', 'Meta'),
            ('llama_4_scout', 'Llama 4 Scout', 'Meta'),
            ('ministral_14b', 'Ministral 14B', 'Mistral'),
            ('mistral_small_24b', 'Mistral Small 24B', 'Mistral'),
            ('qwen3_235b', 'Qwen3 235B', 'Alibaba'),
        ],
    },
    'medical': {
        'closed': [
            ('gpt4o', 'GPT-4o', 'OpenAI'),
            ('gpt4o_mini_rerun', 'GPT-4o-mini', 'OpenAI'),
            ('gpt_5_2', 'GPT-5.2', 'OpenAI'),
            ('claude_haiku', 'Claude Haiku', 'Anthropic'),
            ('claude_opus', 'Claude Opus', 'Anthropic'),
            ('gemini_flash', 'Gemini Flash', 'Google'),
        ],
        'open': [
            ('deepseek_v3_1', 'DeepSeek V3.1', 'DeepSeek'),
            ('kimi_k2', 'Kimi K2', 'Moonshot'),
            ('llama_4_maverick', 'Llama 4 Maverick', 'Meta'),
            ('llama_4_scout', 'Llama 4 Scout', 'Meta'),
            ('ministral_14b', 'Ministral 14B', 'Mistral'),
            ('mistral_small_24b', 'Mistral Small 24B', 'Mistral'),
            ('qwen3_235b', 'Qwen3 235B', 'Alibaba'),
        ],
    },
}

# Vendor colors
VENDOR_COLORS = {
    'OpenAI': '#10a37f',
    'Anthropic': '#d4a574',
    'Google': '#4285f4',
    'Meta': '#0668E1',
    'DeepSeek': '#1a1a2e',
    'Moonshot': '#7c3aed',
    'Mistral': '#ff6b35',
    'Alibaba': '#ff6a00',
}

DOMAIN_COLORS = {
    'philosophy': '#6366f1',
    'medical': '#ef4444',
}


# ============================================================
# Data Loading
# ============================================================

def load_model_data(domain, model_type, file_key):
    """Load a model's JSON data file."""
    subdir = DATA_DIR / domain / f"{model_type}_models"
    for f in subdir.iterdir():
        if f.suffix == '.json' and file_key in f.name and 'checkpoint' not in f.name and 'recovered' not in f.name:
            if file_key == 'gpt4o' and 'mini' in f.name:
                continue
            if file_key == 'gpt4o' and 'rerun' in f.name:
                continue
            if file_key == 'gpt4o_mini_rerun' and 'rerun' not in f.name:
                continue
            if file_key == 'gpt4o_mini' and 'rerun' in f.name and file_key == 'gpt4o_mini':
                continue
            with open(f, encoding='utf-8') as fh:
                return json.load(fh)
    return None


def extract_drcis(data, n_trials=50):
    """Extract per-trial dRCI (cold) values."""
    trials = data.get('trials', [])[:n_trials]
    drcis = []
    for t in trials:
        dr = t.get('delta_rci', {})
        if isinstance(dr, dict):
            val = dr.get('cold')
            if val is not None:
                drcis.append(val)
        elif isinstance(dr, (int, float)):
            drcis.append(dr)
    return np.array(drcis)


def extract_position_drcis(data, n_trials=50):
    """Extract position-level dRCI from alignment data."""
    trials = data.get('trials', [])[:n_trials]
    # Each trial has alignments with per-position similarities
    n_positions = 30
    position_drcis = {p: [] for p in range(n_positions)}

    for t in trials:
        aligns = t.get('alignments', {})
        if not isinstance(aligns, dict):
            continue
        cold_sims = aligns.get('cold', aligns.get('true_vs_cold', []))
        scram_sims = aligns.get('scrambled', aligns.get('true_vs_scrambled', []))
        true_sims = aligns.get('true', aligns.get('true_self', []))

        if isinstance(cold_sims, list) and len(cold_sims) >= n_positions:
            for p in range(min(len(cold_sims), n_positions)):
                # dRCI at position p = true_self[p] - cold[p] (or just cold similarity)
                if isinstance(true_sims, list) and len(true_sims) > p:
                    pos_drci = true_sims[p] - cold_sims[p]
                else:
                    pos_drci = cold_sims[p]
                position_drcis[p].append(pos_drci)

    return position_drcis


def load_all_data():
    """Load all 25 model-domain runs."""
    results = []
    for domain in ['philosophy', 'medical']:
        for mtype in ['closed', 'open']:
            for file_key, display_name, vendor in MODELS[domain][mtype]:
                data = load_model_data(domain, mtype, file_key)
                if data is None:
                    print(f"  WARNING: Could not load {file_key} ({domain}/{mtype})")
                    continue
                drcis = extract_drcis(data)
                if len(drcis) == 0:
                    print(f"  WARNING: No dRCI data for {file_key} ({domain})")
                    continue
                results.append({
                    'file_key': file_key,
                    'display_name': display_name,
                    'vendor': vendor,
                    'domain': domain,
                    'model_type': mtype,
                    'drcis': drcis,
                    'mean_drci': np.mean(drcis),
                    'std_drci': np.std(drcis, ddof=1),
                    'se_drci': np.std(drcis, ddof=1) / np.sqrt(len(drcis)),
                    'n_trials': len(drcis),
                    'data': data,
                })
                print(f"  Loaded {display_name} ({domain}/{mtype}): n={len(drcis)}, mean={np.mean(drcis):.4f}")
    return results


# ============================================================
# Figure 1: Dataset Overview Heatmap
# ============================================================

def fig1_dataset_overview(results):
    """Heatmap: 14 models × 2 domains, colored by mean ΔRCI."""
    # Get unique models (by display_name) and their domain data
    model_data = {}
    for r in results:
        name = r['display_name']
        if name not in model_data:
            model_data[name] = {'vendor': r['vendor'], 'philosophy': None, 'medical': None}
        model_data[name][r['domain']] = r['mean_drci']

    # Sort by vendor then name
    vendor_order = ['OpenAI', 'Anthropic', 'Google', 'DeepSeek', 'Moonshot', 'Meta', 'Mistral', 'Alibaba']
    sorted_models = sorted(model_data.items(),
                          key=lambda x: (vendor_order.index(x[1]['vendor']) if x[1]['vendor'] in vendor_order else 99, x[0]))

    names = [m[0] for m in sorted_models]
    vendors = [m[1]['vendor'] for m in sorted_models]
    phil_vals = [m[1]['philosophy'] for m in sorted_models]
    med_vals = [m[1]['medical'] for m in sorted_models]

    fig, ax = plt.subplots(figsize=(8, 7))

    # Build matrix
    matrix = np.full((len(names), 2), np.nan)
    for i, (pv, mv) in enumerate(zip(phil_vals, med_vals)):
        if pv is not None:
            matrix[i, 0] = pv
        if mv is not None:
            matrix[i, 1] = mv

    # Plot heatmap
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-0.15, vmax=0.45)

    # Annotate cells
    for i in range(len(names)):
        for j in range(2):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, '—', ha='center', va='center', fontsize=9, color='gray')
            else:
                color = 'white' if val < 0 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=9, color=color)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Philosophy', 'Medical'], fontsize=11)
    ax.set_yticks(range(len(names)))
    # Add vendor prefix
    ylabels = [f'{v[:3]}  {n}' for n, v in zip(names, vendors)]
    ax.set_yticklabels(ylabels, fontsize=9)

    # Color y-labels by vendor
    for i, v in enumerate(vendors):
        ax.get_yticklabels()[i].set_color(VENDOR_COLORS.get(v, 'black'))

    plt.colorbar(im, ax=ax, label='Mean ΔRCI', shrink=0.8)
    ax.set_title('Paper 2 Dataset: Mean ΔRCI by Model and Domain\n(14 models, 25 model-domain runs, 50 trials each)', fontsize=12)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig1_dataset_overview.png')
    plt.close()
    print("  Saved fig1_dataset_overview.png")


# ============================================================
# Figure 2: Domain Comparison (Box/Violin Plots)
# ============================================================

def fig2_domain_comparison(results):
    """Philosophy vs Medical ΔRCI distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 1.5]})

    # Left: Overall domain comparison (violin)
    phil_means = [r['mean_drci'] for r in results if r['domain'] == 'philosophy']
    med_means = [r['mean_drci'] for r in results if r['domain'] == 'medical']

    ax = axes[0]
    parts = ax.violinplot([phil_means, med_means], positions=[0, 1], showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        color = DOMAIN_COLORS['philosophy'] if i == 0 else DOMAIN_COLORS['medical']
        pc.set_facecolor(color)
        pc.set_alpha(0.3)
    parts['cmeans'].set_color('black')
    parts['cmedians'].set_color('gray')

    # Overlay individual points
    for r in results:
        x = 0 if r['domain'] == 'philosophy' else 1
        color = DOMAIN_COLORS[r['domain']]
        ax.scatter(x + np.random.uniform(-0.1, 0.1), r['mean_drci'],
                  c=color, s=40, alpha=0.7, edgecolors='white', linewidth=0.5, zorder=5)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Philosophy\n(12 models)', 'Medical\n(13 models)'])
    ax.set_ylabel('Mean ΔRCI')
    ax.set_title('Domain Comparison')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

    # Mann-Whitney test
    u_stat, p_val = stats.mannwhitneyu(phil_means, med_means, alternative='two-sided')
    ax.text(0.5, 0.95, f'U={u_stat:.0f}, p={p_val:.3f}',
           transform=ax.transAxes, ha='center', va='top', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Right: Per-model paired comparison (models in both domains)
    ax = axes[1]
    # Find models with data in both domains
    model_pairs = {}
    for r in results:
        name = r['display_name']
        if name not in model_pairs:
            model_pairs[name] = {}
        model_pairs[name][r['domain']] = r

    paired = {k: v for k, v in model_pairs.items() if 'philosophy' in v and 'medical' in v}

    if paired:
        names_sorted = sorted(paired.keys())
        x = np.arange(len(names_sorted))
        width = 0.35

        phil_bars = [paired[n]['philosophy']['mean_drci'] for n in names_sorted]
        med_bars = [paired[n]['medical']['mean_drci'] for n in names_sorted]
        phil_err = [paired[n]['philosophy']['se_drci'] for n in names_sorted]
        med_err = [paired[n]['medical']['se_drci'] for n in names_sorted]

        ax.bar(x - width/2, phil_bars, width, yerr=phil_err, label='Philosophy',
              color=DOMAIN_COLORS['philosophy'], alpha=0.7, capsize=3)
        ax.bar(x + width/2, med_bars, width, yerr=med_err, label='Medical',
              color=DOMAIN_COLORS['medical'], alpha=0.7, capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels(names_sorted, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Mean ΔRCI')
        ax.set_title(f'Cross-Domain Comparison\n({len(paired)} models in both domains)')
        ax.legend()
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig2_domain_comparison.png')
    plt.close()
    print("  Saved fig2_domain_comparison.png")


# ============================================================
# Figure 3: Vendor Signatures
# ============================================================

def fig3_vendor_signatures(results):
    """Bar chart: Mean ΔRCI by vendor with error bars."""
    # Group by vendor
    vendor_data = {}
    for r in results:
        v = r['vendor']
        if v not in vendor_data:
            vendor_data[v] = []
        vendor_data[v].append(r['mean_drci'])

    # Sort by mean
    sorted_vendors = sorted(vendor_data.items(), key=lambda x: np.mean(x[1]), reverse=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    names = [v[0] for v in sorted_vendors]
    means = [np.mean(v[1]) for v in sorted_vendors]
    sems = [np.std(v[1], ddof=1) / np.sqrt(len(v[1])) if len(v[1]) > 1 else 0 for v in sorted_vendors]
    counts = [len(v[1]) for v in sorted_vendors]
    colors = [VENDOR_COLORS.get(n, '#666666') for n in names]

    bars = ax.bar(range(len(names)), means, yerr=sems, capsize=5,
                 color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)

    # Overlay individual points
    for i, (vendor, drcis) in enumerate(sorted_vendors):
        for d in drcis:
            ax.scatter(i + np.random.uniform(-0.15, 0.15), d,
                      c='black', s=20, alpha=0.4, zorder=5)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f'{n}\n(n={c})' for n, c in zip(names, counts)], fontsize=10)
    ax.set_ylabel('Mean ΔRCI')
    ax.set_title('Vendor Signatures: Context Sensitivity by Model Provider')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

    # ANOVA
    groups = [v[1] for v in sorted_vendors if len(v[1]) > 1]
    if len(groups) >= 2:
        f_stat, p_val = stats.f_oneway(*groups)
        ax.text(0.98, 0.95, f'F({len(groups)-1},{sum(len(g) for g in groups)-len(groups)})={f_stat:.2f}\np={p_val:.4f}',
               transform=ax.transAxes, ha='right', va='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig3_vendor_signatures.png')
    plt.close()
    print("  Saved fig3_vendor_signatures.png")


# ============================================================
# Figure 4: Position Patterns (Philosophy vs Medical)
# ============================================================

def fig4_position_patterns(results):
    """Line plot: ΔRCI across 30 positions by domain."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, domain in enumerate(['philosophy', 'medical']):
        ax = axes[ax_idx]
        domain_results = [r for r in results if r['domain'] == domain]

        all_position_data = []
        for r in domain_results:
            pos_drcis = extract_position_drcis(r['data'])
            if pos_drcis and any(len(v) > 0 for v in pos_drcis.values()):
                all_position_data.append((r['display_name'], r['vendor'], pos_drcis))

        if not all_position_data:
            ax.text(0.5, 0.5, f'No position data\navailable for {domain}',
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'{domain.title()} Domain')
            continue

        # Plot individual model curves (thin lines)
        positions = range(1, 31)
        grand_means = np.zeros(30)
        n_models = 0

        for name, vendor, pos_data in all_position_data:
            means = []
            for p in range(30):
                vals = pos_data.get(p, [])
                means.append(np.mean(vals) if vals else np.nan)
            means = np.array(means)
            if not np.all(np.isnan(means)):
                ax.plot(positions, means, alpha=0.3, linewidth=1,
                       color=VENDOR_COLORS.get(vendor, '#666666'), label=name)
                grand_means += np.nan_to_num(means)
                n_models += 1

        # Grand mean (thick line)
        if n_models > 0:
            grand_means /= n_models
            ax.plot(positions, grand_means, linewidth=3,
                   color=DOMAIN_COLORS[domain], label='Grand Mean', zorder=10)

        ax.set_xlabel('Position')
        ax.set_ylabel('ΔRCI')
        ax.set_title(f'{domain.title()} Domain ({n_models} models)')
        ax.set_xlim(1, 30)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

        # Annotate P30
        if n_models > 0:
            ax.axvline(x=30, color='red', linestyle=':', alpha=0.3)
            ax.text(29.5, ax.get_ylim()[1] * 0.9, 'P30\n(summary)',
                   ha='right', fontsize=8, color='red', alpha=0.7)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig4_position_patterns.png')
    plt.close()
    print("  Saved fig4_position_patterns.png")


# ============================================================
# Figure 5: Information Hierarchy (TRUE > SCRAMBLED > COLD)
# ============================================================

def fig5_information_hierarchy(results):
    """Show TRUE > SCRAMBLED > COLD hierarchy across models."""
    fig, ax = plt.subplots(figsize=(12, 6))

    models_data = []
    for r in results:
        cold_vals = []
        scram_vals = []
        for t in r['data'].get('trials', [])[:50]:
            dr = t.get('delta_rci', {})
            if isinstance(dr, dict):
                cv = dr.get('cold')
                sv = dr.get('scrambled')
                if cv is not None:
                    cold_vals.append(cv)
                if sv is not None:
                    scram_vals.append(sv)

        if cold_vals and scram_vals:
            models_data.append({
                'name': f"{r['display_name']}\n({r['domain'][:4]})",
                'cold': np.mean(cold_vals),
                'scrambled': np.mean(scram_vals),
                'domain': r['domain'],
                'vendor': r['vendor'],
                'hierarchy_holds': np.mean(scram_vals) > np.mean(cold_vals),
            })

    # Sort by cold dRCI
    models_data.sort(key=lambda x: x['cold'], reverse=True)

    x = np.arange(len(models_data))
    width = 0.35

    cold_bars = [m['cold'] for m in models_data]
    scram_bars = [m['scrambled'] for m in models_data]
    names = [m['name'] for m in models_data]

    ax.bar(x - width/2, scram_bars, width, label='ΔRCI (SCRAMBLED)',
          color='#f59e0b', alpha=0.7)
    ax.bar(x + width/2, cold_bars, width, label='ΔRCI (COLD)',
          color='#3b82f6', alpha=0.7)

    # Mark where hierarchy breaks
    n_holds = sum(1 for m in models_data if m['hierarchy_holds'])
    n_total = len(models_data)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, ha='center', fontsize=7)
    ax.set_ylabel('Mean ΔRCI')
    ax.set_title(f'Information Hierarchy: SCRAMBLED vs COLD\n(Hierarchy holds for {n_holds}/{n_total} model-domain runs)')
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig5_information_hierarchy.png')
    plt.close()
    print("  Saved fig5_information_hierarchy.png")


# ============================================================
# Figure 6: Model Rankings
# ============================================================

def fig6_model_rankings(results):
    """All models ranked by overall ΔRCI with confidence intervals."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for ax_idx, domain in enumerate(['philosophy', 'medical']):
        ax = axes[ax_idx]
        domain_results = [r for r in results if r['domain'] == domain]
        domain_results.sort(key=lambda x: x['mean_drci'], reverse=True)

        y_pos = range(len(domain_results))
        means = [r['mean_drci'] for r in domain_results]
        ci_low = [r['mean_drci'] - 1.96 * r['se_drci'] for r in domain_results]
        ci_high = [r['mean_drci'] + 1.96 * r['se_drci'] for r in domain_results]
        errors = [[m - lo for m, lo in zip(means, ci_low)],
                  [hi - m for m, hi in zip(means, ci_high)]]
        colors = [VENDOR_COLORS.get(r['vendor'], '#666666') for r in domain_results]
        names = [r['display_name'] for r in domain_results]
        types = ['(C)' if r['model_type'] == 'closed' else '(O)' for r in domain_results]

        ax.barh(y_pos, means, xerr=errors, capsize=3,
               color=colors, alpha=0.8, edgecolor='white', linewidth=1)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([f'{n} {t}' for n, t in zip(names, types)], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Mean ΔRCI')
        ax.set_title(f'{domain.title()} Domain ({len(domain_results)} models)')
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

        # Annotate mean
        overall_mean = np.mean(means)
        ax.axvline(x=overall_mean, color='red', linestyle=':', alpha=0.5)
        ax.text(overall_mean, len(domain_results) - 0.5,
               f'mean={overall_mean:.3f}', fontsize=8, color='red', ha='center')

    # Add vendor legend
    vendor_patches = [mpatches.Patch(color=VENDOR_COLORS[v], label=v)
                     for v in ['OpenAI', 'Anthropic', 'Google', 'DeepSeek', 'Moonshot', 'Meta', 'Mistral', 'Alibaba']]
    axes[1].legend(handles=vendor_patches, loc='lower right', fontsize=8, title='Vendor')

    plt.suptitle('Model Rankings by Mean ΔRCI (95% CI)\n(C)=Closed, (O)=Open', fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig6_model_rankings.png')
    plt.close()
    print("  Saved fig6_model_rankings.png")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("PAPER 2 FIGURE GENERATION")
    print("=" * 60)

    print("\nLoading all model data...")
    results = load_all_data()
    print(f"\nLoaded {len(results)} model-domain runs")
    print(f"  Philosophy: {sum(1 for r in results if r['domain'] == 'philosophy')}")
    print(f"  Medical: {sum(1 for r in results if r['domain'] == 'medical')}")

    print("\nGenerating figures...")
    print("\n[1/6] Dataset Overview Heatmap")
    fig1_dataset_overview(results)

    print("\n[2/6] Domain Comparison")
    fig2_domain_comparison(results)

    print("\n[3/6] Vendor Signatures")
    fig3_vendor_signatures(results)

    print("\n[4/6] Position Patterns")
    fig4_position_patterns(results)

    print("\n[5/6] Information Hierarchy")
    fig5_information_hierarchy(results)

    print("\n[6/6] Model Rankings")
    fig6_model_rankings(results)

    print("\n" + "=" * 60)
    print(f"ALL FIGURES SAVED TO: {FIG_DIR}")
    print("=" * 60)

    # Print summary stats for the paper
    print("\n--- PAPER 2 SUMMARY STATISTICS ---")
    phil_means = [r['mean_drci'] for r in results if r['domain'] == 'philosophy']
    med_means = [r['mean_drci'] for r in results if r['domain'] == 'medical']
    print(f"Philosophy: mean={np.mean(phil_means):.4f}, std={np.std(phil_means):.4f}, n={len(phil_means)}")
    print(f"Medical:    mean={np.mean(med_means):.4f}, std={np.std(med_means):.4f}, n={len(med_means)}")

    # Vendor ANOVA
    vendor_groups = {}
    for r in results:
        v = r['vendor']
        if v not in vendor_groups:
            vendor_groups[v] = []
        vendor_groups[v].append(r['mean_drci'])
    groups = [v for v in vendor_groups.values() if len(v) > 1]
    if len(groups) >= 2:
        f_stat, p_val = stats.f_oneway(*groups)
        print(f"Vendor ANOVA: F={f_stat:.2f}, p={p_val:.6f}")


if __name__ == '__main__':
    main()
