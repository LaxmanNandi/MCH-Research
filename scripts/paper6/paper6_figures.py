#!/usr/bin/env python3
"""
Paper 6: Conservation Law Figures
==================================
Generates 4 publication-quality figures for Paper 6.
Mahashivaratri edition.
"""
import csv
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')
BASE = Path("C:/Users/barla/mch_experiments")
FIG_DIR = BASE / "docs" / "figures" / "paper6"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Global style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.linewidth': 1.0,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'figure.dpi': 300,
})

# =============================================================
# LOAD DATA
# =============================================================
data = []
with open(BASE / "data" / "paper6" / "conservation_product_test.csv", encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        row['drci'] = float(row['drci'])
        row['var_ratio'] = float(row['var_ratio'])
        row['product'] = float(row['product'])
        data.append(row)

med = [d for d in data if d['domain'] == 'Medical']
phil = [d for d in data if d['domain'] == 'Philosophy']

K_med = np.mean([d['product'] for d in med])
K_phil = np.mean([d['product'] for d in phil])
cv_med = np.std([d['product'] for d in med], ddof=1) / K_med
cv_phil = np.std([d['product'] for d in phil], ddof=1) / K_phil

print(f"Medical:    N={len(med)}, K={K_med:.4f}, CV={cv_med:.4f}")
print(f"Philosophy: N={len(phil)}, K={K_phil:.4f}, CV={cv_phil:.4f}")

# Pretty model names
def pretty_name(model):
    names = {
        'deepseek_v3_1': 'DeepSeek V3.1',
        'gemini_flash': 'Gemini Flash',
        'llama_4_maverick': 'Llama Maverick',
        'llama_4_scout': 'Llama Scout',
        'qwen3_235b': 'Qwen3 235B',
        'mistral_small_24b': 'Mistral Small',
        'ministral_14b': 'Ministral 14B',
        'kimi_k2': 'Kimi K2',
        'claude_haiku': 'Claude Haiku',
        'gpt4o': 'GPT-4o',
        'gpt4o_mini': 'GPT-4o Mini',
    }
    return names.get(model, model.replace('_', ' ').title())


# =============================================================
# FIGURE 1: CONSERVATION LAW — SCATTER WITH HYPERBOLAS
# =============================================================
def fig1_conservation_hyperbolas():
    fig, ax = plt.subplots(figsize=(10, 8))

    # Hyperbola curves: Var_Ratio = K / ΔRCI
    drci_range = np.linspace(0.15, 0.50, 200)
    vr_med = K_med / drci_range
    vr_phil = K_phil / drci_range

    ax.plot(drci_range, vr_med, '--', color='#c0392b', linewidth=2, alpha=0.6,
            label=f'Medical: K = {K_med:.3f}')
    ax.plot(drci_range, vr_phil, '--', color='#2471a3', linewidth=2, alpha=0.6,
            label=f'Philosophy: K = {K_phil:.3f}')

    # Fill between hyperbolas (conservation gap)
    ax.fill_between(drci_range, vr_phil, vr_med, alpha=0.06, color='gray')

    # Scatter points
    for d in med:
        ax.scatter(d['drci'], d['var_ratio'], c='#e74c3c', s=120,
                   edgecolors='black', linewidths=0.8, zorder=10, marker='o')
    for d in phil:
        ax.scatter(d['drci'], d['var_ratio'], c='#3498db', s=120,
                   edgecolors='black', linewidths=0.8, zorder=10, marker='s')

    # Models that appear in both domains — add suffix to disambiguate
    dual_domain = {'deepseek_v3_1', 'gemini_flash', 'llama_4_maverick'}

    # Labels with offsets to avoid overlap
    label_offsets = {
        ('gemini_flash', 'Medical'): (0.008, 0.06),
        ('llama_4_scout', 'Medical'): (0.008, 0.08),
        ('llama_4_maverick', 'Medical'): (-0.07, 0.06),
        ('qwen3_235b', 'Medical'): (0.008, 0.05),
        ('ministral_14b', 'Medical'): (0.008, -0.07),
        ('kimi_k2', 'Medical'): (0.008, -0.07),
        ('mistral_small_24b', 'Medical'): (-0.08, -0.04),
        ('deepseek_v3_1', 'Medical'): (0.02, -0.07),
        ('gemini_flash', 'Philosophy'): (0.008, 0.05),
        ('claude_haiku', 'Philosophy'): (0.008, -0.06),
        ('deepseek_v3_1', 'Philosophy'): (-0.08, 0.04),
        ('gpt4o', 'Philosophy'): (-0.06, -0.04),
        ('gpt4o_mini', 'Philosophy'): (0.008, -0.06),
        ('llama_4_maverick', 'Philosophy'): (-0.07, -0.04),
    }

    for d in data:
        key = (d['model'], d['domain'])
        dx, dy = label_offsets.get(key, (0.008, 0.03))
        # Add domain tag for models in both domains
        label = pretty_name(d['model'])
        if d['model'] in dual_domain:
            tag = 'Med' if d['domain'] == 'Medical' else 'Phil'
            label = f"{label} ({tag})"
        ax.annotate(label,
                    (d['drci'], d['var_ratio']),
                    xytext=(d['drci'] + dx, d['var_ratio'] + dy),
                    fontsize=7.5, ha='left', va='bottom',
                    arrowprops=dict(arrowstyle='-', color='gray', alpha=0.4, lw=0.5)
                    if abs(dx) > 0.02 or abs(dy) > 0.06 else None)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
               markersize=11, markeredgecolor='black', markeredgewidth=0.8,
               label='Medical'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#3498db',
               markersize=11, markeredgecolor='black', markeredgewidth=0.8,
               label='Philosophy'),
        Line2D([0], [0], linestyle='--', color='#c0392b', linewidth=2,
               label=f'$K_{{med}}$ = {K_med:.3f}'),
        Line2D([0], [0], linestyle='--', color='#2471a3', linewidth=2,
               label=f'$K_{{phil}}$ = {K_phil:.3f}'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
              framealpha=0.9, edgecolor='gray')

    ax.set_xlabel(r'$\Delta$RCI (Context Sensitivity)', fontsize=13)
    ax.set_ylabel('Var_Ratio (Output Variance)', fontsize=13)
    ax.set_title(
        r'Conservation Law: $\Delta$RCI $\times$ Var_Ratio $\approx$ K(domain)' + '\n'
        f'Medical K = {K_med:.3f} (CV = {cv_med:.3f}), '
        f'Philosophy K = {K_phil:.3f} (CV = {cv_phil:.3f})',
        fontsize=13
    )

    ax.set_xlim(0.20, 0.48)
    ax.set_ylim(0.7, 1.8)

    plt.tight_layout()
    path = FIG_DIR / "fig1_conservation_law_hyperbolas.png"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path.name}")


# =============================================================
# FIGURE 2: PRODUCT DISTRIBUTION BY DOMAIN
# =============================================================
def fig2_product_distribution():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, subset, color, domain, K, cv in [
        (ax1, med, '#e74c3c', 'Medical', K_med, cv_med),
        (ax2, phil, '#3498db', 'Philosophy', K_phil, cv_phil),
    ]:
        products = [d['product'] for d in subset]
        mean_p = np.mean(products)
        std_p = np.std(products, ddof=1)

        # Individual bars
        names = [pretty_name(d['model']) for d in subset]
        x = range(len(subset))
        ax.bar(x, products, color=color, edgecolor='black', linewidth=0.5, alpha=0.85)

        # Mean line
        ax.axhline(y=mean_p, color='black', linestyle='-', linewidth=2, alpha=0.8)

        # ±1 SD shading
        ax.axhspan(mean_p - std_p, mean_p + std_p, alpha=0.15, color=color)

        # Labels
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel(r'Product = $\Delta$RCI $\times$ Var_Ratio', fontsize=11)
        ax.set_title(
            f'{domain} (N={len(subset)})\n'
            f'K = {mean_p:.3f} $\\pm$ {std_p:.3f}, CV = {cv:.3f}',
            fontsize=12
        )

        # Annotate mean and SD
        ax.text(0.97, 0.95,
                f'Mean = {mean_p:.3f}\n$\\sigma$ = {std_p:.3f}\nCV = {cv:.3f}',
                transform=ax.transAxes, fontsize=10, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='gray', alpha=0.9))

        ax.set_ylim(0, max(products) * 1.25)

    fig.suptitle(
        r'Product Distribution: Tight Clustering Within Domains ($\Delta$RCI $\times$ Var_Ratio)',
        fontsize=14, y=1.02
    )
    plt.tight_layout()
    path = FIG_DIR / "fig2_product_distribution.png"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path.name}")


# =============================================================
# FIGURE 3: DOMAIN CONSTANTS COMPARISON
# =============================================================
def fig3_domain_constants():
    fig, ax = plt.subplots(figsize=(6, 6))

    med_products = [d['product'] for d in med]
    phil_products = [d['product'] for d in phil]

    means = [K_med, K_phil]
    stds = [np.std(med_products, ddof=1), np.std(phil_products, ddof=1)]
    # 95% CI = mean ± t * (std / sqrt(n))
    ci_med = stats.t.ppf(0.975, len(med_products)-1) * stds[0] / np.sqrt(len(med_products))
    ci_phil = stats.t.ppf(0.975, len(phil_products)-1) * stds[1] / np.sqrt(len(phil_products))
    cis = [ci_med, ci_phil]

    colors = ['#e74c3c', '#3498db']
    labels = ['Medical\n(N=8)', 'Philosophy\n(N=6)']

    bars = ax.bar([0, 1], means, yerr=cis, color=colors,
                  edgecolor='black', linewidth=0.8, alpha=0.85,
                  capsize=8, error_kw={'linewidth': 2, 'capthick': 2})

    # Individual data points (jittered)
    np.random.seed(42)
    for i, products in enumerate([med_products, phil_products]):
        jitter = np.random.uniform(-0.15, 0.15, len(products))
        ax.scatter([i + j for j in jitter], products,
                   c='black', s=30, alpha=0.5, zorder=5)

    # Significance bracket
    y_max = max(means) + max(cis) + 0.03
    bracket_y = y_max + 0.02
    ax.plot([0, 0, 1, 1], [bracket_y - 0.01, bracket_y, bracket_y, bracket_y - 0.01],
            'k-', linewidth=1.5)
    ax.text(0.5, bracket_y + 0.005, 'p = 0.003 **', ha='center', va='bottom',
            fontsize=12, fontweight='bold')

    # Value labels on bars
    for i, (m, ci) in enumerate(zip(means, cis)):
        ax.text(i, m + ci + 0.01, f'K = {m:.3f}', ha='center', va='bottom',
                fontsize=12, fontweight='bold')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel(r'Conservation Constant K = $\Delta$RCI $\times$ Var_Ratio', fontsize=12)
    ax.set_title('Domain-Specific Conservation Constants\n'
                 r'$\Delta$RCI $\times$ Var_Ratio $\approx$ K(domain)',
                 fontsize=13)
    ax.set_ylim(0, bracket_y + 0.06)

    plt.tight_layout()
    path = FIG_DIR / "fig3_domain_constants.png"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path.name}")


# =============================================================
# FIGURE 4: TAXONOMY OVERLAY ON CONSERVATION LAW
# =============================================================
def fig4_taxonomy_overlay():
    fig, ax = plt.subplots(figsize=(10, 8))

    # Paper 5 class assignments (Medical only — where taxonomy was defined)
    class_map = {
        'deepseek_v3_1': ('IDEAL', '#27ae60', 'o'),
        'ministral_14b': ('IDEAL', '#27ae60', 'o'),
        'kimi_k2': ('IDEAL', '#27ae60', 'o'),
        'mistral_small_24b': ('IDEAL', '#27ae60', 'o'),
        'gemini_flash': ('EMPTY', '#f39c12', 'D'),
        'llama_4_scout': ('DANGEROUS', '#e74c3c', 'X'),
        'llama_4_maverick': ('DANGEROUS', '#e74c3c', 'X'),
        'qwen3_235b': ('RICH', '#3498db', '^'),
    }

    # Hyperbola curves
    drci_range = np.linspace(0.15, 0.50, 200)
    vr_med = K_med / drci_range
    ax.plot(drci_range, vr_med, '--', color='gray', linewidth=2, alpha=0.5,
            label=f'K(Medical) = {K_med:.3f}')

    # Philosophy runs as small gray dots (context, not primary)
    for d in phil:
        ax.scatter(d['drci'], d['var_ratio'], c='lightgray', s=50,
                   edgecolors='gray', linewidths=0.5, zorder=3, marker='s')
        ax.annotate(pretty_name(d['model']),
                    (d['drci'], d['var_ratio']),
                    fontsize=7, color='gray', alpha=0.6,
                    xytext=(5, -8), textcoords='offset points')

    # Medical runs colored by class
    for d in med:
        cls, color, marker = class_map.get(d['model'], ('UNKNOWN', 'gray', 'o'))
        ax.scatter(d['drci'], d['var_ratio'], c=color, s=160,
                   edgecolors='black', linewidths=1.0, zorder=10, marker=marker)

    # Labels for medical models with smart offsets
    med_label_offsets = {
        'gemini_flash': (8, 10),
        'llama_4_scout': (8, 8),
        'llama_4_maverick': (-85, 8),
        'qwen3_235b': (8, -15),
        'ministral_14b': (8, -15),
        'kimi_k2': (8, 8),
        'mistral_small_24b': (-90, -10),
        'deepseek_v3_1': (8, -15),
    }

    for d in med:
        cls, color, _ = class_map.get(d['model'], ('UNKNOWN', 'gray', 'o'))
        dx, dy = med_label_offsets.get(d['model'], (8, 5))
        ax.annotate(
            f"{pretty_name(d['model'])}\n({cls})",
            (d['drci'], d['var_ratio']),
            xytext=(dx, dy), textcoords='offset points',
            fontsize=8, fontweight='bold', color=color,
            arrowprops=dict(arrowstyle='->', color=color, alpha=0.6, lw=1.0)
        )

    # Deviation from hyperbola annotation
    # Compute residuals
    deviations = []
    for d in med:
        expected_vr = K_med / d['drci']
        actual_vr = d['var_ratio']
        deviation = abs(actual_vr - expected_vr) / expected_vr * 100
        deviations.append((d['model'], deviation, class_map.get(d['model'], ('?',))[0]))

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#27ae60',
               markersize=12, markeredgecolor='black', markeredgewidth=1, label='IDEAL'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#f39c12',
               markersize=11, markeredgecolor='black', markeredgewidth=1, label='EMPTY'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='#e74c3c',
               markersize=13, markeredgecolor='black', markeredgewidth=1, label='DANGEROUS'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#3498db',
               markersize=12, markeredgecolor='black', markeredgewidth=1, label='RICH'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgray',
               markersize=9, markeredgecolor='gray', markeredgewidth=0.5,
               label='Philosophy (ref)'),
        Line2D([0], [0], linestyle='--', color='gray', linewidth=2,
               label=f'K = {K_med:.3f} hyperbola'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
              framealpha=0.9, edgecolor='gray')

    ax.set_xlabel(r'$\Delta$RCI (Context Sensitivity)', fontsize=13)
    ax.set_ylabel('Var_Ratio (Output Variance)', fontsize=13)
    ax.set_title(
        'Four-Class Safety Taxonomy and the Conservation Law\n'
        'All classes obey the domain-specific information budget',
        fontsize=13
    )

    ax.set_xlim(0.20, 0.48)
    ax.set_ylim(0.7, 1.8)

    plt.tight_layout()
    path = FIG_DIR / "fig4_taxonomy_overlay.png"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path.name}")


# =============================================================
# MAIN
# =============================================================
if __name__ == "__main__":
    print("Generating Paper 6 Conservation Law Figures...")
    print("=" * 50)

    print("\nFigure 1: Conservation Law with Hyperbolas")
    fig1_conservation_hyperbolas()

    print("\nFigure 2: Product Distribution by Domain")
    fig2_product_distribution()

    print("\nFigure 3: Domain Constants Comparison")
    fig3_domain_constants()

    print("\nFigure 4: Taxonomy Overlay")
    fig4_taxonomy_overlay()

    print("\n" + "=" * 50)
    print("All 4 figures saved to docs/figures/paper6/")
    print("=" * 50)
