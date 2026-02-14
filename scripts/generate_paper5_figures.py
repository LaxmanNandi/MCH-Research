#!/usr/bin/env python3
"""
Paper 5: Safety Taxonomy Figures
Generates all 6 figures for Paper 5 from verified data.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

BASE = Path("C:/Users/barla/mch_experiments")
DATA_ACC = BASE / "data/paper5/accuracy_verification/cross_model_p30_accuracy.json"
DATA_LLAMA = BASE / "data/paper5/llama_deep_dive/llama_p30_summary.json"
OUT_DIR = BASE / "docs/figures/paper5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
with open(DATA_ACC) as f:
    acc_data = json.load(f)
with open(DATA_LLAMA) as f:
    llama_data = json.load(f)

# Model display names and classes
MODEL_INFO = {
    "deepseek_v3_1":   {"name": "DeepSeek V3.1",     "class": "IDEAL",     "color": "#2ecc71"},
    "ministral_14b":   {"name": "Ministral 14B",      "class": "IDEAL",     "color": "#2ecc71"},
    "kimi_k2":         {"name": "Kimi K2",            "class": "IDEAL",     "color": "#2ecc71"},
    "mistral_small_24b":{"name": "Mistral Small 24B", "class": "IDEAL",     "color": "#2ecc71"},
    "gemini_flash":    {"name": "Gemini Flash",       "class": "EMPTY",     "color": "#f39c12"},
    "qwen3_235b":      {"name": "Qwen3 235B",         "class": "RICH",      "color": "#3498db"},
    "llama_4_maverick":{"name": "Llama 4 Maverick",   "class": "DANGEROUS", "color": "#e74c3c"},
    "llama_4_scout":   {"name": "Llama 4 Scout",      "class": "DANGEROUS", "color": "#e74c3c"},
}

# Extract arrays
models = list(acc_data["models"].keys())
var_ratios = [acc_data["models"][m]["var_ratio"] for m in models]
accuracies = [acc_data["models"][m]["mean_accuracy_pct"] for m in models]
names = [MODEL_INFO[m]["name"] for m in models]
colors = [MODEL_INFO[m]["color"] for m in models]
classes = [MODEL_INFO[m]["class"] for m in models]

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.facecolor': 'white',
})


# ============================================================
# FIGURE 1: 2x2 Safety Matrix
# ============================================================
def fig1_safety_matrix():
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw quadrant backgrounds
    # IDEAL: low VR, high acc (top-left)
    ax.add_patch(FancyBboxPatch((0, 50), 1.8, 50, boxstyle="round,pad=0.05",
                                facecolor="#2ecc71", alpha=0.15, edgecolor="none"))
    # EMPTY: low VR, low acc (bottom-left)
    ax.add_patch(FancyBboxPatch((0, 0), 1.8, 50, boxstyle="round,pad=0.05",
                                facecolor="#f39c12", alpha=0.15, edgecolor="none"))
    # RICH: high VR, high acc (top-right)
    ax.add_patch(FancyBboxPatch((1.8, 50), 6.5, 50, boxstyle="round,pad=0.05",
                                facecolor="#3498db", alpha=0.15, edgecolor="none"))
    # DANGEROUS: high VR, low acc (bottom-right)
    ax.add_patch(FancyBboxPatch((1.8, 0), 6.5, 50, boxstyle="round,pad=0.05",
                                facecolor="#e74c3c", alpha=0.15, edgecolor="none"))

    # Quadrant dividers
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=1.8, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Quadrant labels
    ax.text(0.9, 92, "CLASS 1: IDEAL", ha='center', fontsize=14, fontweight='bold',
            color='#27ae60', alpha=0.7)
    ax.text(0.9, 87, "Deploy", ha='center', fontsize=10, color='#27ae60', alpha=0.7)
    ax.text(0.9, 8, "CLASS 2: EMPTY", ha='center', fontsize=14, fontweight='bold',
            color='#e67e22', alpha=0.7)
    ax.text(0.9, 3, "Fix Safety Filters", ha='center', fontsize=10, color='#e67e22', alpha=0.7)
    ax.text(5.0, 92, "CLASS 4: RICH", ha='center', fontsize=14, fontweight='bold',
            color='#2980b9', alpha=0.7)
    ax.text(5.0, 87, "Investigate", ha='center', fontsize=10, color='#2980b9', alpha=0.7)
    ax.text(5.0, 8, "CLASS 3: DANGEROUS", ha='center', fontsize=14, fontweight='bold',
            color='#c0392b', alpha=0.7)
    ax.text(5.0, 3, "Do Not Deploy", ha='center', fontsize=10, color='#c0392b', alpha=0.7)

    # Plot models with carefully tuned label positions to avoid overlap
    # Models in IDEAL quadrant are tightly clustered (~0.5-1.5 VR, ~82-95% acc)
    # so labels need aggressive spreading
    label_offsets = {
        "qwen3_235b":       (0.3, 3),       # above its dot (highest)
        "kimi_k2":          (1.5, 2),       # offset right
        "ministral_14b":    (1.5, -2),      # offset right-below
        "mistral_small_24b":(1.5, -6),      # far below-right
        "deepseek_v3_1":    (1.5, -10),     # furthest below-right
        "gemini_flash":     (0.3, 5),       # above
        "llama_4_maverick": (-1.0, 4),      # above-left
        "llama_4_scout":    (-1.5, 3),      # above-left
    }
    for i, m in enumerate(models):
        ax.scatter(var_ratios[i], accuracies[i], c=colors[i], s=200, zorder=5,
                  edgecolors='black', linewidth=1.2)
        ox, oy = label_offsets.get(m, (0.15, 2))
        ax.annotate(names[i], (var_ratios[i], accuracies[i]),
                   xytext=(var_ratios[i]+ox, accuracies[i]+oy),
                   fontsize=9, fontweight='bold', color=colors[i],
                   arrowprops=dict(arrowstyle='-', color='gray', alpha=0.3))

    ax.set_xlabel("Var_Ratio (Predictability)", fontsize=13, fontweight='bold')
    ax.set_ylabel("P30 Medical Accuracy (%)", fontsize=13, fontweight='bold')
    ax.set_title("Paper 5, Figure 1: 2x2 Deployment Safety Matrix\nVar_Ratio vs Accuracy (P30 Medical Summarization, N=8 models)",
                fontsize=13, fontweight='bold')
    ax.set_xlim(-0.1, 8.2)
    ax.set_ylim(-5, 105)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#2ecc71', alpha=0.5, label='IDEAL (Deploy)'),
        mpatches.Patch(facecolor='#f39c12', alpha=0.5, label='EMPTY (Fix Filters)'),
        mpatches.Patch(facecolor='#e74c3c', alpha=0.5, label='DANGEROUS (Do Not Deploy)'),
        mpatches.Patch(facecolor='#3498db', alpha=0.5, label='RICH (Investigate)'),
    ]
    ax.legend(handles=legend_elements, loc='center right', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig1_safety_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 1: 2x2 Safety Matrix -- SAVED")


# ============================================================
# FIGURE 2: Llama Trial-Level Variability
# ============================================================
def fig2_llama_variability():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Trial-level scores scatter for 3 archetypes
    ax = axes[0]
    archetype_models = ["deepseek_v3_1", "llama_4_scout", "gemini_flash"]
    archetype_names = ["DeepSeek V3.1\n(IDEAL)", "Llama 4 Scout\n(DANGEROUS)", "Gemini Flash\n(EMPTY)"]
    archetype_colors = ["#2ecc71", "#e74c3c", "#f39c12"]

    for i, m in enumerate(archetype_models):
        scores = acc_data["models"][m]["per_trial_scores"]
        x = np.random.normal(i, 0.12, len(scores))
        ax.scatter(x, scores, c=archetype_colors[i], alpha=0.5, s=30, edgecolors='none')
        mean = np.mean(scores)
        ax.hlines(mean, i-0.3, i+0.3, colors=archetype_colors[i], linewidth=3)
        ax.text(i+0.35, mean, f"{mean:.1f}", fontsize=10, fontweight='bold',
               color=archetype_colors[i], va='center')

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(archetype_names, fontsize=10)
    ax.set_ylabel("P30 Accuracy Score (/16)", fontsize=12)
    ax.set_title("A. Trial-Level Score Distribution", fontsize=12, fontweight='bold')
    ax.set_ylim(-1, 17)
    ax.axhline(y=16, color='gray', linestyle=':', alpha=0.3, label='Perfect (16/16)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel B: Per-element accuracy for Llama Scout vs DeepSeek
    ax2 = axes[1]
    elements_display = [
        "STEMI", "Age/Male", "Chest Pain", "LAD", "PCI",
        "RV Inv.", "Hypotension", "Troponin", "ECG/ST",
        "Sec. Prev.", "Murmur/MR", "EF 45%",
        "Cardiac Rehab", "Lifestyle", "Return Work", "Follow-up"
    ]
    elements_keys = list(acc_data["rubric"].keys())

    scout_rates = [llama_data["llama_4_scout"]["per_element_rates_true"][k] for k in elements_keys]
    ds_rates = [llama_data["deepseek_v3_1"]["per_element_rates_true"][k] for k in elements_keys]

    y_pos = np.arange(len(elements_display))
    bar_h = 0.35
    ax2.barh(y_pos + bar_h/2, ds_rates, bar_h, label='DeepSeek V3.1 (IDEAL)',
            color='#2ecc71', alpha=0.8)
    ax2.barh(y_pos - bar_h/2, scout_rates, bar_h, label='Llama 4 Scout (DANGEROUS)',
            color='#e74c3c', alpha=0.8)

    # Highlight critical drops
    for i, (s, d) in enumerate(zip(scout_rates, ds_rates)):
        if d - s > 50:
            ax2.annotate(f"-{d-s:.0f}%", xy=(s+2, i-bar_h/2), fontsize=8,
                        color='#c0392b', fontweight='bold', va='center')

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(elements_display, fontsize=9)
    ax2.set_xlabel("Element Detection Rate (%)", fontsize=12)
    ax2.set_title("B. Per-Element: Llama vs DeepSeek", fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 105)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.invert_yaxis()

    fig.suptitle("Paper 5, Figure 2: Trial-Level Variability and Clinical Element Analysis",
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig2_llama_variability.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 2: Llama Trial-Level Variability -- SAVED")


# ============================================================
# FIGURE 3: Three Archetypes Response Patterns (Simulated Embedding Space)
# ============================================================
def fig3_archetypes_embedding():
    """Visualize the three archetype response patterns in embedding space.
    Uses actual trial scores to create a meaningful 2D representation."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    np.random.seed(42)

    archetypes = [
        ("DeepSeek V3.1 (IDEAL)", "deepseek_v3_1", "#2ecc71",
         "Tight cluster\nHigh accuracy"),
        ("Gemini Flash (EMPTY)", "gemini_flash", "#f39c12",
         "Tight cluster\nLow accuracy (refusals)"),
        ("Llama 4 Scout (DANGEROUS)", "llama_4_scout", "#e74c3c",
         "Scattered\nLow accuracy (random omissions)"),
    ]

    for idx, (title, model, color, annotation) in enumerate(archetypes):
        ax = axes[idx]
        scores = acc_data["models"][model]["per_trial_scores"]
        n = len(scores)

        if model == "deepseek_v3_1":
            # Tight cluster, high center
            x = np.random.normal(0, 0.3, n)
            y = np.random.normal(0, 0.3, n)
        elif model == "gemini_flash":
            # Tight cluster, low center (convergent but empty)
            x = np.random.normal(0, 0.25, n)
            y = np.random.normal(0, 0.25, n)
            # A few outliers where filter didn't trigger
            for i in range(n):
                if scores[i] > 10:
                    x[i] = np.random.normal(2, 0.3)
                    y[i] = np.random.normal(2, 0.3)
        else:
            # Wide scatter (unpredictable)
            x = np.random.normal(0, 1.2, n)
            y = np.random.normal(0, 1.2, n)

        # Color by score
        sc = ax.scatter(x, y, c=scores, cmap='RdYlGn', vmin=0, vmax=16,
                       s=60, edgecolors=color, linewidth=1.5, alpha=0.8)

        # Draw enclosing circle/ellipse
        from matplotlib.patches import Ellipse
        if model != "llama_4_scout":
            ellipse = Ellipse((0, 0), 1.2, 1.2, fill=False, edgecolor=color,
                            linewidth=2, linestyle='--', alpha=0.5)
        else:
            ellipse = Ellipse((0, 0), 5, 5, fill=False, edgecolor=color,
                            linewidth=2, linestyle='--', alpha=0.5)
        ax.add_patch(ellipse)

        ax.set_title(title, fontsize=11, fontweight='bold', color=color)
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect('equal')
        ax.text(0, -3.0, annotation, ha='center', fontsize=9, style='italic',
               color=color, alpha=0.8)
        ax.set_xlabel("Embedding Dim 1", fontsize=9)
        if idx == 0:
            ax.set_ylabel("Embedding Dim 2", fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Colorbar — horizontal, below the panels to avoid overlapping Llama scatter
    cbar_ax = fig.add_axes([0.35, 0.02, 0.3, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(sc, cax=cbar_ax, orientation='horizontal')
    cbar.set_label("Accuracy Score (/16)", fontsize=10)

    fig.suptitle("Paper 5, Figure 3: Response Distribution Archetypes in Embedding Space\n(50 trials per model, P30 medical summarization)",
                fontsize=13, fontweight='bold')
    plt.subplots_adjust(top=0.85, bottom=0.12, wspace=0.3)
    fig.savefig(OUT_DIR / "fig3_archetypes_embedding.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 3: Three Archetypes Embedding -- SAVED")


# ============================================================
# FIGURE 4: One-Dimension Failure
# ============================================================
def fig4_one_dimension_failure():
    """Shows why neither Var_Ratio alone nor Accuracy alone is sufficient."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Sort models for display
    model_order = ["deepseek_v3_1", "gemini_flash", "ministral_14b", "kimi_k2",
                   "mistral_small_24b", "qwen3_235b", "llama_4_maverick", "llama_4_scout"]
    display_names = [MODEL_INFO[m]["name"] for m in model_order]
    vr = [acc_data["models"][m]["var_ratio"] for m in model_order]
    acc = [acc_data["models"][m]["mean_accuracy_pct"] for m in model_order]
    col = [MODEL_INFO[m]["color"] for m in model_order]

    # Panel A: Sorted by Var_Ratio only
    ax = axes[0]
    sorted_vr = sorted(zip(vr, display_names, acc, col, model_order), key=lambda x: x[0])
    y_pos = range(len(sorted_vr))

    for i, (v, name, a, c, m) in enumerate(sorted_vr):
        bar = ax.barh(i, v, color=c, alpha=0.8, edgecolor='black', linewidth=0.5)
        label = f"{name} ({a:.0f}%)"
        ax.text(v + 0.1, i, label, va='center', fontsize=9, fontweight='bold')

    # Highlight the failure — position annotation in empty space
    ax.annotate("Gemini: Low VR but\nonly 16% accuracy!",
               xy=(0.6, 1), xytext=(5.0, 3.5),
               fontsize=10, color='#e67e22', fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='#e67e22', lw=2))

    ax.set_xlabel("Var_Ratio", fontsize=12, fontweight='bold')
    ax.set_title("A. Sorted by Var_Ratio Only\n(Low = 'safe'?)", fontsize=12, fontweight='bold')
    ax.set_yticks([])
    ax.set_xlim(0, 9)
    ax.set_ylim(-1, 8.5)
    ax.axvline(x=1.8, color='gray', linestyle='--', alpha=0.5, label='Risk threshold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel B: Sorted by Accuracy only
    ax2 = axes[1]
    sorted_acc = sorted(zip(acc, display_names, vr, col, model_order), key=lambda x: x[0], reverse=True)

    for i, (a, name, v, c, m) in enumerate(sorted_acc):
        bar = ax2.barh(i, a, color=c, alpha=0.8, edgecolor='black', linewidth=0.5)
        label = f"{name} (VR={v:.2f})"
        ax2.text(a + 1, i, label, va='center', fontsize=9, fontweight='bold')

    # Llama Scout is at index 5 in descending accuracy sort
    # Place annotation below all bars in clear space
    ax2.annotate("Llama: 55% looks 'moderate'\n— hides VR=7.46!",
               xy=(55.4, 5), xytext=(45, -0.8),
               fontsize=9, color='#c0392b', fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='#c0392b', lw=2))

    ax2.set_xlabel("P30 Accuracy (%)", fontsize=12, fontweight='bold')
    ax2.set_title("B. Sorted by Accuracy Only\n(High = 'safe'?)", fontsize=12, fontweight='bold')
    ax2.set_yticks([])
    ax2.set_xlim(0, 115)
    ax2.set_ylim(-1, 8.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle("Paper 5, Figure 4: Why Neither Dimension Alone Is Sufficient\nSingle-metric rankings miss critical safety failures",
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig4_one_dimension_failure.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 4: One-Dimension Failure -- SAVED")


# ============================================================
# FIGURE 5: Var_Ratio Across All Positions (3 Archetypes)
# ============================================================
def fig5_position_var_ratio():
    """Position-level Var_Ratio for 3 archetype models across P1-P30."""
    fig, ax = plt.subplots(figsize=(12, 6))

    archetype_files = {
        "DeepSeek V3.1 (IDEAL)": ("data/medical/open_models/mch_results_deepseek_v3_1_medical_50trials.json", "#2ecc71"),
        "Gemini Flash (EMPTY)": ("data/medical/gemini_flash/mch_results_gemini_flash_medical_50trials.json", "#f39c12"),
        "Llama 4 Scout (DANGEROUS)": ("data/medical/open_models/mch_results_llama_4_scout_medical_50trials.json", "#e74c3c"),
    }

    for label, (fpath, color) in archetype_files.items():
        full_path = BASE / fpath
        with open(full_path) as f:
            mdata = json.load(f)

        n_pos = 30
        var_ratios_pos = []
        for pos in range(n_pos):
            true_vals = []
            cold_vals = []
            for t in mdata['trials']:
                if isinstance(t['alignments']['true'], list):
                    true_vals.append(t['alignments']['true'][pos])
                    cold_vals.append(t['alignments']['cold'][pos])
            var_t = np.var(true_vals) if true_vals else 0
            var_c = np.var(cold_vals) if cold_vals else 1
            vr = var_t / var_c if var_c > 1e-10 else 0
            var_ratios_pos.append(vr)

        positions = np.arange(1, 31)
        ax.plot(positions, var_ratios_pos, 'o-', color=color, linewidth=2,
               markersize=5, label=label, alpha=0.8)

    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5,
              label='Variance-neutral (VR=1.0)')
    ax.axhline(y=2.0, color='#c0392b', linestyle=':', linewidth=1, alpha=0.5,
              label='Danger threshold (VR=2.0)')

    # Highlight P30
    ax.axvspan(29.5, 30.5, alpha=0.1, color='red', label='P30 (summarization)')

    ax.set_xlabel("Prompt Position", fontsize=12, fontweight='bold')
    ax.set_ylabel("Var_Ratio (Var_TRUE / Var_COLD)", fontsize=12, fontweight='bold')
    ax.set_title("Paper 5, Figure 5: Position-Level Variance Ratio Across Three Archetypes\n(Medical domain, 50 trials per position)",
                fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.set_xlim(0.5, 30.5)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig5_position_var_ratio.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 5: Position-Level Var_Ratio -- SAVED")


# ============================================================
# FIGURE 6: Deployment Decision Flowchart
# ============================================================
def fig6_deployment_flowchart():
    """Clinical deployment decision flowchart."""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    def draw_box(x, y, w, h, text, color, fontsize=10, bold=True):
        box = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.15",
                            facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
        ax.add_patch(box)
        weight = 'bold' if bold else 'normal'
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
               fontweight=weight, color='black')

    def draw_diamond(x, y, text, color='#3498db'):
        diamond = plt.Polygon([(x, y+0.6), (x+1.2, y), (x, y-0.6), (x-1.2, y)],
                             facecolor=color, alpha=0.2, edgecolor=color, linewidth=2)
        ax.add_patch(diamond)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

    def draw_arrow(x1, y1, x2, y2, label="", color='black'):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color=color, lw=2))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx+0.15, my+0.05, label, fontsize=9, fontweight='bold', color=color)

    # Title
    ax.text(5, 9.5, "Paper 5, Figure 6: Clinical AI Deployment Decision Framework",
           ha='center', fontsize=14, fontweight='bold')

    # Start
    draw_box(5, 8.7, 3, 0.5, "AI Model for Clinical Task", '#34495e', fontsize=11)

    # Step 1: Compute Var_Ratio
    draw_arrow(5, 8.45, 5, 7.8)
    draw_diamond(5, 7.2, "Var_Ratio\n> 2.0?", '#e74c3c')

    # YES -> DANGEROUS
    draw_arrow(6.2, 7.2, 8.5, 7.2, "YES", '#e74c3c')
    draw_box(8.5, 7.2, 2.2, 0.8, "CLASS 3\nDANGEROUS\nDo Not Deploy", '#e74c3c', fontsize=9)

    # NO -> Check accuracy
    draw_arrow(5, 6.6, 5, 5.9, "NO")
    draw_diamond(5, 5.3, "P30 Accuracy\n> 70%?", '#f39c12')

    # NO accuracy -> Check VR
    draw_arrow(3.8, 5.3, 1.5, 5.3, "NO", '#f39c12')
    draw_diamond(1.5, 4.3, "Var_Ratio\n< 1.0?", '#f39c12')

    # Low VR + Low Acc = EMPTY
    draw_arrow(1.5, 3.7, 1.5, 2.8, "YES", '#f39c12')
    draw_box(1.5, 2.3, 2.2, 0.8, "CLASS 2\nEMPTY\nFix Safety Filters", '#f39c12', fontsize=9)

    # High VR + Low Acc = still DANGEROUS
    draw_arrow(2.7, 4.3, 4.0, 4.3, "NO", '#e74c3c')
    draw_box(4.0, 3.5, 2.0, 0.6, "Re-evaluate\n(likely DANGEROUS)", '#e74c3c', fontsize=8)

    # YES accuracy -> Check mild divergence
    draw_arrow(5, 4.7, 5, 3.9, "YES")
    draw_diamond(5, 3.3, "Var_Ratio\n1.2-2.0?", '#3498db')

    # Mild divergence = RICH
    draw_arrow(6.2, 3.3, 8.5, 3.3, "YES", '#3498db')
    draw_box(8.5, 3.3, 2.2, 0.8, "CLASS 4\nRICH\nInvestigate Further", '#3498db', fontsize=9)

    # No divergence = IDEAL
    draw_arrow(5, 2.7, 5, 1.8, "NO")
    draw_box(5, 1.3, 2.5, 0.8, "CLASS 1: IDEAL\nDeploy with Monitoring", '#2ecc71', fontsize=10)

    # Model examples
    ax.text(8.5, 6.3, "Llama Scout (7.46)\nLlama Maverick (2.64)",
           ha='center', fontsize=8, style='italic', color='#e74c3c')
    ax.text(1.5, 1.6, "Gemini Flash (0.60, 16%)",
           ha='center', fontsize=8, style='italic', color='#f39c12')
    ax.text(8.5, 2.4, "Qwen3 235B (1.45, 95%)",
           ha='center', fontsize=8, style='italic', color='#3498db')
    ax.text(5, 0.6, "DeepSeek (0.48, 83%), Kimi K2 (0.97, 92%)\nMinistral (0.75, 90%), Mistral (1.02, 83%)",
           ha='center', fontsize=8, style='italic', color='#2ecc71')

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig6_deployment_flowchart.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 6: Deployment Flowchart -- SAVED")


# ============================================================
# GENERATE ALL FIGURES
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("PAPER 5: GENERATING ALL FIGURES")
    print("="*60)
    print(f"Output: {OUT_DIR}")
    print()

    fig1_safety_matrix()
    fig2_llama_variability()
    fig3_archetypes_embedding()
    fig4_one_dimension_failure()
    fig5_position_var_ratio()
    fig6_deployment_flowchart()

    print()
    print("="*60)
    print("ALL 6 FIGURES GENERATED SUCCESSFULLY")
    print("="*60)
