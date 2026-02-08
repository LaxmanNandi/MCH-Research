import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = Path("C:/Users/barla/mch_experiments")
ANALYSIS_DIR = BASE_DIR / "analysis"
OUT_DIR = BASE_DIR / "docs" / "figures" / "paper3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POSITION_DATA = ANALYSIS_DIR / "position_drci_data.csv"
TREND_DATA = ANALYSIS_DIR / "position30_analysis" / "position30_trend_comparison.csv"
SUMMARY_DATA = ANALYSIS_DIR / "position_analysis_summary.csv"
TYPE2_DATA = ANALYSIS_DIR / "type2_scaling_points.csv"

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def _domain_colors(domain):
    return "#1f77b4" if domain == "philosophy" else "#d62728"


def _plot_domain(ax, df, domain):
    domain_df = df[df["domain"] == domain]
    models = domain_df["model"].unique()

    # Individual model curves (thin)
    for model in models:
        mdf = domain_df[domain_df["model"] == model].sort_values("position")
        ax.plot(
            mdf["position"],
            mdf["mean_drci_cold"],
            color=_domain_colors(domain),
            alpha=0.25,
            linewidth=1.0,
        )

    # Grand mean with SEM
    mean_by_pos = domain_df.groupby("position")["mean_drci_cold"].mean().sort_index()
    sem_by_pos = (
        domain_df.groupby("position")["mean_drci_cold"].std().sort_index()
        / math.sqrt(len(models))
    )

    ax.plot(
        mean_by_pos.index,
        mean_by_pos.values,
        color="black",
        linewidth=2.2,
        label="Grand mean",
    )
    ax.fill_between(
        mean_by_pos.index,
        mean_by_pos.values - sem_by_pos.values,
        mean_by_pos.values + sem_by_pos.values,
        color="black",
        alpha=0.15,
    )

    # Bin boundaries
    ax.axvline(x=10.5, color="gray", linestyle=":", alpha=0.6)
    ax.axvline(x=20.5, color="gray", linestyle=":", alpha=0.6)

    ax.set_xlim(1, 30)
    ax.set_xlabel("Prompt position")
    ax.set_ylabel("ΔRCI (1 - cold alignment)")
    ax.set_title(f"{domain.title()} domain")
    ax.grid(True, alpha=0.3)


def _load_type2_points():
    if not TYPE2_DATA.exists():
        return None

    df = pd.read_csv(TYPE2_DATA)
    if "position" not in df.columns or "z_score" not in df.columns:
        return None

    df = df.dropna(subset=["position", "z_score"]).copy()
    df = df[df["position"] > 1]
    if df.shape[0] < 2:
        return None

    return df


# ---------------------------------------------------------------------
# Figure 1: Position-dependent ΔRCI by domain (two panels)
# ---------------------------------------------------------------------

df = pd.read_csv(POSITION_DATA)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
_plot_domain(axes[0], df, "philosophy")
_plot_domain(axes[1], df, "medical")

fig.suptitle("Position-dependent ΔRCI by domain", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig1_position_drci_domains.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------
# Figure 2: Domain grand means over position
# ---------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))

for domain in ["philosophy", "medical"]:
    domain_df = df[df["domain"] == domain]
    models = domain_df["model"].unique()
    mean_by_pos = domain_df.groupby("position")["mean_drci_cold"].mean().sort_index()
    sem_by_pos = (
        domain_df.groupby("position")["mean_drci_cold"].std().sort_index()
        / math.sqrt(len(models))
    )

    ax.plot(
        mean_by_pos.index,
        mean_by_pos.values,
        color=_domain_colors(domain),
        linewidth=2.5,
        label=f"{domain.title()} grand mean",
    )
    ax.fill_between(
        mean_by_pos.index,
        mean_by_pos.values - sem_by_pos.values,
        mean_by_pos.values + sem_by_pos.values,
        color=_domain_colors(domain),
        alpha=0.15,
    )

ax.axvline(x=10.5, color="gray", linestyle=":", alpha=0.6)
ax.axvline(x=20.5, color="gray", linestyle=":", alpha=0.6)
ax.set_xlim(1, 30)
ax.set_xlabel("Prompt position")
ax.set_ylabel("ΔRCI (grand mean)")
ax.set_title("Domain comparison of position-dependent ΔRCI")
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUT_DIR / "fig2_domain_grand_mean.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------
# Figure 6: Type 2 scaling law
# ---------------------------------------------------------------------

type2_df = _load_type2_points()

if type2_df is None:
    # Fallback: illustrative fit anchored at P10 and P30
    P10 = 10
    P30 = 30
    z10 = -0.59
    z30 = 2.01

    positions = np.array([P10, P30])
    z_scores = np.array([z10, z30])
    label = "Observed (P10/P30 anchor)"
else:
    positions = type2_df["position"].to_numpy()
    z_scores = type2_df["z_score"].to_numpy()
    label = "Observed"

logx = np.log(positions - 1)

alpha = np.polyfit(logx, z_scores, 1)[0]
beta = np.polyfit(logx, z_scores, 1)[1]

x_curve = np.log(np.arange(3, 31) - 1)
curve = alpha * x_curve + beta

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(x_curve, curve, color="#1f77b4", linewidth=2.5, label="Log fit")
ax.scatter(logx, z_scores, color="#d62728", s=60, zorder=5, label=label)

for p, z in zip(positions, z_scores):
    ax.annotate(f"P{int(p)}", (math.log(p - 1), z), textcoords="offset points", xytext=(8, -12))

ax.set_xlabel("log(P - 1)")
ax.set_ylabel("Type 2 effect (Z-score)")
ax.set_title("Type 2 scaling (log fit)")
ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()
fig.savefig(OUT_DIR / "fig6_type2_scaling.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------
# Figure 7: Kimi K2 vs other philosophy models
# ---------------------------------------------------------------------

trend = pd.read_csv(TREND_DATA)
summary = pd.read_csv(SUMMARY_DATA)

trend_phil = trend[trend["domain"] == "philosophy"][["model", "slope_pos_29"]]
summary_phil = summary[summary["domain"] == "philosophy"][["model", "disruption_mean"]]

merged = pd.merge(trend_phil, summary_phil, on="model", how="inner")

fig, ax = plt.subplots(figsize=(7.5, 5))

for _, row in merged.iterrows():
    is_kimi = row["model"] == "Kimi K2"
    ax.scatter(
        row["slope_pos_29"],
        row["disruption_mean"],
        color="#d62728" if is_kimi else "#7f7f7f",
        s=80 if is_kimi else 40,
        alpha=0.9,
        zorder=5 if is_kimi else 3,
    )
    if is_kimi:
        ax.annotate("Kimi K2", (row["slope_pos_29"], row["disruption_mean"]),
                    textcoords="offset points", xytext=(8, -6))

ax.axhline(y=0, color="black", linewidth=0.8)
ax.set_xlabel("Slope (positions 1-29)")
ax.set_ylabel("Disruption sensitivity (mean)")
ax.set_title("Scale-dependent accumulation: Kimi K2 vs peers")
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUT_DIR / "fig7_kimi_k2_scale.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("Figures saved to", OUT_DIR)
