# -*- coding: utf-8 -*-
"""
11_make_extra_figures.py
Generates the three remaining figures in the unified publication style:
  Figure 2  -> fig_rq1_reproducibility   (RQ1: ICC vs CV; Bland-Altman bias)
  Figure S1 -> fig_feature_screening      (feature selection)
  Figure S2 -> fig_rq3_interaction        (Group x Pipeline interaction z-values)
Dependencies: numpy, pandas, matplotlib (+ local figstyle.py).
"""
import sys
from pathlib import Path
import warnings, logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle
from figstyle import despine, panel, GRID, GREY
warnings.filterwarnings("ignore"); logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
figstyle.apply()

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "outputs"; FIG = BASE / "figures"
MC = {"FA": "#0072B2", "MD": "#E69F00", "AD": "#009E73", "RD": "#CC79A7"}
DISP = {"FA": "FA", "MD": "MD", "RD": "RD", "AD": "AxD"}
METRICS = ["FA", "MD", "RD", "AD"]


def save(fig, stem):
    for ext in ("png", "pdf", "svg"):
        kw = dict(bbox_inches="tight", facecolor="white")
        if ext == "png":
            kw["dpi"] = 600
        fig.savefig(FIG / f"{stem}.{ext}", **kw)
    plt.close(fig)


# ---- Figure 2: RQ1 reproducibility ----
def fig_rq1():
    icc = pd.read_csv(OUT / "rq1_reproducibility" / "icc_all_features.csv")
    icc = icc[icc.feature == "mean"][["tract", "metric", "icc_3_1"]]
    cv = pd.read_csv(OUT / "rq1_reproducibility" / "cv_all_features.csv")
    cv = cv[cv.feature == "mean"][["tract", "metric", "mean_cv_percent"]]
    d = icc.merge(cv, on=["tract", "metric"])
    ba = pd.read_csv(OUT / "rq1_reproducibility" / "bland_altman_all_features.csv")
    ba = ba[ba.feature == "mean"].copy()
    pm = {("pipeline_01", "pipeline_02"): "P02-P01", ("pipeline_01", "pipeline_03"): "P03-P01",
          ("pipeline_02", "pipeline_03"): "P03-P02"}
    ba["pair"] = [pm[(a, b)] for a, b in zip(ba.pipeline_1, ba.pipeline_2)]
    ba["rel"] = 100 * ba["bias_pipeline2_minus_pipeline1"] / ba["mean_pair_value"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), gridspec_kw={"width_ratios": [1, 1]})
    ax = axes[0]
    ax.axhline(0.90, color=GREY, ls=(0, (4, 4)), lw=0.8, zorder=1)
    ax.axvline(5.0, color=GREY, ls=(0, (4, 4)), lw=0.8, zorder=1)
    for m in METRICS:
        s = d[d.metric == m]
        ax.scatter(s["mean_cv_percent"], s["icc_3_1"], s=22, color=MC[m], alpha=0.85,
                   edgecolor="white", lw=0.4, zorder=3, label=DISP[m])
    low = d.nsmallest(4, "icc_3_1")
    for _, r in low.iterrows():
        _lo = {"CG_left_FA": (-5, 7), "CG_right_FA": (7, -2), "CC_7_MD": (7, 4), "CC_7_AxD": (7, -3)}
        _lbl = f"{r['tract']}_{DISP[r['metric']]}"
        ax.annotate(_lbl, (r["mean_cv_percent"], r["icc_3_1"]), textcoords="offset points",
                    xytext=_lo.get(_lbl, (6, -2)), fontsize=6, color="#555",
                    ha="right" if r["mean_cv_percent"] > 4 else "left")
    ax.set_xlabel("within-subject CV (%)"); ax.set_ylabel("consistency ICC(3,1)")
    ax.set_ylim(0.70, 1.0); ax.set_xlim(0, max(6, d.mean_cv_percent.max() * 1.1))
    despine(ax); panel(ax, "A")
    axb = axes[1]
    axb.axhline(0, color="#9a9a9a", lw=0.8, zorder=1)
    pairs = ["P02-P01", "P03-P01", "P03-P02"]
    rng = np.random.default_rng(0)
    for i, pr in enumerate(pairs):
        sub = ba[ba.pair == pr]
        xj = i + (rng.random(len(sub)) - 0.5) * 0.5
        for m in METRICS:
            ss = sub[sub.metric == m]
            xx = i + (rng.random(len(ss)) - 0.5) * 0.5
            axb.scatter(xx, ss["rel"], s=14, color=MC[m], alpha=0.7, edgecolor="white", lw=0.3, zorder=3)
        axb.scatter(i, sub["rel"].median(), marker="D", s=42, color="#222", zorder=5)
    axb.set_xticks(range(len(pairs))); axb.set_xticklabels(pairs)
    axb.set_ylabel("relative Bland-Altman bias (%)")
    axb.set_xlim(-0.5, 2.5)
    despine(axb, which=("left",)); axb.tick_params(bottom=False); panel(axb, "B")
    handles = [Line2D([0], [0], marker="o", color="w", mfc=MC[m], ms=6, label=DISP[m]) for m in METRICS] \
        + [Line2D([0], [0], marker="D", color="w", mfc="#222", ms=6, label="per-pair median")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=5, frameon=False, fontsize=7.6)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig_rq1_reproducibility")


# ---- Figure S1: feature screening ----
def fig_feat():
    fl = pd.read_csv(OUT / "feature_screening" / "feature_level_summary.csv")
    order = fl.sort_values("overall_rank_score")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    for _, r in fl.iterrows():
        size = max(20, (r["mean_auc"] - 0.5) * 1400)
        is_primary = r["feature"] == "mean"
        ax.scatter(r["mean_abs_g"], r["mean_relative_pipeline_range_percent"], s=size,
                   color="#0072B2" if is_primary else "#bfd3e6",
                   edgecolor="#0a3d62" if is_primary else "#7f7f7f", lw=0.8, alpha=0.9, zorder=3)
        _off = {"sd": (7, 4), "distal_mean": (7, 3), "median": (-4, -12), "mean": (9, -9),
                "middle_mean": (5, 9), "proximal_mean": (7, 3)}
        ax.annotate(r["feature"], (r["mean_abs_g"], r["mean_relative_pipeline_range_percent"]),
                    textcoords="offset points", xytext=_off.get(r["feature"], (6, 4)), fontsize=6.6,
                    fontweight="bold" if is_primary else "normal", color="#0a3d62" if is_primary else "#222")
    ax.set_xlabel("mean |Hedges g| (AD vs CN)")
    ax.set_ylabel("mean relative cross-pipeline range (%)")
    ax.text(0.02, 0.97, "bubble area ∝ mean AUC − 0.5", transform=ax.transAxes, va="top", fontsize=6.6, color="#777")
    despine(ax); panel(ax, "A")
    axb = axes[1]
    axb.xaxis.grid(True, color=GRID, lw=0.6)
    y = np.arange(len(order))[::-1]
    cols = ["#0072B2" if f == "mean" else "#9bb8d4" for f in order["feature"]]
    axb.barh(y, order["overall_rank_score"], color=cols, edgecolor="white", height=0.66, zorder=3)
    axb.set_yticks(y); axb.set_yticklabels(order["feature"])
    axb.set_xlabel("composite rank score (lower = better)")
    axb.tick_params(left=False)
    despine(axb, which=("bottom",)); panel(axb, "B")
    fig.tight_layout()
    save(fig, "fig_feature_screening")


# ---- Figure S2: Group x Pipeline interaction ----
def fig_int():
    g = pd.read_csv(OUT / "rq3_biological_robustness" / "group_pipeline_interactions.csv")
    g = g[(g.feature == "mean") & g.z_value.notna()].copy()
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.axhspan(-1.96, 1.96, color="#f3f3f3", zorder=0)
    ax.axhline(0, color="#9a9a9a", lw=0.8, zorder=1)
    for ln in (-1.96, 1.96):
        ax.axhline(ln, color=GREY, ls=(0, (4, 4)), lw=0.8, zorder=1)
    rng = np.random.default_rng(1)
    for i, m in enumerate(METRICS):
        s = g[g.metric == m]
        xx = i + (rng.random(len(s)) - 0.5) * 0.55
        ax.scatter(xx, s["z_value"], s=16, color=MC[m], alpha=0.75, edgecolor="white", lw=0.3, zorder=3)
    ax.set_xticks(range(len(METRICS))); ax.set_xticklabels([DISP[m] for m in METRICS])
    ax.set_ylabel("Group × Pipeline interaction (z)")
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-3.0, 2.55)
    ax.tick_params(bottom=False)
    despine(ax, which=("left",))
    fig.text(0.5, 0.0, "Shaded band: |z| < 1.96 (nominal p > 0.05).  No interaction term survived FDR correction (0 / 80).",
             ha="center", fontsize=7.2, color="#555")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig_rq3_interaction")


if __name__ == "__main__":
    fig_rq1(); print("done fig_rq1_reproducibility")
    fig_feat(); print("done fig_feature_screening")
    fig_int(); print("done fig_rq3_interaction")
