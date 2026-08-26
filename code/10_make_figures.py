# -*- coding: utf-8 -*-
"""
10_make_figures.py
==================
Regenerate all figures in a unified, publication-grade minimalist style
(Helvetica/Arial typography, trimmed/offset spines, light reference grids,
clean box-free annotations). Descriptive titles are NOT drawn on the images;
they are written to figures/figure_titles.txt for use as captions.

Inputs : data/all_pipelines_tractometry_features_master.csv  and  outputs/*
Outputs: figures/*.{png,pdf,svg}  and  figures/figure_titles.txt
Dependencies: NumPy, pandas, matplotlib (+ local figstyle.py).
"""
import sys
import warnings
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle
from figstyle import despine, panel, GRID, GREY

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
figstyle.apply()

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "outputs"
FIG = BASE / "figures"
FIG.mkdir(parents=True, exist_ok=True)
DATA = BASE / "data" / "all_pipelines_tractometry_features_master.csv"

df = pd.read_csv(DATA)
PIPES = ["pipeline_01", "pipeline_02", "pipeline_03"]
PL = ["P01", "P02", "P03"]
SUBJ = sorted(df.subject.unique())
grp = {s: g for s, g in df.drop_duplicates("subject")[["subject", "group"]].values}
labels = np.array([1 if grp[s] == "AD" else 0 for s in SUBJ])

MC = {"FA": "#0072B2", "MD": "#E69F00", "AD": "#009E73", "RD": "#CC79A7"}
DISP = {"FA": "FA", "MD": "MD", "RD": "RD", "AD": "AxD"}
PC = {"P01": "#4a1486", "P02": "#807dba", "P03": "#b2abd2"}

TITLES = {
    "fig_variance_decomposition": {"title": "Variance decomposition: biological signal dominates pipeline-induced variability"},
    "fig_rq3_effectsize_stability": {"title": "Alzheimer-related effect sizes are robust to pipeline choice",
                                     "A": "Effect-size stability across pipelines",
                                     "B": "Biomarker concordance (pipeline 01 vs 03)"},
    "fig_rq4_auc_stability": {"title": "Diagnostic performance is stable across pipelines"},
    "fig_rq4_roc_robust": {"title": "ROC curves of the most robust biomarkers coincide across pipelines"},
    "fig_tost_equivalence": {"title": "Aggregate equivalence: pipeline differences fall within negligible margins",
                             "A": "Effect-size equivalence (Delta Hedges g between pipelines)",
                             "B": "Diagnostic equivalence (Delta AUC between pipelines)"},
    "fig_rq2_pipeline_effect": {"title": "Preprocessing systematically shifts absolute DTI values (RQ2)",
                                "A": "Pipeline 03 vs 01 - signed relative value shift",
                                "B": "Direction and magnitude of the shift by metric"},
    "fig_supp_icc_comparison": {"title": "Reproducibility under consistency vs absolute agreement (ICC)"},
}


def save(fig, stem):
    for ext in ("png", "pdf", "svg"):
        kw = dict(bbox_inches="tight", facecolor="white")
        if ext == "png":
            kw["dpi"] = 600
        fig.savefig(FIG / f"{stem}.{ext}", **kw)
    plt.close(fig)


def mat(tr, me, feat="mean"):
    sub = df[(df.tract == tr) & (df.metric == me)]
    return sub.pivot_table(index="subject", columns="pipeline", values=feat).reindex(index=SUBJ, columns=PIPES).values


# ============ FIG 1: Variance decomposition ============
def fig_var():
    ov = pd.read_csv(OUT / "variance_decomposition" / "variance_decomposition_overall.csv").iloc[0]
    vm = pd.read_csv(OUT / "variance_decomposition" / "variance_decomposition_by_metric.csv").set_index("metric")
    rows = ["Overall", "FA", "MD", "RD", "AD"]
    comps = ["individual", "disease", "pipeline", "interaction", "residual"]
    data = {"Overall": [ov[c] for c in comps]}
    for m in ["FA", "MD", "RD", "AD"]:
        data[m] = [vm.loc[m, c] for c in comps]
    labels_c = ["Individual (subject)", "Disease (AD vs CN)", "Pipeline", "Group x Pipeline", "Residual"]
    cols = ["#9ecae1", "#1b7837", "#d55e00", "#f0c54a", "#d9d9d9"]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    y = np.arange(len(rows))[::-1]
    left = np.zeros(len(rows))
    M = np.array([data[r] for r in rows])
    for j, (c, col) in enumerate(zip(labels_c, cols)):
        ax.barh(y, M[:, j], left=left, color=col, edgecolor="white", height=0.66, label=c, zorder=3)
        for i, val in enumerate(M[:, j]):
            wide = val >= (6 if j == 2 else 7)
            if wide and j in (0, 1, 2):
                ax.text(left[i] + val / 2, y[i], f"{val:.0f}", ha="center", va="center",
                        fontsize=7, color="white", zorder=4)
            elif j == 2 and val >= 0.4:
                ax.text(left[i] + val + 1.0, y[i], f"{val:.0f}", ha="left", va="center",
                        fontsize=6.5, color="#b0560a", zorder=4)
        left += M[:, j]
    ax.set_yticks(y); ax.set_yticklabels(["AxD" if r == "AD" else r for r in rows])
    ax.set_xlim(0, 100); ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("Variance explained (% of total sum of squares)")
    ax.tick_params(left=False)
    despine(ax, which=("bottom",))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=5, frameon=False,
              fontsize=7.6, handlelength=1.1, columnspacing=1.3, handletextpad=0.5)
    ax.text(0.0, 1.03, "Orange numbers = pipeline share (%).   Group\u00d7Pipeline \u2248 0.1% \u2014 present but not visible at this scale.",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.6, color="#8a8a8a")
    save(fig, "fig_variance_decomposition")


# ============ FIG 2: RQ3 effect-size & ranking stability ============
def fig_rq3():
    dc = pd.read_csv(OUT / "rq3_biological_robustness" / "direction_consistency.csv")
    g = dc[dc.feature == "mean"].rename(columns={"pipeline_01_g": "g_P01", "pipeline_02_g": "g_P02", "pipeline_03_g": "g_P03"})
    summ = pd.read_csv(OUT / "rq3_biological_robustness" / "rq3_feature_summary.csv")
    srow = summ[summ.feature == "mean"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), gridspec_kw={"width_ratios": [1.05, 1]})
    ax = axes[0]
    ax.axhline(0, color=GREY, lw=0.8, ls=(0, (4, 4)), zorder=1)
    for _, r in g.iterrows():
        ax.plot([0, 1, 2], [r["g_P01"], r["g_P02"], r["g_P03"]], "-", color=MC[r["metric"]],
                alpha=0.5, lw=1.0, marker="o", ms=2.6, mfc=MC[r["metric"]], mec="none", zorder=3)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(PL); ax.set_xlim(-0.15, 2.15)
    ax.set_ylabel("Hedges g  (AD vs CN)"); ax.set_xlabel("Pipeline")
    despine(ax); panel(ax, "A")
    ax2 = axes[1]
    lim = [g[["g_P01", "g_P03"]].min().min() - 0.12, g[["g_P01", "g_P03"]].max().max() + 0.12]
    ax2.plot(lim, lim, ls=(0, (4, 4)), color=GREY, lw=0.8, zorder=1)
    for _, r in g.iterrows():
        ax2.scatter(r["g_P01"], r["g_P03"], s=20, color=MC[r["metric"]], alpha=0.85, edgecolor="white", lw=0.4, zorder=3)
    ax2.set_xlim(lim); ax2.set_ylim(lim); ax2.set_aspect("equal")
    ax2.set_xlabel("Hedges g, pipeline 01"); ax2.set_ylabel("Hedges g, pipeline 03")
    ax2.text(0.04, 0.97, f"ranking $\\rho$ = {srow['mean_ranking_spearman_rho']:.2f}\nsame direction = {srow['same_direction_percent']:.1f}%",
             transform=ax2.transAxes, fontsize=7.5, va="top", color="#444")
    despine(ax2); panel(ax2, "B")
    handles = [Line2D([0], [0], color=MC[m], lw=2.0, marker="o", ms=4, mfc=MC[m], mec="none", label=DISP[m]) for m in ["FA", "MD", "RD", "AD"]]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=4, frameon=False,
               fontsize=8, columnspacing=2.0, handlelength=1.6, handletextpad=0.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig_rq3_effectsize_stability")


# ============ FIG 3: RQ4 AUC stability forest ============
def fig_rq4_forest():
    a = pd.read_csv(OUT / "rq4_diagnostic_utility" / "auc_stability_summary.csv")
    a = a[a.feature == "mean"].rename(columns={"auc_pipeline_01": "auc_P01", "auc_pipeline_02": "auc_P02", "auc_pipeline_03": "auc_P03"})
    a["bm"] = a["tract"] + "_" + a["metric"].map(DISP)
    a = a.sort_values("mean_auc")
    mean_auc_range = a["auc_range"].mean()
    fig, ax = plt.subplots(figsize=(5.6, 8.4))
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    for i, (_, r) in enumerate(a.iterrows()):
        aucs = [r["auc_P01"], r["auc_P02"], r["auc_P03"]]
        ax.plot([min(aucs), max(aucs)], [i, i], color="#dcdcdc", lw=2.2, zorder=2, solid_capstyle="round")
        for p, col in zip(["P01", "P02", "P03"], [PC["P01"], PC["P02"], PC["P03"]]):
            ax.scatter(r[f"auc_{p}"], i, s=18, color=col, zorder=3, edgecolor="white", lw=0.3)
    ax.axvline(0.5, color=GREY, ls=(0, (4, 4)), lw=0.8, zorder=1)
    ax.set_yticks(range(len(a))); ax.set_yticklabels(a["bm"], fontsize=6.4)
    ax.set_ylim(-1, len(a)); ax.set_xlim(0.45, 0.85); ax.set_xticks(np.arange(0.5, 0.86, 0.1))
    ax.set_xlabel("ROC-AUC (AD vs CN)")
    ax.tick_params(left=False)
    ax.text(0.46, len(a) - 1.2, f"mean AUC range = {mean_auc_range:.3f};  38/40 vary $\\leq$ 0.10",
            fontsize=7.2, va="top", color="#555")
    despine(ax, which=("bottom",))
    handles = [Line2D([0], [0], marker="o", color="w", mfc=PC[p], ms=6, label=p) for p in ["P01", "P02", "P03"]] \
        + [Line2D([0], [0], color="#dcdcdc", lw=2.2, label="pipeline range")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.005), ncol=4, frameon=False, fontsize=7.6)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "fig_rq4_auc_stability")


# ============ FIG 4: ROC overlays for robust biomarkers ============
def roc_curve(scores, lab):
    s = np.asarray(scores, float)
    pos = s[lab == 1]; neg = s[lab == 0]
    auc = ((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
    if auc < 0.5:
        s = -s
    thr = np.r_[np.inf, np.sort(np.unique(s))[::-1], -np.inf]
    P = (lab == 1).sum(); N = (lab == 0).sum()
    tpr = []; fpr = []
    for t in thr:
        pred = s >= t
        tpr.append(((pred) & (lab == 1)).sum() / P); fpr.append(((pred) & (lab == 0)).sum() / N)
    return np.array(fpr), np.array(tpr), max(auc, 1 - auc)


def fig_roc():
    sel = [("UF_right", "RD"), ("UF_left", "RD"), ("UF_right", "MD"), ("UF_left", "MD"), ("UF_right", "FA"), ("UF_left", "FA")]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))
    for ax, (tr, me) in zip(axes.ravel(), sel):
        M = mat(tr, me)
        ax.plot([0, 1], [0, 1], ls=(0, (4, 4)), color=GREY, lw=0.8)
        _ls = ["-", (0, (5, 2)), (0, (1.3, 1.5))]
        _pc = ["#3f007d", "#6a51a3", "#9e7bc0"]
        for j, p in enumerate(PL):
            fpr, tpr, auc = roc_curve(M[:, j], labels)
            ax.step(fpr, tpr, where="post", color=_pc[j], lw=1.7, ls=_ls[j], label=f"{p}  {auc:.2f}", solid_capstyle="round")
        ax.text(0.5, 1.02, f"{tr}_{DISP[me]}", ha="center", va="bottom", fontsize=8, fontweight="bold", transform=ax.transAxes)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        ax.legend(loc="lower right", frameon=False, fontsize=6.6, handlelength=1.0, labelspacing=0.3, borderpad=0.2)
        despine(ax, offset=3)
    for ax in axes[-1]:
        ax.set_xlabel("False positive rate")
    for ax in axes[:, 0]:
        ax.set_ylabel("True positive rate")
    fig.tight_layout(w_pad=1.6, h_pad=1.4)
    save(fig, "fig_rq4_roc_robust")


# ============ FIG 5: TOST equivalence ============
def fig_tost():
    gA = pd.read_csv(OUT / "equivalence_tost" / "tost_effect_size_aggregate.csv")
    gm = pd.read_csv(OUT / "equivalence_tost" / "tost_effect_size_by_metric.csv")
    au = pd.read_csv(OUT / "equivalence_tost" / "tost_auc_aggregate.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={"width_ratios": [1, 0.82]})
    ax = axes[0]
    rows = []
    for sc, lab in [("ALL", "All pairs (pooled)"), ("03-02", "P03-P02"), ("03-01", "P03-P01"), ("02-01", "P02-P01")]:
        r = gA[gA.scope == sc].iloc[0]; rows.append((lab, r["mean"], r["ci_lo"], r["ci_hi"], "pair"))
    for _, r in gm.iterrows():
        rows.append((f"  {DISP[r['metric']]}", r["mean"], r["ci_lo"], r["ci_hi"], "metric"))
    y = np.arange(len(rows))[::-1]
    h1 = ax.axvspan(-0.2, 0.2, color="#eef3ee", zorder=0, label="margin $\\pm$0.2")
    h2 = ax.axvspan(-0.1, 0.1, color="#d6e8d6", zorder=0, label="margin $\\pm$0.1")
    ax.axvline(0, color="#9a9a9a", lw=0.8, zorder=1)
    for i, (lab, m, lo, hi, kind) in enumerate(rows):
        col = "#1b7837" if kind == "pair" else "#7a7a7a"
        ax.plot([lo, hi], [y[i], y[i]], color=col, lw=1.8, solid_capstyle="round", zorder=3)
        ax.scatter(m, y[i], s=20, color=col, zorder=4)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(-0.35, 0.35); ax.set_xlabel("Delta Hedges g  (between pipelines)")
    ax.tick_params(left=False)
    despine(ax, which=("bottom",)); panel(ax, "A", dx=-44)
    axb = axes[1]
    rows2 = []
    for sc, lab in [("ALL", "All pairs (pooled)"), ("03-02", "P03-P02"), ("03-01", "P03-P01"), ("02-01", "P02-P01")]:
        r = au[au.scope == sc].iloc[0]; rows2.append((lab, r["mean"], r["ci_lo"], r["ci_hi"]))
    y2 = np.arange(len(rows2))[::-1]
    h3 = axb.axvspan(-0.05, 0.05, color="#d6e8d6", zorder=0, label="margin $\\pm$0.05")
    axb.axvline(0, color="#9a9a9a", lw=0.8, zorder=1)
    for i, (lab, m, lo, hi) in enumerate(rows2):
        axb.plot([lo, hi], [y2[i], y2[i]], color="#1b7837", lw=1.8, solid_capstyle="round", zorder=3)
        axb.scatter(m, y2[i], s=20, color="#1b7837", zorder=4)
    axb.set_yticks(y2); axb.set_yticklabels([r[0] for r in rows2])
    axb.set_xlim(-0.09, 0.09); axb.set_xlabel("Delta AUC  (between pipelines)")
    axb.tick_params(left=False)
    despine(axb, which=("bottom",)); panel(axb, "B", dx=-44)
    fig.legend(handles=[h1, h2, h3], loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=3, frameon=False, fontsize=7.6)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig_tost_equivalence")


# ============ FIG 6: RQ2 pipeline effect ============
def fig_rq2():
    pp = pd.read_csv(OUT / "rq2_pipeline_effect_v2" / "rq2_signed_relative_changes.csv")
    pp = pp[pp.feature == "mean"].copy()
    pp["contrast"] = pp["contrast"].str.replace("P", "", regex=False)
    pp["rel_pct"] = pp["median_signed_symmetric_percent"]
    TR = sorted(df.tract.unique()); ME = ["FA", "MD", "RD", "AD"]
    H = pp[pp.contrast == "03-01"].pivot_table(index="tract", columns="metric", values="rel_pct").reindex(index=TR, columns=ME)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.2), gridspec_kw={"width_ratios": [1, 1.05]})
    ax = axes[0]
    vmax = np.nanmax(np.abs(H.values))
    im = ax.imshow(H.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(ME))); ax.set_xticklabels([DISP[m] for m in ME])
    ax.set_yticks(range(len(TR))); ax.set_yticklabels(TR, fontsize=7)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(len(TR)):
        for j in range(len(ME)):
            v = H.values[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.6,
                    color="white" if abs(v) > vmax * 0.6 else "#222")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("signed relative change (%)", fontsize=7.2); cb.outline.set_visible(False)
    cb.ax.tick_params(length=2)
    panel(ax, "A", dx=-30)
    axb = axes[1]
    axb.yaxis.grid(True, color=GRID, lw=0.6)
    cons = ["02-01", "03-01", "03-02"]
    x = np.arange(len(ME)); w = 0.26
    bcol = ["#cfcfcf", "#8c8c8c", "#3f3f3f"]
    for k, c in enumerate(cons):
        vals = [pp[(pp.contrast == c) & (pp.metric == m)]["rel_pct"].median() for m in ME]
        axb.bar(x + (k - 1) * w, vals, w, color=bcol[k], edgecolor="white", zorder=3)
    axb.axhline(0, color="#9a9a9a", lw=0.8)
    axb.set_xticks(x); axb.set_xticklabels([DISP[m] for m in ME])
    axb.set_ylabel("median signed relative change (%)")
    despine(axb, which=("left",)); axb.tick_params(bottom=False)
    panel(axb, "B")
    handles = [Patch(facecolor=bcol[k]) for k in range(3)]
    axb.legend(handles=handles, labels=["P02 vs P01", "P03 vs P01", "P03 vs P02"], loc="upper center",
               bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False, fontsize=7.2, title="contrast",
               title_fontsize=7.4, columnspacing=1.1, handlelength=1.2, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig_rq2_pipeline_effect")


# ============ SUPP: ICC(3,1) vs ICC(2,1) ============
def fig_icc():
    im = pd.read_csv(OUT / "rq1_reproducibility_supplement" / "icc_consistency_vs_absolute_by_metric.csv")
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    x = np.arange(len(im)); w = 0.38
    b1 = ax.bar(x - w / 2, im["mean_icc31"], w, label="ICC(3,1) consistency", color="#3f3f3f", edgecolor="white", zorder=3)
    b2 = ax.bar(x + w / 2, im["mean_icc21"], w, label="ICC(2,1) absolute agreement", color="#bdbdbd", edgecolor="white", zorder=3)
    ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=7.6, color="#333")
    ax.bar_label(b2, fmt="%.2f", padding=2, fontsize=7.6, color="#333")
    ax.axhline(0.75, color=GREY, ls=(0, (5, 5)), lw=0.9, zorder=2)
    ax.text(1.015, 0.75, "good\n(0.75)", transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=6.8, color="#9a9a9a", linespacing=1.2)
    ax.set_xticks(x); ax.set_xticklabels([DISP[m] for m in im["metric"]], fontsize=9)
    ax.set_ylim(0, 1.0); ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("ICC"); ax.margins(x=0.08)
    ax.tick_params(bottom=False)
    despine(ax, which=("left",))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False,
              fontsize=8, handlelength=1.4, columnspacing=2.0)
    save(fig, "fig_supp_icc_comparison")


def write_titles():
    lines = ["FIGURE TITLES AND PANEL CAPTIONS",
             "(kept out of the figure images; use as figure captions / titles)",
             "=" * 72, ""]
    order = ["fig_variance_decomposition", "fig_rq3_effectsize_stability", "fig_rq4_auc_stability",
             "fig_rq4_roc_robust", "fig_tost_equivalence", "fig_rq2_pipeline_effect", "fig_supp_icc_comparison"]
    for stem in order:
        t = TITLES[stem]
        lines.append(f"[{stem}]")
        lines.append(f"  Title: {t['title']}")
        for k in ("A", "B"):
            if k in t:
                lines.append(f"  Panel {k}: {t[k]}")
        lines.append("")
    (FIG / "figure_titles.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    for fn in [fig_var, fig_rq3, fig_rq4_forest, fig_roc, fig_tost, fig_rq2, fig_icc]:
        fn(); print("done", fn.__name__)
    write_titles()
    print("ALL FIGURES SAVED to", FIG)
