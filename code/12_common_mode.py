# -*- coding: utf-8 -*-
"""
12_common_mode.py
Direct test of the common-mode hypothesis: the preprocessing-induced shift in
absolute DTI values is the same in AD and CN (additive, group-independent),
so it cancels in the between-group contrast.

For each biomarker (tract x metric, primary feature) and each pipeline pair we
compute the signed symmetric relative shift (%) SEPARATELY in the AD and CN
groups, then compare them. Common-mode => shift_AD ~ shift_CN (points on the
identity line; differential component ~ 0).

Does NOT modify any existing analysis output. New outputs only:
  outputs/common_mode/common_mode_shifts.csv
  figures/fig_common_mode.{png,pdf,svg}
Dependencies: numpy, pandas, matplotlib (+ local figstyle.py, statlib.py).
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
from statlib import t_ppf, pearson_r_p
warnings.filterwarnings("ignore"); logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
figstyle.apply()

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "all_pipelines_tractometry_features_master.csv"
OUTDIR = BASE / "outputs" / "common_mode"; OUTDIR.mkdir(parents=True, exist_ok=True)
FIG = BASE / "figures"
MC = {"FA": "#0072B2", "MD": "#E69F00", "AD": "#009E73", "RD": "#CC79A7"}
METRICS = ["FA", "MD", "RD", "AD"]
DISP = {"FA": "FA", "MD": "MD", "RD": "RD", "AD": "AxD"}
PAIRS = [("P02-P01", "pipeline_01", "pipeline_02"), ("P03-P01", "pipeline_01", "pipeline_03"),
         ("P03-P02", "pipeline_02", "pipeline_03")]


def sym_pct(a, b):
    return 100.0 * (b - a) / ((a + b) / 2.0)


def main():
    df = pd.read_csv(DATA)
    subj = df.drop_duplicates("subject")[["subject", "group"]]
    grp = dict(zip(subj.subject, subj.group))
    tracts = sorted(df.tract.unique())
    rows = []
    for tr in tracts:
        for me in METRICS:
            sub = df[(df.tract == tr) & (df.metric == me)]
            piv = sub.pivot_table(index="subject", columns="pipeline", values="mean")
            piv["group"] = [grp[s] for s in piv.index]
            mAD = piv[piv.group == "AD"].mean(numeric_only=True)
            mCN = piv[piv.group == "CN"].mean(numeric_only=True)
            sdpool = piv.drop(columns="group").std(ddof=1)  # across all subjects per pipeline
            for lbl, pi, pj in PAIRS:
                shift_AD = sym_pct(mAD[pi], mAD[pj])
                shift_CN = sym_pct(mCN[pi], mCN[pj])
                # standardized (pooled SD at reference pipeline pi) -> SD units
                ref_sd = sdpool[pi]
                std_AD = (mAD[pj] - mAD[pi]) / ref_sd
                std_CN = (mCN[pj] - mCN[pi]) / ref_sd
                rows.append(dict(tract=tr, metric=me, pair=lbl,
                                 shift_AD_pct=shift_AD, shift_CN_pct=shift_CN,
                                 diff_pct=shift_AD - shift_CN, mean_pct=(shift_AD + shift_CN) / 2,
                                 std_AD=std_AD, std_CN=std_CN, diff_std=std_AD - std_CN))
    d = pd.DataFrame(rows)
    d.to_csv(OUTDIR / "common_mode_shifts.csv", index=False)

    # ---- headline statistics ----
    r, _ = pearson_r_p(d.shift_AD_pct.values, d.shift_CN_pct.values)
    slope = np.polyfit(d.shift_CN_pct.values, d.shift_AD_pct.values, 1)[0]
    diff = d.diff_pct.values
    md = diff.mean(); se = diff.std(ddof=1) / np.sqrt(len(diff))
    ci = t_ppf(0.95, len(diff) - 1) * se
    cm_ratio = np.median(np.abs(d.diff_pct)) / np.median(np.abs(d.mean_pct))
    # variance of the group-specific shift explained by the shared shift
    ss_tot = np.var(np.r_[d.shift_AD_pct, d.shift_CN_pct])
    ss_diff = np.var(d.diff_pct) / 2
    shared_var_frac = 1 - ss_diff / ss_tot
    diff_std = d.diff_std.values
    md_s = diff_std.mean(); ci_s = t_ppf(0.95, len(diff_std) - 1) * diff_std.std(ddof=1) / np.sqrt(len(diff_std))

    L = []
    L.append("DIRECT COMMON-MODE TEST")
    L.append("=" * 64)
    L.append(f"Units: signed symmetric relative shift (%), AD vs CN, per biomarker-pair (n={len(d)}).")
    L.append(f"AD-shift vs CN-shift: Pearson r = {r:.3f}; regression slope = {slope:.3f} (identity = 1).")
    L.append(f"Differential shift (AD - CN): mean = {md:+.3f}% (90% CI {md-ci:+.3f} to {md+ci:+.3f}).")
    L.append(f"Differential / shared shift (median ratio) = {cm_ratio:.3f}  -> group-specific part is ~{cm_ratio*100:.0f}% of the shared shift.")
    L.append(f"Shared shift explains {shared_var_frac*100:.1f}% of the variance in group-specific shifts.")
    L.append(f"Standardized differential (SD units): mean = {md_s:+.4f} (90% CI {md_s-ci_s:+.4f} to {md_s+ci_s:+.4f}); margin 0.1 -> {'EQUIVALENT' if abs(md_s)+ci_s<0.1 else 'n.s.'}.")
    (OUTDIR / "common_mode_report.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), gridspec_kw={"width_ratios": [1.05, 1]})
    ax = axes[0]
    lim = [min(d.shift_AD_pct.min(), d.shift_CN_pct.min()) - 0.6, max(d.shift_AD_pct.max(), d.shift_CN_pct.max()) + 0.6]
    ax.plot(lim, lim, ls=(0, (4, 4)), color=GREY, lw=0.9, zorder=1)
    for m in METRICS:
        s = d[d.metric == m]
        ax.scatter(s.shift_CN_pct, s.shift_AD_pct, s=20, color=MC[m], alpha=0.8, edgecolor="white", lw=0.4, zorder=3, label=m)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("pipeline shift in CN (%)"); ax.set_ylabel("pipeline shift in AD (%)")
    ax.text(0.04, 0.96, f"r = {r:.2f}\nslope = {slope:.2f}", transform=ax.transAxes, fontsize=7.5, va="top", color="#444")
    despine(ax); panel(ax, "A")
    axb = axes[1]
    band = 1.0
    axb.axhspan(-band, band, color="#f0f0f0", zorder=0)
    axb.axhline(0, color="#9a9a9a", lw=0.8, zorder=1)
    rng = np.random.default_rng(2)
    for i, m in enumerate(METRICS):
        s = d[d.metric == m]
        xx = i + (rng.random(len(s)) - 0.5) * 0.55
        axb.scatter(xx, s.diff_pct, s=16, color=MC[m], alpha=0.75, edgecolor="white", lw=0.3, zorder=3)
    axb.set_xticks(range(len(METRICS))); axb.set_xticklabels([DISP[m] for m in METRICS])
    axb.set_ylabel("differential shift, AD − CN (%)")
    axb.set_xlim(-0.5, 3.5)
    axb.text(0.015, 0.97, f"mean = {md:+.2f}% (≈0); shaded |Δ| < {band:.0f}%", transform=axb.transAxes, fontsize=6.8, va="top", color="#555")
    axb.tick_params(bottom=False); despine(axb, which=("left",)); panel(axb, "B")
    handles = [Line2D([0], [0], marker="o", color="w", mfc=MC[m], ms=6, label=DISP[m]) for m in METRICS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"fig_common_mode.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("\nfigure: fig_common_mode ; csv: common_mode_shifts.csv")


if __name__ == "__main__":
    main()
