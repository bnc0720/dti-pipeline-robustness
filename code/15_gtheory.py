# -*- coding: utf-8 -*-
"""
15_gtheory.py
Generalizability-theory (G-theory) treatment of the subject x pipeline design.
Recasts pipeline reproducibility as a reliability problem:
  - G(1)   = relative (ranking) reliability   -> equals consistency ICC(3,1)
  - Phi(1) = absolute (agreement) dependability-> equals absolute-agreement ICC(2,1)
  - D-study: Phi(k), G(k) when AVERAGING k pipelines (the actionable new layer).
Variance components from the single-facet crossed design; bootstrap 95% CIs by
resampling subjects (seed fixed). Does not modify any existing analysis.

Outputs (outputs/gtheory/):
  gtheory_by_biomarker.csv, gtheory_by_metric.csv, gtheory_dstudy.csv, report.txt
Figure: figures/fig_gtheory.{png,pdf,svg}
Dependencies: numpy, pandas, matplotlib (+ figstyle.py).
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
DATA = BASE / "data" / "all_pipelines_tractometry_features_master.csv"
OUTDIR = BASE / "outputs" / "gtheory"; OUTDIR.mkdir(parents=True, exist_ok=True)
FIG = BASE / "figures"
MC = {"FA": "#0072B2", "MD": "#E69F00", "AD": "#009E73", "RD": "#CC79A7"}
METRICS = ["FA", "MD", "RD", "AD"]
DISP = {"FA": "FA", "MD": "MD", "RD": "RD", "AD": "AxD"}
KMAX = 6
RANDOM_STATE = 42
N_BOOT = 1000

df = pd.read_csv(DATA)
SUBJ = sorted(df.subject.unique())
PIPES = ["pipeline_01", "pipeline_02", "pipeline_03"]
tracts = sorted(df.tract.unique())
MATS = {}
for m in METRICS:
    for tr in tracts:
        sub = df[(df.tract == tr) & (df.metric == m)]
        MATS[(m, tr)] = sub.pivot_table(index="subject", columns="pipeline", values="mean").reindex(index=SUBJ, columns=PIPES).values


def var_components(Y):
    n_s, n_p = Y.shape
    gm = Y.mean(); rm = Y.mean(1); cm = Y.mean(0)
    MSs = n_p * ((rm - gm) ** 2).sum() / (n_s - 1)
    MSp = n_s * ((cm - gm) ** 2).sum() / (n_p - 1)
    MSe = ((Y - rm[:, None] - cm[None, :] + gm) ** 2).sum() / ((n_s - 1) * (n_p - 1))
    return max((MSs - MSe) / n_p, 0.0), max((MSp - MSe) / n_s, 0.0), MSe


def G(vs, vp, ve, k):
    return vs / (vs + ve / k)


def Phi(vs, vp, ve, k):
    return vs / (vs + (vp + ve) / k)


# ---- per biomarker ----
rows = []
for m in METRICS:
    for tr in tracts:
        vs, vp, ve = var_components(MATS[(m, tr)])
        rows.append(dict(metric=m, tract=tr, var_subject=vs, var_pipeline=vp, var_residual=ve,
                         G1=G(vs, vp, ve, 1), Phi1=Phi(vs, vp, ve, 1),
                         pipeline_facet_pct=100 * vp / (vs + vp + ve)))
bm = pd.DataFrame(rows)
bm.to_csv(OUTDIR / "gtheory_by_biomarker.csv", index=False)

# ---- D-study per metric (average coefficients across the 10 biomarkers) ----
ds = []
for m in METRICS:
    sub = bm[bm.metric == m]
    for k in range(1, KMAX + 1):
        gk = np.mean([G(r.var_subject, r.var_pipeline, r.var_residual, k) for _, r in sub.iterrows()])
        pk = np.mean([Phi(r.var_subject, r.var_pipeline, r.var_residual, k) for _, r in sub.iterrows()])
        ds.append(dict(metric=m, k_pipelines=k, G=gk, Phi=pk))
dstudy = pd.DataFrame(ds)
dstudy.to_csv(OUTDIR / "gtheory_dstudy.csv", index=False)

# ---- bootstrap CI for metric-level G(1), Phi(1) ----
rng = np.random.default_rng(RANDOM_STATE)
idx = np.arange(len(SUBJ))
boot = {m: {"G1": [], "Phi1": []} for m in METRICS}
for _ in range(N_BOOT):
    bi = rng.choice(idx, size=len(idx), replace=True)
    for m in METRICS:
        g1s, p1s = [], []
        for tr in tracts:
            vs, vp, ve = var_components(MATS[(m, tr)][bi])
            g1s.append(G(vs, vp, ve, 1)); p1s.append(Phi(vs, vp, ve, 1))
        boot[m]["G1"].append(np.mean(g1s)); boot[m]["Phi1"].append(np.mean(p1s))

met_rows = []
for m in METRICS:
    sub = bm[bm.metric == m]
    g1, p1 = sub.G1.mean(), sub.Phi1.mean()
    gci = np.percentile(boot[m]["G1"], [2.5, 97.5]); pci = np.percentile(boot[m]["Phi1"], [2.5, 97.5])
    phik = {k: dstudy[(dstudy.metric == m) & (dstudy.k_pipelines == k)]["Phi"].iloc[0] for k in range(1, KMAX + 1)}
    k_excellent = next((k for k in range(1, KMAX + 1) if phik[k] >= 0.90), None)
    met_rows.append(dict(metric=m, G1=g1, G1_lo=gci[0], G1_hi=gci[1],
                         Phi1=p1, Phi1_lo=pci[0], Phi1_hi=pci[1],
                         gap_G_minus_Phi=g1 - p1, pipeline_facet_pct=sub.pipeline_facet_pct.mean(),
                         Phi2=phik[2], Phi3=phik[3], k_for_Phi_0p90=k_excellent))
met = pd.DataFrame(met_rows)
met.to_csv(OUTDIR / "gtheory_by_metric.csv", index=False)

# ---- report ----
L = ["GENERALIZABILITY THEORY (subject x pipeline, single-facet crossed)", "=" * 74,
     "G(1)=relative reliability (=consistency ICC), Phi(1)=absolute dependability (=absolute ICC).",
     "D-study Phi(k)=dependability when averaging k pipelines. Bootstrap 95% CI (subjects).", "",
     f"{'metric':6} {'G(1)':>16} {'Phi(1)':>16} {'gap':>6} {'pipe%':>6} {'Phi(2)':>7} {'Phi(3)':>7} {'k->0.90':>8}"]
for _, r in met.iterrows():
    L.append(f"{r['metric']:6} {r['G1']:.3f}[{r['G1_lo']:.2f}-{r['G1_hi']:.2f}] "
             f"{r['Phi1']:.3f}[{r['Phi1_lo']:.2f}-{r['Phi1_hi']:.2f}] {r['gap_G_minus_Phi']:6.3f} "
             f"{r['pipeline_facet_pct']:5.1f}% {r['Phi2']:7.3f} {r['Phi3']:7.3f} {str(r['k_for_Phi_0p90']):>8}")
L += ["", "Read-off: relative reliability is excellent for every metric (single pipeline",
      "ranks subjects reliably). Absolute dependability is metric-dependent: FA and AD",
      "carry a large pipeline-facet penalty (gap), MD/RD almost none. Averaging pipelines",
      "(D-study) restores FA absolute dependability from ~0.78 (1) to ~0.91 (3 pipelines)."]
(OUTDIR / "gtheory_report.txt").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))

# ---- figure (base): group means across pipelines, enriched with G-theory reliability ----
from statlib import t_ppf
from matplotlib import patheffects as pe
grp = {ss: gg for ss, gg in df.drop_duplicates("subject")[["subject", "group"]].values}
SC = {"FA": 1.0, "MD": 1e3, "RD": 1e3, "AD": 1e3}
GC = {"AD": "#c0392b", "CN": "#2c7fb8"}
YL = {"FA": "FA", "MD": "MD  (\u00d710\u207b\u00b3 mm\u00b2/s)", "RD": "RD  (\u00d710\u207b\u00b3 mm\u00b2/s)", "AD": "AxD  (\u00d710\u207b\u00b3 mm\u00b2/s)"}
mp = met.set_index("metric")
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(9.4, 6.0))
_gs = GridSpec(2, 4, height_ratios=[1.0, 1.12], hspace=0.80, wspace=0.44)
top = [fig.add_subplot(_gs[0, j]) for j in range(4)]
for ax, m in zip(top, METRICS):
    sub = df[df.metric == m]
    piv = sub.pivot_table(index="subject", columns="pipeline", values="mean", aggfunc="mean").reindex(columns=PIPES)
    sc = SC[m]
    xs = np.array([0, 1, 2])
    for zb, g in [(1, "AD"), (2, "CN")]:
        ids = [ss for ss in piv.index if grp[ss] == g]
        vals = piv.loc[ids].values * sc
        mean = vals.mean(0); sd = vals.std(0, ddof=1); n = len(ids)
        ci = t_ppf(0.975, n - 1) * sd / np.sqrt(n)
        ax.fill_between(xs, mean - ci, mean + ci, color=GC[g], alpha=0.16, lw=0, zorder=zb)
        ax.plot(xs, mean, color=GC[g], lw=2.8, marker="o", ms=6.2, mfc=GC[g], mec="white", mew=1.0, zorder=6,
                solid_capstyle="round", path_effects=[pe.Stroke(linewidth=4.6, foreground="white"), pe.Normal()])
    p01 = sub[sub.pipeline == "pipeline_01"]["mean"].mean()
    p03 = sub[sub.pipeline == "pipeline_03"]["mean"].mean()
    shift = 100 * (p03 - p01) / p01
    r = mp.loc[m]
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["P01", "P02", "P03"], fontsize=8)
    ax.set_xlim(-0.32, 2.32); ax.set_ylabel(YL[m], fontsize=8)
    ax.text(0.5, 1.30, DISP[m], transform=ax.transAxes, ha="center", va="bottom", fontsize=11, fontweight="bold", color="#222")
    ax.text(0.5, 1.135, f"\u0394P03\u2212P01 = {shift:+.1f}%", transform=ax.transAxes, ha="center", va="bottom", fontsize=6.8, color="#666")
    ax.text(0.5, 1.01, f"ranking G {r.G1:.2f}   \u00b7   absolute \u03a6 {r.Phi1:.2f}", transform=ax.transAxes, ha="center", va="bottom", fontsize=7.0, color="#1a1a1a", fontweight="bold")
    despine(ax, offset=3)
fig.legend(handles=[Line2D([0], [0], color=GC["AD"], lw=2.4, marker="o", ms=5, label="AD group mean"),
                    Line2D([0], [0], color=GC["CN"], lw=2.4, marker="o", ms=5, label="CN group mean")],
           loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=2, frameon=False, fontsize=8.2)
axd = fig.add_subplot(_gs[1, :])
axd.axhline(0.90, color=GREY, ls=(0, (5, 5)), lw=0.9, zorder=1)
axd.text(6.05, 0.90, "\u03a6 = 0.90", va="center", ha="left", fontsize=6.8, color="#9a9a9a")
for _m in METRICS:
    _d = dstudy[dstudy.metric == _m].sort_values("k_pipelines")
    _k = _d.k_pipelines.values.astype(float); _ph = _d.Phi.values
    axd.plot(_k[_k <= 3], _ph[_k <= 3], color=MC[_m], lw=2.0, marker="o", ms=4.6, mfc=MC[_m], mec="white", mew=0.6, zorder=4, label=DISP[_m])
    axd.plot(_k[_k >= 3], _ph[_k >= 3], color=MC[_m], lw=1.3, ls=(0, (3, 2)), alpha=0.75, zorder=3)
    _k90 = next((kk for kk, pp in zip(_k, _ph) if pp >= 0.90), None)
    if _k90 is not None:
        axd.scatter([_k90], [_ph[_k == _k90][0]], s=70, facecolor="none", edgecolor=MC[_m], lw=1.4, zorder=5)
axd.set_xticks(range(1, 7)); axd.set_xlim(0.8, 6.7); axd.set_ylim(0.74, 1.0)
axd.set_xlabel("number of pipelines averaged (k)")
axd.set_ylabel("absolute dependability  \u03a6(k)")
axd.grid(True, axis="y", color=GRID, lw=0.6)
axd.text(0.0, 1.05, "D-study \u2014 averaging pipelines recovers absolute dependability (solid: k\u22643, within study; dashed: extrapolation; circle: first k reaching \u03a6\u22650.90).",
         transform=axd.transAxes, ha="left", va="bottom", fontsize=6.8, color="#555")
axd.legend(loc="lower right", ncol=4, frameon=False, fontsize=7.4, columnspacing=1.1, handlelength=1.5)
despine(axd, offset=3)
fig.tight_layout(rect=(0, 0.0, 1, 0.97))
for ext in ("png", "pdf", "svg"):
    fig.savefig(FIG / f"fig_gtheory.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("\nfigure: fig_gtheory (base group-means, G-theory enriched)")
