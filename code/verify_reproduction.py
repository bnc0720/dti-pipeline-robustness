# -*- coding: utf-8 -*-
"""
verify_reproduction.py
======================
Audits the reproduction in two ways and writes ../VERIFICATION.md:

  (1) INDEPENDENT AGREEMENT - the NumPy-only reconstruction (statlib) is checked
      against the committed SciPy/statsmodels outputs (ICC and Hedges g),
      demonstrating that two independent implementations agree.
  (2) HEADLINE NUMBERS - every reported value is checked against the manuscript
      drafts within tolerance.

Run after run_all.py. Dependencies: numpy, pandas (+ local statlib).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from statlib import icc_two_way, hedges_g

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "outputs"
DATA = BASE / "data" / "all_pipelines_tractometry_features_master.csv"
PIPES = ["pipeline_01", "pipeline_02", "pipeline_03"]
METRICS = ["FA", "MD", "RD", "AD"]

df = pd.read_csv(DATA)
subjects = sorted(df.subject.unique())
group_map = {s: g for s, g in df.drop_duplicates("subject")[["subject", "group"]].values}
labels = np.array([1 if group_map[s] == "AD" else 0 for s in subjects])
tracts = sorted(df.tract.unique())


def mat(tr, me, feat="mean"):
    sub = df[(df.tract == tr) & (df.metric == me)]
    return sub.pivot_table(index="subject", columns="pipeline", values=feat).reindex(index=subjects, columns=PIPES).values


checks = []  # (section, name, expected, got, ok)


def chk(section, name, expected, got, ok):
    checks.append((section, name, expected, got, bool(ok)))


# ---------------------------------------------------------------- (1) agreement
icc_comm = pd.read_csv(OUT / "rq1_reproducibility" / "icc_all_features.csv")
icc_comm = icc_comm[icc_comm.feature == "mean"].set_index(["tract", "metric"])["icc_3_1"]
dirc = pd.read_csv(OUT / "rq3_biological_robustness" / "direction_consistency.csv")
dirc = dirc[dirc.feature == "mean"].set_index(["tract", "metric"])

icc_diffs, g_diffs = [], []
for tr in tracts:
    for me in METRICS:
        Y = mat(tr, me)
        icc31, _ = icc_two_way(Y)
        icc_diffs.append(abs(icc31 - icc_comm.loc[(tr, me)]))
        my_g = [hedges_g(Y[labels == 1, p], Y[labels == 0, p]) for p in range(3)]
        comm_g = [dirc.loc[(tr, me), f"pipeline_0{p+1}_g"] for p in range(3)]
        g_diffs += [abs(a - b) for a, b in zip(my_g, comm_g)]

chk("Independent agreement", "ICC(3,1) NumPy vs committed SciPy (max abs diff)",
    "< 1e-6", f"{max(icc_diffs):.2e}", max(icc_diffs) < 1e-6)
chk("Independent agreement", "Hedges g NumPy vs committed (max abs diff)",
    "< 1e-6", f"{max(g_diffs):.2e}", max(g_diffs) < 1e-6)


# ---------------------------------------------------------------- (2) headline
def near(a, b, tol):
    return abs(a - b) <= tol

# variance decomposition
vov = pd.read_csv(OUT / "variance_decomposition" / "variance_decomposition_overall.csv").iloc[0]
vme = pd.read_csv(OUT / "variance_decomposition" / "variance_decomposition_by_metric.csv").set_index("metric")
chk("Variance", "biological ~ 91%", "91", f"{vov['biological']:.2f}", near(vov["biological"], 91, 1.5))
chk("Variance", "pipeline ~ 5%", "5", f"{vov['pipeline']:.2f}", near(vov["pipeline"], 5, 1.0))
chk("Variance", "Group x Pipeline ~ 0.14%", "0.14", f"{vov['interaction']:.3f}", near(vov["interaction"], 0.14, 0.1))
chk("Variance", "FA pipeline ~ 11%", "11", f"{vme.loc['FA','pipeline']:.2f}", near(vme.loc["FA", "pipeline"], 11, 1.5))

# ICC consistency vs absolute
icm = pd.read_csv(OUT / "rq1_reproducibility_supplement" / "icc_consistency_vs_absolute_by_metric.csv").set_index("metric")
icb = pd.read_csv(OUT / "rq1_reproducibility_supplement" / "icc_consistency_vs_absolute_by_biomarker.csv")
chk("Reproducibility", "mean ICC(3,1) ~ 0.94", "0.94", f"{icb['icc_3_1_consistency'].mean():.3f}", near(icb["icc_3_1_consistency"].mean(), 0.94, 0.01))
chk("Reproducibility", "mean ICC(2,1) ~ 0.87", "0.87", f"{icb['icc_2_1_absolute'].mean():.3f}", near(icb["icc_2_1_absolute"].mean(), 0.87, 0.01))
chk("Reproducibility", "FA ICC 0.92 -> 0.78", "0.92/0.78", f"{icm.loc['FA','mean_icc31']:.2f}/{icm.loc['FA','mean_icc21']:.2f}", near(icm.loc["FA", "mean_icc31"], 0.92, 0.015) and near(icm.loc["FA", "mean_icc21"], 0.78, 0.02))
chk("Reproducibility", "AD ICC 0.93 -> 0.84", "0.93/0.84", f"{icm.loc['AD','mean_icc31']:.2f}/{icm.loc['AD','mean_icc21']:.2f}", near(icm.loc["AD", "mean_icc31"], 0.925, 0.02) and near(icm.loc["AD", "mean_icc21"], 0.84, 0.02))
chk("Reproducibility", "biomarkers < 0.75: absolute=4, consistency=0", "4 / 0",
    f"{int((icb['icc_2_1_absolute']<0.75).sum())} / {int((icb['icc_3_1_consistency']<0.75).sum())}",
    int((icb["icc_2_1_absolute"] < 0.75).sum()) == 4 and int((icb["icc_3_1_consistency"] < 0.75).sum()) == 0)

# RQ2 omnibus
om = pd.read_csv(OUT / "rq2_pipeline_effect_v2" / "rq2_omnibus_pipeline_tests.csv")
nsig = int(om["fdr_significant_primary"].sum())
chk("RQ2 pipeline effect", "omnibus FDR-significant 34/40", "34", f"{nsig}", nsig == 34)

# RQ3
rsum = pd.read_csv(OUT / "rq3_biological_robustness" / "rq3_feature_summary.csv")
rmean = rsum[rsum.feature == "mean"].iloc[0]
chk("RQ3 biological robustness", "Group x Pipeline interactions 0/40", "0", f"{int(rmean['n_fdr_significant_interactions'])}", int(rmean["n_fdr_significant_interactions"]) == 0)
chk("RQ3 biological robustness", "ranking rho ~ 0.90", "0.90", f"{rmean['mean_ranking_spearman_rho']:.3f}", near(rmean["mean_ranking_spearman_rho"], 0.90, 0.02))
chk("RQ3 biological robustness", "same direction 87.5%", "87.5", f"{rmean['same_direction_percent']:.1f}", near(rmean["same_direction_percent"], 87.5, 0.1))

# RQ4
au = pd.read_csv(OUT / "rq4_diagnostic_utility" / "auc_stability_summary.csv")
au = au[au.feature == "mean"]
chk("RQ4 diagnostic utility", "mean AUC range ~ 0.045", "0.045", f"{au['auc_range'].mean():.4f}", near(au["auc_range"].mean(), 0.045, 0.005))

# TOST
es = pd.read_csv(OUT / "equivalence_tost" / "tost_effect_size_aggregate.csv")
ac = pd.read_csv(OUT / "equivalence_tost" / "tost_auc_aggregate.csv")
es_ok = bool(es["equivalent_0.1"].all())
ac_ok = bool(ac["equivalent_0.05"].all())
chk("Equivalence (TOST)", "aggregate Delta g equivalent at strict 0.1 (all pairs)", "all True", str(es_ok), es_ok)
chk("Equivalence (TOST)", "aggregate Delta AUC equivalent at 0.05 (all pairs)", "all True", str(ac_ok), ac_ok)


# ---- common-mode (direct test) ----
from statlib import pearson_r_p as _pr, t_ppf as _tp
cm = pd.read_csv(OUT / "common_mode" / "common_mode_shifts.csv")
_r, _ = _pr(cm["shift_AD_pct"].values, cm["shift_CN_pct"].values)
_slope = float(np.polyfit(cm["shift_CN_pct"].values, cm["shift_AD_pct"].values, 1)[0])
_ds = cm["diff_std"].values
_md = _ds.mean(); _ci = _tp(0.95, len(_ds) - 1) * _ds.std(ddof=1) / np.sqrt(len(_ds))
chk("Common-mode", "AD-shift vs CN-shift Pearson r ~ 0.94", "~0.94", f"{_r:.2f}", abs(_r - 0.94) < 0.03)
chk("Common-mode", "regression slope ~ 1 (identity)", "~1.0", f"{_slope:.2f}", abs(_slope - 1.0) < 0.25)
chk("Common-mode", "standardized differential equivalent at 0.1", "equivalent", f"{_md:+.3f}+/-{_ci:.3f}", (abs(_md) + _ci) < 0.1)


# ---- metric-deviation summary ----
mdv = pd.read_csv(OUT / "metric_deviation" / "metric_deviation_summary.csv").set_index("metric")
chk("Metric deviation", "FA largest mean |delta %| ~ 3.9", "~3.9", f"{mdv.loc['FA','mean_abs_pct']:.2f}", near(mdv.loc["FA","mean_abs_pct"], 3.9, 0.3))
chk("Metric deviation", "MD smallest mean |delta %| ~ 0.7", "~0.7", f"{mdv.loc['MD','mean_abs_pct']:.2f}", near(mdv.loc["MD","mean_abs_pct"], 0.7, 0.3))
chk("Metric deviation", "FA is the most pipeline-deviant metric", "FA", mdv["mean_abs_pct"].idxmax(), mdv["mean_abs_pct"].idxmax() == "FA")


# ---- per-pipeline value table ----
pv = pd.read_csv(OUT / "pipeline_value_table" / "pipeline_value_by_metric.csv").set_index("metric")
chk("Pipeline value table", "FA P03 vs P01 ~ -5.7%", "~-5.7", f"{pv.loc['FA','pct_P03_vs_P01']:.1f}", near(pv.loc["FA","pct_P03_vs_P01"], -5.7, 0.5))
chk("Pipeline value table", "MD has the smallest |shift| under P03", "MD", min(METRICS, key=lambda m: abs(pv.loc[m,"pct_P03_vs_P01"])), min(METRICS, key=lambda m: abs(pv.loc[m,"pct_P03_vs_P01"])) == "MD")


# ---- generalizability theory (G-theory) ----
gt = pd.read_csv(OUT / "gtheory" / "gtheory_by_metric.csv").set_index("metric")
chk("G-theory", "FA relative G(1) ~ 0.92 (=consistency ICC)", "0.92", f"{gt.loc['FA','G1']:.2f}", near(gt.loc['FA','G1'], 0.92, 0.02))
chk("G-theory", "FA absolute Phi(1) ~ 0.78 (=absolute ICC)", "0.78", f"{gt.loc['FA','Phi1']:.2f}", near(gt.loc['FA','Phi1'], 0.78, 0.02))
chk("G-theory", "FA reaches Phi>=0.90 by averaging 3 pipelines", "3", str(int(gt.loc['FA','k_for_Phi_0p90'])), int(gt.loc['FA','k_for_Phi_0p90']) == 3)


# ---------------------------------------------------------------- write report
n_pass = sum(1 for *_, ok in checks if ok)
n_tot = len(checks)
lines = ["# Reproduction verification", "",
         f"**{n_pass}/{n_tot} checks passed.**", "",
         "Generated by `code/verify_reproduction.py` after `run_all.py`.", ""]
section = None
for sec, name, exp, got, ok in checks:
    if sec != section:
        lines += ["", f"## {sec}", "", "| Check | Expected | Got | Status |", "|---|---|---|---|"]
        section = sec
    lines.append(f"| {name} | {exp} | {got} | {'✅ PASS' if ok else '❌ FAIL'} |")
(BASE / "VERIFICATION.md").write_text("\n".join(lines), encoding="utf-8")

print(f"{n_pass}/{n_tot} checks passed")
for sec, name, exp, got, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: expected {exp}, got {got}")
if n_pass != n_tot:
    sys.exit(1)
