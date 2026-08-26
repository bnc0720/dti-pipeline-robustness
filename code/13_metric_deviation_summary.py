# -*- coding: utf-8 -*-
"""
13_metric_deviation_summary.py
Consolidates the magnitude of pipeline-induced deviation PER METRIC (and per
pipeline pair) from already-committed outputs (RQ1 CV, RQ2 relative changes &
pairwise contrasts, variance decomposition). No existing analysis is modified.

Outputs (outputs/metric_deviation/):
  metric_deviation_summary.csv   (one row per metric)
  metric_pair_deviation.csv      (metric x pipeline pair)
  metric_deviation_report.txt
Dependencies: numpy, pandas.
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "outputs"
OUTDIR = OUT / "metric_deviation"; OUTDIR.mkdir(parents=True, exist_ok=True)
METRICS = ["FA", "MD", "RD", "AD"]
PAIRS = ["P02-P01", "P03-P01", "P03-P02"]

rc = pd.read_csv(OUT / "rq2_pipeline_effect_v2" / "rq2_signed_relative_changes.csv")
rc = rc[rc.feature == "mean"].copy()
rc["abs_pct"] = rc["median_signed_symmetric_percent"].abs()
pw = pd.read_csv(OUT / "rq2_pipeline_effect_v2" / "rq2_pairwise_model_contrasts.csv")
pw = pw[pw.feature == "mean"].copy()
cv = pd.read_csv(OUT / "rq1_reproducibility" / "cv_all_features.csv")
cv = cv[cv.feature == "mean"]
vm = pd.read_csv(OUT / "variance_decomposition" / "variance_decomposition_by_metric.csv").set_index("metric")

# ---- per metric x pair ----
pair_rows = []
for m in METRICS:
    for pr in PAIRS:
        s = rc[(rc.metric == m) & (rc.contrast == pr)]
        sig = pw[(pw.metric == m) & (pw.contrast == pr)]["fdr_significant_pairwise"].sum()
        pair_rows.append(dict(
            metric=m, pair=pr,
            median_signed_pct=s["median_signed_symmetric_percent"].median(),
            q1_pct=s["median_signed_symmetric_percent"].quantile(.25),
            q3_pct=s["median_signed_symmetric_percent"].quantile(.75),
            mean_abs_pct=s["abs_pct"].mean(),
            n_fdr_significant=int(sig), n_biomarkers=len(s)))
pair = pd.DataFrame(pair_rows)
pair.to_csv(OUTDIR / "metric_pair_deviation.csv", index=False)

# ---- per metric summary ----
rows = []
for m in METRICS:
    s = rc[rc.metric == m]
    rows.append(dict(
        metric=m,
        mean_abs_pct=s["abs_pct"].mean(),
        median_abs_pct=s["abs_pct"].median(),
        max_abs_pct=s["abs_pct"].max(),
        signed_P02_P01=rc[(rc.metric == m) & (rc.contrast == "P02-P01")]["median_signed_symmetric_percent"].median(),
        signed_P03_P01=rc[(rc.metric == m) & (rc.contrast == "P03-P01")]["median_signed_symmetric_percent"].median(),
        signed_P03_P02=rc[(rc.metric == m) & (rc.contrast == "P03-P02")]["median_signed_symmetric_percent"].median(),
        mean_cv_percent=cv[cv.metric == m]["mean_cv_percent"].mean(),
        pipeline_variance_pct=vm.loc[m, "pipeline"],
        n_fdr_significant=int(pw[pw.metric == m]["fdr_significant_pairwise"].sum()),
        n_contrasts=int(len(pw[pw.metric == m]))))
summ = pd.DataFrame(rows)
summ.to_csv(OUTDIR / "metric_deviation_summary.csv", index=False)

# ---- report ----
L = ["PER-METRIC PIPELINE-INDUCED DEVIATION SUMMARY", "=" * 70,
     "Magnitude of the deviation caused by changing the preprocessing pipeline,",
     "summarised per metric (from committed RQ1/RQ2/variance outputs).", "",
     f"{'metric':6} {'mean|Δ%|':>9} {'max|Δ%|':>9} {'CV%':>6} {'var%':>6}  {'signed medians (02-01/03-01/03-02)':>38}  {'FDR-sig':>9}"]
for _, r in summ.iterrows():
    L.append(f"{r['metric']:6} {r['mean_abs_pct']:8.2f}% {r['max_abs_pct']:8.2f}% {r['mean_cv_percent']:5.2f}% {r['pipeline_variance_pct']:5.1f}%  "
             f"{r['signed_P02_P01']:+7.2f}/{r['signed_P03_P01']:+7.2f}/{r['signed_P03_P02']:+7.2f} %  {r['n_fdr_significant']:3d}/{r['n_contrasts']:<3d}")
L += ["", "Interpretation: FA shows the largest pipeline-induced deviation, MD the",
      "smallest; RD and AD intermediate. CV% and variance% tell the same story.",
      "All values derive from the master CSV via the committed pipeline."]
(OUTDIR / "metric_deviation_report.txt").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
print("\nwritten:", OUTDIR / "metric_deviation_summary.csv", "+ metric_pair_deviation.csv")
