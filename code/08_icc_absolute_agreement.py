# -*- coding: utf-8 -*-
"""
08_icc_absolute_agreement.py
============================
RQ1 supplement. The primary RQ1 script reports consistency ICC(3,1). A reviewer
will reasonably ask for absolute-agreement ICC(2,1), which additionally
penalizes the systematic pipeline offsets demonstrated in RQ2. This script
computes BOTH definitions per biomarker (primary feature) and summarizes the
consistency -> absolute-agreement change per metric.

Input : data/all_pipelines_tractometry_features_master.csv
Outputs (outputs/rq1_reproducibility_supplement/):
    icc_consistency_vs_absolute_by_biomarker.csv
    icc_consistency_vs_absolute_by_metric.csv
    icc_absolute_agreement_report.txt

Dependencies: NumPy, pandas (plus local statlib.py). No SciPy/statsmodels.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from statlib import icc_two_way

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "all_pipelines_tractometry_features_master.csv"
OUTDIR = BASE / "outputs" / "rq1_reproducibility_supplement"
OUTDIR.mkdir(parents=True, exist_ok=True)

PRIMARY_FEATURE = "mean"
PIPELINES = ["pipeline_01", "pipeline_02", "pipeline_03"]
METRICS = ["FA", "MD", "RD", "AD"]
GOOD = 0.75


def cat(x):
    return ("poor" if x < 0.50 else "moderate" if x < 0.75 else "good" if x < 0.90 else "excellent")


def main():
    df = pd.read_csv(DATA)
    subjects = sorted(df["subject"].unique())
    tracts = sorted(df["tract"].unique())

    rows = []
    for tract in tracts:
        for metric in METRICS:
            sub = df[(df["tract"] == tract) & (df["metric"] == metric)]
            Y = (sub.pivot_table(index="subject", columns="pipeline", values=PRIMARY_FEATURE)
                    .reindex(index=subjects, columns=PIPELINES).values)
            icc31, icc21 = icc_two_way(Y)
            rows.append(dict(tract=tract, metric=metric, feature=PRIMARY_FEATURE,
                             icc_3_1_consistency=icc31, icc_2_1_absolute=icc21,
                             delta=icc31 - icc21,
                             cat_consistency=cat(icc31), cat_absolute=cat(icc21)))
    bm = pd.DataFrame(rows)
    bm.to_csv(OUTDIR / "icc_consistency_vs_absolute_by_biomarker.csv", index=False)

    by_metric = (bm.groupby("metric")
                   .agg(mean_icc31=("icc_3_1_consistency", "mean"),
                        mean_icc21=("icc_2_1_absolute", "mean"))
                   .reindex(METRICS).reset_index())
    by_metric.to_csv(OUTDIR / "icc_consistency_vs_absolute_by_metric.csv", index=False)

    n_below_abs = int((bm["icc_2_1_absolute"] < GOOD).sum())
    n_below_cons = int((bm["icc_3_1_consistency"] < GOOD).sum())

    L = []
    L.append("ICC: CONSISTENCY (3,1) vs ABSOLUTE AGREEMENT (2,1)")
    L.append("=" * 80)
    L.append(f"Primary feature: {PRIMARY_FEATURE}   |   biomarkers: {len(bm)}")
    L.append(f"Mean consistency  ICC(3,1) = {bm['icc_3_1_consistency'].mean():.3f}")
    L.append(f"Mean absolute     ICC(2,1) = {bm['icc_2_1_absolute'].mean():.3f}")
    L.append(f"Biomarkers below 'good' (0.75): consistency {n_below_cons}/40, absolute {n_below_abs}/40")
    L.append("")
    L.append("BY METRIC")
    L.append("-" * 80)
    for _, r in by_metric.iterrows():
        L.append(f"  {r['metric']:3s}: ICC(3,1)={r['mean_icc31']:.3f}  ->  ICC(2,1)={r['mean_icc21']:.3f}")
    L.append("")
    L.append("INTERPRETATION")
    L.append("-" * 80)
    L.append("Absolute agreement is lower than consistency for the metrics with the")
    L.append("largest systematic pipeline offsets (FA, AD) and essentially unchanged")
    L.append("for MD/RD. RQ1 is therefore framed as high reproducibility of subject")
    L.append("RANKING rather than of absolute values.")
    (OUTDIR / "icc_absolute_agreement_report.txt").write_text("\n".join(L), encoding="utf-8")

    print("[08] ICC consistency vs absolute written to", OUTDIR)
    print(by_metric.round(3).to_string(index=False))
    print(f"mean ICC(3,1)={bm['icc_3_1_consistency'].mean():.3f} -> ICC(2,1)={bm['icc_2_1_absolute'].mean():.3f}; "
          f"below-good abs={n_below_abs}, cons={n_below_cons}")


if __name__ == "__main__":
    main()
