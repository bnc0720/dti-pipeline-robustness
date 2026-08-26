# -*- coding: utf-8 -*-
"""
09_equivalence_tost.py
======================
Equivalence (TOST) testing of pipeline differences. A non-significant
Group x Pipeline interaction is only absence of evidence; this script provides
positive evidence that pipeline differences fall within pre-registered
negligibility margins.

Two levels:
  * AGGREGATE  - biomarker treated as the sampling unit; the 40 per-biomarker
                 between-pipeline differences are tested as one sample
                 (well powered). Margins: Hedges-g delta = 0.2 (Cohen 'small',
                 pre-registered) and the stricter 0.1; AUC delta = 0.05.
  * BIOMARKER  - within-subject bootstrap (subjects resampled with their three
                 pipeline values kept together), 90% CI per biomarker
                 (under-powered; reported as an honest limitation).

Input : data/all_pipelines_tractometry_features_master.csv
Outputs (outputs/equivalence_tost/):
    tost_effect_size_aggregate.csv
    tost_effect_size_by_metric.csv
    tost_auc_aggregate.csv
    tost_biomarker_level.csv
    tost_report.txt

Reproducibility: bootstrap uses a fixed seed (RANDOM_STATE = 42).
Dependencies: NumPy, pandas (plus local statlib.py). No SciPy/statsmodels.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from statlib import hedges_g, auc_oriented, tost_one_sample

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "all_pipelines_tractometry_features_master.csv"
OUTDIR = BASE / "outputs" / "equivalence_tost"
OUTDIR.mkdir(parents=True, exist_ok=True)

PRIMARY_FEATURE = "mean"
PIPELINES = ["pipeline_01", "pipeline_02", "pipeline_03"]
METRICS = ["FA", "MD", "RD", "AD"]
PAIRS = [("02-01", 1, 0), ("03-01", 2, 0), ("03-02", 2, 1)]   # label, j, i  -> g[j]-g[i]
G_MARGINS = [0.2, 0.1]
AUC_MARGIN = 0.05
N_BOOT = 2000
RANDOM_STATE = 42


def build_matrices(df):
    subjects = sorted(df["subject"].unique())
    group_map = {s: g for s, g in df.drop_duplicates("subject")[["subject", "group"]].values}
    labels = np.array([1 if group_map[s] == "AD" else 0 for s in subjects])
    tracts = sorted(df["tract"].unique())
    mats = {}
    for tract in tracts:
        for metric in METRICS:
            sub = df[(df["tract"] == tract) & (df["metric"] == metric)]
            Y = (sub.pivot_table(index="subject", columns="pipeline", values=PRIMARY_FEATURE)
                    .reindex(index=subjects, columns=PIPELINES).values)
            mats[(tract, metric)] = Y
    return subjects, labels, tracts, mats


def g_per_pipeline(Y, labels):
    return np.array([hedges_g(Y[labels == 1, p], Y[labels == 0, p]) for p in range(Y.shape[1])])


def auc_per_pipeline(Y, labels):
    return np.array([auc_oriented(Y[:, p], labels) for p in range(Y.shape[1])])


def aggregate_rows(delta_by_pair, margins, margin_main):
    """delta_by_pair: dict label -> np.array of per-biomarker deltas."""
    rows = []
    pooled = np.concatenate([delta_by_pair[lbl] for lbl, _, _ in PAIRS])
    scopes = [("ALL", pooled)] + [(lbl, delta_by_pair[lbl]) for lbl, _, _ in PAIRS]
    for scope, vals in scopes:
        rec = dict(scope=scope, n=len(vals), mean=float(np.mean(vals)))
        # 90% CI from the main-margin TOST (alpha=0.05 -> 90% CI)
        base = tost_one_sample(vals, margin_main)
        rec["ci_lo"] = base["ci_lo"]
        rec["ci_hi"] = base["ci_hi"]
        for m in margins:
            r = tost_one_sample(vals, m)
            rec[f"p_tost_{m}"] = r["p_tost"]
            rec[f"equivalent_{m}"] = r["equivalent"]
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(DATA)
    subjects, labels, tracts, mats = build_matrices(df)
    biomarkers = [(t, m) for t in tracts for m in METRICS]

    # ---- point estimates per biomarker ----
    g_pipe = {bm: g_per_pipeline(mats[bm], labels) for bm in biomarkers}
    auc_pipe = {bm: auc_per_pipeline(mats[bm], labels) for bm in biomarkers}

    # ---- effect-size aggregate (margins 0.2 and 0.1; 90% CI at strict 0.1) ----
    dg = {lbl: np.array([g_pipe[bm][j] - g_pipe[bm][i] for bm in biomarkers]) for lbl, j, i in PAIRS}
    es_agg = aggregate_rows(dg, G_MARGINS, margin_main=0.1)
    es_agg.to_csv(OUTDIR / "tost_effect_size_aggregate.csv", index=False)

    # ---- effect-size by metric (pool 3 pairs x 10 biomarkers; margin 0.1) ----
    rows = []
    for metric in METRICS:
        vals = np.concatenate([[g_pipe[(t, metric)][j] - g_pipe[(t, metric)][i] for t in tracts]
                               for _, j, i in PAIRS])
        r = tost_one_sample(vals, 0.1)
        rows.append({"metric": metric, "n": len(vals), "mean": r["mean"],
                     "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                     "p_tost_0.1": r["p_tost"], "equivalent_0.1": r["equivalent"]})
    pd.DataFrame(rows).to_csv(OUTDIR / "tost_effect_size_by_metric.csv", index=False)

    # ---- AUC aggregate (margin 0.05) ----
    dauc = {lbl: np.array([auc_pipe[bm][j] - auc_pipe[bm][i] for bm in biomarkers]) for lbl, j, i in PAIRS}
    auc_agg = aggregate_rows(dauc, [AUC_MARGIN], margin_main=AUC_MARGIN)
    auc_agg.to_csv(OUTDIR / "tost_auc_aggregate.csv", index=False)

    # ---- biomarker-level within-subject bootstrap (under-powered) ----
    rng = np.random.default_rng(RANDOM_STATE)
    idx_ad = np.where(labels == 1)[0]
    idx_cn = np.where(labels == 0)[0]
    brows = []
    for (t, m) in biomarkers:
        Y = mats[(t, m)]
        boot = {lbl: np.empty(N_BOOT) for lbl, _, _ in PAIRS}
        for b in range(N_BOOT):
            ra = rng.choice(idx_ad, size=len(idx_ad), replace=True)
            rc = rng.choice(idx_cn, size=len(idx_cn), replace=True)
            gp = np.array([hedges_g(Y[ra, p], Y[rc, p]) for p in range(Y.shape[1])])
            for lbl, j, i in PAIRS:
                boot[lbl][b] = gp[j] - gp[i]
        for lbl, j, i in PAIRS:
            lo, hi = np.percentile(boot[lbl], [5, 95])
            brows.append({"tract": t, "metric": m, "pair": lbl,
                          "delta_g": g_pipe[(t, m)][j] - g_pipe[(t, m)][i],
                          "boot_ci_lo": lo, "boot_ci_hi": hi, "half_width": (hi - lo) / 2,
                          "equivalent_0.2": bool(lo > -0.2 and hi < 0.2)})
    bdf = pd.DataFrame(brows)
    bdf.to_csv(OUTDIR / "tost_biomarker_level.csv", index=False)

    # ---- report ----
    L = []
    L.append("TOST EQUIVALENCE REPORT")
    L.append("=" * 80)
    L.append("Margins: Hedges-g delta = 0.2 (pre-registered) and 0.1 (strict);")
    L.append("AUC delta = 0.05. Aggregate level = biomarker as sampling unit,")
    L.append("90% CI (alpha=0.05 per one-sided test). Equivalent if 90% CI is")
    L.append("fully inside the margin.")
    L.append("")
    L.append("EFFECT-SIZE EQUIVALENCE (aggregate)")
    L.append("-" * 80)
    for _, r in es_agg.iterrows():
        L.append(f"  {r['scope']:7s}: mean dg={r['mean']:+.3f}  90%CI[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]  "
                 f"equiv@0.2={r['equivalent_0.2']}  equiv@0.1={r['equivalent_0.1']}  "
                 f"(p_TOST@0.1={r['p_tost_0.1']:.2e})")
    L.append("")
    L.append("DIAGNOSTIC (AUC) EQUIVALENCE (aggregate, margin 0.05)")
    L.append("-" * 80)
    for _, r in auc_agg.iterrows():
        L.append(f"  {r['scope']:7s}: mean dAUC={r['mean']:+.4f}  90%CI[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]  "
                 f"equiv@0.05={r['equivalent_0.05']}  (p_TOST={r['p_tost_0.05']:.2e})")
    L.append("")
    n_equiv = int(bdf["equivalent_0.2"].sum())
    L.append("BIOMARKER LEVEL (within-subject bootstrap, margin 0.2)")
    L.append("-" * 80)
    L.append(f"  Individually equivalent pair-tests: {n_equiv}/{len(bdf)} "
             f"(median 90% CI half-width = {bdf['half_width'].median():.3f}).")
    L.append("  The biomarker-level test is under-powered (half-width ~ margin);")
    L.append("  the aggregate test is the powered, primary equivalence result.")
    (OUTDIR / "tost_report.txt").write_text("\n".join(L), encoding="utf-8")

    print("[09] TOST equivalence written to", OUTDIR)
    print(es_agg[["scope", "mean", "ci_lo", "ci_hi", "equivalent_0.1"]].round(3).to_string(index=False))
    print(auc_agg[["scope", "mean", "ci_lo", "ci_hi", "equivalent_0.05"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
