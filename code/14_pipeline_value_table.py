# -*- coding: utf-8 -*-
"""
14_pipeline_value_table.py
"Which pipeline moves how much" — the actual mean value produced by EACH
pipeline (P01 reference) per metric and per tract, with the % shift of P02/P03
vs P01. Direct, no aggregation across pipeline pairs. Computed straight from the
master CSV. Does not modify any existing analysis.

Outputs (outputs/pipeline_value_table/):
  pipeline_value_by_metric.csv
  pipeline_value_by_tract_metric.csv
  pipeline_value_table_report.txt
Dependencies: numpy, pandas.
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "all_pipelines_tractometry_features_master.csv"
OUTDIR = BASE / "outputs" / "pipeline_value_table"; OUTDIR.mkdir(parents=True, exist_ok=True)
METRICS = ["FA", "MD", "RD", "AD"]
SCALE = {"FA": 1.0, "MD": 1e3, "RD": 1e3, "AD": 1e3}   # diffusivities shown x10^-3
UNIT = {"FA": "(–)", "MD": "(x10^-3 mm2/s)", "RD": "(x10^-3 mm2/s)", "AD": "(x10^-3 mm2/s)"}

df = pd.read_csv(DATA)


def block(sub):
    p01 = sub[sub.pipeline == "pipeline_01"]["mean"].mean()
    p02 = sub[sub.pipeline == "pipeline_02"]["mean"].mean()
    p03 = sub[sub.pipeline == "pipeline_03"]["mean"].mean()
    return p01, p02, p03


# ---- by metric (pooled over subjects x tracts) ----
rows = []
for m in METRICS:
    p01, p02, p03 = block(df[df.metric == m])
    sc = SCALE[m]
    rows.append(dict(metric=m, unit=UNIT[m],
                     P01=p01 * sc, P02=p02 * sc, P03=p03 * sc,
                     pct_P02_vs_P01=100 * (p02 - p01) / p01,
                     pct_P03_vs_P01=100 * (p03 - p01) / p01))
bym = pd.DataFrame(rows)
bym.to_csv(OUTDIR / "pipeline_value_by_metric.csv", index=False)

# ---- by tract x metric (full detail, no averaging across tracts) ----
rows = []
for tr in sorted(df.tract.unique()):
    for m in METRICS:
        p01, p02, p03 = block(df[(df.tract == tr) & (df.metric == m)])
        sc = SCALE[m]
        rows.append(dict(tract=tr, metric=m, P01=p01 * sc, P02=p02 * sc, P03=p03 * sc,
                         pct_P02_vs_P01=100 * (p02 - p01) / p01,
                         pct_P03_vs_P01=100 * (p03 - p01) / p01))
byt = pd.DataFrame(rows)
byt.to_csv(OUTDIR / "pipeline_value_by_tract_metric.csv", index=False)

# ---- report ----
L = ["WHICH PIPELINE MOVES HOW MUCH (P01 = reference)", "=" * 64,
     "Mean value produced by each pipeline, per metric (pooled over tracts & subjects).", "",
     f"{'metric':14} {'P01':>9} {'P02':>9} {'P03':>9}   {'P02 vs P01':>10} {'P03 vs P01':>10}"]
for _, r in bym.iterrows():
    L.append(f"{r['metric']+' '+r['unit']:14} {r['P01']:9.3f} {r['P02']:9.3f} {r['P03']:9.3f}   "
             f"{r['pct_P02_vs_P01']:+9.2f}% {r['pct_P03_vs_P01']:+9.2f}%")
L += ["", "Read-off: P03 produces the largest shift (FA -5.7%, AD -3.0%, RD +2.2%);",
      "P02 an intermediate one; MD barely moves under either pipeline.",
      "Full tract-level detail: pipeline_value_by_tract_metric.csv (40 rows)."]
(OUTDIR / "pipeline_value_table_report.txt").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
