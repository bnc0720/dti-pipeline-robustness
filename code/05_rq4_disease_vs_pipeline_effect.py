# -*- coding: utf-8 -*-
"""
05_rq4_disease_vs_pipeline_effect.py

RQ4:
    Mekkora az Alzheimer-kórhoz kapcsolódó biológiai hatás a preprocessing
    pipeline által okozott technikai eltéréshez képest?

Új elemzés, a korábbi statisztikai pipeline-tól függetlenül.

Bemenet:
    data/all_pipelines_tractometry_features_master.csv

Kimenetek:
    outputs/rq4_disease_vs_pipeline_effect/
        disease_pipeline_ratio_all_features.csv
        disease_pipeline_ratio_primary_mean.csv
        rq4_feature_summary.csv
        rq4_metric_summary_primary_mean.csv
        rq4_tract_summary_primary_mean.csv
        rq4_top_biomarkers_primary_mean.csv
        rq4_report.txt

Fő mutató:
    Disease-to-Pipeline Ratio = |AD-CN különbség| / |legnagyobb pipeline-közti eltérés|

Értelmezés:
    Ratio > 1:
        a betegséghez kapcsolódó különbség nagyobb, mint a pipeline által okozott eltérés.
    Ratio > 2:
        a betegségjel legalább kétszer akkora, mint a technikai pipeline-hatás.
    Ratio > 3:
        erős biológiai dominancia.
"""

from pathlib import Path
import itertools
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# 1. Útvonalak
# ---------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_DIR / "data" / "all_pipelines_tractometry_features_master.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "rq4_disease_vs_pipeline_effect"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# 2. Beolvasás
# ---------------------------------------------------------------------

df = pd.read_csv(INPUT_CSV)

FEATURES = ["mean", "median", "sd", "proximal_mean", "middle_mean", "distal_mean"]
PRIMARY_FEATURE = "mean"

required_cols = ["subject", "group", "pipeline", "tract", "metric"] + FEATURES
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Hiányzó oszlopok: {missing_cols}")

long_df = df.melt(
    id_vars=["subject", "group", "pipeline", "tract", "metric"],
    value_vars=FEATURES,
    var_name="feature",
    value_name="value"
)


# ---------------------------------------------------------------------
# 3. Segédfüggvények
# ---------------------------------------------------------------------

def hedges_g(x_ad, x_cn):
    x_ad = np.asarray(x_ad, dtype=float)
    x_cn = np.asarray(x_cn, dtype=float)

    n1 = len(x_ad)
    n0 = len(x_cn)

    if n1 < 2 or n0 < 2:
        return np.nan

    sd1 = np.std(x_ad, ddof=1)
    sd0 = np.std(x_cn, ddof=1)

    pooled_var = ((n1 - 1) * sd1**2 + (n0 - 1) * sd0**2) / (n1 + n0 - 2)

    if pooled_var <= 0:
        return np.nan

    d = (np.mean(x_ad) - np.mean(x_cn)) / np.sqrt(pooled_var)
    correction = 1 - (3 / (4 * (n1 + n0) - 9))
    return d * correction


def classify_ratio(ratio):
    if pd.isna(ratio):
        return "undefined"
    if ratio < 1:
        return "pipeline_effect_larger_or_equal"
    elif ratio < 2:
        return "disease_effect_slightly_larger"
    elif ratio < 3:
        return "disease_effect_moderately_larger"
    else:
        return "disease_effect_strongly_larger"


# ---------------------------------------------------------------------
# 4. Disease-to-pipeline ratio számítás
# ---------------------------------------------------------------------

rows = []

for tract, metric, feature in itertools.product(
    sorted(long_df["tract"].unique()),
    sorted(long_df["metric"].unique()),
    FEATURES
):
    sub = long_df[
        (long_df["tract"] == tract) &
        (long_df["metric"] == metric) &
        (long_df["feature"] == feature)
    ].copy()

    # Pipeline-onkénti AD-CN különbségek
    disease_diffs = []
    hedges_values = []

    for pipeline in sorted(sub["pipeline"].unique()):
        sp = sub[sub["pipeline"] == pipeline]
        ad = sp.loc[sp["group"] == "AD", "value"].dropna()
        cn = sp.loc[sp["group"] == "CN", "value"].dropna()

        diff = ad.mean() - cn.mean()
        g = hedges_g(ad, cn)

        disease_diffs.append(diff)
        hedges_values.append(g)

    mean_disease_difference = np.nanmean(disease_diffs)
    mean_abs_disease_difference = np.nanmean(np.abs(disease_diffs))
    max_abs_disease_difference = np.nanmax(np.abs(disease_diffs))
    mean_abs_g = np.nanmean(np.abs(hedges_values))

    # Pipeline effect: subject szinten pipeline-ok közötti range
    wide = sub.pivot_table(index="subject", columns="pipeline", values="value")

    subject_pipeline_ranges = wide.max(axis=1) - wide.min(axis=1)
    mean_abs_pipeline_difference = subject_pipeline_ranges.mean()
    median_abs_pipeline_difference = subject_pipeline_ranges.median()
    max_abs_pipeline_difference = subject_pipeline_ranges.max()

    # Alternatív pipeline effect: pipeline group átlagok közötti range
    pipeline_means = sub.groupby("pipeline")["value"].mean()
    pipeline_mean_range = pipeline_means.max() - pipeline_means.min()

    # Fő ratio: átlagos abszolút disease különbség / átlagos subject-szintű pipeline range
    ratio_subject_level = (
        mean_abs_disease_difference / mean_abs_pipeline_difference
        if mean_abs_pipeline_difference != 0 else np.nan
    )

    # Másodlagos ratio: disease különbség / pipeline group mean range
    ratio_group_mean_level = (
        mean_abs_disease_difference / pipeline_mean_range
        if pipeline_mean_range != 0 else np.nan
    )

    rows.append({
        "tract": tract,
        "metric": metric,
        "feature": feature,
        "mean_disease_difference_AD_minus_CN": mean_disease_difference,
        "mean_abs_disease_difference": mean_abs_disease_difference,
        "max_abs_disease_difference": max_abs_disease_difference,
        "mean_abs_hedges_g": mean_abs_g,
        "mean_abs_pipeline_difference_subject_level": mean_abs_pipeline_difference,
        "median_abs_pipeline_difference_subject_level": median_abs_pipeline_difference,
        "max_abs_pipeline_difference_subject_level": max_abs_pipeline_difference,
        "pipeline_mean_range": pipeline_mean_range,
        "disease_to_pipeline_ratio_subject_level": ratio_subject_level,
        "disease_to_pipeline_ratio_group_mean_level": ratio_group_mean_level,
        "ratio_category": classify_ratio(ratio_subject_level),
    })

ratio_df = pd.DataFrame(rows)
ratio_df.to_csv(OUTPUT_DIR / "disease_pipeline_ratio_all_features.csv", index=False)

primary_df = ratio_df[ratio_df["feature"] == PRIMARY_FEATURE].copy()
primary_df = primary_df.sort_values(
    ["disease_to_pipeline_ratio_subject_level", "mean_abs_hedges_g"],
    ascending=[False, False]
)
primary_df.to_csv(OUTPUT_DIR / "disease_pipeline_ratio_primary_mean.csv", index=False)


# ---------------------------------------------------------------------
# 5. Összesítő táblák
# ---------------------------------------------------------------------

feature_summary = (
    ratio_df.groupby("feature")
    .agg(
        n_biomarkers=("disease_to_pipeline_ratio_subject_level", "count"),
        mean_ratio=("disease_to_pipeline_ratio_subject_level", "mean"),
        median_ratio=("disease_to_pipeline_ratio_subject_level", "median"),
        max_ratio=("disease_to_pipeline_ratio_subject_level", "max"),
        mean_abs_g=("mean_abs_hedges_g", "mean"),
        mean_abs_disease_difference=("mean_abs_disease_difference", "mean"),
        mean_abs_pipeline_difference=("mean_abs_pipeline_difference_subject_level", "mean"),
        n_ratio_gt_1=("disease_to_pipeline_ratio_subject_level", lambda x: int((x > 1).sum())),
        n_ratio_gt_2=("disease_to_pipeline_ratio_subject_level", lambda x: int((x > 2).sum())),
        n_ratio_gt_3=("disease_to_pipeline_ratio_subject_level", lambda x: int((x > 3).sum())),
    )
    .reset_index()
)

feature_summary["percent_ratio_gt_1"] = feature_summary["n_ratio_gt_1"] / feature_summary["n_biomarkers"] * 100
feature_summary["percent_ratio_gt_2"] = feature_summary["n_ratio_gt_2"] / feature_summary["n_biomarkers"] * 100
feature_summary["percent_ratio_gt_3"] = feature_summary["n_ratio_gt_3"] / feature_summary["n_biomarkers"] * 100
feature_summary = feature_summary.sort_values("median_ratio", ascending=False)
feature_summary.to_csv(OUTPUT_DIR / "rq4_feature_summary.csv", index=False)

metric_summary = (
    primary_df.groupby("metric")
    .agg(
        n_biomarkers=("disease_to_pipeline_ratio_subject_level", "count"),
        mean_ratio=("disease_to_pipeline_ratio_subject_level", "mean"),
        median_ratio=("disease_to_pipeline_ratio_subject_level", "median"),
        max_ratio=("disease_to_pipeline_ratio_subject_level", "max"),
        mean_abs_g=("mean_abs_hedges_g", "mean"),
        n_ratio_gt_1=("disease_to_pipeline_ratio_subject_level", lambda x: int((x > 1).sum())),
        n_ratio_gt_2=("disease_to_pipeline_ratio_subject_level", lambda x: int((x > 2).sum())),
    )
    .reset_index()
)
metric_summary["percent_ratio_gt_1"] = metric_summary["n_ratio_gt_1"] / metric_summary["n_biomarkers"] * 100
metric_summary["percent_ratio_gt_2"] = metric_summary["n_ratio_gt_2"] / metric_summary["n_biomarkers"] * 100
metric_summary = metric_summary.sort_values("median_ratio", ascending=False)
metric_summary.to_csv(OUTPUT_DIR / "rq4_metric_summary_primary_mean.csv", index=False)

tract_summary = (
    primary_df.groupby("tract")
    .agg(
        n_biomarkers=("disease_to_pipeline_ratio_subject_level", "count"),
        mean_ratio=("disease_to_pipeline_ratio_subject_level", "mean"),
        median_ratio=("disease_to_pipeline_ratio_subject_level", "median"),
        max_ratio=("disease_to_pipeline_ratio_subject_level", "max"),
        mean_abs_g=("mean_abs_hedges_g", "mean"),
        n_ratio_gt_1=("disease_to_pipeline_ratio_subject_level", lambda x: int((x > 1).sum())),
        n_ratio_gt_2=("disease_to_pipeline_ratio_subject_level", lambda x: int((x > 2).sum())),
    )
    .reset_index()
)
tract_summary["percent_ratio_gt_1"] = tract_summary["n_ratio_gt_1"] / tract_summary["n_biomarkers"] * 100
tract_summary["percent_ratio_gt_2"] = tract_summary["n_ratio_gt_2"] / tract_summary["n_biomarkers"] * 100
tract_summary = tract_summary.sort_values("median_ratio", ascending=False)
tract_summary.to_csv(OUTPUT_DIR / "rq4_tract_summary_primary_mean.csv", index=False)

top_biomarkers = primary_df.sort_values(
    ["disease_to_pipeline_ratio_subject_level", "mean_abs_hedges_g"],
    ascending=[False, False]
).head(20)

top_biomarkers.to_csv(OUTPUT_DIR / "rq4_top_biomarkers_primary_mean.csv", index=False)


# ---------------------------------------------------------------------
# 6. Szöveges riport
# ---------------------------------------------------------------------

primary_mean_ratio = primary_df["disease_to_pipeline_ratio_subject_level"].mean()
primary_median_ratio = primary_df["disease_to_pipeline_ratio_subject_level"].median()
primary_gt1 = (primary_df["disease_to_pipeline_ratio_subject_level"] > 1).mean() * 100
primary_gt2 = (primary_df["disease_to_pipeline_ratio_subject_level"] > 2).mean() * 100
primary_gt3 = (primary_df["disease_to_pipeline_ratio_subject_level"] > 3).mean() * 100

lines = []

lines.append("RQ4 DISEASE VS PIPELINE EFFECT REPORT")
lines.append("=" * 80)
lines.append("")
lines.append("Kutatási kérdés:")
lines.append("Mekkora az Alzheimer-kórhoz kapcsolódó biológiai hatás a preprocessing pipeline által okozott technikai eltéréshez képest?")
lines.append("")
lines.append("Fő mutató:")
lines.append("Disease-to-Pipeline Ratio = |AD-CN különbség| / |pipeline-ok közötti subject-szintű eltérés|")
lines.append("")
lines.append("Értelmezés:")
lines.append("Ratio > 1: a betegségjel nagyobb, mint a pipeline-hatás")
lines.append("Ratio > 2: a betegségjel legalább kétszer akkora")
lines.append("Ratio > 3: erős biológiai dominancia")
lines.append("")

lines.append("PRIMARY FEATURE: mean")
lines.append("-" * 80)
lines.append(f"Number of biomarkers: {primary_df.shape[0]}")
lines.append(f"Mean disease-to-pipeline ratio: {primary_mean_ratio:.3f}")
lines.append(f"Median disease-to-pipeline ratio: {primary_median_ratio:.3f}")
lines.append(f"Biomarkers with ratio > 1: {primary_gt1:.1f}%")
lines.append(f"Biomarkers with ratio > 2: {primary_gt2:.1f}%")
lines.append(f"Biomarkers with ratio > 3: {primary_gt3:.1f}")
lines.append("")

lines.append("FEATURE-LEVEL SUMMARY")
lines.append("-" * 80)
for _, row in feature_summary.iterrows():
    lines.append(
        f"{row['feature']}: "
        f"median ratio={row['median_ratio']:.3f}, "
        f"mean ratio={row['mean_ratio']:.3f}, "
        f"ratio>1={row['percent_ratio_gt_1']:.1f}%, "
        f"ratio>2={row['percent_ratio_gt_2']:.1f}%, "
        f"mean_abs_g={row['mean_abs_g']:.3f}"
    )

lines.append("")
lines.append("METRIC-LEVEL SUMMARY, PRIMARY MEAN")
lines.append("-" * 80)
for _, row in metric_summary.iterrows():
    lines.append(
        f"{row['metric']}: "
        f"median ratio={row['median_ratio']:.3f}, "
        f"mean ratio={row['mean_ratio']:.3f}, "
        f"ratio>1={row['percent_ratio_gt_1']:.1f}%, "
        f"mean_abs_g={row['mean_abs_g']:.3f}"
    )

lines.append("")
lines.append("TRACT-LEVEL SUMMARY, PRIMARY MEAN")
lines.append("-" * 80)
for _, row in tract_summary.iterrows():
    lines.append(
        f"{row['tract']}: "
        f"median ratio={row['median_ratio']:.3f}, "
        f"mean ratio={row['mean_ratio']:.3f}, "
        f"ratio>1={row['percent_ratio_gt_1']:.1f}%, "
        f"mean_abs_g={row['mean_abs_g']:.3f}"
    )

lines.append("")
lines.append("TOP BIOMARKERS, PRIMARY MEAN")
lines.append("-" * 80)
for _, row in top_biomarkers.iterrows():
    lines.append(
        f"{row['tract']} {row['metric']}: "
        f"ratio={row['disease_to_pipeline_ratio_subject_level']:.3f}, "
        f"mean_abs_g={row['mean_abs_hedges_g']:.3f}, "
        f"disease_diff={row['mean_abs_disease_difference']:.6g}, "
        f"pipeline_diff={row['mean_abs_pipeline_difference_subject_level']:.6g}"
    )

lines.append("")
lines.append("INTERPRETÁCIÓS MEGJEGYZÉS")
lines.append("-" * 80)
lines.append(
    "Ez az elemzés nem azt vizsgálja, hogy a pipeline-hatás szignifikáns-e, hanem azt, "
    "hogy az AD-CN biológiai különbség nagysága hogyan viszonyul a pipeline-választásból "
    "származó technikai eltéréshez. Ez közvetlenül kiegészíti az RQ2 és RQ3 eredményeit."
)
lines.append("")
lines.append(
    "A magas ratio azt jelzi, hogy az adott biomarker esetén a betegséghez kapcsolódó jel "
    "nagyobb, mint a preprocessing pipeline-ok közötti eltérés. Alacsony ratio esetén "
    "az adott biomarker biológiai interpretációja érzékenyebb lehet a pipeline-választásra."
)
lines.append("")
lines.append("Kimeneti fájlok:")
for file in sorted(OUTPUT_DIR.glob("*")):
    lines.append(f"- {file.name}")

with open(OUTPUT_DIR / "rq4_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\n".join(lines))
