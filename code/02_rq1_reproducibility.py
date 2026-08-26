# -*- coding: utf-8 -*-
"""
02_rq1_reproducibility.py

RQ1:
    Mennyire reprodukálhatók a tractometriai biomarkerek különböző
    preprocessing pipeline-ok között?

Új elemzés, a korábbi statisztikai pipeline-tól függetlenül.

Bemenet:
    data/all_pipelines_tractometry_features_master.csv

Kimenetek:
    outputs/rq1_reproducibility/
        icc_all_features.csv
        cv_all_features.csv
        pairwise_correlations_all_features.csv
        bland_altman_all_features.csv
        feature_reproducibility_summary.csv
        tract_reproducibility_summary.csv
        metric_reproducibility_summary.csv
        rq1_report.txt

Módszertani döntés:
    ICC(3,1) jellegű two-way mixed, consistency, single-measure ICC kerül számításra,
    mivel ugyanazon alanyokat ugyanazon fix pipeline-okkal hasonlítjuk össze.
"""

from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------
# 1. Útvonalak
# ---------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_DIR / "data" / "all_pipelines_tractometry_features_master.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "rq1_reproducibility"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# 2. Beolvasás
# ---------------------------------------------------------------------

df = pd.read_csv(INPUT_CSV)

FEATURES = ["mean", "median", "sd", "proximal_mean", "middle_mean", "distal_mean"]

required_cols = ["subject", "group", "pipeline", "tract", "metric"] + FEATURES
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Hiányzó oszlopok: {missing_cols}")


# ---------------------------------------------------------------------
# 3. ICC(3,1) függvény
# ---------------------------------------------------------------------

def icc_3_1(data):
    """
    ICC(3,1) számítása:
        two-way mixed effects
        consistency
        single measurement

    Bemenet:
        data: pandas DataFrame
              sorok = alanyok
              oszlopok = pipeline-ok

    Képlet:
        ICC(3,1) = (MS_subject - MS_error) /
                   (MS_subject + (k - 1) * MS_error)

    ahol:
        k = raterek / pipeline-ok száma
    """

    data = data.dropna(axis=0)

    n, k = data.shape

    if n < 2 or k < 2:
        return np.nan

    values = data.to_numpy(dtype=float)

    grand_mean = np.mean(values)
    subject_means = np.mean(values, axis=1)
    pipeline_means = np.mean(values, axis=0)

    ss_subject = k * np.sum((subject_means - grand_mean) ** 2)
    ss_pipeline = n * np.sum((pipeline_means - grand_mean) ** 2)
    ss_total = np.sum((values - grand_mean) ** 2)
    ss_error = ss_total - ss_subject - ss_pipeline

    df_subject = n - 1
    df_error = (n - 1) * (k - 1)

    if df_subject <= 0 or df_error <= 0:
        return np.nan

    ms_subject = ss_subject / df_subject
    ms_error = ss_error / df_error

    denominator = ms_subject + (k - 1) * ms_error

    if denominator == 0:
        return np.nan

    return (ms_subject - ms_error) / denominator


def icc_category(icc):
    if pd.isna(icc):
        return "undefined"
    if icc < 0.50:
        return "poor"
    elif icc < 0.75:
        return "moderate"
    elif icc < 0.90:
        return "good"
    else:
        return "excellent"


# ---------------------------------------------------------------------
# 4. ICC és CV számítás
# ---------------------------------------------------------------------

icc_rows = []
cv_rows = []

for tract, metric, feature in itertools.product(
    sorted(df["tract"].unique()),
    sorted(df["metric"].unique()),
    FEATURES
):
    sub = df[(df["tract"] == tract) & (df["metric"] == metric)]

    wide = sub.pivot_table(
        index="subject",
        columns="pipeline",
        values=feature
    )

    # ICC
    icc_value = icc_3_1(wide)

    icc_rows.append({
        "tract": tract,
        "metric": metric,
        "feature": feature,
        "n_subjects": wide.shape[0],
        "n_pipelines": wide.shape[1],
        "icc_3_1": icc_value,
        "icc_category": icc_category(icc_value)
    })

    # CV
    row_means = wide.mean(axis=1)
    row_sds = wide.std(axis=1, ddof=1)
    cvs = row_sds / row_means.abs().replace(0, np.nan) * 100

    cv_rows.append({
        "tract": tract,
        "metric": metric,
        "feature": feature,
        "n_subjects": wide.shape[0],
        "mean_cv_percent": cvs.mean(),
        "median_cv_percent": cvs.median(),
        "max_cv_percent": cvs.max()
    })

icc_df = pd.DataFrame(icc_rows)
cv_df = pd.DataFrame(cv_rows)

icc_df.to_csv(OUTPUT_DIR / "icc_all_features.csv", index=False)
cv_df.to_csv(OUTPUT_DIR / "cv_all_features.csv", index=False)


# ---------------------------------------------------------------------
# 5. Pairwise pipeline korrelációk
# ---------------------------------------------------------------------

corr_rows = []

pipeline_pairs = list(itertools.combinations(sorted(df["pipeline"].unique()), 2))

for tract, metric, feature in itertools.product(
    sorted(df["tract"].unique()),
    sorted(df["metric"].unique()),
    FEATURES
):
    sub = df[(df["tract"] == tract) & (df["metric"] == metric)]

    wide = sub.pivot_table(
        index="subject",
        columns="pipeline",
        values=feature
    )

    for p1, p2 in pipeline_pairs:
        x = wide[p1]
        y = wide[p2]

        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)

        corr_rows.append({
            "tract": tract,
            "metric": metric,
            "feature": feature,
            "pipeline_1": p1,
            "pipeline_2": p2,
            "n_subjects": len(wide),
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p
        })

corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv(OUTPUT_DIR / "pairwise_correlations_all_features.csv", index=False)


# ---------------------------------------------------------------------
# 6. Bland–Altman összefoglalók
# ---------------------------------------------------------------------

ba_rows = []

for tract, metric, feature in itertools.product(
    sorted(df["tract"].unique()),
    sorted(df["metric"].unique()),
    FEATURES
):
    sub = df[(df["tract"] == tract) & (df["metric"] == metric)]

    wide = sub.pivot_table(
        index="subject",
        columns="pipeline",
        values=feature
    )

    for p1, p2 in pipeline_pairs:
        diff = wide[p2] - wide[p1]
        pair_mean = (wide[p1] + wide[p2]) / 2

        bias = diff.mean()
        sd_diff = diff.std(ddof=1)
        loa_lower = bias - 1.96 * sd_diff
        loa_upper = bias + 1.96 * sd_diff
        mean_abs_diff = diff.abs().mean()

        ba_rows.append({
            "tract": tract,
            "metric": metric,
            "feature": feature,
            "pipeline_1": p1,
            "pipeline_2": p2,
            "n_subjects": len(wide),
            "bias_pipeline2_minus_pipeline1": bias,
            "sd_difference": sd_diff,
            "loa_lower": loa_lower,
            "loa_upper": loa_upper,
            "mean_absolute_difference": mean_abs_diff,
            "mean_pair_value": pair_mean.mean()
        })

ba_df = pd.DataFrame(ba_rows)
ba_df.to_csv(OUTPUT_DIR / "bland_altman_all_features.csv", index=False)


# ---------------------------------------------------------------------
# 7. Összesítő táblák
# ---------------------------------------------------------------------

merged = icc_df.merge(cv_df, on=["tract", "metric", "feature", "n_subjects"], how="left")

feature_summary = (
    merged.groupby("feature")
    .agg(
        mean_icc=("icc_3_1", "mean"),
        median_icc=("icc_3_1", "median"),
        min_icc=("icc_3_1", "min"),
        max_icc=("icc_3_1", "max"),
        mean_cv_percent=("mean_cv_percent", "mean"),
        median_cv_percent=("median_cv_percent", "median"),
        n_excellent_icc=("icc_category", lambda x: int((x == "excellent").sum())),
        n_good_or_better_icc=("icc_category", lambda x: int(((x == "good") | (x == "excellent")).sum())),
        n_total=("icc_3_1", "count")
    )
    .reset_index()
)

feature_summary["percent_good_or_better_icc"] = (
    feature_summary["n_good_or_better_icc"] / feature_summary["n_total"] * 100
)

feature_summary = feature_summary.sort_values(["mean_icc", "mean_cv_percent"], ascending=[False, True])
feature_summary.to_csv(OUTPUT_DIR / "feature_reproducibility_summary.csv", index=False)

tract_summary = (
    merged.groupby("tract")
    .agg(
        mean_icc=("icc_3_1", "mean"),
        median_icc=("icc_3_1", "median"),
        mean_cv_percent=("mean_cv_percent", "mean"),
        median_cv_percent=("median_cv_percent", "median"),
        n_total=("icc_3_1", "count")
    )
    .reset_index()
    .sort_values("mean_icc", ascending=False)
)
tract_summary.to_csv(OUTPUT_DIR / "tract_reproducibility_summary.csv", index=False)

metric_summary = (
    merged.groupby("metric")
    .agg(
        mean_icc=("icc_3_1", "mean"),
        median_icc=("icc_3_1", "median"),
        mean_cv_percent=("mean_cv_percent", "mean"),
        median_cv_percent=("median_cv_percent", "median"),
        n_total=("icc_3_1", "count")
    )
    .reset_index()
    .sort_values("mean_icc", ascending=False)
)
metric_summary.to_csv(OUTPUT_DIR / "metric_reproducibility_summary.csv", index=False)


# ---------------------------------------------------------------------
# 8. RQ1 riport
# ---------------------------------------------------------------------

lines = []

lines.append("RQ1 REPRODUCIBILITY REPORT")
lines.append("=" * 80)
lines.append("")
lines.append("Kutatási kérdés:")
lines.append("Mennyire reprodukálhatók a tractometriai biomarkerek különböző preprocessing pipeline-ok között?")
lines.append("")
lines.append("Módszerek:")
lines.append("- ICC(3,1): two-way mixed, consistency, single measurement")
lines.append("- CV%: alanyon belüli variabilitás a három pipeline között")
lines.append("- Pairwise Pearson/Spearman korrelációk")
lines.append("- Bland–Altman bias és limits of agreement")
lines.append("")

lines.append("FEATURE-LEVEL REPRODUCIBILITY")
lines.append("-" * 80)

for _, row in feature_summary.iterrows():
    lines.append(
        f"{row['feature']}: "
        f"mean ICC={row['mean_icc']:.3f}, "
        f"median ICC={row['median_icc']:.3f}, "
        f"mean CV={row['mean_cv_percent']:.2f}%, "
        f"good/excellent ICC={row['percent_good_or_better_icc']:.1f}% "
        f"({int(row['n_good_or_better_icc'])}/{int(row['n_total'])})"
    )

lines.append("")
lines.append("TRACT-LEVEL REPRODUCIBILITY")
lines.append("-" * 80)

for _, row in tract_summary.iterrows():
    lines.append(
        f"{row['tract']}: "
        f"mean ICC={row['mean_icc']:.3f}, "
        f"median ICC={row['median_icc']:.3f}, "
        f"mean CV={row['mean_cv_percent']:.2f}%"
    )

lines.append("")
lines.append("METRIC-LEVEL REPRODUCIBILITY")
lines.append("-" * 80)

for _, row in metric_summary.iterrows():
    lines.append(
        f"{row['metric']}: "
        f"mean ICC={row['mean_icc']:.3f}, "
        f"median ICC={row['median_icc']:.3f}, "
        f"mean CV={row['mean_cv_percent']:.2f}%"
    )

lines.append("")
lines.append("INTERPRETÁCIÓS MEGJEGYZÉS")
lines.append("-" * 80)
lines.append(
    "Az RQ1 elemzés célja annak meghatározása, hogy a pipeline választása mellett "
    "a biomarkerek mennyire stabilak. A magas ICC azt jelzi, hogy a pipeline-ok "
    "hasonlóan rangsorolják az alanyokat, míg az alacsony CV kis abszolút pipeline-közti "
    "eltérést jelez."
)
lines.append("")
lines.append(
    "Az ICC kategóriák értelmezése: <0.50 poor, 0.50–0.75 moderate, "
    "0.75–0.90 good, >0.90 excellent."
)
lines.append("")
lines.append("Kimeneti fájlok:")
for file in sorted(OUTPUT_DIR.glob("*")):
    lines.append(f"- {file.name}")

with open(OUTPUT_DIR / "rq1_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\n".join(lines))
