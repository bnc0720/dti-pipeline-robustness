# -*- coding: utf-8 -*-
"""
01_feature_screening.py

Cél:
    A master tractometry CSV-ben szereplő 6 feature objektív összehasonlítása.

Feature-ek:
    - mean
    - median
    - sd
    - proximal_mean
    - middle_mean
    - distal_mean

Fő kérdés:
    Mely feature-ek hordozzák a legerősebb és legstabilabb Alzheimer-kórhoz kapcsolódó jelet?

Bemenet:
    data/all_pipelines_tractometry_features_master.csv

Kimenetek:
    outputs/feature_screening/
        feature_screening_summary.csv
        feature_level_summary.csv
        effect_sizes_all_features.csv
        auc_all_features.csv
        pipeline_sensitivity_all_features.csv
        reproducibility_cv_all_features.csv
        feature_screening_report.txt

Megjegyzés:
    Ez még nem a fő hipotézisvizsgálat, hanem előszűrő / karakterizáló elemzés.
"""

from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------
# 1. Útvonalak
# ---------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_CSV = PROJECT_DIR / "data" / "all_pipelines_tractometry_features_master.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "feature_screening"
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

df["group_binary"] = df["group"].map({"CN": 0, "AD": 1})

if df["group_binary"].isna().any():
    raise ValueError("A group oszlop csak 'AD' és 'CN' értékeket tartalmazhat.")


# ---------------------------------------------------------------------
# 3. Segédfüggvények
# ---------------------------------------------------------------------

def hedges_g(x_ad, x_cn):
    """
    Hedges g számítása AD - CN iránnyal.
    Pozitív érték: AD > CN
    Negatív érték: AD < CN
    """
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

    pooled_sd = np.sqrt(pooled_var)
    d = (np.mean(x_ad) - np.mean(x_cn)) / pooled_sd

    # small sample correction
    correction = 1 - (3 / (4 * (n1 + n0) - 9))
    return d * correction


def safe_auc(y_true, values):
    """
    AUC számítása úgy, hogy az irányt is kezeljük.
    Visszaad:
        auc_oriented: mindig >= 0.5
        auc_raw: eredeti AUC
        direction: higher_values_AD vagy lower_values_AD
    """
    y_true = np.asarray(y_true)
    values = np.asarray(values, dtype=float)

    if len(np.unique(y_true)) != 2:
        return np.nan, np.nan, "undefined"

    try:
        auc_raw = roc_auc_score(y_true, values)
    except Exception:
        return np.nan, np.nan, "failed"

    if auc_raw >= 0.5:
        return auc_raw, auc_raw, "higher_values_AD"
    else:
        return 1 - auc_raw, auc_raw, "lower_values_AD"


# ---------------------------------------------------------------------
# 4. Effect size minden feature-re
# ---------------------------------------------------------------------

effect_rows = []

for pipeline, tract, metric, feature in itertools.product(
    sorted(df["pipeline"].unique()),
    sorted(df["tract"].unique()),
    sorted(df["metric"].unique()),
    FEATURES
):
    sub = df[
        (df["pipeline"] == pipeline) &
        (df["tract"] == tract) &
        (df["metric"] == metric)
    ]

    ad_values = sub.loc[sub["group"] == "AD", feature].dropna()
    cn_values = sub.loc[sub["group"] == "CN", feature].dropna()

    g = hedges_g(ad_values, cn_values)

    effect_rows.append({
        "pipeline": pipeline,
        "tract": tract,
        "metric": metric,
        "feature": feature,
        "n_AD": len(ad_values),
        "n_CN": len(cn_values),
        "mean_AD": ad_values.mean(),
        "mean_CN": cn_values.mean(),
        "hedges_g_AD_minus_CN": g,
        "abs_g": abs(g) if pd.notna(g) else np.nan,
        "direction": "AD_greater_CN" if pd.notna(g) and g > 0 else "AD_lower_CN" if pd.notna(g) and g < 0 else "undefined"
    })

effect_df = pd.DataFrame(effect_rows)
effect_df.to_csv(OUTPUT_DIR / "effect_sizes_all_features.csv", index=False)


# ---------------------------------------------------------------------
# 5. AUC minden feature-re
# ---------------------------------------------------------------------

auc_rows = []

for pipeline, tract, metric, feature in itertools.product(
    sorted(df["pipeline"].unique()),
    sorted(df["tract"].unique()),
    sorted(df["metric"].unique()),
    FEATURES
):
    sub = df[
        (df["pipeline"] == pipeline) &
        (df["tract"] == tract) &
        (df["metric"] == metric)
    ].copy()

    auc_oriented, auc_raw, direction = safe_auc(sub["group_binary"], sub[feature])

    auc_rows.append({
        "pipeline": pipeline,
        "tract": tract,
        "metric": metric,
        "feature": feature,
        "n": len(sub),
        "auc_oriented": auc_oriented,
        "auc_raw": auc_raw,
        "auc_abs_minus_0_5": abs(auc_oriented - 0.5) if pd.notna(auc_oriented) else np.nan,
        "direction": direction
    })

auc_df = pd.DataFrame(auc_rows)
auc_df.to_csv(OUTPUT_DIR / "auc_all_features.csv", index=False)


# ---------------------------------------------------------------------
# 6. Pipeline sensitivity
# ---------------------------------------------------------------------
# Egyszerű, modellfüggetlen érzékenységi mutató:
# subject × tract × metric szinten a három pipeline közötti range / abs(mean across pipelines)
# Minél kisebb, annál kevésbé pipeline-érzékeny.

pipeline_sensitivity_rows = []

for tract, metric, feature in itertools.product(
    sorted(df["tract"].unique()),
    sorted(df["metric"].unique()),
    FEATURES
):
    sub = df[(df["tract"] == tract) & (df["metric"] == metric)]

    wide = sub.pivot_table(
        index=["subject", "group"],
        columns="pipeline",
        values=feature
    )

    row_ranges = wide.max(axis=1) - wide.min(axis=1)
    row_means = wide.mean(axis=1).abs()

    relative_range = row_ranges / row_means.replace(0, np.nan)

    pipeline_sensitivity_rows.append({
        "tract": tract,
        "metric": metric,
        "feature": feature,
        "n_subjects": wide.shape[0],
        "mean_pipeline_range": row_ranges.mean(),
        "median_pipeline_range": row_ranges.median(),
        "mean_relative_pipeline_range_percent": relative_range.mean() * 100,
        "median_relative_pipeline_range_percent": relative_range.median() * 100,
    })

pipeline_sensitivity_df = pd.DataFrame(pipeline_sensitivity_rows)
pipeline_sensitivity_df.to_csv(OUTPUT_DIR / "pipeline_sensitivity_all_features.csv", index=False)


# ---------------------------------------------------------------------
# 7. Reproducibility CV
# ---------------------------------------------------------------------
# Feature szintű egyszerű CV a három pipeline között subject × tract × metric szinten.

cv_rows = []

for tract, metric, feature in itertools.product(
    sorted(df["tract"].unique()),
    sorted(df["metric"].unique()),
    FEATURES
):
    sub = df[(df["tract"] == tract) & (df["metric"] == metric)]

    wide = sub.pivot_table(
        index=["subject", "group"],
        columns="pipeline",
        values=feature
    )

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

cv_df = pd.DataFrame(cv_rows)
cv_df.to_csv(OUTPUT_DIR / "reproducibility_cv_all_features.csv", index=False)


# ---------------------------------------------------------------------
# 8. Feature-level összesítés
# ---------------------------------------------------------------------

effect_feature_summary = (
    effect_df.groupby("feature")
    .agg(
        mean_abs_g=("abs_g", "mean"),
        median_abs_g=("abs_g", "median"),
        max_abs_g=("abs_g", "max"),
        n_large_effects=("abs_g", lambda x: int((x >= 0.8).sum())),
        n_medium_effects=("abs_g", lambda x: int((x >= 0.5).sum())),
    )
    .reset_index()
)

auc_feature_summary = (
    auc_df.groupby("feature")
    .agg(
        mean_auc=("auc_oriented", "mean"),
        median_auc=("auc_oriented", "median"),
        max_auc=("auc_oriented", "max"),
        n_auc_ge_0_75=("auc_oriented", lambda x: int((x >= 0.75).sum())),
        n_auc_ge_0_80=("auc_oriented", lambda x: int((x >= 0.80).sum())),
    )
    .reset_index()
)

sensitivity_feature_summary = (
    pipeline_sensitivity_df.groupby("feature")
    .agg(
        mean_relative_pipeline_range_percent=("mean_relative_pipeline_range_percent", "mean"),
        median_relative_pipeline_range_percent=("median_relative_pipeline_range_percent", "median"),
    )
    .reset_index()
)

cv_feature_summary = (
    cv_df.groupby("feature")
    .agg(
        mean_cv_percent=("mean_cv_percent", "mean"),
        median_cv_percent=("median_cv_percent", "median"),
        max_cv_percent=("max_cv_percent", "max")
    )
    .reset_index()
)

feature_level_summary = (
    effect_feature_summary
    .merge(auc_feature_summary, on="feature", how="left")
    .merge(sensitivity_feature_summary, on="feature", how="left")
    .merge(cv_feature_summary, on="feature", how="left")
)

# Rangszámok: nagyobb abs_g és AUC jobb; kisebb CV és pipeline range jobb.
feature_level_summary["rank_effect"] = feature_level_summary["mean_abs_g"].rank(ascending=False, method="min")
feature_level_summary["rank_auc"] = feature_level_summary["mean_auc"].rank(ascending=False, method="min")
feature_level_summary["rank_cv"] = feature_level_summary["mean_cv_percent"].rank(ascending=True, method="min")
feature_level_summary["rank_pipeline_sensitivity"] = feature_level_summary["mean_relative_pipeline_range_percent"].rank(ascending=True, method="min")

feature_level_summary["overall_rank_score"] = (
    feature_level_summary["rank_effect"] +
    feature_level_summary["rank_auc"] +
    feature_level_summary["rank_cv"] +
    feature_level_summary["rank_pipeline_sensitivity"]
)

feature_level_summary = feature_level_summary.sort_values("overall_rank_score")

feature_level_summary.to_csv(OUTPUT_DIR / "feature_level_summary.csv", index=False)
feature_level_summary.to_csv(OUTPUT_DIR / "feature_screening_summary.csv", index=False)


# ---------------------------------------------------------------------
# 9. Top biomarkerek feature szerint
# ---------------------------------------------------------------------

top_effects = (
    effect_df.sort_values("abs_g", ascending=False)
    .groupby("feature")
    .head(10)
    .reset_index(drop=True)
)
top_effects.to_csv(OUTPUT_DIR / "top10_effects_by_feature.csv", index=False)

top_auc = (
    auc_df.sort_values("auc_oriented", ascending=False)
    .groupby("feature")
    .head(10)
    .reset_index(drop=True)
)
top_auc.to_csv(OUTPUT_DIR / "top10_auc_by_feature.csv", index=False)


# ---------------------------------------------------------------------
# 10. Szöveges riport
# ---------------------------------------------------------------------

lines = []
lines.append("FEATURE SCREENING REPORT")
lines.append("=" * 80)
lines.append("")
lines.append("Cél: a 6 tractometriai feature objektív összehasonlítása.")
lines.append("")
lines.append("Vizsgált feature-ek:")
for f in FEATURES:
    lines.append(f"- {f}")
lines.append("")

lines.append("FEATURE-LEVEL SUMMARY")
lines.append("-" * 80)

display_cols = [
    "feature",
    "mean_abs_g",
    "median_abs_g",
    "mean_auc",
    "median_auc",
    "mean_cv_percent",
    "mean_relative_pipeline_range_percent",
    "overall_rank_score"
]

for _, row in feature_level_summary.iterrows():
    lines.append(
        f"{row['feature']}: "
        f"mean_abs_g={row['mean_abs_g']:.3f}, "
        f"median_abs_g={row['median_abs_g']:.3f}, "
        f"mean_AUC={row['mean_auc']:.3f}, "
        f"median_AUC={row['median_auc']:.3f}, "
        f"mean_CV={row['mean_cv_percent']:.2f}%, "
        f"pipeline_range={row['mean_relative_pipeline_range_percent']:.2f}%, "
        f"rank_score={row['overall_rank_score']:.0f}"
    )

lines.append("")
lines.append("INTERPRETÁCIÓS MEGJEGYZÉS")
lines.append("-" * 80)
lines.append(
    "Az overall_rank_score csak előszűrő mutató. Nem tekintendő formális statisztikai "
    "tesztnek. Célja annak feltárása, hogy mely feature-ek érdemesek fő elemzésre."
)
lines.append("")
lines.append(
    "A mean/median/sd feature-ek globális tractus-összefoglalók, míg a proximal_mean, "
    "middle_mean és distal_mean along-tract lokalizációs feature-ek. Ezeket az értelmezésben "
    "külön kategóriaként kell kezelni."
)
lines.append("")
lines.append("Kimeneti fájlok:")
for file in sorted(OUTPUT_DIR.glob("*")):
    lines.append(f"- {file.name}")

with open(OUTPUT_DIR / "feature_screening_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\n".join(lines))
