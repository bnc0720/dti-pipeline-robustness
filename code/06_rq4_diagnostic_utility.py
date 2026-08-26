# -*- coding: utf-8 -*-
"""
06_rq4_diagnostic_utility.py

RQ4:
    Megváltoztatja-e a preprocessing pipeline a diagnosztikai teljesítményt?

Fő cél:
    Nem prediktív modell építése, hanem biomarker-szintű diagnosztikai robusztusság
    vizsgálata pipeline-ok között.

Bemenet:
    data/all_pipelines_tractometry_features_master.csv

Kimenetek:
    outputs/rq4_diagnostic_utility/
        auc_by_pipeline.csv
        auc_stability_summary.csv
        top_auc_biomarkers.csv
        roc_curve_points.csv
        selected_roc_curve_points.csv
        selected_robust_biomarkers_auc.csv
        figure_auc_stability_data.csv
        rq4_report.txt

Primer feature:
    mean

Elemzett biomarkerek:
    10 tract × 4 metric = 40 biomarker

Fontos:
    A ROC görbékhez kiválasztott biomarkerek nem AUC alapján kerülnek kiválasztásra,
    hanem az RQ3-ban azonosított biológiai robusztusság alapján.
"""

from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


# ---------------------------------------------------------------------
# 1. Útvonalak
# ---------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_DIR / "data" / "all_pipelines_tractometry_features_master.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "rq4_diagnostic_utility"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# 2. Paraméterek
# ---------------------------------------------------------------------

FEATURE = "mean"
N_BOOTSTRAP = 2000
RANDOM_STATE = 42

# RQ3 alapján kiválasztott biológiailag robusztus biomarkerek,
# NEM az AUC alapján kiválasztva.
SELECTED_ROC_BIOMARKERS = [
    ("UF_right", "RD"),
    ("UF_right", "FA"),
    ("UF_right", "MD"),
    ("UF_left", "FA"),
    ("UF_left", "RD"),
    ("UF_left", "MD"),
]


# ---------------------------------------------------------------------
# 3. Beolvasás
# ---------------------------------------------------------------------

df = pd.read_csv(INPUT_CSV)

required_cols = ["subject", "group", "pipeline", "tract", "metric", FEATURE]
missing_cols = [c for c in required_cols if c not in df.columns]

if missing_cols:
    raise ValueError(f"Hiányzó oszlopok: {missing_cols}")

df = df.copy()
df["group_binary"] = df["group"].map({"CN": 0, "AD": 1})

if df["group_binary"].isna().any():
    raise ValueError("A group oszlop csak 'AD' és 'CN' értékeket tartalmazhat.")

pipelines = sorted(df["pipeline"].unique())
tracts = sorted(df["tract"].unique())
metrics = sorted(df["metric"].unique())


# ---------------------------------------------------------------------
# 4. Segédfüggvények
# ---------------------------------------------------------------------

def oriented_auc(y_true, values):
    """
    AUC számítása orientációval.
    Ha az AD-ben alacsonyabb érték jelez betegséget, az AUC_raw < 0.5 lehet.
    Az auc_oriented mindig >=0.5, a direction jelzi az irányt.
    """
    y_true = np.asarray(y_true)
    values = np.asarray(values, dtype=float)

    auc_raw = roc_auc_score(y_true, values)

    if auc_raw >= 0.5:
        return auc_raw, auc_raw, "higher_values_AD", values
    else:
        return 1 - auc_raw, auc_raw, "lower_values_AD", -values


def bootstrap_auc_ci(y_true, values, n_bootstrap=2000, random_state=42):
    """
    Bootstrap 95% CI az orientált AUC-ra.
    Stratifikált bootstrap: AD és CN csoportból külön mintavételezünk.
    """
    rng = np.random.default_rng(random_state)

    y_true = np.asarray(y_true)
    values = np.asarray(values, dtype=float)

    idx_cn = np.where(y_true == 0)[0]
    idx_ad = np.where(y_true == 1)[0]

    boot_aucs = []

    for _ in range(n_bootstrap):
        sample_cn = rng.choice(idx_cn, size=len(idx_cn), replace=True)
        sample_ad = rng.choice(idx_ad, size=len(idx_ad), replace=True)
        sample_idx = np.concatenate([sample_cn, sample_ad])

        y_b = y_true[sample_idx]
        v_b = values[sample_idx]

        try:
            auc_b, _, _, _ = oriented_auc(y_b, v_b)
            boot_aucs.append(auc_b)
        except Exception:
            continue

    if len(boot_aucs) == 0:
        return np.nan, np.nan

    return (
        float(np.percentile(boot_aucs, 2.5)),
        float(np.percentile(boot_aucs, 97.5))
    )


def get_roc_points(y_true, values):
    """
    ROC görbe pontok orientált értékekkel.
    """
    auc_or, auc_raw, direction, oriented_values = oriented_auc(y_true, values)
    fpr, tpr, thresholds = roc_curve(y_true, oriented_values)

    return auc_or, auc_raw, direction, fpr, tpr, thresholds


# ---------------------------------------------------------------------
# 5. AUC pipeline-onként minden biomarkerre
# ---------------------------------------------------------------------

auc_rows = []
roc_rows = []

for pipeline, tract, metric in itertools.product(pipelines, tracts, metrics):
    sub = df[
        (df["pipeline"] == pipeline) &
        (df["tract"] == tract) &
        (df["metric"] == metric)
    ].copy()

    y = sub["group_binary"].values
    values = sub[FEATURE].values

    auc_or, auc_raw, direction, fpr, tpr, thresholds = get_roc_points(y, values)
    ci_low, ci_high = bootstrap_auc_ci(
        y,
        values,
        n_bootstrap=N_BOOTSTRAP,
        random_state=RANDOM_STATE
    )

    auc_rows.append({
        "pipeline": pipeline,
        "tract": tract,
        "metric": metric,
        "feature": FEATURE,
        "biomarker": f"{tract}_{metric}",
        "n": len(sub),
        "n_AD": int((sub["group"] == "AD").sum()),
        "n_CN": int((sub["group"] == "CN").sum()),
        "auc": auc_or,
        "auc_raw": auc_raw,
        "auc_ci_low": ci_low,
        "auc_ci_high": ci_high,
        "direction": direction,
        "abs_auc_minus_0_5": abs(auc_or - 0.5)
    })

    for i in range(len(fpr)):
        roc_rows.append({
            "pipeline": pipeline,
            "tract": tract,
            "metric": metric,
            "feature": FEATURE,
            "biomarker": f"{tract}_{metric}",
            "fpr": fpr[i],
            "tpr": tpr[i],
            "threshold": thresholds[i],
            "auc": auc_or,
            "direction": direction
        })

auc_df = pd.DataFrame(auc_rows)
roc_df = pd.DataFrame(roc_rows)

auc_df.to_csv(OUTPUT_DIR / "auc_by_pipeline.csv", index=False)
roc_df.to_csv(OUTPUT_DIR / "roc_curve_points.csv", index=False)


# ---------------------------------------------------------------------
# 6. AUC stabilitás pipeline-ok között
# ---------------------------------------------------------------------

stability_rows = []

for tract, metric in itertools.product(tracts, metrics):
    sub = auc_df[
        (auc_df["tract"] == tract) &
        (auc_df["metric"] == metric)
    ].copy()

    auc_values = sub.set_index("pipeline")["auc"].reindex(pipelines)

    mean_auc = auc_values.mean()
    median_auc = auc_values.median()
    auc_range = auc_values.max() - auc_values.min()
    auc_sd = auc_values.std(ddof=1)

    direction_values = sub.set_index("pipeline")["direction"].reindex(pipelines).tolist()
    same_direction_all = len(set(direction_values)) == 1

    stability_rows.append({
        "tract": tract,
        "metric": metric,
        "feature": FEATURE,
        "biomarker": f"{tract}_{metric}",
        "auc_pipeline_01": auc_values.get("pipeline_01", np.nan),
        "auc_pipeline_02": auc_values.get("pipeline_02", np.nan),
        "auc_pipeline_03": auc_values.get("pipeline_03", np.nan),
        "mean_auc": mean_auc,
        "median_auc": median_auc,
        "auc_range": auc_range,
        "auc_sd": auc_sd,
        "same_direction_all_pipelines": same_direction_all,
        "pipeline_01_direction": direction_values[0],
        "pipeline_02_direction": direction_values[1],
        "pipeline_03_direction": direction_values[2],
    })

stability_df = pd.DataFrame(stability_rows)

# Diagnosztikai robusztussági score:
# magas mean_auc és alacsony auc_range legyen előnyös.
# Egyszerű, transzparens rangsor:
stability_df["rank_mean_auc"] = stability_df["mean_auc"].rank(ascending=False, method="min")
stability_df["rank_auc_range"] = stability_df["auc_range"].rank(ascending=True, method="min")
stability_df["diagnostic_robustness_rank_score"] = (
    stability_df["rank_mean_auc"] + stability_df["rank_auc_range"]
)

stability_df = stability_df.sort_values(
    ["diagnostic_robustness_rank_score", "mean_auc"],
    ascending=[True, False]
)

stability_df.to_csv(OUTPUT_DIR / "auc_stability_summary.csv", index=False)
stability_df.to_csv(OUTPUT_DIR / "figure_auc_stability_data.csv", index=False)


# ---------------------------------------------------------------------
# 7. Top AUC biomarkerek és RQ3-selected biomarkerek
# ---------------------------------------------------------------------

top_auc = stability_df.sort_values(
    ["mean_auc", "auc_range"],
    ascending=[False, True]
).head(20)

top_auc.to_csv(OUTPUT_DIR / "top_auc_biomarkers.csv", index=False)

selected_mask = stability_df.apply(
    lambda row: (row["tract"], row["metric"]) in SELECTED_ROC_BIOMARKERS,
    axis=1
)

selected_auc_df = stability_df[selected_mask].copy()
selected_auc_df.to_csv(OUTPUT_DIR / "selected_robust_biomarkers_auc.csv", index=False)

selected_biomarker_names = set(selected_auc_df["biomarker"])
selected_roc_df = roc_df[roc_df["biomarker"].isin(selected_biomarker_names)].copy()
selected_roc_df.to_csv(OUTPUT_DIR / "selected_roc_curve_points.csv", index=False)


# ---------------------------------------------------------------------
# 8. Összefoglalók
# ---------------------------------------------------------------------

overall_summary = {
    "n_biomarkers": stability_df.shape[0],
    "mean_auc_all_biomarkers": stability_df["mean_auc"].mean(),
    "median_auc_all_biomarkers": stability_df["mean_auc"].median(),
    "mean_auc_range_all_biomarkers": stability_df["auc_range"].mean(),
    "median_auc_range_all_biomarkers": stability_df["auc_range"].median(),
    "n_mean_auc_ge_0_70": int((stability_df["mean_auc"] >= 0.70).sum()),
    "n_mean_auc_ge_0_75": int((stability_df["mean_auc"] >= 0.75).sum()),
    "n_mean_auc_ge_0_80": int((stability_df["mean_auc"] >= 0.80).sum()),
    "n_auc_range_le_0_05": int((stability_df["auc_range"] <= 0.05).sum()),
    "n_auc_range_le_0_10": int((stability_df["auc_range"] <= 0.10).sum()),
}

overall_summary_df = pd.DataFrame([overall_summary])
overall_summary_df.to_csv(OUTPUT_DIR / "diagnostic_robustness_summary.csv", index=False)

metric_summary = (
    stability_df.groupby("metric")
    .agg(
        n_biomarkers=("biomarker", "count"),
        mean_auc=("mean_auc", "mean"),
        median_auc=("mean_auc", "median"),
        mean_auc_range=("auc_range", "mean"),
        median_auc_range=("auc_range", "median"),
        n_auc_ge_0_75=("mean_auc", lambda x: int((x >= 0.75).sum())),
        n_range_le_0_05=("auc_range", lambda x: int((x <= 0.05).sum()))
    )
    .reset_index()
    .sort_values("mean_auc", ascending=False)
)

metric_summary.to_csv(OUTPUT_DIR / "metric_diagnostic_summary.csv", index=False)

tract_summary = (
    stability_df.groupby("tract")
    .agg(
        n_biomarkers=("biomarker", "count"),
        mean_auc=("mean_auc", "mean"),
        median_auc=("mean_auc", "median"),
        mean_auc_range=("auc_range", "mean"),
        median_auc_range=("auc_range", "median"),
        n_auc_ge_0_75=("mean_auc", lambda x: int((x >= 0.75).sum())),
        n_range_le_0_05=("auc_range", lambda x: int((x <= 0.05).sum()))
    )
    .reset_index()
    .sort_values("mean_auc", ascending=False)
)

tract_summary.to_csv(OUTPUT_DIR / "tract_diagnostic_summary.csv", index=False)


# ---------------------------------------------------------------------
# 9. Szöveges riport
# ---------------------------------------------------------------------

lines = []

lines.append("RQ4 DIAGNOSTIC UTILITY REPORT")
lines.append("=" * 80)
lines.append("")
lines.append("Kutatási kérdés:")
lines.append("Megváltoztatja-e a preprocessing pipeline a diagnosztikai teljesítményt?")
lines.append("")
lines.append("Primer elemzés:")
lines.append("- Feature: mean")
lines.append("- Biomarkerek: 10 tract × 4 metric = 40 biomarker")
lines.append("- Minden biomarkerre pipeline-onként ROC-AUC + bootstrap 95% CI")
lines.append("- Diagnosztikai robusztusság: mean AUC, AUC range, AUC SD")
lines.append("")
lines.append("Fontos módszertani megjegyzés:")
lines.append(
    "A ROC-görbékhez kiválasztott biomarkerek az RQ3 biológiai robusztussági "
    "eredményei alapján kerültek kiválasztásra, nem az AUC alapján."
)
lines.append("")

lines.append("OVERALL DIAGNOSTIC ROBUSTNESS")
lines.append("-" * 80)
lines.append(f"Number of biomarkers: {overall_summary['n_biomarkers']}")
lines.append(f"Mean AUC across biomarkers: {overall_summary['mean_auc_all_biomarkers']:.3f}")
lines.append(f"Median AUC across biomarkers: {overall_summary['median_auc_all_biomarkers']:.3f}")
lines.append(f"Mean AUC range across pipelines: {overall_summary['mean_auc_range_all_biomarkers']:.3f}")
lines.append(f"Median AUC range across pipelines: {overall_summary['median_auc_range_all_biomarkers']:.3f}")
lines.append(f"Biomarkers with mean AUC ≥ 0.70: {overall_summary['n_mean_auc_ge_0_70']}/{overall_summary['n_biomarkers']}")
lines.append(f"Biomarkers with mean AUC ≥ 0.75: {overall_summary['n_mean_auc_ge_0_75']}/{overall_summary['n_biomarkers']}")
lines.append(f"Biomarkers with mean AUC ≥ 0.80: {overall_summary['n_mean_auc_ge_0_80']}/{overall_summary['n_biomarkers']}")
lines.append(f"Biomarkers with AUC range ≤ 0.05: {overall_summary['n_auc_range_le_0_05']}/{overall_summary['n_biomarkers']}")
lines.append(f"Biomarkers with AUC range ≤ 0.10: {overall_summary['n_auc_range_le_0_10']}/{overall_summary['n_biomarkers']}")
lines.append("")

lines.append("METRIC-LEVEL DIAGNOSTIC SUMMARY")
lines.append("-" * 80)
for _, row in metric_summary.iterrows():
    lines.append(
        f"{row['metric']}: "
        f"mean AUC={row['mean_auc']:.3f}, "
        f"median AUC={row['median_auc']:.3f}, "
        f"mean AUC range={row['mean_auc_range']:.3f}, "
        f"AUC≥0.75={int(row['n_auc_ge_0_75'])}/{int(row['n_biomarkers'])}"
    )

lines.append("")
lines.append("TRACT-LEVEL DIAGNOSTIC SUMMARY")
lines.append("-" * 80)
for _, row in tract_summary.iterrows():
    lines.append(
        f"{row['tract']}: "
        f"mean AUC={row['mean_auc']:.3f}, "
        f"median AUC={row['median_auc']:.3f}, "
        f"mean AUC range={row['mean_auc_range']:.3f}, "
        f"AUC≥0.75={int(row['n_auc_ge_0_75'])}/{int(row['n_biomarkers'])}"
    )

lines.append("")
lines.append("TOP DIAGNOSTICALLY ROBUST BIOMARKERS")
lines.append("-" * 80)
for _, row in stability_df.head(15).iterrows():
    lines.append(
        f"{row['biomarker']}: "
        f"mean AUC={row['mean_auc']:.3f}, "
        f"AUC range={row['auc_range']:.3f}, "
        f"P01={row['auc_pipeline_01']:.3f}, "
        f"P02={row['auc_pipeline_02']:.3f}, "
        f"P03={row['auc_pipeline_03']:.3f}"
    )

lines.append("")
lines.append("RQ3-SELECTED ROBUST BIOMARKERS FOR ROC VISUALIZATION")
lines.append("-" * 80)
for _, row in selected_auc_df.iterrows():
    lines.append(
        f"{row['biomarker']}: "
        f"mean AUC={row['mean_auc']:.3f}, "
        f"AUC range={row['auc_range']:.3f}, "
        f"P01={row['auc_pipeline_01']:.3f}, "
        f"P02={row['auc_pipeline_02']:.3f}, "
        f"P03={row['auc_pipeline_03']:.3f}"
    )

lines.append("")
lines.append("INTERPRETÁCIÓS MEGJEGYZÉS")
lines.append("-" * 80)
lines.append(
    "Az RQ4 eredmények értelmezésekor az AUC abszolút nagysága mellett az AUC range "
    "kulcsfontosságú, mert ez mutatja meg, hogy a pipeline-választás mennyire változtatja "
    "meg a diagnosztikai teljesítményt."
)
lines.append("")
lines.append(
    "A teljes elemzés minden biomarkerre lefutott. A ROC-görbék ábrázolása csak a "
    "kézirat olvashatósága miatt korlátozódik az RQ3-ban előre azonosított robusztus biomarkerekre."
)
lines.append("")
lines.append("Kimeneti fájlok:")
for file in sorted(OUTPUT_DIR.glob("*")):
    lines.append(f"- {file.name}")

with open(OUTPUT_DIR / "rq4_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\n".join(lines))
