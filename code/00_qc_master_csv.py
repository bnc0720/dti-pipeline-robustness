# -*- coding: utf-8 -*-
"""
00_qc_master_csv.py

Cél:
    A master tractometry CSV teljes alapellenőrzése az új statisztikai elemzés előtt.

Bemenet:
    data/all_pipelines_tractometry_features_master.csv

Kimenetek:
    outputs/qc/dataset_summary.txt
    outputs/qc/subject_counts.csv
    outputs/qc/group_counts.csv
    outputs/qc/pipeline_counts.csv
    outputs/qc/tract_counts.csv
    outputs/qc/metric_counts.csv
    outputs/qc/missing_values.csv
    outputs/qc/duplicate_rows.csv
    outputs/qc/completeness_check.csv
    outputs/qc/feature_value_summary.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# 1. Útvonalak
# ---------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_CSV = PROJECT_DIR / "data" / "all_pipelines_tractometry_features_master.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "qc"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# 2. Beolvasás
# ---------------------------------------------------------------------

if not INPUT_CSV.exists():
    raise FileNotFoundError(f"Nem található a bemeneti fájl: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)


# ---------------------------------------------------------------------
# 3. Elvárt oszlopok ellenőrzése
# ---------------------------------------------------------------------

expected_columns = [
    "subject",
    "group",
    "pipeline",
    "tract",
    "metric",
    "mean",
    "median",
    "sd",
    "proximal_mean",
    "middle_mean",
    "distal_mean",
]

missing_expected_columns = [col for col in expected_columns if col not in df.columns]
extra_columns = [col for col in df.columns if col not in expected_columns]

if missing_expected_columns:
    raise ValueError(f"Hiányzó kötelező oszlopok: {missing_expected_columns}")


# ---------------------------------------------------------------------
# 4. Alap objektumok
# ---------------------------------------------------------------------

subjects = sorted(df["subject"].dropna().unique())
groups = sorted(df["group"].dropna().unique())
pipelines = sorted(df["pipeline"].dropna().unique())
tracts = sorted(df["tract"].dropna().unique())
metrics = sorted(df["metric"].dropna().unique())

n_subjects = len(subjects)
n_groups = len(groups)
n_pipelines = len(pipelines)
n_tracts = len(tracts)
n_metrics = len(metrics)

feature_columns = ["mean", "median", "sd", "proximal_mean", "middle_mean", "distal_mean"]


# ---------------------------------------------------------------------
# 5. Count táblák
# ---------------------------------------------------------------------

subject_counts = (
    df.groupby(["subject", "group"])
    .size()
    .reset_index(name="n_rows")
    .sort_values(["group", "subject"])
)

group_counts = (
    df[["subject", "group"]]
    .drop_duplicates()
    .groupby("group")
    .size()
    .reset_index(name="n_subjects")
)

pipeline_counts = df.groupby("pipeline").size().reset_index(name="n_rows")
tract_counts = df.groupby("tract").size().reset_index(name="n_rows")
metric_counts = df.groupby("metric").size().reset_index(name="n_rows")

subject_counts.to_csv(OUTPUT_DIR / "subject_counts.csv", index=False)
group_counts.to_csv(OUTPUT_DIR / "group_counts.csv", index=False)
pipeline_counts.to_csv(OUTPUT_DIR / "pipeline_counts.csv", index=False)
tract_counts.to_csv(OUTPUT_DIR / "tract_counts.csv", index=False)
metric_counts.to_csv(OUTPUT_DIR / "metric_counts.csv", index=False)


# ---------------------------------------------------------------------
# 6. Hiányzó értékek
# ---------------------------------------------------------------------

missing_values = (
    df.isna()
    .sum()
    .reset_index()
)
missing_values.columns = ["column", "n_missing"]
missing_values["percent_missing"] = missing_values["n_missing"] / len(df) * 100
missing_values.to_csv(OUTPUT_DIR / "missing_values.csv", index=False)


# ---------------------------------------------------------------------
# 7. Duplikált sorok
# ---------------------------------------------------------------------

duplicate_mask = df.duplicated()
duplicates = df.loc[duplicate_mask].copy()
duplicates.to_csv(OUTPUT_DIR / "duplicate_rows.csv", index=False)


# ---------------------------------------------------------------------
# 8. Kombinációk teljességének ellenőrzése
# ---------------------------------------------------------------------

expected_index = pd.MultiIndex.from_product(
    [subjects, pipelines, tracts, metrics],
    names=["subject", "pipeline", "tract", "metric"]
).to_frame(index=False)

observed_index = df[["subject", "pipeline", "tract", "metric"]].drop_duplicates()

completeness = expected_index.merge(
    observed_index,
    on=["subject", "pipeline", "tract", "metric"],
    how="left",
    indicator=True
)

completeness["present"] = completeness["_merge"].eq("both")
completeness = completeness.drop(columns=["_merge"])
completeness.to_csv(OUTPUT_DIR / "completeness_check.csv", index=False)

n_expected_combinations = len(expected_index)
n_observed_combinations = len(observed_index)
n_missing_combinations = int((~completeness["present"]).sum())


# ---------------------------------------------------------------------
# 9. Feature summary
# ---------------------------------------------------------------------

feature_value_summary = (
    df.groupby(["pipeline", "tract", "metric"])[feature_columns]
    .agg(["count", "mean", "std", "median", "min", "max"])
)

feature_value_summary.to_csv(OUTPUT_DIR / "feature_value_summary.csv")


# ---------------------------------------------------------------------
# 10. Numerikus sanity check
# ---------------------------------------------------------------------

numeric_issues = []

for col in feature_columns:
    if col in df.columns:
        n_inf = np.isinf(df[col]).sum()
        n_nan = df[col].isna().sum()
        numeric_issues.append({
            "feature": col,
            "n_nan": int(n_nan),
            "n_inf": int(n_inf),
            "min": df[col].min(),
            "max": df[col].max(),
        })

numeric_issues = pd.DataFrame(numeric_issues)
numeric_issues.to_csv(OUTPUT_DIR / "numeric_sanity_check.csv", index=False)


# ---------------------------------------------------------------------
# 11. Szöveges riport
# ---------------------------------------------------------------------

summary_lines = []

summary_lines.append("MASTER CSV QC REPORT")
summary_lines.append("=" * 80)
summary_lines.append("")
summary_lines.append(f"Input file: {INPUT_CSV}")
summary_lines.append(f"Total rows: {len(df)}")
summary_lines.append(f"Total columns: {len(df.columns)}")
summary_lines.append("")

summary_lines.append("DATASET STRUCTURE")
summary_lines.append("-" * 80)
summary_lines.append(f"Number of subjects: {n_subjects}")
summary_lines.append(f"Groups: {groups}")
summary_lines.append(f"Number of groups: {n_groups}")
summary_lines.append(f"Pipelines: {pipelines}")
summary_lines.append(f"Number of pipelines: {n_pipelines}")
summary_lines.append(f"Number of tracts: {n_tracts}")
summary_lines.append(f"Tracts: {tracts}")
summary_lines.append(f"Number of metrics: {n_metrics}")
summary_lines.append(f"Metrics: {metrics}")
summary_lines.append("")

summary_lines.append("GROUP COUNTS")
summary_lines.append("-" * 80)
for _, row in group_counts.iterrows():
    summary_lines.append(f"{row['group']}: {row['n_subjects']} subjects")
summary_lines.append("")

summary_lines.append("EXPECTED COMPLETENESS")
summary_lines.append("-" * 80)
summary_lines.append(f"Expected combinations: {n_expected_combinations}")
summary_lines.append(f"Observed unique combinations: {n_observed_combinations}")
summary_lines.append(f"Missing combinations: {n_missing_combinations}")
summary_lines.append("")

summary_lines.append("MISSING VALUES")
summary_lines.append("-" * 80)
for _, row in missing_values.iterrows():
    summary_lines.append(
        f"{row['column']}: {row['n_missing']} missing "
        f"({row['percent_missing']:.2f}%)"
    )
summary_lines.append("")

summary_lines.append("DUPLICATES")
summary_lines.append("-" * 80)
summary_lines.append(f"Duplicate rows: {len(duplicates)}")
summary_lines.append("")

summary_lines.append("NUMERIC SANITY CHECK")
summary_lines.append("-" * 80)
for _, row in numeric_issues.iterrows():
    summary_lines.append(
        f"{row['feature']}: NaN={row['n_nan']}, Inf={row['n_inf']}, "
        f"min={row['min']}, max={row['max']}"
    )
summary_lines.append("")

summary_lines.append("QC STATUS")
summary_lines.append("-" * 80)

qc_pass = True

if n_missing_combinations > 0:
    qc_pass = False
    summary_lines.append("WARNING: Missing subject × pipeline × tract × metric combinations detected.")

if len(duplicates) > 0:
    qc_pass = False
    summary_lines.append("WARNING: Duplicate rows detected.")

if missing_values["n_missing"].sum() > 0:
    qc_pass = False
    summary_lines.append("WARNING: Missing values detected.")

if qc_pass:
    summary_lines.append("PASS: No missing combinations, duplicate rows, or missing values detected.")
else:
    summary_lines.append("CHECK REQUIRED: One or more QC warnings were detected.")

summary_lines.append("")
summary_lines.append("Generated output files:")
for file in sorted(OUTPUT_DIR.glob("*")):
    summary_lines.append(f"- {file.name}")

with open(OUTPUT_DIR / "dataset_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print("\n".join(summary_lines))
