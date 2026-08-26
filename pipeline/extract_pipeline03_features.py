import pandas as pd
import numpy as np
from pathlib import Path

PROJECT = Path("/path/to/adni_project")  # <-- EDIT: set to your project root
PIPELINE = "pipeline_03"

TRACTS = [
    "FX_left","FX_right",
    "CG_left","CG_right",
    "UF_left","UF_right",
    "ILF_left","ILF_right",
    "IFO_left","IFO_right",
    "CC_2","CC_7"
]

METRICS = ["FA", "MD", "RD", "AD"]

rows = []

for subj_dir in sorted((PROJECT / PIPELINE).glob("*")):
    if not subj_dir.is_dir():
        continue

    subj = subj_dir.name
    group = subj.split("_")[0]
    out = subj_dir / "tractometry_TOM"

    for metric in METRICS:
        csv_file = out / f"{subj}_{PIPELINE}_{metric}_tractometry.csv"

        if not csv_file.exists():
            print(f"Missing: {csv_file}")
            continue

        df = pd.read_csv(csv_file, sep=";")

        for tract in TRACTS:
            if tract not in df.columns:
                print(f"Missing tract column: {subj} {metric} {tract}")
                continue

            profile = df[tract].to_numpy(dtype=float)

            rows.append({
                "subject": subj,
                "group": group,
                "pipeline": PIPELINE,
                "tract": tract,
                "metric": metric,
                "mean": np.mean(profile),
                "median": np.median(profile),
                "sd": np.std(profile, ddof=1),
                "proximal_mean": np.mean(profile[:33]),
                "middle_mean": np.mean(profile[33:66]),
                "distal_mean": np.mean(profile[66:]),
            })

features = pd.DataFrame(rows)

outfile = PROJECT / "pipeline_03_tractometry_features_master.csv"
features.to_csv(outfile, index=False)

print(features.head())
print(f"Saved: {outfile}")
print(f"Rows: {len(features)}")
print("Subjects:", features["subject"].nunique() if len(features) else 0)
print("Tracts:", sorted(features["tract"].unique()) if len(features) else [])
print("Metrics:", sorted(features["metric"].unique()) if len(features) else [])
