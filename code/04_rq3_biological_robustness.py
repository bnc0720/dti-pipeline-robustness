# -*- coding: utf-8 -*-
"""
04_rq3_biological_robustness.py

RQ3:
    Megváltoztatja-e a preprocessing pipeline az Alzheimer-kórhoz kapcsolódó
    biológiai következtetéseket?

Új elemzés, a korábbi statisztikai pipeline-tól függetlenül.

Bemenet:
    data/all_pipelines_tractometry_features_master.csv

Kimenetek:
    outputs/rq3_biological_robustness/
        group_pipeline_interactions.csv
        effect_sizes_by_pipeline.csv
        direction_consistency.csv
        effect_size_stability.csv
        biomarker_ranking_stability.csv
        top_biomarker_overlap.csv
        rq3_report.txt

Fő bizonyítéklánc:
    1. Group × Pipeline interakciók vegyes modellből
    2. Pipeline-onkénti AD vs CN effektusméretek
    3. Effektusirány stabilitása
    4. Effektusméret-stabilitás
    5. Biomarker-rangsor stabilitása

Primer feature:
    mean

Exploratív feature:
    proximal_mean
"""

from pathlib import Path
import itertools
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------
# 1. Útvonalak
# ---------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_DIR / "data" / "all_pipelines_tractometry_features_master.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "rq3_biological_robustness"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# 2. Beolvasás
# ---------------------------------------------------------------------

df = pd.read_csv(INPUT_CSV)

FEATURES = ["mean", "proximal_mean"]
PRIMARY_FEATURE = "mean"
EXPLORATORY_FEATURE = "proximal_mean"

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

long_df["group"] = pd.Categorical(long_df["group"], categories=["CN", "AD"])
long_df["pipeline"] = pd.Categorical(
    long_df["pipeline"],
    categories=["pipeline_01", "pipeline_02", "pipeline_03"]
)


# ---------------------------------------------------------------------
# 3. Segédfüggvények
# ---------------------------------------------------------------------

def hedges_g(x_ad, x_cn):
    """
    Hedges g számítása AD - CN iránnyal.
    Pozitív: AD > CN
    Negatív: AD < CN
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
    correction = 1 - (3 / (4 * (n1 + n0) - 9))
    return d * correction


def fit_interaction_model(sub):
    """
    Mixed model:
        value ~ group + pipeline + group:pipeline + (1|subject)

    RQ3-ban az interakciós tagokra fókuszálunk:
        C(group)[T.AD]:C(pipeline)[T.pipeline_02]
        C(group)[T.AD]:C(pipeline)[T.pipeline_03]
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.mixedlm(
                "value ~ C(group) + C(pipeline) + C(group):C(pipeline)",
                data=sub,
                groups=sub["subject"]
            )
            fit = model.fit(reml=False, method="lbfgs", maxiter=500, disp=False)
        return fit, ""
    except Exception as e:
        return None, str(e)


def direction_from_g(g):
    if pd.isna(g):
        return "undefined"
    if g > 0:
        return "AD_greater_CN"
    if g < 0:
        return "AD_lower_CN"
    return "zero"


# ---------------------------------------------------------------------
# 4. Group × Pipeline interakciók
# ---------------------------------------------------------------------

interaction_rows = []

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

    fit, error = fit_interaction_model(sub)

    if fit is None:
        interaction_rows.append({
            "tract": tract,
            "metric": metric,
            "feature": feature,
            "term": "MODEL_FAILED",
            "estimate": np.nan,
            "std_error": np.nan,
            "z_value": np.nan,
            "p_value": np.nan,
            "n_observations": sub.shape[0],
            "n_subjects": sub["subject"].nunique(),
            "converged": False,
            "error": error
        })
    else:
        for term in fit.params.index:
            if "C(group)[T.AD]:C(pipeline)" in term:
                interaction_rows.append({
                    "tract": tract,
                    "metric": metric,
                    "feature": feature,
                    "term": term,
                    "estimate": fit.params.get(term, np.nan),
                    "std_error": fit.bse.get(term, np.nan),
                    "z_value": fit.tvalues.get(term, np.nan),
                    "p_value": fit.pvalues.get(term, np.nan),
                    "n_observations": sub.shape[0],
                    "n_subjects": sub["subject"].nunique(),
                    "converged": bool(fit.converged),
                    "error": ""
                })

interaction_df = pd.DataFrame(interaction_rows)

interaction_df["p_fdr_by_feature"] = np.nan
interaction_df["fdr_significant"] = False

for feature in FEATURES:
    mask = (
        (interaction_df["feature"] == feature) &
        interaction_df["p_value"].notna() &
        (interaction_df["term"] != "MODEL_FAILED")
    )
    if mask.sum() > 0:
        _, p_fdr, _, _ = multipletests(interaction_df.loc[mask, "p_value"], method="fdr_bh")
        interaction_df.loc[mask, "p_fdr_by_feature"] = p_fdr
        interaction_df.loc[mask, "fdr_significant"] = p_fdr < 0.05

interaction_df.to_csv(OUTPUT_DIR / "group_pipeline_interactions.csv", index=False)


# ---------------------------------------------------------------------
# 5. Pipeline-onkénti AD vs CN effektusméretek
# ---------------------------------------------------------------------

effect_rows = []

for pipeline, tract, metric, feature in itertools.product(
    sorted(long_df["pipeline"].cat.categories),
    sorted(long_df["tract"].unique()),
    sorted(long_df["metric"].unique()),
    FEATURES
):
    sub = long_df[
        (long_df["pipeline"] == pipeline) &
        (long_df["tract"] == tract) &
        (long_df["metric"] == metric) &
        (long_df["feature"] == feature)
    ].copy()

    ad_values = sub.loc[sub["group"] == "AD", "value"].dropna()
    cn_values = sub.loc[sub["group"] == "CN", "value"].dropna()

    g = hedges_g(ad_values, cn_values)

    try:
        t_stat, p_value = stats.ttest_ind(ad_values, cn_values, equal_var=False)
    except Exception:
        t_stat, p_value = np.nan, np.nan

    effect_rows.append({
        "pipeline": pipeline,
        "tract": tract,
        "metric": metric,
        "feature": feature,
        "n_AD": len(ad_values),
        "n_CN": len(cn_values),
        "mean_AD": ad_values.mean(),
        "mean_CN": cn_values.mean(),
        "difference_AD_minus_CN": ad_values.mean() - cn_values.mean(),
        "hedges_g_AD_minus_CN": g,
        "abs_g": abs(g) if pd.notna(g) else np.nan,
        "direction": direction_from_g(g),
        "welch_t": t_stat,
        "p_value": p_value
    })

effect_df = pd.DataFrame(effect_rows)

# FDR pipeline-onként + feature-önként, mert ezek külön AD-CN családok
effect_df["p_fdr_by_pipeline_feature"] = np.nan
effect_df["fdr_significant"] = False

for pipeline, feature in itertools.product(effect_df["pipeline"].unique(), FEATURES):
    mask = (
        (effect_df["pipeline"] == pipeline) &
        (effect_df["feature"] == feature) &
        effect_df["p_value"].notna()
    )
    if mask.sum() > 0:
        _, p_fdr, _, _ = multipletests(effect_df.loc[mask, "p_value"], method="fdr_bh")
        effect_df.loc[mask, "p_fdr_by_pipeline_feature"] = p_fdr
        effect_df.loc[mask, "fdr_significant"] = p_fdr < 0.05

effect_df.to_csv(OUTPUT_DIR / "effect_sizes_by_pipeline.csv", index=False)


# ---------------------------------------------------------------------
# 6. Irányazonosság és effektusméret-stabilitás
# ---------------------------------------------------------------------

consistency_rows = []
stability_rows = []

for tract, metric, feature in itertools.product(
    sorted(effect_df["tract"].unique()),
    sorted(effect_df["metric"].unique()),
    FEATURES
):
    sub = effect_df[
        (effect_df["tract"] == tract) &
        (effect_df["metric"] == metric) &
        (effect_df["feature"] == feature)
    ].copy()

    directions = sub.set_index("pipeline")["direction"].to_dict()
    gs = sub.set_index("pipeline")["hedges_g_AD_minus_CN"].to_dict()
    abs_gs = sub.set_index("pipeline")["abs_g"].to_dict()

    direction_values = [directions.get(p) for p in ["pipeline_01", "pipeline_02", "pipeline_03"]]
    valid_dirs = [d for d in direction_values if d not in ["undefined", None, "zero"]]

    same_direction_all = len(set(valid_dirs)) == 1 if len(valid_dirs) == 3 else False

    g_values = np.array([gs.get(p, np.nan) for p in ["pipeline_01", "pipeline_02", "pipeline_03"]], dtype=float)
    abs_g_values = np.abs(g_values)

    consistency_rows.append({
        "tract": tract,
        "metric": metric,
        "feature": feature,
        "pipeline_01_direction": directions.get("pipeline_01", "missing"),
        "pipeline_02_direction": directions.get("pipeline_02", "missing"),
        "pipeline_03_direction": directions.get("pipeline_03", "missing"),
        "same_direction_all_pipelines": same_direction_all,
        "pipeline_01_g": gs.get("pipeline_01", np.nan),
        "pipeline_02_g": gs.get("pipeline_02", np.nan),
        "pipeline_03_g": gs.get("pipeline_03", np.nan),
        "mean_abs_g": np.nanmean(abs_g_values),
        "min_abs_g": np.nanmin(abs_g_values),
        "max_abs_g": np.nanmax(abs_g_values),
        "range_abs_g": np.nanmax(abs_g_values) - np.nanmin(abs_g_values),
        "sd_abs_g": np.nanstd(abs_g_values, ddof=1)
    })

    stability_rows.append({
        "tract": tract,
        "metric": metric,
        "feature": feature,
        "mean_abs_g": np.nanmean(abs_g_values),
        "range_abs_g": np.nanmax(abs_g_values) - np.nanmin(abs_g_values),
        "sd_abs_g": np.nanstd(abs_g_values, ddof=1),
        "relative_range_abs_g_percent": (
            (np.nanmax(abs_g_values) - np.nanmin(abs_g_values)) /
            np.nanmean(abs_g_values) * 100
            if np.nanmean(abs_g_values) != 0 else np.nan
        )
    })

direction_df = pd.DataFrame(consistency_rows)
stability_df = pd.DataFrame(stability_rows)

direction_df.to_csv(OUTPUT_DIR / "direction_consistency.csv", index=False)
stability_df.to_csv(OUTPUT_DIR / "effect_size_stability.csv", index=False)


# ---------------------------------------------------------------------
# 7. Biomarker-rangsor stabilitás
# ---------------------------------------------------------------------
# Biomarker = tract + metric + feature
# Pipeline-onként abs_g alapján rangsorolunk.
# Ezután Spearman-korrelációt számolunk a pipeline rangsorok között.

ranking_rows = []
overlap_rows = []

for feature in FEATURES:
    sub = effect_df[effect_df["feature"] == feature].copy()
    sub["biomarker"] = sub["tract"] + "_" + sub["metric"] + "_" + sub["feature"]

    ranking_wide = sub.pivot_table(
        index="biomarker",
        columns="pipeline",
        values="abs_g"
    )

    # Spearman rangkorrelációk
    for p1, p2 in itertools.combinations(["pipeline_01", "pipeline_02", "pipeline_03"], 2):
        valid_pair = ranking_wide[[p1, p2]].dropna()
        rho, pval = stats.spearmanr(valid_pair[p1], valid_pair[p2])

        ranking_rows.append({
            "feature": feature,
            "pipeline_1": p1,
            "pipeline_2": p2,
            "n_biomarkers": valid_pair.shape[0],
            "spearman_rho_abs_g_ranking": rho,
            "p_value": pval
        })

    # Top-k overlap
    for k in [5, 10, 15]:
        top_sets = {}
        for p in ["pipeline_01", "pipeline_02", "pipeline_03"]:
            tmp = sub[sub["pipeline"] == p].sort_values("abs_g", ascending=False)
            top_sets[p] = set(tmp.head(k)["biomarker"])

        overlap_12 = len(top_sets["pipeline_01"] & top_sets["pipeline_02"])
        overlap_13 = len(top_sets["pipeline_01"] & top_sets["pipeline_03"])
        overlap_23 = len(top_sets["pipeline_02"] & top_sets["pipeline_03"])
        overlap_all = len(top_sets["pipeline_01"] & top_sets["pipeline_02"] & top_sets["pipeline_03"])

        overlap_rows.append({
            "feature": feature,
            "top_k": k,
            "overlap_pipeline01_02": overlap_12,
            "overlap_pipeline01_03": overlap_13,
            "overlap_pipeline02_03": overlap_23,
            "overlap_all_three": overlap_all,
            "overlap_all_three_percent": overlap_all / k * 100
        })

ranking_df = pd.DataFrame(ranking_rows)
overlap_df = pd.DataFrame(overlap_rows)

ranking_df.to_csv(OUTPUT_DIR / "biomarker_ranking_stability.csv", index=False)
overlap_df.to_csv(OUTPUT_DIR / "top_biomarker_overlap.csv", index=False)


# ---------------------------------------------------------------------
# 8. Összesítések a riporthoz
# ---------------------------------------------------------------------

primary_interaction = interaction_df[
    (interaction_df["feature"] == PRIMARY_FEATURE) &
    (interaction_df["term"] != "MODEL_FAILED")
]

primary_direction = direction_df[direction_df["feature"] == PRIMARY_FEATURE]
primary_stability = stability_df[stability_df["feature"] == PRIMARY_FEATURE]
primary_ranking = ranking_df[ranking_df["feature"] == PRIMARY_FEATURE]
primary_overlap = overlap_df[overlap_df["feature"] == PRIMARY_FEATURE]

expl_interaction = interaction_df[
    (interaction_df["feature"] == EXPLORATORY_FEATURE) &
    (interaction_df["term"] != "MODEL_FAILED")
]

summary_rows = []

for feature in FEATURES:
    inter = interaction_df[(interaction_df["feature"] == feature) & (interaction_df["term"] != "MODEL_FAILED")]
    direc = direction_df[direction_df["feature"] == feature]
    stab = stability_df[stability_df["feature"] == feature]
    rank = ranking_df[ranking_df["feature"] == feature]

    summary_rows.append({
        "feature": feature,
        "n_interaction_tests": inter.shape[0],
        "n_fdr_significant_interactions": int(inter["fdr_significant"].sum()),
        "percent_fdr_significant_interactions": inter["fdr_significant"].mean() * 100 if inter.shape[0] else np.nan,
        "same_direction_percent": direc["same_direction_all_pipelines"].mean() * 100,
        "mean_abs_g": direc["mean_abs_g"].mean(),
        "mean_range_abs_g": direc["range_abs_g"].mean(),
        "mean_relative_range_abs_g_percent": stab["relative_range_abs_g_percent"].mean(),
        "mean_ranking_spearman_rho": rank["spearman_rho_abs_g_ranking"].mean()
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTPUT_DIR / "rq3_feature_summary.csv", index=False)


# ---------------------------------------------------------------------
# 9. Top stabil biomarkerek
# ---------------------------------------------------------------------

stable_candidates = primary_direction[
    (primary_direction["same_direction_all_pipelines"] == True)
].copy()

stable_candidates = stable_candidates.sort_values(
    ["mean_abs_g", "range_abs_g"],
    ascending=[False, True]
)

stable_candidates.to_csv(OUTPUT_DIR / "stable_biomarker_candidates_primary_mean.csv", index=False)


# ---------------------------------------------------------------------
# 10. Szöveges riport
# ---------------------------------------------------------------------

lines = []

lines.append("RQ3 BIOLOGICAL ROBUSTNESS REPORT")
lines.append("=" * 80)
lines.append("")
lines.append("Kutatási kérdés:")
lines.append("Megváltoztatja-e a preprocessing pipeline az Alzheimer-kórhoz kapcsolódó biológiai következtetéseket?")
lines.append("")
lines.append("Bizonyítéklánc:")
lines.append("1. Group × Pipeline interakciók")
lines.append("2. Pipeline-onkénti AD vs CN effektusméretek")
lines.append("3. Effektusirány stabilitása")
lines.append("4. Effektusméret-stabilitás")
lines.append("5. Biomarker-rangsor stabilitása")
lines.append("")

lines.append("PRIMARY FEATURE: mean")
lines.append("-" * 80)
lines.append(f"Interaction tests: {primary_interaction.shape[0]}")
lines.append(
    f"FDR-significant Group×Pipeline interactions: "
    f"{int(primary_interaction['fdr_significant'].sum())}/{primary_interaction.shape[0]} "
    f"({primary_interaction['fdr_significant'].mean() * 100:.1f}%)"
)
lines.append(
    f"Same AD-CN effect direction across all pipelines: "
    f"{primary_direction['same_direction_all_pipelines'].mean() * 100:.1f}% "
    f"({int(primary_direction['same_direction_all_pipelines'].sum())}/{primary_direction.shape[0]})"
)
lines.append(f"Mean absolute Hedges g across biomarkers: {primary_direction['mean_abs_g'].mean():.3f}")
lines.append(f"Mean absolute g range across pipelines: {primary_direction['range_abs_g'].mean():.3f}")
lines.append(f"Mean ranking Spearman rho across pipeline pairs: {primary_ranking['spearman_rho_abs_g_ranking'].mean():.3f}")
lines.append("")

lines.append("TOP-K BIOMARKER OVERLAP, PRIMARY FEATURE")
lines.append("-" * 80)
for _, row in primary_overlap.iterrows():
    lines.append(
        f"Top {int(row['top_k'])}: "
        f"all-three overlap={int(row['overlap_all_three'])}/{int(row['top_k'])} "
        f"({row['overlap_all_three_percent']:.1f}%)"
    )

lines.append("")
lines.append("FEATURE-LEVEL SUMMARY")
lines.append("-" * 80)
for _, row in summary_df.iterrows():
    lines.append(
        f"{row['feature']}: "
        f"interactions FDR-significant={int(row['n_fdr_significant_interactions'])}/{int(row['n_interaction_tests'])} "
        f"({row['percent_fdr_significant_interactions']:.1f}%), "
        f"same_direction={row['same_direction_percent']:.1f}%, "
        f"mean_abs_g={row['mean_abs_g']:.3f}, "
        f"mean_g_range={row['mean_range_abs_g']:.3f}, "
        f"ranking_rho={row['mean_ranking_spearman_rho']:.3f}"
    )

lines.append("")
lines.append("TOP STABLE BIOMARKER CANDIDATES, PRIMARY MEAN")
lines.append("-" * 80)

for _, row in stable_candidates.head(15).iterrows():
    lines.append(
        f"{row['tract']} {row['metric']}: "
        f"mean_abs_g={row['mean_abs_g']:.3f}, "
        f"range_abs_g={row['range_abs_g']:.3f}, "
        f"g_P01={row['pipeline_01_g']:.3f}, "
        f"g_P02={row['pipeline_02_g']:.3f}, "
        f"g_P03={row['pipeline_03_g']:.3f}"
    )

lines.append("")
lines.append("INTERPRETÁCIÓS MEGJEGYZÉS")
lines.append("-" * 80)
lines.append(
    "Az RQ3 akkor támasztja alá a biológiai robusztusságot, ha kevés Group×Pipeline "
    "interakció szignifikáns, az effektusirányok nagyrészt azonosak, az effektusméretek "
    "stabilak, és a biomarker-rangsorok erősen korrelálnak a pipeline-ok között."
)
lines.append("")
lines.append(
    "Fontos: a nem szignifikáns interakció önmagában nem bizonyíték a robusztusságra. "
    "Ezért az értelmezés effektusméret-, irány- és rangsorstabilitási mutatókkal együtt történik."
)
lines.append("")
lines.append("Kimeneti fájlok:")
for file in sorted(OUTPUT_DIR.glob("*")):
    lines.append(f"- {file.name}")

with open(OUTPUT_DIR / "rq3_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\n".join(lines))
