# -*- coding: utf-8 -*-
"""
03_rq2_pipeline_effect_v2.py

RQ2:
    Megváltoztatja-e a preprocessing pipeline az abszolút DTI-értékeket?

PRIMER KÉRDÉS
-------------
Van-e a diagnosztikai csoportra korrigált, általános pipeline-hatás
a primer 'mean' tractometriai feature 40 tractus × metrika biomarkerére?

PRIMER MODELL
-------------
Teljes modell:
    value ~ C(group) + C(pipeline) + (1|subject)

Redukált modell:
    value ~ C(group) + (1|subject)

Primer omnibus teszt:
    Maximum likelihood likelihood-ratio test (df=2)
    a teljes és redukált modell között.

Primer multiplicitási család:
    10 tractus × 4 metrika = 40 omnibus teszt
    Benjamini–Hochberg FDR.

Szekunder modellalapú pairwise kontrasztok:
    P02 − P01
    P03 − P01
    P03 − P02

Szekunder multiplicitási család:
    40 biomarker × 3 kontraszt = 120 teszt
    Benjamini–Hochberg FDR.

Numerikus stabilitás:
    FA: modellillesztéshez ×10^2 skála
    AD/RD/MD: modellillesztéshez ×10^5 skála
    A koefficiensek az exportban eredeti és skálázott formában is szerepelnek.

Deskriptív hatásnagyság:
    Subject-szintű signed symmetric percent difference:
    200 × (P2 − P1) / (|P1| + |P2|)

Bemenet:
    data/all_pipelines_tractometry_features_master.csv

Kimenetek:
    outputs/rq2_pipeline_effect_v2/
        rq2_omnibus_pipeline_tests.csv
        rq2_pairwise_model_contrasts.csv
        rq2_signed_relative_changes.csv
        rq2_model_diagnostics.csv
        rq2_primary_metric_summary.csv
        rq2_primary_tract_summary.csv
        rq2_report.txt
"""

from pathlib import Path
from typing import Iterable
import math
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2, norm
from statsmodels.stats.multitest import multipletests


# =============================================================================
# 1. PATHS
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_DIR / "data" / "all_pipelines_tractometry_features_master.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "rq2_pipeline_effect_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. ANALYSIS SETTINGS
# =============================================================================

PRIMARY_FEATURE = "mean"

GROUP_ORDER = ["CN", "AD"]
PIPELINE_ORDER = ["pipeline_01", "pipeline_02", "pipeline_03"]

PIPELINE_PAIRS = [
    ("pipeline_01", "pipeline_02", "P02-P01"),
    ("pipeline_01", "pipeline_03", "P03-P01"),
    ("pipeline_02", "pipeline_03", "P03-P02"),
]

FULL_FORMULA = "analysis_value ~ C(group) + C(pipeline)"
REDUCED_FORMULA = "analysis_value ~ C(group)"

METRIC_SCALE_FACTORS = {
    "FA": 1e2,
    "AD": 1e5,
    "RD": 1e5,
    "MD": 1e5,
}

OPTIMIZERS = ["lbfgs", "powell", "cg"]
MAXITER = 1000
ALPHA = 0.05


# =============================================================================
# 3. HELPERS
# =============================================================================

def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(df.columns))
    if missing:
        raise ValueError("Hiányzó oszlop(ok): " + ", ".join(missing))


def fit_mixed_model(formula: str, sub: pd.DataFrame):
    """
    ML MixedLM random subject intercepttel.

    Több optimizerrel próbálkozik. Minden warningot és hibát rögzít,
    de nem nyom el csendben.
    """
    attempts = []

    for method in OPTIMIZERS:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            try:
                model = smf.mixedlm(
                    formula,
                    data=sub,
                    groups=sub["subject"],
                )

                fit = model.fit(
                    reml=False,
                    method=method,
                    maxiter=MAXITER,
                    disp=False,
                )

                warning_messages = [str(w.message) for w in caught]

                attempts.append({
                    "optimizer": method,
                    "fit": fit,
                    "warnings": warning_messages,
                    "error": "",
                })

                if bool(fit.converged) and np.isfinite(fit.llf):
                    return fit, method, warning_messages, attempts

            except Exception as exc:
                attempts.append({
                    "optimizer": method,
                    "fit": None,
                    "warnings": [str(w.message) for w in caught],
                    "error": str(exc),
                })

    # Ha egyik optimizer sem jelzett konvergenciát, de van véges illesztés,
    # azt visszaadjuk, egyértelmű diagnosztikai jelöléssel.
    for attempt in attempts:
        fit = attempt["fit"]
        if fit is not None and np.isfinite(fit.llf):
            return (
                fit,
                attempt["optimizer"],
                attempt["warnings"],
                attempts,
            )

    return None, "", [], attempts


def warnings_to_text(messages) -> str:
    unique = []
    for message in messages:
        if message not in unique:
            unique.append(message)
    return " | ".join(unique)


def attempts_to_text(attempts) -> str:
    parts = []

    for attempt in attempts:
        fit = attempt["fit"]
        converged = (
            bool(fit.converged)
            if fit is not None
            else False
        )

        error = attempt["error"]
        warning_text = warnings_to_text(attempt["warnings"])

        parts.append(
            f"{attempt['optimizer']}: "
            f"converged={converged}; "
            f"error={error or 'none'}; "
            f"warnings={warning_text or 'none'}"
        )

    return " || ".join(parts)


def fixed_effect_covariance(fit: object) -> pd.DataFrame:
    fe_names = list(fit.fe_params.index)
    covariance = fit.cov_params()
    return covariance.loc[fe_names, fe_names]


def linear_contrast(
    fit: object,
    weights: dict,
) -> dict:
    """
    Lineáris kontraszt a fixed-effect koefficiensekből.
    """
    fe = fit.fe_params
    covariance = fixed_effect_covariance(fit)
    names = list(fe.index)

    contrast_vector = np.array(
        [weights.get(name, 0.0) for name in names],
        dtype=float,
    )

    estimate = float(contrast_vector @ fe.to_numpy())
    variance = float(
        contrast_vector
        @ covariance.to_numpy()
        @ contrast_vector
    )

    variance = max(variance, 0.0)
    std_error = math.sqrt(variance)

    if std_error > 0:
        z_value = estimate / std_error
        p_value = 2.0 * norm.sf(abs(z_value))
    else:
        z_value = np.nan
        p_value = np.nan

    ci_low = estimate - 1.96 * std_error
    ci_high = estimate + 1.96 * std_error

    return {
        "estimate": estimate,
        "std_error": std_error,
        "z_value": z_value,
        "p_value": p_value,
        "ci_low_95": ci_low,
        "ci_high_95": ci_high,
    }


def fdr_bh(series: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=series.index, dtype=float)
    mask = series.notna()

    if mask.sum() > 0:
        result.loc[mask] = multipletests(
            series.loc[mask].to_numpy(),
            method="fdr_bh",
        )[1]

    return result


def hu(value, digits=3) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}".replace(".", ",")


# =============================================================================
# 4. LOAD AND AUDIT DATA
# =============================================================================

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"Nem található a master CSV:\n{INPUT_CSV}"
    )

df = pd.read_csv(INPUT_CSV)

required_columns = [
    "subject",
    "group",
    "pipeline",
    "tract",
    "metric",
    PRIMARY_FEATURE,
]

require_columns(df, required_columns)

df["group"] = pd.Categorical(
    df["group"],
    categories=GROUP_ORDER,
    ordered=True,
)

df["pipeline"] = pd.Categorical(
    df["pipeline"],
    categories=PIPELINE_ORDER,
    ordered=True,
)

if df[required_columns].isna().any().any():
    raise ValueError(
        "Hiányzó érték található a primer RQ2 elemzéshez szükséges oszlopokban."
    )

duplicate_mask = df.duplicated(
    ["subject", "pipeline", "tract", "metric"],
    keep=False,
)

if duplicate_mask.any():
    raise ValueError(
        "Duplikált subject × pipeline × tract × metric kombináció található."
    )

tracts = sorted(df["tract"].astype(str).unique())
metrics = sorted(df["metric"].astype(str).unique())

expected_rows = (
    df["subject"].nunique()
    * len(PIPELINE_ORDER)
    * len(tracts)
    * len(metrics)
)

if len(df) != expected_rows:
    raise ValueError(
        "A master adatbázis nem teljesen kiegyensúlyozott. "
        f"Várt sorok: {expected_rows}; tényleges sorok: {len(df)}."
    )


# =============================================================================
# 5. PRIMARY OMNIBUS MODELS AND PAIRWISE MODEL CONTRASTS
# =============================================================================

omnibus_rows = []
contrast_rows = []
diagnostic_rows = []

term_p02 = "C(pipeline)[T.pipeline_02]"
term_p03 = "C(pipeline)[T.pipeline_03]"

for tract in tracts:
    for metric in metrics:
        sub = df.loc[
            (df["tract"].astype(str) == tract)
            & (df["metric"].astype(str) == metric)
        ].copy()

        scale_factor = METRIC_SCALE_FACTORS.get(metric, 1.0)
        sub["analysis_value"] = (
            pd.to_numeric(sub[PRIMARY_FEATURE], errors="coerce")
            * scale_factor
        )

        full_fit, full_optimizer, full_warnings, full_attempts = fit_mixed_model(
            FULL_FORMULA,
            sub,
        )

        reduced_fit, reduced_optimizer, reduced_warnings, reduced_attempts = fit_mixed_model(
            REDUCED_FORMULA,
            sub,
        )

        model_ok = full_fit is not None and reduced_fit is not None

        if model_ok:
            lr_statistic = max(
                0.0,
                2.0 * (full_fit.llf - reduced_fit.llf),
            )
            lr_df = 2
            omnibus_p = chi2.sf(lr_statistic, lr_df)

            random_variance_scaled = float(full_fit.cov_re.iloc[0, 0])
            residual_variance_scaled = float(full_fit.scale)
            random_variance = (
                random_variance_scaled / (scale_factor ** 2)
            )
            residual_variance = (
                residual_variance_scaled / (scale_factor ** 2)
            )
            variance_total = random_variance + residual_variance

            random_effect_fraction = (
                random_variance / variance_total
                if variance_total > 0
                else np.nan
            )

            omnibus_rows.append({
                "tract": tract,
                "metric": metric,
                "feature": PRIMARY_FEATURE,
                "full_formula": FULL_FORMULA,
                "reduced_formula": REDUCED_FORMULA,
                "lr_statistic": lr_statistic,
                "lr_df": lr_df,
                "p_value": omnibus_p,
                "n_observations": int(len(sub)),
                "n_subjects": int(sub["subject"].nunique()),
                "full_converged": bool(full_fit.converged),
                "reduced_converged": bool(reduced_fit.converged),
            })

            contrast_definitions = [
                (
                    "P02-P01",
                    {
                        term_p02: 1.0,
                    },
                ),
                (
                    "P03-P01",
                    {
                        term_p03: 1.0,
                    },
                ),
                (
                    "P03-P02",
                    {
                        term_p02: -1.0,
                        term_p03: 1.0,
                    },
                ),
            ]

            for contrast_name, weights in contrast_definitions:
                result = linear_contrast(
                    full_fit,
                    weights,
                )

                contrast_rows.append({
                    "tract": tract,
                    "metric": metric,
                    "feature": PRIMARY_FEATURE,
                    "contrast": contrast_name,
                    "scale_factor": scale_factor,
                    "estimate_scaled": result["estimate"],
                    "std_error_scaled": result["std_error"],
                    "ci_low_95_scaled": result["ci_low_95"],
                    "ci_high_95_scaled": result["ci_high_95"],
                    "estimate": result["estimate"] / scale_factor,
                    "std_error": result["std_error"] / scale_factor,
                    "z_value": result["z_value"],
                    "p_value": result["p_value"],
                    "ci_low_95": result["ci_low_95"] / scale_factor,
                    "ci_high_95": result["ci_high_95"] / scale_factor,
                    "n_observations": int(len(sub)),
                    "n_subjects": int(sub["subject"].nunique()),
                })

            diagnostic_rows.append({
                "tract": tract,
                "metric": metric,
                "feature": PRIMARY_FEATURE,
                "model_ok": True,
                "full_optimizer": full_optimizer,
                "reduced_optimizer": reduced_optimizer,
                "full_converged": bool(full_fit.converged),
                "reduced_converged": bool(reduced_fit.converged),
                "full_llf": float(full_fit.llf),
                "reduced_llf": float(reduced_fit.llf),
                "full_aic": float(full_fit.aic),
                "full_bic": float(full_fit.bic),
                "scale_factor": scale_factor,
                "random_intercept_variance_scaled": random_variance_scaled,
                "residual_variance_scaled": residual_variance_scaled,
                "random_intercept_variance": random_variance,
                "residual_variance": residual_variance,
                "random_effect_fraction": random_effect_fraction,
                "full_warnings": warnings_to_text(full_warnings),
                "reduced_warnings": warnings_to_text(reduced_warnings),
                "full_attempt_log": attempts_to_text(full_attempts),
                "reduced_attempt_log": attempts_to_text(reduced_attempts),
                "error": "",
            })

        else:
            omnibus_rows.append({
                "tract": tract,
                "metric": metric,
                "feature": PRIMARY_FEATURE,
                "full_formula": FULL_FORMULA,
                "reduced_formula": REDUCED_FORMULA,
                "lr_statistic": np.nan,
                "lr_df": 2,
                "p_value": np.nan,
                "n_observations": int(len(sub)),
                "n_subjects": int(sub["subject"].nunique()),
                "full_converged": False,
                "reduced_converged": False,
            })

            diagnostic_rows.append({
                "tract": tract,
                "metric": metric,
                "feature": PRIMARY_FEATURE,
                "model_ok": False,
                "full_optimizer": full_optimizer,
                "reduced_optimizer": reduced_optimizer,
                "full_converged": False,
                "reduced_converged": False,
                "full_llf": np.nan,
                "reduced_llf": np.nan,
                "full_aic": np.nan,
                "full_bic": np.nan,
                "scale_factor": scale_factor,
                "random_intercept_variance_scaled": np.nan,
                "residual_variance_scaled": np.nan,
                "random_intercept_variance": np.nan,
                "residual_variance": np.nan,
                "random_effect_fraction": np.nan,
                "full_warnings": warnings_to_text(full_warnings),
                "reduced_warnings": warnings_to_text(reduced_warnings),
                "full_attempt_log": attempts_to_text(full_attempts),
                "reduced_attempt_log": attempts_to_text(reduced_attempts),
                "error": "A teljes vagy redukált modell nem volt illeszthető.",
            })


omnibus_df = pd.DataFrame(omnibus_rows)
contrast_df = pd.DataFrame(contrast_rows)
diagnostics_df = pd.DataFrame(diagnostic_rows)

# Primer FDR family: 40 omnibus teszt.
omnibus_df["p_fdr_primary_40"] = fdr_bh(
    omnibus_df["p_value"]
)
omnibus_df["fdr_significant_primary"] = (
    omnibus_df["p_fdr_primary_40"] < ALPHA
)

# Szekunder FDR family: 120 pairwise modellkontraszt.
contrast_df["p_fdr_pairwise_120"] = fdr_bh(
    contrast_df["p_value"]
)
contrast_df["fdr_significant_pairwise"] = (
    contrast_df["p_fdr_pairwise_120"] < ALPHA
)

# Másodlagos, metrikán belüli FDR is exportálásra kerül,
# de a fő kéziratban a 120 tesztes family legyen az elsődleges.
contrast_df["p_fdr_within_metric_30"] = np.nan

for metric in metrics:
    mask = contrast_df["metric"].astype(str) == metric
    contrast_df.loc[
        mask,
        "p_fdr_within_metric_30",
    ] = fdr_bh(
        contrast_df.loc[mask, "p_value"]
    )

contrast_df["fdr_significant_within_metric"] = (
    contrast_df["p_fdr_within_metric_30"] < ALPHA
)


# =============================================================================
# 6. SUBJECT-LEVEL SIGNED SYMMETRIC RELATIVE CHANGES
# =============================================================================

relative_rows = []

for tract in tracts:
    for metric in metrics:
        sub = df.loc[
            (df["tract"].astype(str) == tract)
            & (df["metric"].astype(str) == metric)
        ].copy()

        wide = sub.pivot(
            index="subject",
            columns="pipeline",
            values=PRIMARY_FEATURE,
        )

        for p1, p2, contrast_name in PIPELINE_PAIRS:
            x1 = pd.to_numeric(wide[p1], errors="coerce")
            x2 = pd.to_numeric(wide[p2], errors="coerce")

            valid = x1.notna() & x2.notna()

            x1 = x1.loc[valid]
            x2 = x2.loc[valid]

            raw_difference = x2 - x1
            denominator = x1.abs() + x2.abs()

            signed_symmetric_percent = (
                200.0
                * raw_difference
                / denominator.replace(0, np.nan)
            )

            relative_rows.append({
                "tract": tract,
                "metric": metric,
                "feature": PRIMARY_FEATURE,
                "contrast": contrast_name,
                "pipeline_1": p1,
                "pipeline_2": p2,
                "n_subjects": int(signed_symmetric_percent.notna().sum()),
                "mean_raw_difference": float(raw_difference.mean()),
                "median_raw_difference": float(raw_difference.median()),
                "sd_raw_difference": float(raw_difference.std(ddof=1)),
                "mean_signed_symmetric_percent": float(
                    signed_symmetric_percent.mean()
                ),
                "median_signed_symmetric_percent": float(
                    signed_symmetric_percent.median()
                ),
                "sd_signed_symmetric_percent": float(
                    signed_symmetric_percent.std(ddof=1)
                ),
                "q1_signed_symmetric_percent": float(
                    signed_symmetric_percent.quantile(0.25)
                ),
                "q3_signed_symmetric_percent": float(
                    signed_symmetric_percent.quantile(0.75)
                ),
                "min_signed_symmetric_percent": float(
                    signed_symmetric_percent.min()
                ),
                "max_signed_symmetric_percent": float(
                    signed_symmetric_percent.max()
                ),
            })

relative_df = pd.DataFrame(relative_rows)


# =============================================================================
# 7. SUMMARIES
# =============================================================================

metric_summary = (
    omnibus_df.groupby("metric", observed=True)
    .agg(
        n_biomarkers=("p_value", "size"),
        n_omnibus_fdr_significant=(
            "fdr_significant_primary",
            "sum",
        ),
        median_lr_statistic=("lr_statistic", "median"),
        min_omnibus_p_fdr=("p_fdr_primary_40", "min"),
    )
    .reset_index()
)

metric_summary["percent_omnibus_fdr_significant"] = (
    100.0
    * metric_summary["n_omnibus_fdr_significant"]
    / metric_summary["n_biomarkers"]
)

pairwise_metric_summary = (
    contrast_df.groupby(
        ["metric", "contrast"],
        observed=True,
    )
    .agg(
        n_biomarkers=("p_value", "size"),
        n_pairwise_fdr_significant=(
            "fdr_significant_pairwise",
            "sum",
        ),
        median_estimate=("estimate", "median"),
        median_ci_width=(
            "ci_high_95",
            lambda values: np.nan,
        ),
        min_pairwise_p_fdr=(
            "p_fdr_pairwise_120",
            "min",
        ),
    )
    .reset_index()
)

# CI width külön számolva.
contrast_df["ci_width_95"] = (
    contrast_df["ci_high_95"]
    - contrast_df["ci_low_95"]
)

ci_width_summary = (
    contrast_df.groupby(
        ["metric", "contrast"],
        observed=True,
    )["ci_width_95"]
    .median()
    .rename("median_ci_width")
    .reset_index()
)

pairwise_metric_summary = (
    pairwise_metric_summary
    .drop(columns=["median_ci_width"])
    .merge(
        ci_width_summary,
        on=["metric", "contrast"],
        how="left",
    )
)

pairwise_metric_summary[
    "percent_pairwise_fdr_significant"
] = (
    100.0
    * pairwise_metric_summary[
        "n_pairwise_fdr_significant"
    ]
    / pairwise_metric_summary["n_biomarkers"]
)

relative_metric_summary = (
    relative_df.groupby(
        ["metric", "contrast"],
        observed=True,
    )
    .agg(
        median_of_tract_medians_percent=(
            "median_signed_symmetric_percent",
            "median",
        ),
        q1_of_tract_medians_percent=(
            "median_signed_symmetric_percent",
            lambda values: values.quantile(0.25),
        ),
        q3_of_tract_medians_percent=(
            "median_signed_symmetric_percent",
            lambda values: values.quantile(0.75),
        ),
    )
    .reset_index()
)

metric_summary = metric_summary.merge(
    pairwise_metric_summary.groupby(
        "metric",
        observed=True,
    )
    .agg(
        n_pairwise_tests=("n_biomarkers", "sum"),
        n_pairwise_fdr_significant=(
            "n_pairwise_fdr_significant",
            "sum",
        ),
    )
    .reset_index(),
    on="metric",
    how="left",
)

metric_summary["percent_pairwise_fdr_significant"] = (
    100.0
    * metric_summary["n_pairwise_fdr_significant"]
    / metric_summary["n_pairwise_tests"]
)

tract_summary = (
    omnibus_df.groupby("tract", observed=True)
    .agg(
        n_biomarkers=("p_value", "size"),
        n_omnibus_fdr_significant=(
            "fdr_significant_primary",
            "sum",
        ),
        min_omnibus_p_fdr=("p_fdr_primary_40", "min"),
    )
    .reset_index()
)

tract_summary["percent_omnibus_fdr_significant"] = (
    100.0
    * tract_summary["n_omnibus_fdr_significant"]
    / tract_summary["n_biomarkers"]
)


# =============================================================================
# 8. SAVE OUTPUTS
# =============================================================================

omnibus_df.to_csv(
    OUTPUT_DIR / "rq2_omnibus_pipeline_tests.csv",
    index=False,
)

contrast_df.to_csv(
    OUTPUT_DIR / "rq2_pairwise_model_contrasts.csv",
    index=False,
)

relative_df.to_csv(
    OUTPUT_DIR / "rq2_signed_relative_changes.csv",
    index=False,
)

diagnostics_df.to_csv(
    OUTPUT_DIR / "rq2_model_diagnostics.csv",
    index=False,
)

metric_summary.to_csv(
    OUTPUT_DIR / "rq2_primary_metric_summary.csv",
    index=False,
)

pairwise_metric_summary.to_csv(
    OUTPUT_DIR / "rq2_pairwise_metric_summary.csv",
    index=False,
)

relative_metric_summary.to_csv(
    OUTPUT_DIR / "rq2_relative_change_metric_summary.csv",
    index=False,
)

tract_summary.to_csv(
    OUTPUT_DIR / "rq2_primary_tract_summary.csv",
    index=False,
)


# =============================================================================
# 9. REPORT
# =============================================================================

n_omnibus = len(omnibus_df)
n_omnibus_sig = int(
    omnibus_df["fdr_significant_primary"].sum()
)

n_pairwise = len(contrast_df)
n_pairwise_sig = int(
    contrast_df["fdr_significant_pairwise"].sum()
)

failed_models = int(
    (~diagnostics_df["model_ok"]).sum()
)

nonconverged_full = int(
    (~diagnostics_df["full_converged"]).sum()
)

nonconverged_reduced = int(
    (~diagnostics_df["reduced_converged"]).sum()
)

report_lines = []

report_lines.append("RQ2 PIPELINE EFFECT REPORT – REVISED PRIMARY ANALYSIS")
report_lines.append("=" * 88)
report_lines.append("")
report_lines.append("Kutatási kérdés:")
report_lines.append(
    "Megváltoztatja-e a preprocessing pipeline az abszolút DTI-értékeket?"
)
report_lines.append("")
report_lines.append("Primer feature:")
report_lines.append(PRIMARY_FEATURE)
report_lines.append("")
report_lines.append("Primer teljes modell:")
report_lines.append("mean ~ C(group) + C(pipeline) + (1|subject)")
report_lines.append("")
report_lines.append("Primer redukált modell:")
report_lines.append("mean ~ C(group) + (1|subject)")
report_lines.append("")
report_lines.append("Primer teszt:")
report_lines.append(
    "ML likelihood-ratio omnibus pipeline-teszt, df=2; "
    "BH-FDR a 40 tractus × metrika tesztre."
)
report_lines.append(
    "Numerikus stabilitás: FA ×10^2, AD/RD/MD ×10^5 skálán; "
    "az exportált fő becslések eredeti egységre visszaalakítva."
)
report_lines.append("")
report_lines.append("PRIMER OMNIBUS EREDMÉNY")
report_lines.append("-" * 88)
report_lines.append(
    f"FDR-szignifikáns biomarkerek: "
    f"{n_omnibus_sig}/{n_omnibus} "
    f"({100*n_omnibus_sig/n_omnibus:.1f}%)"
)
report_lines.append("")

for _, row in metric_summary.sort_values(
    "percent_omnibus_fdr_significant",
    ascending=False,
).iterrows():
    report_lines.append(
        f"{row['metric']}: "
        f"{int(row['n_omnibus_fdr_significant'])}/"
        f"{int(row['n_biomarkers'])} "
        f"({row['percent_omnibus_fdr_significant']:.1f}%), "
        f"min FDR={row['min_omnibus_p_fdr']:.3g}"
    )

report_lines.append("")
report_lines.append("SZEKUNDER MODELLALAPÚ PAIRWISE KONTRASZTOK")
report_lines.append("-" * 88)
report_lines.append(
    f"FDR-szignifikáns kontrasztok: "
    f"{n_pairwise_sig}/{n_pairwise} "
    f"({100*n_pairwise_sig/n_pairwise:.1f}%)"
)
report_lines.append(
    "FDR family: mind a 120 pairwise modellkontraszt együtt."
)
report_lines.append("")

for _, row in pairwise_metric_summary.sort_values(
    ["metric", "contrast"]
).iterrows():
    report_lines.append(
        f"{row['metric']} {row['contrast']}: "
        f"{int(row['n_pairwise_fdr_significant'])}/"
        f"{int(row['n_biomarkers'])} "
        f"({row['percent_pairwise_fdr_significant']:.1f}%), "
        f"median estimate={row['median_estimate']:.6g}, "
        f"min FDR={row['min_pairwise_p_fdr']:.3g}"
    )

report_lines.append("")
report_lines.append("DESKRIPTÍV SIGNED SZIMMETRIKUS RELATÍV ELTÉRÉSEK")
report_lines.append("-" * 88)

for _, row in relative_metric_summary.sort_values(
    ["metric", "contrast"]
).iterrows():
    report_lines.append(
        f"{row['metric']} {row['contrast']}: "
        f"tract-mediánok mediánja="
        f"{row['median_of_tract_medians_percent']:.2f}%, "
        f"IQR="
        f"{row['q1_of_tract_medians_percent']:.2f}–"
        f"{row['q3_of_tract_medians_percent']:.2f}%"
    )

report_lines.append("")
report_lines.append("MODELLDIAGNOSZTIKA")
report_lines.append("-" * 88)
report_lines.append(f"Sikertelen modellpárok: {failed_models}/40")
report_lines.append(
    f"Nem konvergált teljes modellek: {nonconverged_full}/40"
)
report_lines.append(
    f"Nem konvergált redukált modellek: {nonconverged_reduced}/40"
)
report_lines.append(
    "A warningok, optimizer-próbálkozások, random intercept variancia "
    "és reziduális variancia a rq2_model_diagnostics.csv fájlban szerepelnek."
)

report_lines.append("")
report_lines.append("ÉRTELMEZÉSI KERET")
report_lines.append("-" * 88)
report_lines.append(
    "Az omnibus teszt azt vizsgálja, hogy a diagnosztikai csoportra "
    "korrigálva van-e bármilyen általános különbség a három pipeline között."
)
report_lines.append(
    "A pairwise modellkontrasztok azt mutatják, hogy mely pipeline-párok "
    "között, milyen irányú és mekkora eltérés becsülhető."
)
report_lines.append(
    "A signed szimmetrikus relatív eltérések deskriptív, subject-szintű "
    "hatásnagyságok; nem helyettesítik a mixed-model inferenciát."
)
report_lines.append(
    "A Group × Pipeline interakció nem része az RQ2 primer modelljének; "
    "annak vizsgálata és biológiai értelmezése az RQ3 feladata."
)

report_lines.append("")
report_lines.append("KIMENETI FÁJLOK")
report_lines.append("-" * 88)

output_names = [
    "rq2_model_diagnostics.csv",
    "rq2_omnibus_pipeline_tests.csv",
    "rq2_pairwise_metric_summary.csv",
    "rq2_pairwise_model_contrasts.csv",
    "rq2_primary_metric_summary.csv",
    "rq2_primary_tract_summary.csv",
    "rq2_relative_change_metric_summary.csv",
    "rq2_signed_relative_changes.csv",
    "rq2_report.txt",
]
for name in output_names:
    report_lines.append(f"- {name}")

report_text = "\n".join(report_lines)

with open(
    OUTPUT_DIR / "rq2_report.txt",
    "w",
    encoding="utf-8",
) as report_file:
    report_file.write(report_text)

print(report_text)
