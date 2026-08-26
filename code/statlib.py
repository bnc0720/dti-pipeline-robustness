# -*- coding: utf-8 -*-
"""
statlib.py
==========
Dependency-light statistical routines (NumPy only) used for the independent,
fully transparent reconstruction of the DTI pipeline-robustness statistics.

Rationale
---------
The headline analyses in this study (variance decomposition, ICC under both
consistency and absolute-agreement definitions, and TOST equivalence testing)
are re-implemented here from first principles using only NumPy, with no
dependence on SciPy, statsmodels or scikit-learn. This provides an
external-package-free reproduction path that any reviewer can run, and serves
as an independent cross-check of the primary analysis scripts (01-06), which
use SciPy / statsmodels / scikit-learn.

Every distributional p-value is obtained from the regularized incomplete beta
function (Numerical Recipes continued-fraction implementation), from which the
Student-t and Fisher-F CDFs are derived.

Author: reconstruction for Imaging Neuroscience submission (2026).
"""
from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------------------
# Regularized incomplete beta function  I_x(a, b)  -> Student-t / F CDFs
# ---------------------------------------------------------------------------
def _betacf(a: float, b: float, x: float, itmax: int = 400, eps: float = 3e-16) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    from math import lgamma, log, exp
    bt = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


# ---------------------------------------------------------------------------
# Student-t distribution
# ---------------------------------------------------------------------------
def t_cdf(t: float, df: float) -> float:
    """CDF of Student-t with df degrees of freedom."""
    x = df / (df + t * t)
    ib = 0.5 * betai(0.5 * df, 0.5, x)
    return 1.0 - ib if t > 0 else ib


def t_sf(t: float, df: float) -> float:
    """Upper-tail (survival) probability of Student-t."""
    return 1.0 - t_cdf(t, df)


def t_ppf(p: float, df: float) -> float:
    """Inverse CDF (quantile) of Student-t via bisection."""
    if p <= 0.0:
        return -np.inf
    if p >= 1.0:
        return np.inf
    lo, hi = -1e4, 1e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def t_two_sided_p(t: float, df: float) -> float:
    """Two-sided p-value for a t-statistic."""
    return 2.0 * t_sf(abs(t), df)


# ---------------------------------------------------------------------------
# Fisher-F distribution (omnibus tests / variance ratios)
# ---------------------------------------------------------------------------
def f_sf(f: float, df1: float, df2: float) -> float:
    """Upper-tail probability of the F distribution."""
    if f <= 0.0:
        return 1.0
    x = df2 / (df2 + df1 * f)
    return betai(0.5 * df2, 0.5 * df1, x)


# ---------------------------------------------------------------------------
# Ranks, Pearson & Spearman correlation with p-values
# ---------------------------------------------------------------------------
def rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties shared), 1-based, NumPy-only."""
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    sa = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def pearson_r_p(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    r = np.corrcoef(x, y)[0, 1]
    if n <= 2 or abs(r) >= 1.0:
        return r, 0.0 if abs(r) >= 1.0 else 1.0
    t = r * np.sqrt((n - 2) / (1 - r * r))
    return r, t_two_sided_p(t, n - 2)


def spearman_r_p(x: np.ndarray, y: np.ndarray):
    return pearson_r_p(rankdata(x), rankdata(y))


# ---------------------------------------------------------------------------
# Effect size: Hedges g (small-sample corrected standardized mean difference)
# ---------------------------------------------------------------------------
def hedges_g(group_a: np.ndarray, group_b: np.ndarray):
    """Hedges g for (mean_a - mean_b). Positive g => a > b."""
    a = np.asarray(group_a, float)
    b = np.asarray(group_b, float)
    na, nb = len(a), len(b)
    sp2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    sp = np.sqrt(sp2)
    if sp == 0:
        return 0.0
    d = (a.mean() - b.mean()) / sp
    J = 1.0 - 3.0 / (4.0 * (na + nb) - 9.0)
    return J * d


# ---------------------------------------------------------------------------
# Intraclass correlation: ICC(3,1) consistency and ICC(2,1) absolute agreement
# (two-way model, single rater/measurement; subjects x raters matrix)
# ---------------------------------------------------------------------------
def icc_two_way(Y: np.ndarray):
    """
    Y : array (n_subjects, k_raters).
    Returns (icc_3_1_consistency, icc_2_1_absolute_agreement).
    """
    Y = np.asarray(Y, float)
    n, k = Y.shape
    grand = Y.mean()
    row_means = Y.mean(axis=1)
    col_means = Y.mean(axis=0)
    SSR = k * ((row_means - grand) ** 2).sum()          # between subjects
    SSC = n * ((col_means - grand) ** 2).sum()           # between raters/pipelines
    SSE = ((Y - row_means[:, None] - col_means[None, :] + grand) ** 2).sum()
    MSR = SSR / (n - 1)
    MSC = SSC / (k - 1)
    MSE = SSE / ((n - 1) * (k - 1))
    icc31 = (MSR - MSE) / (MSR + (k - 1) * MSE)
    icc21 = (MSR - MSE) / (MSR + (k - 1) * MSE + (k / n) * (MSC - MSE))
    return icc31, icc21


# ---------------------------------------------------------------------------
# Variance decomposition (split-plot / repeated-measures sum-of-squares, eta^2)
# Factors: Disease (group, between) ; Individual = Subject(Group) ;
#          Pipeline (within) ; Group x Pipeline ; Residual = Subject(G) x Pipeline
# Balanced design assumed (equal n per group, equal pipelines per subject).
# ---------------------------------------------------------------------------
def variance_partition(Y: np.ndarray, group_vector: np.ndarray) -> dict:
    """
    Y            : (n_subjects, b_pipelines) values for one biomarker/feature.
    group_vector : (n_subjects,) group labels.
    Returns dict of sum-of-squares and percent-of-total (eta^2) for
    individual / disease / pipeline / interaction / residual.
    """
    Y = np.asarray(Y, float)
    g = np.asarray(group_vector)
    b = Y.shape[1]
    grand = Y.mean()
    SS_total = ((Y - grand) ** 2).sum()
    pipe_means = Y.mean(axis=0)
    SS_pipeline = Y.shape[0] * ((pipe_means - grand) ** 2).sum()
    SS_disease = SS_individual = SS_interaction = 0.0
    for lev in np.unique(g):
        idx = g == lev
        Yg = Y[idx]
        ng = idx.sum()
        gm = Yg.mean()
        SS_disease += b * ng * (gm - grand) ** 2
        subj_means = Yg.mean(axis=1)
        SS_individual += b * ((subj_means - gm) ** 2).sum()
        cell = Yg.mean(axis=0)
        SS_interaction += ng * ((cell - gm - pipe_means + grand) ** 2).sum()
    SS_residual = SS_total - SS_disease - SS_individual - SS_pipeline - SS_interaction
    out = dict(individual=SS_individual, disease=SS_disease, pipeline=SS_pipeline,
               interaction=SS_interaction, residual=SS_residual, total=SS_total)
    for key in ("individual", "disease", "pipeline", "interaction", "residual"):
        out[f"pct_{key}"] = 100.0 * out[key] / SS_total if SS_total > 0 else np.nan
    return out


# ---------------------------------------------------------------------------
# TOST equivalence test (one-sample, on a vector of differences) using Student-t
# ---------------------------------------------------------------------------
def tost_one_sample(values: np.ndarray, margin: float, alpha: float = 0.05) -> dict:
    """
    Two one-sided tests for equivalence of the mean of `values` to zero,
    within +/- margin. Reports the (1-2*alpha) CI used for the equivalence
    decision (90% CI for alpha=0.05).
    """
    v = np.asarray(values, float)
    n = len(v)
    mean = v.mean()
    sd = v.std(ddof=1)
    se = sd / np.sqrt(n)
    df = n - 1
    # one-sided tests
    t_lower = (mean - (-margin)) / se          # H0: mean <= -margin
    t_upper = (mean - margin) / se             # H0: mean >=  margin
    p_lower = t_sf(t_lower, df)                # reject if mean > -margin
    p_upper = t_cdf(t_upper, df)               # reject if mean <  margin
    p_tost = max(p_lower, p_upper)
    tcrit = t_ppf(1 - alpha, df)               # for (1-2alpha) CI
    ci_lo = mean - tcrit * se
    ci_hi = mean + tcrit * se
    equivalent = (ci_lo > -margin) and (ci_hi < margin)
    return dict(n=n, mean=mean, sd=sd, se=se, df=df,
                ci_lo=ci_lo, ci_hi=ci_hi, p_tost=p_tost,
                p_lower=p_lower, p_upper=p_upper,
                margin=margin, equivalent=bool(equivalent))


# ---------------------------------------------------------------------------
# ROC AUC (Mann-Whitney U normalization, direction-free) + bootstrap CI
# ---------------------------------------------------------------------------
def auc_oriented(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC = max(AUC, 1-AUC); labels in {0,1}, 1 = positive (AD)."""
    s = np.asarray(scores, float)
    y = np.asarray(labels)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    a = gt + 0.5 * eq
    return max(a, 1 - a)
