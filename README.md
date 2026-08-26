# DTI preprocessing-pipeline robustness in Alzheimer's disease — reproducible analysis

This repository contains the complete, end-to-end reproducible analysis code for
the study of how diffusion-MRI preprocessing pipeline choice affects the
reproducibility, biological interpretation, and diagnostic utility of
tractometry biomarkers in Alzheimer's disease (AD vs cognitively normal, CN).

Everything in `outputs/` and `figures/` can be regenerated from the single
master data file with `python run_all.py`.

## Study design (data structure)

| Factor | Levels |
|---|---|
| Subjects | 30 (15 AD, 15 CN), within-subject design |
| Pipelines | 3 (`pipeline_01` reference, `pipeline_02`, `pipeline_03`) |
| Tracts | 10 (CC_2, CC_7, CG L/R, IFO L/R, ILF L/R, UF L/R) |
| Metrics | 4 (FA, MD, RD, AD) |
| Features | 6 per tract (mean, median, sd, proximal/middle/distal mean) |
| Biomarker | one tract × metric (40 total); primary feature = tract `mean` |

`data/all_pipelines_tractometry_features_master.csv` = 30×3×10×4 = 3,600 rows.

> **Metric naming:** in the data columns `AD` denotes **axial diffusivity**; the manuscript labels this **AxD** to avoid collision with Alzheimer's disease (AD).

## Repository layout

```
imaging_neuroscience_repro/
├── data/         master CSV (+ DATA_AVAILABILITY note)
├── pipeline/     preprocessing scripts (DICOM → preprocessed DWI → TractSeg tractometry)
├── code/         analysis scripts (00–15) + statlib.py
├── outputs/      generated CSVs and text reports (per research question)
├── figures/      generated figures (PNG 600 dpi / PDF / SVG)
├── run_all.py    runs the whole pipeline (00 → 15)
└── requirements.txt
```

## Preprocessing pipeline (DICOM → tractometry)

The `pipeline/` folder contains the upstream preprocessing that turns ADNI
diffusion DICOMs into the per-subject tractometry CSVs aggregated in
`data/all_pipelines_tractometry_features_master.csv`. Three pipelines of
increasing complexity are built:

- **P01** (minimal): DICOM→NIfTI (dcm2niix), tensor fit (MRtrix3), TractSeg tractometry.
- **P02**: adds FSL topup + eddy distortion/motion correction, FSL BET mask, FSL dtifit.
- **P03**: adds MP-PCA denoising + Gibbs-ringing removal (MRtrix3), topup+eddy via the
  `dwifslpreproc` wrapper, ANTs N4 bias-field correction, and MRtrix3 tensor fit.

External tools required: **dcm2niix, FSL** (topup, eddy, BET, dtifit), **MRtrix3**
(dwidenoise, mrdegibbs, dwifslpreproc, dwi2tensor, tensor2metric), **ANTs** (N4),
and **TractSeg**. Exact tool versions are reported in the manuscript Methods. Each
script sets a project root at the top (`PROJECT=...`) — edit it to your local path.
These steps require the ADNI source images (controlled access) and are provided
for transparency and reuse, not as a turnkey rerun.

## Analysis scripts

| Script | Purpose | Dependencies |
|---|---|---|
| `00_qc_master_csv.py` | data QC (completeness, duplicates, sanity) | numpy, pandas |
| `01_feature_screening.py` | select primary feature (tract mean) | numpy, pandas, scipy |
| `02_rq1_reproducibility.py` | RQ1: ICC(3,1), CV, correlations, Bland–Altman | scipy |
| `03_rq2_pipeline_effect.py` | RQ2: linear mixed models, omnibus + pairwise | statsmodels |
| `04_rq3_biological_robustness.py` | RQ3: Group×Pipeline interaction, effect-size & ranking stability | scipy |
| `05_rq4_disease_vs_pipeline_effect.py` | RQ4 support analysis | scipy |
| `06_rq4_diagnostic_utility.py` | RQ4: ROC-AUC per pipeline + bootstrap CI | scikit-learn |
| **`07_variance_decomposition.py`** | variance partition (η², biological vs pipeline) | **numpy only** |
| **`08_icc_absolute_agreement.py`** | RQ1 supplement: ICC(2,1) absolute agreement | **numpy only** |
| **`09_equivalence_tost.py`** | TOST equivalence (effect size + AUC) | **numpy only** |
| `10_make_figures.py` | all figures, CSV-driven | numpy, pandas, matplotlib |
| `11_make_extra_figures.py` | Fig 2 (reproducibility: ICC/CV, Bland–Altman), S1 (feature screening), S3 (Group×Pipeline interaction) | numpy, pandas, matplotlib |
| **`12_common_mode.py`** | direct common-mode test (pipeline shift in AD vs CN) + figure | **numpy only** |
| **`13_metric_deviation_summary.py`** | per-metric pipeline-deviation summary table (RQ1/RQ2/variance) | **numpy only** |
| **`14_pipeline_value_table.py`** | per-pipeline absolute-value table (manuscript Table) | numpy, pandas |
| **`15_gtheory.py`** | generalizability theory: relative G(1) & absolute Φ(k), decision study; Fig 8 | numpy, pandas, matplotlib |
| `figstyle.py` | shared publication figure style (fonts, despined axes) | matplotlib |
| `verify_reproduction.py` | audits the reproduction; writes `VERIFICATION.md` | numpy, pandas |
| `statlib.py` | dependency-light statistics (incomplete-beta → t/F CDFs, ICC, Hedges g, SS partition, TOST) | **numpy only** |

Scripts **07–13, `figstyle.py` and `statlib.py` were newly written for this release** to make
the previously pickle-bound results (variance decomposition, TOST, ICC absolute
agreement) fully reproducible. They depend only on NumPy/pandas/matplotlib and
provide an **independent, external-package-free reproduction** of the headline
statistics — an internal cross-check of the SciPy/statsmodels/scikit-learn
scripts (01–06).

## How to reproduce

```bash
pip install -r requirements.txt
python run_all.py                 # full pipeline (00 → 15)
# or, dependency-light reconstruction + figures only:
python run_all.py --recompute-only   # needs only numpy, pandas, matplotlib
```

## Headline results (regenerated, verified)

- **Variance decomposition** (`outputs/variance_decomposition/`): biological
  (individual + disease) ≈ **90.9 %**, pipeline ≈ **5.2 %**, Group×Pipeline ≈
  **0.14 %**, residual ≈ **3.8 %**. FA most pipeline-sensitive (≈11 %), MD/RD
  most robust (≈1–2 %).
- **Reproducibility** (RQ1): mean consistency ICC(3,1) = **0.94**; absolute
  agreement ICC(2,1) = **0.87** (FA 0.92→0.78, AD 0.93→0.84; MD/RD unchanged);
  4/40 biomarkers below 0.75 under absolute agreement, 0/40 under consistency.
- **Pipeline effect on absolute values** (RQ2): omnibus FDR-significant for
  **34/40** biomarkers; FA shifts most (P03 vs P01 median −5.7 %).
- **Per-metric deviation magnitude** (`outputs/metric_deviation/`): mean |Δ| per
  pipeline change — FA **3.9 %** (var 11 %), AD 2.0 %, RD 1.9 %, MD **0.7 %**
  (var 1 %); FA least transferable, MD most stable.
- **Biological robustness** (RQ3): **0/40** significant Group×Pipeline
  interactions; effect direction same in 87.5 %; ranking ρ = 0.90.
- **Common-mode** (`outputs/common_mode/`): the pipeline-induced shift is near-identical in AD and CN (Pearson r = **0.94**, slope ≈ **1.14**; standardized differential equivalent within ±0.1) — additive and group-independent.
- **Equivalence** (TOST, `outputs/equivalence_tost/`): aggregate Δ Hedges g
  equivalent at the strict ±0.1 margin on all three pipeline pairs; Δ AUC
  equivalent at ±0.05.
- **Diagnostic utility** (RQ4): mean AUC range across pipelines = **0.045**;
  38/40 biomarkers vary ≤ 0.10.

## Data availability

The data are derived from ADNI (Alzheimer's Disease Neuroimaging Initiative) and
are subject to the ADNI Data Use Agreement; they cannot be redistributed openly.
See `data/DATA_AVAILABILITY.md`. All analysis **code** in
this repository is openly available.

## License

Code is released under the MIT License (`LICENSE`). The data are governed by the
ADNI Data Use Agreement and are **not** covered by this license.
