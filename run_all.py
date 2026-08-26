# -*- coding: utf-8 -*-
"""
run_all.py
Reproduce the full analysis end-to-end from the master data file.

Usage:
    python run_all.py                  # full pipeline (00 -> 12)
    python run_all.py --recompute-only # NumPy-only reconstruction + figures
                                       # (07, 08, 09, 12, 10, 11)

Stages 00-06 (QC, feature screening, RQ1-RQ4) require scipy / statsmodels /
scikit-learn. Stages 07-12 (variance decomposition, ICC absolute agreement,
TOST equivalence, common-mode test, and all figures) require only
numpy / pandas / matplotlib and provide an independent reproduction path.
"""
import sys
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parent
CODE = BASE / "code"

PRIMARY = [
    "00_qc_master_csv.py",
    "01_feature_screening.py",
    "02_rq1_reproducibility.py",
    "03_rq2_pipeline_effect.py",
    "04_rq3_biological_robustness.py",
    "05_rq4_disease_vs_pipeline_effect.py",
    "06_rq4_diagnostic_utility.py",
]
ANALYSIS = [
    "07_variance_decomposition.py",
    "08_icc_absolute_agreement.py",
    "09_equivalence_tost.py",
    "12_common_mode.py",
    "13_metric_deviation_summary.py",
    "14_pipeline_value_table.py",
    "15_gtheory.py",
]
FIGURES = [
    "10_make_figures.py",
    "11_make_extra_figures.py",
]


def run(scripts):
    for s in scripts:
        print(f"\n{'='*70}\n>>> {s}\n{'='*70}")
        r = subprocess.run([sys.executable, str(CODE / s)], cwd=str(BASE))
        if r.returncode != 0:
            print(f"!!! {s} exited with code {r.returncode}")
            sys.exit(r.returncode)


def main():
    recompute_only = "--recompute-only" in sys.argv
    scripts = (ANALYSIS + FIGURES) if recompute_only else (PRIMARY + ANALYSIS + FIGURES)
    run(scripts)
    print("\nDONE. Outputs in outputs/, figures in figures/.")
    print("Verify with:  python code/verify_reproduction.py")


if __name__ == "__main__":
    main()
