#!/bin/bash

set -e

PROJECT="/path/to/adni_project"  # <-- EDIT: set to your project root
PIPELINE="$PROJECT/pipeline_02"
QC="$PROJECT/qc"
LOGDIR="$PROJECT/logs"

mkdir -p "$QC" "$LOGDIR"

OUTCSV="$QC/02_02_topup_qc.csv"
WARN="$LOGDIR/02_02_topup_warnings.log"

echo "subject,b0_pair_exists,topup_fieldcoef,topup_movpar,topup_corrected_b0,corrected_b0_size" > "$OUTCSV"
> "$WARN"

for subject_dir in "$PIPELINE"/*; do

    [ -d "$subject_dir" ] || continue

    SUBJECT=$(basename "$subject_dir")
    PREPROC="$subject_dir/preproc"

    B0PAIR="$PREPROC/b0_pair_PA_AP.nii.gz"
    ACQP="$PREPROC/acqparams.txt"

    echo "Running TOPUP: $SUBJECT"

    if [ ! -f "$B0PAIR" ] || [ ! -f "$ACQP" ]; then
        echo "WARNING: Missing b0_pair or acqparams for $SUBJECT" >> "$WARN"
        echo "$SUBJECT,no,no,no,no,NA" >> "$OUTCSV"
        continue
    fi

    cd "$PREPROC"

    topup \
        --imain=b0_pair_PA_AP.nii.gz \
        --datain=acqparams.txt \
        --config=b02b0.cnf \
        --subsamp=1 \
        --out=topup_results \
        --iout=topup_corrected_b0

    fieldcoef="no"
    movpar="no"
    corrected="no"
    corrected_size="NA"

    [ -f "$PREPROC/topup_results_fieldcoef.nii.gz" ] && fieldcoef="yes"
    [ -f "$PREPROC/topup_results_movpar.txt" ] && movpar="yes"
    [ -f "$PREPROC/topup_corrected_b0.nii.gz" ] && corrected="yes"

    if [ -f "$PREPROC/topup_corrected_b0.nii.gz" ]; then
        corrected_size=$(mrinfo "$PREPROC/topup_corrected_b0.nii.gz" -size | tr ' ' 'x')
    else
        echo "WARNING: TOPUP corrected b0 missing for $SUBJECT" >> "$WARN"
    fi

    echo "$SUBJECT,yes,$fieldcoef,$movpar,$corrected,$corrected_size" >> "$OUTCSV"

done

echo "TOPUP finished."
echo "QC: $OUTCSV"
echo "Warnings: $WARN"
