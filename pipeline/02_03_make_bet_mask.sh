#!/bin/bash

set -e

PROJECT="/path/to/adni_project"  # <-- EDIT: set to your project root
PIPELINE="$PROJECT/pipeline_02"
QC="$PROJECT/qc"
LOGDIR="$PROJECT/logs"

mkdir -p "$QC" "$LOGDIR"

OUTCSV="$QC/02_03_bet_mask_qc.csv"
WARN="$LOGDIR/02_03_bet_mask_warnings.log"

echo "subject,input_b0,mask_exists,mask_size" > "$OUTCSV"
> "$WARN"

for subject_dir in "$PIPELINE"/*; do
    [ -d "$subject_dir" ] || continue

    SUBJECT=$(basename "$subject_dir")
    PREPROC="$subject_dir/preproc"

    B0="$PREPROC/PA_b0_mean.nii.gz"

    echo "BET mask: $SUBJECT"

    if [ ! -f "$B0" ]; then
        echo "WARNING: Missing PA_b0_mean for $SUBJECT" >> "$WARN"
        echo "$SUBJECT,no,no,NA" >> "$OUTCSV"
        continue
    fi

    cd "$PREPROC"

    bet PA_b0_mean.nii.gz nodif_brain -m -f 0.25

    if [ -f "$PREPROC/nodif_brain_mask.nii.gz" ]; then
        mask_exists="yes"
        mask_size=$(mrinfo "$PREPROC/nodif_brain_mask.nii.gz" -size | tr ' ' 'x')
    else
        mask_exists="no"
        mask_size="NA"
        echo "WARNING: BET mask not created for $SUBJECT" >> "$WARN"
    fi

    echo "$SUBJECT,yes,$mask_exists,$mask_size" >> "$OUTCSV"
done

echo "BET mask generation finished."
echo "QC: $OUTCSV"
echo "Warnings: $WARN"
