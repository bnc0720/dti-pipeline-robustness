#!/bin/bash

set -e

PROJECT="/path/to/adni_project"  # <-- EDIT: set to your project root
PIPELINE="$PROJECT/pipeline_03"
QC="$PROJECT/qc"
LOGDIR="$PROJECT/logs"

mkdir -p "$QC" "$LOGDIR"

OUTCSV="$QC/03_05_mask_qc.csv"
WARN="$LOGDIR/03_05_mask_warnings.log"

echo "subject,status,mask_exists,biascorr_size,mask_size" > "$OUTCSV"
> "$WARN"

for subject_dir in "$PIPELINE"/*; do

    [ -d "$subject_dir" ] || continue

    SUBJECT=$(basename "$subject_dir")
    PREPROC="$subject_dir/preproc"

    INPUT="$PREPROC/PA_biascorr.mif"
    MASK="$PREPROC/mask.mif"

    echo "V3 dwi2mask: $SUBJECT"

    if [ -f "$MASK" ]; then
        mask_size=$(mrinfo "$MASK" -size | tr ' ' 'x')
        input_size=$(mrinfo "$INPUT" -size | tr ' ' 'x')
        echo "$SUBJECT,skipped_existing,yes,$input_size,$mask_size" >> "$OUTCSV"
        continue
    fi

    if [ ! -f "$INPUT" ]; then
        echo "WARNING: Missing PA_biascorr.mif for $SUBJECT" >> "$WARN"
        echo "$SUBJECT,missing_input,no,NA,NA" >> "$OUTCSV"
        continue
    fi

    cd "$PREPROC"

    dwi2mask PA_biascorr.mif mask.mif

    if [ -f "$MASK" ]; then
        input_size=$(mrinfo "$INPUT" -size | tr ' ' 'x')
        mask_size=$(mrinfo "$MASK" -size | tr ' ' 'x')
        echo "$SUBJECT,completed,yes,$input_size,$mask_size" >> "$OUTCSV"
    else
        echo "WARNING: mask.mif not created for $SUBJECT" >> "$WARN"
        echo "$SUBJECT,failed,no,NA,NA" >> "$OUTCSV"
    fi

done

echo "03_05 mask generation finished."
echo "QC: $OUTCSV"
echo "Warnings: $WARN"
