#!/bin/bash

set -e

PROJECT="/path/to/adni_project"  # <-- EDIT: set to your project root
PIPELINE="$PROJECT/pipeline_03"
QC="$PROJECT/qc"
LOGDIR="$PROJECT/logs"

mkdir -p "$QC" "$LOGDIR"

OUTCSV="$QC/03_04_biascorr_qc.csv"
WARN="$LOGDIR/03_04_biascorr_warnings.log"

echo "subject,status,output_exists,output_size" > "$OUTCSV"
> "$WARN"

for subject_dir in "$PIPELINE"/*; do

    [ -d "$subject_dir" ] || continue

    SUBJECT=$(basename "$subject_dir")
    PREPROC="$subject_dir/preproc"

    INPUT="$PREPROC/PA_preprocessed.mif"
    OUTPUT="$PREPROC/PA_biascorr.mif"

    echo "Bias correction: $SUBJECT"

    if [ -f "$OUTPUT" ]; then
        output_size=$(mrinfo "$OUTPUT" -size | tr ' ' 'x')
        echo "$SUBJECT,skipped_existing,yes,$output_size" >> "$OUTCSV"
        continue
    fi

    if [ ! -f "$INPUT" ]; then
        echo "WARNING: Missing PA_preprocessed.mif for $SUBJECT" >> "$WARN"
        echo "$SUBJECT,missing_input,no,NA" >> "$OUTCSV"
        continue
    fi

    cd "$PREPROC"

    dwibiascorrect ants \
        PA_preprocessed.mif \
        PA_biascorr.mif

    if [ -f "$OUTPUT" ]; then
        output_size=$(mrinfo "$OUTPUT" -size | tr ' ' 'x')
        echo "$SUBJECT,completed,yes,$output_size" >> "$OUTCSV"
    else
        echo "WARNING: Bias corrected output missing for $SUBJECT" >> "$WARN"
        echo "$SUBJECT,failed,no,NA" >> "$OUTCSV"
    fi

done

echo "03_04 bias correction finished."
echo "QC: $OUTCSV"
echo "Warnings: $WARN"
