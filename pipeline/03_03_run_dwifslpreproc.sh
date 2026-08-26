#!/bin/bash

set -e

PROJECT="/path/to/adni_project"  # <-- EDIT: set to your project root
PIPELINE="$PROJECT/pipeline_03"
QC="$PROJECT/qc"
LOGDIR="$PROJECT/logs"

mkdir -p "$QC" "$LOGDIR"

OUTCSV="$QC/03_03_dwifslpreproc_qc.csv"
WARN="$LOGDIR/03_03_dwifslpreproc_warnings.log"

echo "subject,status,output_exists,output_size" > "$OUTCSV"
> "$WARN"

for subject_dir in "$PIPELINE"/*; do

    [ -d "$subject_dir" ] || continue

    SUBJECT=$(basename "$subject_dir")
    PREPROC="$subject_dir/preproc"

    INPUT="$PREPROC/PA_denoised_degibbs_b0_b1000.mif"
    SE_EPI="$PREPROC/se_epi_PA_AP.mif"
    OUTPUT="$PREPROC/PA_preprocessed.mif"

    echo "dwifslpreproc: $SUBJECT"

    if [ -f "$OUTPUT" ]; then
        output_size=$(mrinfo "$OUTPUT" -size | tr ' ' 'x')
        echo "$SUBJECT,skipped_existing,yes,$output_size" >> "$OUTCSV"
        echo "Skipping $SUBJECT, output already exists."
        continue
    fi

    if [ ! -f "$INPUT" ] || [ ! -f "$SE_EPI" ]; then
        echo "WARNING: Missing input or se_epi for $SUBJECT" >> "$WARN"
        echo "$SUBJECT,missing_input,no,NA" >> "$OUTCSV"
        continue
    fi

    cd "$PREPROC"

    dwifslpreproc PA_denoised_degibbs_b0_b1000.mif PA_preprocessed.mif \
        -rpe_pair \
        -se_epi se_epi_PA_AP.mif \
        -pe_dir j \
        -eddy_options " --repol " \
        -readout_time 0.0632499

    if [ -f "$OUTPUT" ]; then
        output_size=$(mrinfo "$OUTPUT" -size | tr ' ' 'x')
        echo "$SUBJECT,completed,yes,$output_size" >> "$OUTCSV"
    else
        echo "WARNING: dwifslpreproc output missing for $SUBJECT" >> "$WARN"
        echo "$SUBJECT,failed,no,NA" >> "$OUTCSV"
    fi

done

echo "03_03 dwifslpreproc batch finished."
echo "QC: $OUTCSV"
echo "Warnings: $WARN"
