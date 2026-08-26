#!/bin/bash

set -e

PROJECT="/path/to/adni_project"  # <-- EDIT: set to your project root
PIPELINE="$PROJECT/pipeline_01"
QC="$PROJECT/qc"
LOGDIR="$PROJECT/logs"

mkdir -p "$QC" "$LOGDIR"

OUTCSV="$QC/06_dti_metrics_qc.csv"
WARN="$LOGDIR/06_dti_metrics_warnings.log"

echo "subject,tensor_exists,FA_exists,MD_exists,AD_exists,RD_exists,FA_size,MD_size,AD_size,RD_size" > "$OUTCSV"
> "$WARN"

for subject_dir in "$PIPELINE"/*; do

    [ -d "$subject_dir" ] || continue

    SUBJECT=$(basename "$subject_dir")
    DWI_DIR="$subject_dir/dwi"

    TENSOR="$DWI_DIR/tensor.nii.gz"

    FA="$DWI_DIR/FA.nii.gz"
    MD="$DWI_DIR/MD.nii.gz"
    AD="$DWI_DIR/AD.nii.gz"
    RD="$DWI_DIR/RD.nii.gz"

    echo "DTI metrics: $SUBJECT"

    if [ ! -f "$TENSOR" ]; then
        echo "WARNING: Missing tensor for $SUBJECT" >> "$WARN"
        echo "$SUBJECT,no,no,no,no,no,NA,NA,NA,NA" >> "$OUTCSV"
        continue
    fi

    tensor2metric "$TENSOR" \
        -fa "$FA" \
        -adc "$MD" \
        -ad "$AD" \
        -rd "$RD"

    FA_exists="no"; MD_exists="no"; AD_exists="no"; RD_exists="no"
    FA_size="NA"; MD_size="NA"; AD_size="NA"; RD_size="NA"

    if [ -f "$FA" ]; then FA_exists="yes"; FA_size=$(mrinfo "$FA" -size | tr ' ' 'x'); fi
    if [ -f "$MD" ]; then MD_exists="yes"; MD_size=$(mrinfo "$MD" -size | tr ' ' 'x'); fi
    if [ -f "$AD" ]; then AD_exists="yes"; AD_size=$(mrinfo "$AD" -size | tr ' ' 'x'); fi
    if [ -f "$RD" ]; then RD_exists="yes"; RD_size=$(mrinfo "$RD" -size | tr ' ' 'x'); fi

    echo "$SUBJECT,yes,$FA_exists,$MD_exists,$AD_exists,$RD_exists,$FA_size,$MD_size,$AD_size,$RD_size" >> "$OUTCSV"

done

echo "DTI metric generation finished."
echo "QC CSV: $OUTCSV"
echo "Warnings: $WARN"
