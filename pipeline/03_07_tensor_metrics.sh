#!/bin/bash

set -e

PROJECT="/path/to/adni_project"  # <-- EDIT: set to your project root
PIPELINE="$PROJECT/pipeline_03"
QC="$PROJECT/qc"
LOGDIR="$PROJECT/logs"

mkdir -p "$QC" "$LOGDIR"

OUTCSV="$QC/03_07_tensor_metrics_qc.csv"
WARN="$LOGDIR/03_07_tensor_metrics_warnings.log"

echo "subject,status,FA,MD,AD,RD,FA_size,MD_size,AD_size,RD_size" > "$OUTCSV"
> "$WARN"

for subject_dir in "$PIPELINE"/*; do

    [ -d "$subject_dir" ] || continue

    SUBJECT=$(basename "$subject_dir")
    PREPROC="$subject_dir/preproc"

    TENSOR="$PREPROC/tensor.mif"

    echo "tensor2metric: $SUBJECT"

    if [ ! -f "$TENSOR" ]; then
        echo "WARNING: Missing tensor.mif for $SUBJECT" >> "$WARN"
        echo "$SUBJECT,missing_tensor,no,no,no,no,NA,NA,NA,NA" >> "$OUTCSV"
        continue
    fi

    cd "$PREPROC"

    tensor2metric tensor.mif \
        -fa FA.mif \
        -ad AD.mif \
        -rd RD.mif \
        -adc MD.mif

    mrconvert FA.mif FA.nii.gz
    mrconvert MD.mif MD.nii.gz
    mrconvert AD.mif AD.nii.gz
    mrconvert RD.mif RD.nii.gz

    FA_size=$(mrinfo FA.mif -size | tr ' ' 'x')
    MD_size=$(mrinfo MD.mif -size | tr ' ' 'x')
    AD_size=$(mrinfo AD.mif -size | tr ' ' 'x')
    RD_size=$(mrinfo RD.mif -size | tr ' ' 'x')

    echo "$SUBJECT,completed,yes,yes,yes,yes,$FA_size,$MD_size,$AD_size,$RD_size" >> "$OUTCSV"

done

echo "03_07 tensor metrics finished."
echo "QC: $OUTCSV"
echo "Warnings: $WARN"
