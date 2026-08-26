#!/bin/bash

set -e

PROJECT="/path/to/adni_project"  # <-- EDIT: set to your project root
PIPELINE="$PROJECT/pipeline_01"
QC="$PROJECT/qc"
LOGDIR="$PROJECT/logs"

mkdir -p "$QC" "$LOGDIR"

OUTCSV="$QC/05_tensor_fit_qc.csv"
WARN="$LOGDIR/05_tensor_fit_warnings.log"

echo "subject,dwi_exists,mask_exists,tensor_exists,dwi_size,mask_size,tensor_size" > "$OUTCSV"
> "$WARN"

for subject_dir in "$PIPELINE"/*; do

    [ -d "$subject_dir" ] || continue

    SUBJECT=$(basename "$subject_dir")
    DWI_DIR="$subject_dir/dwi"

    DWI="$DWI_DIR/dwi_b0_b1000.nii.gz"
    BVEC="$DWI_DIR/dwi_b0_b1000.bvec"
    BVAL="$DWI_DIR/dwi_b0_b1000.bval"
    MASK="$DWI_DIR/mask.nii.gz"
    TENSOR="$DWI_DIR/tensor.nii.gz"

    echo "Tensor fitting: $SUBJECT"

    if [ ! -f "$DWI" ] || [ ! -f "$BVEC" ] || [ ! -f "$BVAL" ]; then
        echo "WARNING: Missing DWI/bvec/bval for $SUBJECT" >> "$WARN"
        echo "$SUBJECT,no,NA,no,NA,NA,NA" >> "$OUTCSV"
        continue
    fi

    if [ ! -f "$MASK" ]; then
        echo "WARNING: Missing mask for $SUBJECT" >> "$WARN"
        echo "$SUBJECT,yes,no,no,$(mrinfo "$DWI" -size | tr ' ' 'x'),NA,NA" >> "$OUTCSV"
        continue
    fi

    dwi2tensor "$DWI" "$TENSOR" \
        -mask "$MASK" \
        -fslgrad "$BVEC" "$BVAL"

    dwi_size=$(mrinfo "$DWI" -size | tr ' ' 'x')
    mask_size=$(mrinfo "$MASK" -size | tr ' ' 'x')

    if [ -f "$TENSOR" ]; then
        tensor_exists="yes"
        tensor_size=$(mrinfo "$TENSOR" -size | tr ' ' 'x')
    else
        tensor_exists="no"
        tensor_size="NA"
        echo "WARNING: Tensor not created for $SUBJECT" >> "$WARN"
    fi

    echo "$SUBJECT,yes,yes,$tensor_exists,$dwi_size,$mask_size,$tensor_size" >> "$OUTCSV"

done

echo "Tensor fitting finished."
echo "QC CSV: $OUTCSV"
echo "Warnings: $WARN"

