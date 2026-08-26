#!/bin/bash

set -e

PROJECT="/path/to/adni_project"  # <-- EDIT: set to your project root
PIPELINE="$PROJECT/pipeline_01"
QC="$PROJECT/qc"
LOGDIR="$PROJECT/logs"

mkdir -p "$QC" "$LOGDIR"

OUTCSV="$QC/04_mask_qc.csv"
WARN="$LOGDIR/04_mask_warnings.log"

echo "subject,dwi_exists,mask_exists,dwi_size,mask_size" > "$OUTCSV"
> "$WARN"

for subject_dir in "$PIPELINE"/*; do

    [ -d "$subject_dir" ] || continue

    SUBJECT=$(basename "$subject_dir")
    DWI_DIR="$subject_dir/dwi"

    DWI="$DWI_DIR/dwi_b0_b1000.nii.gz"
    BVEC="$DWI_DIR/dwi_b0_b1000.bvec"
    BVAL="$DWI_DIR/dwi_b0_b1000.bval"
    MASK="$DWI_DIR/mask.nii.gz"

    echo "Processing mask: $SUBJECT"

    if [ ! -f "$DWI" ] || [ ! -f "$BVEC" ] || [ ! -f "$BVAL" ]; then
        echo "WARNING: Missing DWI/bvec/bval for $SUBJECT" >> "$WARN"
        echo "$SUBJECT,no,no,NA,NA" >> "$OUTCSV"
        continue
    fi

    dwi2mask "$DWI" "$MASK" \
        -fslgrad "$BVEC" "$BVAL"

    dwi_size=$(mrinfo "$DWI" -size | tr ' ' 'x')

    if [ -f "$MASK" ]; then
        mask_exists="yes"
        mask_size=$(mrinfo "$MASK" -size | tr ' ' 'x')
    else
        mask_exists="no"
        mask_size="NA"
        echo "WARNING: Mask not created for $SUBJECT" >> "$WARN"
    fi

    echo "$SUBJECT,yes,$mask_exists,$dwi_size,$mask_size" >> "$OUTCSV"

done

echo "Mask generation finished."
echo "QC CSV: $OUTCSV"
echo "Warnings: $WARN"
