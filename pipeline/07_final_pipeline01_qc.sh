#!/bin/bash

set -e

PROJECT="/path/to/adni_project"  # <-- EDIT: set to your project root
PIPELINE="$PROJECT/pipeline_01"
QC="$PROJECT/qc"
LOGDIR="$PROJECT/logs"

mkdir -p "$QC" "$LOGDIR"

OUTCSV="$QC/07_final_pipeline01_qc.csv"
WARN="$LOGDIR/07_final_pipeline01_warnings.log"

echo "subject,DWI,mask,tensor,FA,MD,AD,RD,DWI_size,mask_size,tensor_size,FA_size,MD_size,AD_size,RD_size,FA_mean,MD_mean,AD_mean,RD_mean" > "$OUTCSV"
> "$WARN"

for subject_dir in "$PIPELINE"/*; do

    [ -d "$subject_dir" ] || continue

    SUBJECT=$(basename "$subject_dir")
    DWI_DIR="$subject_dir/dwi"

    DWI="$DWI_DIR/dwi_b0_b1000.nii.gz"
    MASK="$DWI_DIR/mask.nii.gz"
    TENSOR="$DWI_DIR/tensor.nii.gz"
    FA="$DWI_DIR/FA.nii.gz"
    MD="$DWI_DIR/MD.nii.gz"
    AD="$DWI_DIR/AD.nii.gz"
    RD="$DWI_DIR/RD.nii.gz"

    get_exists () {
        [ -f "$1" ] && echo "yes" || echo "no"
    }

    get_size () {
        [ -f "$1" ] && mrinfo "$1" -size | tr ' ' 'x' || echo "NA"
    }

    get_mean () {
        local img="$1"
        local mask="$2"
        if [ -f "$img" ] && [ -f "$mask" ]; then
            mrstats "$img" -mask "$mask" -output mean
        else
            echo "NA"
        fi
    }

    DWI_exists=$(get_exists "$DWI")
    MASK_exists=$(get_exists "$MASK")
    TENSOR_exists=$(get_exists "$TENSOR")
    FA_exists=$(get_exists "$FA")
    MD_exists=$(get_exists "$MD")
    AD_exists=$(get_exists "$AD")
    RD_exists=$(get_exists "$RD")

    if [ "$FA_exists" = "no" ] || [ "$MD_exists" = "no" ] || [ "$AD_exists" = "no" ] || [ "$RD_exists" = "no" ]; then
        echo "WARNING: Missing final DTI metric for $SUBJECT" >> "$WARN"
    fi

    DWI_size=$(get_size "$DWI")
    MASK_size=$(get_size "$MASK")
    TENSOR_size=$(get_size "$TENSOR")
    FA_size=$(get_size "$FA")
    MD_size=$(get_size "$MD")
    AD_size=$(get_size "$AD")
    RD_size=$(get_size "$RD")

    FA_mean=$(get_mean "$FA" "$MASK")
    MD_mean=$(get_mean "$MD" "$MASK")
    AD_mean=$(get_mean "$AD" "$MASK")
    RD_mean=$(get_mean "$RD" "$MASK")

    echo "$SUBJECT,$DWI_exists,$MASK_exists,$TENSOR_exists,$FA_exists,$MD_exists,$AD_exists,$RD_exists,$DWI_size,$MASK_size,$TENSOR_size,$FA_size,$MD_size,$AD_size,$RD_size,$FA_mean,$MD_mean,$AD_mean,$RD_mean" >> "$OUTCSV"

done

echo "Final pipeline_01 QC finished."
echo "CSV: $OUTCSV"
echo "Warnings: $WARN"
