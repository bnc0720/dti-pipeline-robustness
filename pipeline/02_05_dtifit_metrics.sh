#!/bin/bash

set -e

PROJECT="/path/to/adni_project"  # <-- EDIT: set to your project root
PIPELINE="$PROJECT/pipeline_02"
QC="$PROJECT/qc"
LOGDIR="$PROJECT/logs"

mkdir -p "$QC" "$LOGDIR"

OUTCSV="$QC/02_05_dtifit_metrics_qc.csv"
WARN="$LOGDIR/02_05_dtifit_metrics_warnings.log"

echo "subject,FA_exists,MD_exists,L1_exists,RD_exists,FA_size,MD_size,RD_size" > "$OUTCSV"
> "$WARN"

for subject_dir in "$PIPELINE"/*; do

    [ -d "$subject_dir" ] || continue

    SUBJECT=$(basename "$subject_dir")
    PREPROC="$subject_dir/preproc"

    echo "DTIFIT: $SUBJECT"

    cd "$PREPROC"

    if [ ! -f "eddy_corrected.nii.gz" ] || \
       [ ! -f "eddy_corrected.eddy_rotated_bvecs" ] || \
       [ ! -f "PA_b0_b1000.bval" ] || \
       [ ! -f "nodif_brain_mask.nii.gz" ]; then

        echo "WARNING: Missing DTIFIT inputs for $SUBJECT" >> "$WARN"
        echo "$SUBJECT,no,no,no,no,NA,NA,NA" >> "$OUTCSV"
        continue
    fi

    dtifit \
        -k eddy_corrected.nii.gz \
        -o dti \
        -m nodif_brain_mask.nii.gz \
        -r eddy_corrected.eddy_rotated_bvecs \
        -b PA_b0_b1000.bval

    # RD számítás
    fslmaths dti_L2.nii.gz -add dti_L3.nii.gz -div 2 dti_RD.nii.gz

    FA_exists="no"
    MD_exists="no"
    L1_exists="no"
    RD_exists="no"

    FA_size="NA"
    MD_size="NA"
    RD_size="NA"

    [ -f "dti_FA.nii.gz" ] && FA_exists="yes"
    [ -f "dti_MD.nii.gz" ] && MD_exists="yes"
    [ -f "dti_L1.nii.gz" ] && L1_exists="yes"
    [ -f "dti_RD.nii.gz" ] && RD_exists="yes"

    [ -f "dti_FA.nii.gz" ] && FA_size=$(mrinfo dti_FA.nii.gz -size | tr ' ' 'x')
    [ -f "dti_MD.nii.gz" ] && MD_size=$(mrinfo dti_MD.nii.gz -size | tr ' ' 'x')
    [ -f "dti_RD.nii.gz" ] && RD_size=$(mrinfo dti_RD.nii.gz -size | tr ' ' 'x')

    echo "$SUBJECT,$FA_exists,$MD_exists,$L1_exists,$RD_exists,$FA_size,$MD_size,$RD_size" >> "$OUTCSV"

done

echo "DTIFIT metrics finished."
echo "QC: $OUTCSV"
echo "Warnings: $WARN"
