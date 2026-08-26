#!/bin/bash

set -e

PROJECT="/path/to/adni_project"  # <-- EDIT: set to your project root
PIPELINE="$PROJECT/pipeline_02"
QC="$PROJECT/qc"
LOGDIR="$PROJECT/logs"

mkdir -p "$QC" "$LOGDIR"

OUTCSV="$QC/02_04_eddy_qc.csv"
WARN="$LOGDIR/02_04_eddy_warnings.log"

echo "subject,eddy_exists,eddy_bvec_exists,eddy_size,eddy_bvec_count" > "$OUTCSV"
> "$WARN"

for subject_dir in "$PIPELINE"/*; do

    [ -d "$subject_dir" ] || continue

    SUBJECT=$(basename "$subject_dir")
    PREPROC="$subject_dir/preproc"

    echo "Running EDDY: $SUBJECT"

    DWI="$PREPROC/PA_b0_b1000.nii.gz"
    MASK="$PREPROC/nodif_brain_mask.nii.gz"
    ACQP="$PREPROC/acqparams.txt"
    INDEX="$PREPROC/index.txt"
    BVEC="$PREPROC/PA_b0_b1000.bvec"
    BVAL="$PREPROC/PA_b0_b1000.bval"

    if [ ! -f "$DWI" ] || [ ! -f "$MASK" ] || [ ! -f "$ACQP" ] || \
       [ ! -f "$INDEX" ] || [ ! -f "$BVEC" ] || [ ! -f "$BVAL" ]; then

        echo "WARNING: Missing eddy inputs for $SUBJECT" >> "$WARN"
        echo "$SUBJECT,no,no,NA,NA" >> "$OUTCSV"
        continue
    fi

    cd "$PREPROC"

    eddy \
        --imain=PA_b0_b1000.nii.gz \
        --mask=nodif_brain_mask.nii.gz \
        --acqp=acqparams.txt \
        --index=index.txt \
        --bvecs=PA_b0_b1000.bvec \
        --bvals=PA_b0_b1000.bval \
        --topup=topup_results \
        --out=eddy_corrected

    eddy_exists="no"
    eddy_bvec_exists="no"
    eddy_size="NA"
    eddy_bvec_count="NA"

    if [ -f "$PREPROC/eddy_corrected.nii.gz" ]; then
        eddy_exists="yes"
        eddy_size=$(mrinfo "$PREPROC/eddy_corrected.nii.gz" -size | tr ' ' 'x')
    else
        echo "WARNING: Missing eddy_corrected.nii.gz for $SUBJECT" >> "$WARN"
    fi

    if [ -f "$PREPROC/eddy_corrected.eddy_rotated_bvecs" ]; then
        eddy_bvec_exists="yes"
        eddy_bvec_count=$(awk 'NR==1{print NF}' "$PREPROC/eddy_corrected.eddy_rotated_bvecs")
    else
        echo "WARNING: Missing rotated bvecs for $SUBJECT" >> "$WARN"
    fi

    echo "$SUBJECT,$eddy_exists,$eddy_bvec_exists,$eddy_size,$eddy_bvec_count" >> "$OUTCSV"

done

echo "EDDY finished."
echo "QC: $OUTCSV"
echo "Warnings: $WARN"

