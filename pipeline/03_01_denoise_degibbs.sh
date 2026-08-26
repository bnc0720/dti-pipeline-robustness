#!/bin/bash

set -e

PROJECT="/path/to/adni_project"  # <-- EDIT: set to your project root
SOURCEDATA="$PROJECT/sourcedata"
PIPELINE="$PROJECT/pipeline_03"
QC="$PROJECT/qc"
LOGDIR="$PROJECT/logs"

mkdir -p "$PIPELINE" "$QC" "$LOGDIR"

OUTCSV="$QC/03_01_denoise_degibbs_qc.csv"
WARN="$LOGDIR/03_01_denoise_degibbs_warnings.log"

echo "subject,input_size,denoised_size,degibbs_size,mif_size" > "$OUTCSV"
> "$WARN"

for subject_dir in "$SOURCEDATA"/*; do

    [ -d "$subject_dir" ] || continue

    SUBJECT=$(basename "$subject_dir")
    NIFTI="$subject_dir/nifti"
    OUTDIR="$PIPELINE/$SUBJECT/preproc"

    mkdir -p "$OUTDIR"

    PA_NII="$NIFTI/PA.nii.gz"
    PA_BVEC="$NIFTI/PA.bvec"
    PA_BVAL="$NIFTI/PA.bval"

    echo "V3 denoise + degibbs: $SUBJECT"

    if [ ! -f "$PA_NII" ] || [ ! -f "$PA_BVEC" ] || [ ! -f "$PA_BVAL" ]; then
        echo "WARNING: Missing PA input files for $SUBJECT" >> "$WARN"
        echo "$SUBJECT,NA,NA,NA,NA" >> "$OUTCSV"
        continue
    fi

    input_size=$(mrinfo "$PA_NII" -size | tr ' ' 'x')

    dwidenoise "$PA_NII" "$OUTDIR/PA_denoised.nii.gz" \
        -noise "$OUTDIR/noise.nii.gz"

    denoised_size=$(mrinfo "$OUTDIR/PA_denoised.nii.gz" -size | tr ' ' 'x')

    mrdegibbs "$OUTDIR/PA_denoised.nii.gz" "$OUTDIR/PA_denoised_degibbs.nii.gz"

    degibbs_size=$(mrinfo "$OUTDIR/PA_denoised_degibbs.nii.gz" -size | tr ' ' 'x')

    mrconvert "$OUTDIR/PA_denoised_degibbs.nii.gz" "$OUTDIR/PA_denoised_degibbs.mif" \
        -fslgrad "$PA_BVEC" "$PA_BVAL"

    mif_size=$(mrinfo "$OUTDIR/PA_denoised_degibbs.mif" -size | tr ' ' 'x')

    echo "$SUBJECT,$input_size,$denoised_size,$degibbs_size,$mif_size" >> "$OUTCSV"

done

echo "V3 module 03_01 finished."
echo "QC: $OUTCSV"
echo "Warnings: $WARN"
