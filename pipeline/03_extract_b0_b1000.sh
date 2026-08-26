#!/bin/bash

set -e

PROJECT="/path/to/adni_project"  # <-- EDIT: set to your project root
SOURCEDATA="$PROJECT/sourcedata"
PIPELINE="$PROJECT/pipeline_01"
QC="$PROJECT/qc"
LOGDIR="$PROJECT/logs"

mkdir -p "$PIPELINE" "$QC" "$LOGDIR"

OUTCSV="$QC/03_extract_b0_b1000_qc.csv"
WARN="$LOGDIR/03_extract_b0_b1000_warnings.log"

echo "subject,input_size,output_size,input_n_bvals,output_n_bvals,output_bvals" > "$OUTCSV"
> "$WARN"

for subject_dir in "$SOURCEDATA"/*; do

    [ -d "$subject_dir" ] || continue

    SUBJECT=$(basename "$subject_dir")
    NIFTI="$subject_dir/nifti"
    OUTDIR="$PIPELINE/$SUBJECT/dwi"

    mkdir -p "$OUTDIR"

    PA_NII="$NIFTI/PA.nii.gz"
    PA_BVEC="$NIFTI/PA.bvec"
    PA_BVAL="$NIFTI/PA.bval"

    if [ ! -f "$PA_NII" ] || [ ! -f "$PA_BVEC" ] || [ ! -f "$PA_BVAL" ]; then
        echo "WARNING: Missing PA files for $SUBJECT" >> "$WARN"
        continue
    fi

    echo "Processing $SUBJECT"

    input_size=$(mrinfo "$PA_NII" -size | tr ' ' 'x')
    input_n_bvals=$(awk '{print NF}' "$PA_BVAL")

    dwiextract "$PA_NII" "$OUTDIR/dwi_b0_b1000.nii.gz" \
        -fslgrad "$PA_BVEC" "$PA_BVAL" \
        -shells 0,1000 \
        -export_grad_fsl "$OUTDIR/dwi_b0_b1000.bvec" "$OUTDIR/dwi_b0_b1000.bval"

    output_size=$(mrinfo "$OUTDIR/dwi_b0_b1000.nii.gz" -size | tr ' ' 'x')
    output_n_bvals=$(awk '{print NF}' "$OUTDIR/dwi_b0_b1000.bval")
    output_bvals=$(cat "$OUTDIR/dwi_b0_b1000.bval" | tr ' ' ';')

    echo "$SUBJECT,$input_size,$output_size,$input_n_bvals,$output_n_bvals,$output_bvals" >> "$OUTCSV"

done

echo "Shell extraction finished."
echo "QC CSV: $OUTCSV"
echo "Warnings: $WARN"
