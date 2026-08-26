#!/bin/bash

set -e

PROJECT="/path/to/adni_project"  # <-- EDIT: set to your project root
SOURCEDATA="$PROJECT/sourcedata"
PIPELINE="$PROJECT/pipeline_03"
QC="$PROJECT/qc"
LOGDIR="$PROJECT/logs"

mkdir -p "$QC" "$LOGDIR"

OUTCSV="$QC/03_02_prepare_dwifslpreproc_qc.csv"
WARN="$LOGDIR/03_02_prepare_dwifslpreproc_warnings.log"

echo "subject,se_epi_size,dwi_b0_b1000_size" > "$OUTCSV"
> "$WARN"

for subject_dir in "$SOURCEDATA"/*; do

    [ -d "$subject_dir" ] || continue

    SUBJECT=$(basename "$subject_dir")

    NIFTI="$subject_dir/nifti"
    PREPROC="$PIPELINE/$SUBJECT/preproc"

    echo "Preparing dwifslpreproc inputs: $SUBJECT"

    PA_MIF="$PREPROC/PA_denoised_degibbs.mif"

    AP_NII="$NIFTI/AP.nii.gz"
    AP_BVEC="$NIFTI/AP.bvec"
    AP_BVAL="$NIFTI/AP.bval"

    if [ ! -f "$PA_MIF" ] || [ ! -f "$AP_NII" ]; then
        echo "WARNING: Missing inputs for $SUBJECT" >> "$WARN"
        echo "$SUBJECT,NA,NA" >> "$OUTCSV"
        continue
    fi

    # 1. b0+b1000 extract
    dwiextract "$PA_MIF" \
        "$PREPROC/PA_denoised_degibbs_b0_b1000.mif" \
        -shells 0,1000

    # 2. PA b0 mean
    dwiextract "$PREPROC/PA_denoised_degibbs_b0_b1000.mif" \
        "$PREPROC/PA_b0s.mif" \
        -bzero

    mrmath "$PREPROC/PA_b0s.mif" mean \
        "$PREPROC/PA_b0_mean.mif" -axis 3

    # 3. AP → mif
    mrconvert "$AP_NII" "$PREPROC/AP.mif" \
        -fslgrad "$AP_BVEC" "$AP_BVAL"

    # 4. AP b0 mean
    dwiextract "$PREPROC/AP.mif" \
        "$PREPROC/AP_b0s.mif" \
        -bzero

    mrmath "$PREPROC/AP_b0s.mif" mean \
        "$PREPROC/AP_b0_mean.mif" -axis 3

    # 5. se_epi pair
    mrcat \
        "$PREPROC/PA_b0_mean.mif" \
        "$PREPROC/AP_b0_mean.mif" \
        "$PREPROC/se_epi_PA_AP.mif" \
        -axis 3

    se_epi_size=$(mrinfo "$PREPROC/se_epi_PA_AP.mif" -size | tr ' ' 'x')
    dwi_size=$(mrinfo "$PREPROC/PA_denoised_degibbs_b0_b1000.mif" -size | tr ' ' 'x')

    echo "$SUBJECT,$se_epi_size,$dwi_size" >> "$OUTCSV"

done

echo "03_02 prepare dwifslpreproc finished."
echo "QC: $OUTCSV"
echo "Warnings: $WARN"
