#!/bin/bash

set -e

PROJECT="/path/to/adni_project"  # <-- EDIT: set to your project root
SOURCEDATA="$PROJECT/sourcedata"
PIPELINE="$PROJECT/pipeline_02"
QC="$PROJECT/qc"
LOGDIR="$PROJECT/logs"

mkdir -p "$PIPELINE" "$QC" "$LOGDIR"

OUTCSV="$QC/02_01_prepare_topup_eddy_inputs_qc.csv"
WARN="$LOGDIR/02_01_prepare_topup_eddy_inputs_warnings.log"

echo "subject,dwi_size,b0_pair_size,index_count,readout_time" > "$OUTCSV"
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
    PA_JSON="$NIFTI/PA.json"

    AP_NII="$NIFTI/AP.nii.gz"
    AP_BVEC="$NIFTI/AP.bvec"
    AP_BVAL="$NIFTI/AP.bval"

    echo "Preparing: $SUBJECT"

    if [ ! -f "$PA_NII" ] || [ ! -f "$PA_BVEC" ] || [ ! -f "$PA_BVAL" ] || [ ! -f "$AP_NII" ]; then
        echo "WARNING: Missing AP/PA input files for $SUBJECT" >> "$WARN"
        continue
    fi

    READOUT=$(grep -m1 '"TotalReadoutTime"' "$PA_JSON" | sed 's/.*: *//; s/,.*//')

    # 1. PA full DWI-ből b0+b1000
    dwiextract "$PA_NII" "$OUTDIR/PA_b0_b1000.nii.gz" \
        -fslgrad "$PA_BVEC" "$PA_BVAL" \
        -shells 0,1000 \
        -export_grad_fsl "$OUTDIR/PA_b0_b1000.bvec" "$OUTDIR/PA_b0_b1000.bval"

    # 2. PA b0 mean
    dwiextract "$OUTDIR/PA_b0_b1000.nii.gz" "$OUTDIR/PA_b0s.nii.gz" \
        -fslgrad "$OUTDIR/PA_b0_b1000.bvec" "$OUTDIR/PA_b0_b1000.bval" \
        -bzero

    mrmath "$OUTDIR/PA_b0s.nii.gz" mean "$OUTDIR/PA_b0_mean.nii.gz" -axis 3

    # 3. AP b0 mean
    dwiextract "$AP_NII" "$OUTDIR/AP_b0s.nii.gz" \
        -fslgrad "$AP_BVEC" "$AP_BVAL" \
        -bzero

    mrmath "$OUTDIR/AP_b0s.nii.gz" mean "$OUTDIR/AP_b0_mean.nii.gz" -axis 3

    # 4. PA + AP b0 pair topuphoz
    mrcat "$OUTDIR/PA_b0_mean.nii.gz" "$OUTDIR/AP_b0_mean.nii.gz" \
        "$OUTDIR/b0_pair_PA_AP.nii.gz" -axis 3

    # 5. acqparams: PA = j, AP = j-
    printf "0 1 0 %s\n0 -1 0 %s\n" "$READOUT" "$READOUT" > "$OUTDIR/acqparams.txt"

    # 6. index.txt: minden PA_b0_b1000 volumen az első acqparams sorhoz tartozik
    N_VOL=$(mrinfo "$OUTDIR/PA_b0_b1000.nii.gz" -size | awk '{print $4}')
    yes 1 | head -n "$N_VOL" | tr '\n' ' ' > "$OUTDIR/index.txt"
    echo "" >> "$OUTDIR/index.txt"

    dwi_size=$(mrinfo "$OUTDIR/PA_b0_b1000.nii.gz" -size | tr ' ' 'x')
    b0_pair_size=$(mrinfo "$OUTDIR/b0_pair_PA_AP.nii.gz" -size | tr ' ' 'x')
    index_count=$(awk '{print NF}' "$OUTDIR/index.txt")

    echo "$SUBJECT,$dwi_size,$b0_pair_size,$index_count,$READOUT" >> "$OUTCSV"

done

echo "Pipeline 02 input preparation finished."
echo "QC: $OUTCSV"
echo "Warnings: $WARN"
