#!/bin/bash

set -e

PROJECT="/path/to/adni_project"  # <-- EDIT: set to your project root
SOURCEDATA="$PROJECT/sourcedata"
QC="$PROJECT/qc"
LOGDIR="$PROJECT/logs"

mkdir -p "$QC" "$LOGDIR"

OUT="$QC/02_nifti_metadata_qc.csv"
WARN="$LOGDIR/02_nifti_metadata_warnings.log"

echo "subject,AP_exists,PA_exists,AP_size,PA_size,AP_phase_encoding,PA_phase_encoding,AP_total_readout,PA_total_readout,AP_n_bvals,PA_n_bvals" > "$OUT"
> "$WARN"

for subject_dir in "$SOURCEDATA"/*; do

    [ -d "$subject_dir" ] || continue

    SUBJECT=$(basename "$subject_dir")
    NIFTI="$subject_dir/nifti"

    AP_NII="$NIFTI/AP.nii.gz"
    PA_NII="$NIFTI/PA.nii.gz"
    AP_BVAL="$NIFTI/AP.bval"
    PA_BVAL="$NIFTI/PA.bval"
    AP_JSON="$NIFTI/AP.json"
    PA_JSON="$NIFTI/PA.json"

    AP_exists="no"
    PA_exists="no"
    AP_size="NA"
    PA_size="NA"
    AP_pe="NA"
    PA_pe="NA"
    AP_readout="NA"
    PA_readout="NA"
    AP_n_bvals="NA"
    PA_n_bvals="NA"

    if [ -f "$AP_NII" ]; then
        AP_exists="yes"
        AP_size=$(mrinfo "$AP_NII" -size | tr ' ' 'x')
    else
        echo "WARNING: $SUBJECT missing AP.nii.gz" >> "$WARN"
    fi

    if [ -f "$PA_NII" ]; then
        PA_exists="yes"
        PA_size=$(mrinfo "$PA_NII" -size | tr ' ' 'x')
    else
        echo "WARNING: $SUBJECT missing PA.nii.gz" >> "$WARN"
    fi

    if [ -f "$AP_JSON" ]; then
        AP_pe=$(grep -m1 '"PhaseEncodingDirection"' "$AP_JSON" | sed 's/.*: *"//; s/".*//')
        AP_readout=$(grep -m1 '"TotalReadoutTime"' "$AP_JSON" | sed 's/.*: *//; s/,.*//')
    else
        echo "WARNING: $SUBJECT missing AP.json" >> "$WARN"
    fi

    if [ -f "$PA_JSON" ]; then
        PA_pe=$(grep -m1 '"PhaseEncodingDirection"' "$PA_JSON" | sed 's/.*: *"//; s/".*//')
        PA_readout=$(grep -m1 '"TotalReadoutTime"' "$PA_JSON" | sed 's/.*: *//; s/,.*//')
    else
        echo "WARNING: $SUBJECT missing PA.json" >> "$WARN"
    fi

    if [ -f "$AP_BVAL" ]; then
        AP_n_bvals=$(awk '{print NF}' "$AP_BVAL")
    else
        echo "WARNING: $SUBJECT missing AP.bval" >> "$WARN"
    fi

    if [ -f "$PA_BVAL" ]; then
        PA_n_bvals=$(awk '{print NF}' "$PA_BVAL")
    else
        echo "WARNING: $SUBJECT missing PA.bval" >> "$WARN"
    fi

    echo "$SUBJECT,$AP_exists,$PA_exists,$AP_size,$PA_size,$AP_pe,$PA_pe,$AP_readout,$PA_readout,$AP_n_bvals,$PA_n_bvals" >> "$OUT"

done

echo "QC finished."
echo "CSV: $OUT"
echo "Warnings: $WARN"
