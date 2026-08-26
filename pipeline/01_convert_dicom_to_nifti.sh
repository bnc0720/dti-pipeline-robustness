#!/bin/bash

set -e

PROJECT="/path/to/adni_project"  # <-- EDIT: set to your project root

RAWDATA="$PROJECT/rawdata"
SOURCEDATA="$PROJECT/sourcedata"
LOGDIR="$PROJECT/logs"

mkdir -p "$SOURCEDATA" "$LOGDIR"

for subject_dir in "$RAWDATA"/*; do

    [ -d "$subject_dir" ] || continue

    SUBJECT=$(basename "$subject_dir")

    echo "======================================="
    echo "Converting: $SUBJECT"
    echo "======================================="

    OUTDIR="$SOURCEDATA/$SUBJECT/nifti"
    mkdir -p "$OUTDIR"

    AP_DICOM="$subject_dir/dicom/dwi_AP"
    PA_DICOM="$subject_dir/dicom/dwi_PA"

    if [ -d "$AP_DICOM" ]; then
        dcm2niix -z y -o "$OUTDIR" -f AP "$AP_DICOM"
    else
        echo "WARNING: Missing AP DICOM for $SUBJECT" | tee -a "$LOGDIR/01_convert_warnings.log"
    fi

    if [ -d "$PA_DICOM" ]; then
        dcm2niix -z y -o "$OUTDIR" -f PA "$PA_DICOM"
    else
        echo "WARNING: Missing PA DICOM for $SUBJECT" | tee -a "$LOGDIR/01_convert_warnings.log"
    fi

done

echo "ALL DICOM TO NIFTI CONVERSIONS FINISHED."
echo "Output: $SOURCEDATA"

