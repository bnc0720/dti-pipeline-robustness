#!/bin/bash

set -e

PROJECT="/path/to/adni_project"  # <-- EDIT: set to your project root
PIPELINE="$PROJECT/pipeline_03"
QC="$PROJECT/qc"
LOGDIR="$PROJECT/logs"

mkdir -p "$QC" "$LOGDIR"

OUTCSV="$QC/03_06_tensor_fit_qc.csv"
WARN="$LOGDIR/03_06_tensor_fit_warnings.log"

echo "subject,status,tensor_exists,input_size,tensor_size" > "$OUTCSV"
> "$WARN"

for subject_dir in "$PIPELINE"/*; do

    [ -d "$subject_dir" ] || continue

    SUBJECT=$(basename "$subject_dir")
    PREPROC="$subject_dir/preproc"

    INPUT="$PREPROC/PA_biascorr.mif"
    MASK="$PREPROC/mask.mif"
    TENSOR="$PREPROC/tensor.mif"

    echo "Tensor fitting: $SUBJECT"

    if [ -f "$TENSOR" ]; then
        input_size=$(mrinfo "$INPUT" -size | tr ' ' 'x')
        tensor_size=$(mrinfo "$TENSOR" -size | tr ' ' 'x')

        echo "$SUBJECT,skipped_existing,yes,$input_size,$tensor_size" >> "$OUTCSV"
        continue
    fi

    if [ ! -f "$INPUT" ] || [ ! -f "$MASK" ]; then
        echo "WARNING: Missing tensor inputs for $SUBJECT" >> "$WARN"
        echo "$SUBJECT,missing_input,no,NA,NA" >> "$OUTCSV"
        continue
    fi

    cd "$PREPROC"

    dwi2tensor PA_biascorr.mif tensor.mif \
        -mask mask.mif

    if [ -f "$TENSOR" ]; then

        input_size=$(mrinfo "$INPUT" -size | tr ' ' 'x')
        tensor_size=$(mrinfo "$TENSOR" -size | tr ' ' 'x')

        echo "$SUBJECT,completed,yes,$input_size,$tensor_size" >> "$OUTCSV"

    else

        echo "WARNING: tensor.mif missing for $SUBJECT" >> "$WARN"
        echo "$SUBJECT,failed,no,NA,NA" >> "$OUTCSV"

    fi

done

echo "03_06 tensor fitting finished."
echo "QC: $OUTCSV"
echo "Warnings: $WARN"
