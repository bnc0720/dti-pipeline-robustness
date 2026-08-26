#!/usr/bin/env bash
set -euo pipefail

PROJECT="/path/to/adni_project"  # <-- EDIT: set to your project root
PIPELINE="pipeline_02"

BUNDLES="FX_left,FX_right,CG_left,CG_right,UF_left,UF_right,ILF_left,ILF_right,IFO_left,IFO_right,CC_2,CC_7"

for SUBJ_DIR in "$PROJECT/$PIPELINE"/*; do
  [ -d "$SUBJ_DIR" ] || continue

  SUBJ=$(basename "$SUBJ_DIR")
  BASE="$SUBJ_DIR/preproc"
  OUT="$SUBJ_DIR/tractometry_TOM"
  LOG="$OUT/${SUBJ}_${PIPELINE}_run.log"
  QC="$OUT/${SUBJ}_${PIPELINE}_streamline_qc.txt"

  echo "========================================"
  echo "Processing $PIPELINE / $SUBJ"
  echo "========================================"

  mkdir -p "$OUT"

  if [ -f "$OUT/${SUBJ}_${PIPELINE}_FA_tractometry.csv" ] && \
     [ -f "$OUT/${SUBJ}_${PIPELINE}_MD_tractometry.csv" ] && \
     [ -f "$OUT/${SUBJ}_${PIPELINE}_RD_tractometry.csv" ] && \
     [ -f "$OUT/${SUBJ}_${PIPELINE}_AD_tractometry.csv" ] && \
     [ -f "$QC" ]; then
    echo "$SUBJ already complete, skipping."
    continue
  fi

  {
    echo "START: $(date)"
    echo "SUBJ=$SUBJ"
    echo "BASE=$BASE"
    echo "OUT=$OUT"

    for f in \
      "$BASE/eddy_corrected.nii.gz" \
      "$BASE/eddy_corrected.eddy_rotated_bvecs" \
      "$BASE/PA_b0_b1000.bval" \
      "$BASE/nodif_brain_mask.nii.gz" \
      "$BASE/dti_FA.nii.gz" \
      "$BASE/dti_MD.nii.gz" \
      "$BASE/dti_RD.nii.gz" \
      "$BASE/dti_L1.nii.gz"; do
      if [ ! -f "$f" ]; then
        echo "MISSING INPUT: $f"
        exit 1
      fi
    done

    echo "1) Convert EDDY-corrected DWI to MRtrix .mif"
    mrconvert "$BASE/eddy_corrected.nii.gz" "$OUT/dwi.mif" \
      -fslgrad "$BASE/eddy_corrected.eddy_rotated_bvecs" "$BASE/PA_b0_b1000.bval" \
      -force

    echo "2) Response function"
    dwi2response tournier "$OUT/dwi.mif" "$OUT/response.txt" \
      -mask "$BASE/nodif_brain_mask.nii.gz" \
      -force

    echo "3) FOD"
    dwi2fod csd "$OUT/dwi.mif" "$OUT/response.txt" "$OUT/fod.mif" \
      -mask "$BASE/nodif_brain_mask.nii.gz" \
      -force

    echo "4) Peaks"
    sh2peaks "$OUT/fod.mif" "$OUT/peaks.mif" -force
    mrconvert "$OUT/peaks.mif" "$OUT/peaks.nii.gz" -force

    echo "5) TractSeg bundle segmentations"
    TractSeg -i "$OUT/peaks.nii.gz" \
      -o "$OUT" \
      --output_type tract_segmentation

    echo "6) TractSeg endings"
    TractSeg -i "$OUT/peaks.nii.gz" \
      -o "$OUT" \
      --output_type endings_segmentation

    echo "7) TractSeg TOM"
    TractSeg -i "$OUT/peaks.nii.gz" \
      -o "$OUT" \
      --output_type TOM \
      --tract_segmentations_path "$OUT/bundle_segmentations"

    echo "8) TOM tracking selected bundles"
    Tracking -i "$OUT/peaks.nii.gz" \
      -o "$OUT" \
      --bundles "$BUNDLES" \
      --nr_fibers 5000 \
      --tracking_format tck

    echo "9) Tractometry FA/MD/RD/AD"
    Tractometry \
      -i "$OUT/TOM_trackings" \
      -e "$OUT/endings_segmentations" \
      -s "$BASE/dti_FA.nii.gz" \
      -o "$OUT/${SUBJ}_${PIPELINE}_FA_tractometry.csv"

    Tractometry \
      -i "$OUT/TOM_trackings" \
      -e "$OUT/endings_segmentations" \
      -s "$BASE/dti_MD.nii.gz" \
      -o "$OUT/${SUBJ}_${PIPELINE}_MD_tractometry.csv"

    Tractometry \
      -i "$OUT/TOM_trackings" \
      -e "$OUT/endings_segmentations" \
      -s "$BASE/dti_RD.nii.gz" \
      -o "$OUT/${SUBJ}_${PIPELINE}_RD_tractometry.csv"

    Tractometry \
      -i "$OUT/TOM_trackings" \
      -e "$OUT/endings_segmentations" \
      -s "$BASE/dti_L1.nii.gz" \
      -o "$OUT/${SUBJ}_${PIPELINE}_AD_tractometry.csv"

    echo "10) Streamline QC"
    echo "Streamline QC for $SUBJ / $PIPELINE" > "$QC"
    echo "Date: $(date)" >> "$QC"
    echo "" >> "$QC"

    for f in "$OUT"/TOM_trackings/*.tck; do
      echo -n "$(basename "$f"): " >> "$QC"
      tckinfo "$f" | grep count >> "$QC"
    done

    echo "END: $(date)"
    echo "DONE: $SUBJ"

  } 2>&1 | tee "$LOG"

done

echo "ALL DONE: $PIPELINE"
