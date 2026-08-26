#!/usr/bin/env bash
set -euo pipefail

SUBJ="AD_001"
BASE="/path/to/adni_project/pipeline_01/${SUBJ}/dwi"  # <-- EDIT: set to your project root
OUT="${BASE}/tractometry_TOM"

BUNDLES="FX_left,FX_right,CG_left,CG_right,UF_left,UF_right,ILF_left,ILF_right,IFO_left,IFO_right,CC_2,CC_7"

mkdir -p "$OUT"

echo "=== Pipeline 01 TOM tractometry test: $SUBJ ==="

echo "1) Convert DWI to MRtrix .mif"
mrconvert "$BASE/dwi_b0_b1000.nii.gz" "$OUT/dwi.mif" \
  -fslgrad "$BASE/dwi_b0_b1000.bvec" "$BASE/dwi_b0_b1000.bval" \
  -force

echo "2) Response function"
dwi2response tournier "$OUT/dwi.mif" "$OUT/response.txt" \
  -mask "$BASE/mask.nii.gz" \
  -force

echo "3) FOD"
dwi2fod csd "$OUT/dwi.mif" "$OUT/response.txt" "$OUT/fod.mif" \
  -mask "$BASE/mask.nii.gz" \
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
for METRIC in FA MD RD AD; do
  Tractometry \
    -i "$OUT/TOM_trackings" \
    -e "$OUT/endings_segmentations" \
    -s "$BASE/${METRIC}.nii.gz" \
    -o "$OUT/${SUBJ}_pipeline_01_${METRIC}_tractometry.csv"
done

echo "10) QC"
echo "Tracking files:"
ls -lh "$OUT/TOM_trackings"

echo "Done: $OUT"
