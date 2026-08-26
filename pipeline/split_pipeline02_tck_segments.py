import numpy as np
import nibabel as nib
from nibabel.streamlines import Tractogram
from pathlib import Path

PROJECT = Path("/path/to/adni_project")  # <-- EDIT: set to your project root
PIPELINE = "pipeline_02"

TRACTS = [
    "FX_left","FX_right",
    "CG_left","CG_right",
    "UF_left","UF_right",
    "ILF_left","ILF_right",
    "IFO_left","IFO_right",
    "CC_2","CC_7"
]

N_POINTS = 100

def resample_streamline(sl, n_points=100):
    sl = np.asarray(sl)

    if len(sl) < 2:
        return None

    distances = np.sqrt(((np.diff(sl, axis=0)) ** 2).sum(axis=1))
    cumulative = np.insert(np.cumsum(distances), 0, 0)

    if cumulative[-1] == 0:
        return None

    new_distances = np.linspace(0, cumulative[-1], n_points)

    new_sl = np.vstack([
        np.interp(new_distances, cumulative, sl[:, dim])
        for dim in range(3)
    ]).T

    return new_sl.astype(np.float32)

for subj_dir in sorted((PROJECT / PIPELINE).glob("*")):
    if not subj_dir.is_dir():
        continue

    subj = subj_dir.name
    base = subj_dir / "tractometry_TOM"
    track_dir = base / "TOM_trackings"

    seg_dir = base / "segment_visualization"
    seg_dir.mkdir(exist_ok=True)

    print(f"\nProcessing {subj}")

    for tract in TRACTS:
        infile = track_dir / f"{tract}.tck"

        if not infile.exists():
            print(f"Missing: {infile}")
            continue

        tck = nib.streamlines.load(str(infile))
        streamlines = list(tck.streamlines)

        proximal = []
        middle = []
        distal = []

        for sl in streamlines:
            r = resample_streamline(sl, N_POINTS)

            if r is None:
                continue

            proximal.append(r[:33])
            middle.append(r[33:66])
            distal.append(r[66:])

        for label, segs in [
            ("proximal", proximal),
            ("middle", middle),
            ("distal", distal),
        ]:
            outfile = seg_dir / f"{tract}_{label}.tck"

            tractogram = Tractogram(
                segs,
                affine_to_rasmm=np.eye(4)
            )

            nib.streamlines.save(
                tractogram,
                str(outfile)
            )

        print(f"{tract}: {len(streamlines)} streamlines split")

print("\nDONE")
