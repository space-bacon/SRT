#!/usr/bin/env python3
"""Turn the longitudinal NLST CT subset into lesion-centred and control slices.

Source is kk08975/NLST on the Hub: 163 participants, most with all three annual
screening rounds, every volume paired with a lesion segmentation. It carries no
outcomes, so nothing here can test early detection. What it does support without
any labels is whether a frozen backbone separates lesion tissue from healthy
tissue in the SAME patient and the same scanner, and whether that separation
tracks across the three rounds.

Controls come from the same volume on purpose. Comparing across patients would
let scanner, site and body habitus stand in for the lesion.

Only the requested patients are downloaded, since the full set is 87.6 GB and
the box does not have it to spare.

    python scripts/get_nlst_long.py --patients 40
"""
import argparse
import collections
import json
import os
import pathlib
import re

REPO = "kk08975/NLST"
# Lung window, the setting a radiologist reads nodules at.
WL, WW = -600.0, 1500.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--patients", type=int, default=40)
    p.add_argument("--slices-per-volume", type=int, default=3,
                   help="lesion slices, matched by the same number of controls")
    p.add_argument("--min-lesion-voxels", type=int, default=20)
    p.add_argument("--control-gap", type=int, default=20,
                   help="slices a control must sit away from any lesion")
    p.add_argument("--out-dir", default="/root/nlst_slices")
    p.add_argument("--manifest", default="/root/nlst_manifest.json")
    return p.parse_args()


def window(vol):
    """HU to 8-bit. Some conversions keep the raw stored values, so shift those."""
    v = vol.astype("float32")
    if v.min() >= -10:  # raw, not yet rescaled to HU
        v = v - 1024.0
    lo, hi = WL - WW / 2, WL + WW / 2
    return (((v.clip(lo, hi) - lo) / (hi - lo)) * 255).astype("uint8")


def main():
    a = parse_args()
    os.environ.setdefault("HF_HOME", "/root/.hf_home")
    import numpy as np
    import nibabel as nib
    from huggingface_hub import HfApi, hf_hub_download
    from PIL import Image

    files = [s.rfilename for s in HfApi().dataset_info(REPO).siblings]
    studies = collections.defaultdict(lambda: {"vol": None, "seg": None})
    for f in files:
        m = re.match(r"(\d+)-\d+/(\d+)-(\d{8})/(.+)$", f)
        if not m or not f.endswith(".nii"):
            continue
        key = (m.group(1), m.group(3))
        if "Segmentation" in f or "label" in f:
            studies[key]["seg"] = f
        else:
            studies[key]["vol"] = f

    by_pat = collections.defaultdict(list)
    for (pat, date), d in studies.items():
        if d["vol"] and d["seg"]:
            by_pat[pat].append((date, d))
    # Prefer participants with all three rounds, since the trajectory is the point.
    chosen = sorted(by_pat, key=lambda p: (-len(by_pat[p]), p))[:a.patients]
    print(f"{len(by_pat)} participants available, taking {len(chosen)}", flush=True)

    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for n, pat in enumerate(chosen, 1):
        for t, (date, d) in enumerate(sorted(by_pat[pat])):
            try:
                vp = hf_hub_download(REPO, d["vol"], repo_type="dataset")
                sp = hf_hub_download(REPO, d["seg"], repo_type="dataset")
                vol = nib.load(vp).get_fdata()
                seg = np.asarray(nib.load(sp).dataobj)
            except Exception as e:
                print(f"  skip {pat} {date}: {type(e).__name__}", flush=True)
                continue
            if vol.shape != seg.shape:
                print(f"  skip {pat} {date}: shape {vol.shape} vs {seg.shape}",
                      flush=True)
                continue

            per_z = (seg > 0).sum(axis=(0, 1))
            lesion_z = np.where(per_z >= a.min_lesion_voxels)[0]
            if len(lesion_z) == 0:
                continue
            top = lesion_z[np.argsort(-per_z[lesion_z])][:a.slices_per_volume]
            far = np.array([z for z in range(vol.shape[2])
                            if np.abs(lesion_z - z).min() > a.control_gap])
            if len(far) == 0:
                continue
            ctrl = far[np.linspace(0, len(far) - 1, min(len(top), len(far))).astype(int)]

            img8 = window(vol)
            for z, has in [(int(z), True) for z in top] + [(int(z), False) for z in ctrl]:
                key = f"{pat}_{date}_z{z:03d}_{'les' if has else 'ctl'}.png"
                Image.fromarray(img8[:, :, z]).convert("RGB").save(out / key)
                rows.append({
                    "modality": "image", "key": key,
                    "caption": "chest computed tomography, axial slice",
                    "patient": pat, "date": date, "timepoint": t,
                    "has_lesion": has, "z": z,
                    "lesion_voxels": int(per_z[z]),
                })
            os.remove(vp)
            os.remove(sp)
        if n % 5 == 0:
            print(f"  {n}/{len(chosen)} participants, {len(rows)} slices", flush=True)

    json.dump({"counts": {"image": len(rows)}, "rows": rows}, open(a.manifest, "w"))
    n_les = sum(1 for r in rows if r["has_lesion"])
    print(f"\n{len(rows)} slices: {n_les} lesion, {len(rows) - n_les} control")
    print(f"  participants {len({r['patient'] for r in rows})}, "
          f"studies {len({(r['patient'], r['date']) for r in rows})}")
    print(f"manifest -> {a.manifest}")


if __name__ == "__main__":
    main()
