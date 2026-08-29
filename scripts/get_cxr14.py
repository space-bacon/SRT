#!/usr/bin/env python3
"""Fetch NIH ChestX-ray14 and emit the manifest shape the encoder already takes.

Detection, not early detection. These labels describe what is visible in the
image in front of you, so nothing here can speak to catching disease before it
is apparent. That needs longitudinal data with outcomes.

Two things are carried through to the manifest because the result is worthless
without them. The official train_val/test lists are patient-disjoint, and any
split we invented ourselves would put the same patient on both sides and inflate
every number. View position is kept because portable AP films are taken of
sicker, bedbound patients, so it is a route by which a probe can score well
without reading any pathology.

The full set is 45 GB in twelve shards. Shards are downloaded in order and each
one is deleted after extraction.

    python scripts/get_cxr14.py --shards 4
"""
import argparse
import csv
import json
import os
import pathlib
import zipfile

REPO = "alkzar90/NIH-Chest-X-ray-dataset"
FINDINGS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
            "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
            "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=4, help="of 12, ~4 GB each")
    ap.add_argument("--out-dir", default="/root/cxr14")
    ap.add_argument("--manifest", default="/root/cxr14_manifest.json")
    a = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/root/.hf_home")
    from huggingface_hub import hf_hub_download

    img_dir = pathlib.Path(a.out_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    meta = {}
    for name in ("Data_Entry_2017_v2020.csv", "test_list.txt", "train_val_list.txt"):
        meta[name] = hf_hub_download(REPO, f"data/{name}", repo_type="dataset")
    test_set = set(open(meta["test_list.txt"]).read().split())
    print(f"official test list: {len(test_set)} images", flush=True)

    rows_by_key = {}
    with open(meta["Data_Entry_2017_v2020.csv"]) as f:
        for r in csv.DictReader(f):
            rows_by_key[r["Image Index"]] = r

    have = set()
    for i in range(1, a.shards + 1):
        fn = f"images_{i:03d}.zip"
        print(f"=== {fn}", flush=True)
        z = hf_hub_download(REPO, f"data/images/{fn}", repo_type="dataset")
        with zipfile.ZipFile(z) as zf:
            for m in zf.namelist():
                if not m.lower().endswith(".png"):
                    continue
                base = os.path.basename(m)
                with zf.open(m) as src, open(img_dir / base, "wb") as dst:
                    dst.write(src.read())
                have.add(base)
        os.remove(z)
        print(f"    {len(have)} images extracted, shard removed", flush=True)

    rows = []
    for key in sorted(have):
        r = rows_by_key.get(key)
        if r is None:
            continue
        labels = [x for x in r["Finding Labels"].split("|") if x != "No Finding"]
        rows.append({
            "modality": "image",
            "key": key,
            "caption": r["Finding Labels"].replace("|", ", "),
            "labels": labels,
            "patient_id": int(r["Patient ID"]),
            "view": r["View Position"],
            "split": "test" if key in test_set else "train",
        })

    json.dump({"counts": {"image": len(rows)}, "rows": rows}, open(a.manifest, "w"))
    n_test = sum(1 for r in rows if r["split"] == "test")
    pats = len({r["patient_id"] for r in rows})
    print(f"\n{len(rows)} images, {pats} patients, {n_test} in official test split")
    print(f"  AP view: {sum(1 for r in rows if r['view'] == 'AP')}")
    for f in FINDINGS:
        n = sum(1 for r in rows if f in r["labels"])
        print(f"    {f:20s} {n:6d}  ({100*n/len(rows):.1f}%)")
    print(f"\nmanifest -> {a.manifest}")


if __name__ == "__main__":
    main()
