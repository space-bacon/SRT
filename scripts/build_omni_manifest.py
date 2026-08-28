"""Build one manifest covering every modality, so the encoder has a single input.

Each row is {modality, key, caption}. Images reference a file on disk, audio and
video reference a dataset row we decode at encode time. Keeping one schema means
the encoder does not branch on dataset quirks, and the gallery is genuinely mixed
rather than three galleries in a trenchcoat.

    python scripts/build_omni_manifest.py --n-image 5000 --n-audio 2000 --n-video 2000
"""
from __future__ import annotations

import argparse
import json
import os

CAP_KEYS = ("caption", "captions", "sentence", "text", "caption_string")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--img-dir", default="/root/val2017")
    p.add_argument("--n-image", type=int, default=5000)
    p.add_argument("--n-audio", type=int, default=2000)
    p.add_argument("--n-video", type=int, default=2000)
    p.add_argument("--out", default="/root/omni_manifest.json")
    return p.parse_args()


def first_caption(v):
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, (list, tuple)) and v:
        return str(v[0]).strip()
    return None


def images(img_dir, n):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(
        "phiyodr/coco2017",
        "data/validation-00000-of-00001-e3c37e369512a3aa.parquet",
        repo_type="dataset")
    t = pq.read_table(p)
    cols = t.column_names
    print(f"  coco2017 columns: {cols}", flush=True)
    fn_col = next((c for c in ("file_name", "image_path", "coco_url", "image_id")
                   if c in cols), None)
    cap_col = next((c for c in CAP_KEYS if c in cols), None)
    if fn_col is None or cap_col is None:
        raise SystemExit(f"cannot map coco columns from {cols}")
    have = set(os.listdir(img_dir))
    rows, seen = [], set()
    for fn, cap in zip(t.column(fn_col).to_pylist(), t.column(cap_col).to_pylist()):
        base = os.path.basename(str(fn))
        c = first_caption(cap)
        if not c or base in seen or base not in have:
            continue
        seen.add(base)
        rows.append({"modality": "image", "key": base, "caption": c})
        if len(rows) >= n:
            break
    return rows


def audio_rows(n):
    """AudioCaps ships a loader script, which datasets 5.x no longer runs, so read
    the metadata CSV and unpack the matching zip directly."""
    import csv
    import zipfile
    from huggingface_hub import hf_hub_download
    meta = hf_hub_download("confit/audiocaps", "metadata/val.csv", repo_type="dataset")
    z = hf_hub_download("confit/audiocaps", "data/val.zip", repo_type="dataset")
    out_dir = "/root/audiocaps_val"
    if not os.path.isdir(out_dir):
        zipfile.ZipFile(z).extractall(out_dir)
    have = {}
    for root, _, files in os.walk(out_dir):
        for f in files:
            if f.lower().endswith((".wav", ".flac", ".mp3", ".ogg")):
                have[os.path.splitext(f)[0]] = os.path.join(root, f)
    rows = []
    with open(meta) as fh:
        for r in csv.DictReader(fh):
            cap = r.get("caption") or r.get("text")
            # Files are named {audiocap_id}.wav, so that key must come first.
            key = r.get("audiocap_id") or r.get("youtube_id") or r.get("file_name")
            if not cap or not key:
                continue
            path = have.get(str(key)) or have.get(os.path.splitext(str(key))[0])
            if not path:
                continue
            rows.append({"modality": "audio", "key": path, "caption": cap.strip()})
            if len(rows) >= n:
                break
    return rows


def video_rows(n):
    """MSR-VTT: captions in a JSON, clips in one zip."""
    import zipfile
    from huggingface_hub import hf_hub_download
    j = hf_hub_download("friedrichor/MSR-VTT", "msrvtt_test_1k.json", repo_type="dataset")
    z = hf_hub_download("friedrichor/MSR-VTT", "MSRVTT_Videos.zip", repo_type="dataset")
    out_dir = "/root/msrvtt_videos"
    if not os.path.isdir(out_dir):
        zipfile.ZipFile(z).extractall(out_dir)
    have = {}
    for root, _, files in os.walk(out_dir):
        for f in files:
            if f.lower().endswith((".mp4", ".avi", ".webm")):
                have[f] = os.path.join(root, f)
    rows = []
    for r in json.load(open(j)):
        cap, vid = r.get("caption"), r.get("video")
        path = have.get(str(vid))
        if not cap or not path:
            continue
        rows.append({"modality": "video", "key": path, "caption": cap.strip()})
        if len(rows) >= n:
            break
    return rows


def main() -> None:
    a = parse_args()
    man = []

    print("images:", flush=True)
    try:
        im = images(a.img_dir, a.n_image)
        print(f"  {len(im)} image rows", flush=True)
        man += im
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}", flush=True)

    print("audio:", flush=True)
    try:
        au = audio_rows(a.n_audio)
        print(f"  {len(au)} audio rows", flush=True)
        man += au
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}", flush=True)

    print("video:", flush=True)
    try:
        vi = video_rows(a.n_video)
        print(f"  {len(vi)} video rows", flush=True)
        man += vi
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}", flush=True)

    counts = {}
    for r in man:
        counts[r["modality"]] = counts.get(r["modality"], 0) + 1
    with open(a.out, "w") as f:
        json.dump({"counts": counts, "rows": man}, f)
    print(f"\nwrote {a.out}: {counts}", flush=True)


if __name__ == "__main__":
    main()
