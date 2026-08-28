#!/usr/bin/env python3
"""Pack audio clips and video frames for the omni demo.

The states already cover image, audio and video, but the Space can only show
photographs because it has no media for the other two. This fetches the source
archives once, transcodes to something small enough to ship, and packs each
modality into one blob plus an offset index, matching the image thumbnails.
"""
import io
import json
import pathlib
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
from huggingface_hub import hf_hub_download

MANIFEST = "artifacts/nla/omni/omni_manifest.json"
OUT = pathlib.Path("artifacts/nla/omni/media")
WORK = pathlib.Path("/tmp/omni_media")
FRAME_AT = "00:00:01"
SIZE = 224


def keys_for(modality):
    rows = json.load(open(MANIFEST))["rows"]
    return [pathlib.PurePosixPath(r["key"]).name
            for r in rows if r["modality"] == modality]


def extract(zip_path, wanted, dest):
    """Pull only the members we need; these archives are ~2GB."""
    dest.mkdir(parents=True, exist_ok=True)
    want = set(wanted)
    got = {}
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            name = pathlib.PurePosixPath(info.filename).name
            if name in want and not info.is_dir():
                target = dest / name
                if not target.exists():
                    with z.open(info) as src, open(target, "wb") as out:
                        out.write(src.read())
                got[name] = target
    return got


def to_mp3(item):
    name, path = item
    out = path.with_suffix(".mp3")
    if not out.exists():
        r = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(path),
             "-ac", "1", "-ar", "22050", "-b:a", "32k", str(out)],
            capture_output=True)
        if r.returncode != 0 or not out.exists():
            return name, None
    data = out.read_bytes()
    # Zero-length source clips produce near-empty audio; drop them.
    return (name, data) if len(data) > 2000 else (name, None)


def to_frame(item):
    name, path = item
    out = path.with_suffix(".jpg")
    if not out.exists():
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", FRAME_AT,
             "-i", str(path), "-frames:v", "1", str(out)], capture_output=True)
    if not out.exists():
        return name, None
    try:
        im = Image.open(out).convert("RGB")
        im.thumbnail((SIZE, SIZE), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=78, optimize=True)
        return name, buf.getvalue()
    except Exception:
        return name, None


def pack(blobs, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    index, offset = {}, 0
    with open(OUT / f"{stem}.bin", "wb") as fh:
        for k, b in blobs.items():
            fh.write(b)
            index[k] = (offset, len(b))
            offset += len(b)
    (OUT / f"{stem}_index.json").write_text(json.dumps(index))
    print(f"  wrote {stem}: {len(index)} items, {offset/1e6:.1f} MB")


def run(modality, repo, member, worker, stem):
    want = keys_for(modality)
    print(f"{modality}: {len(want)} wanted")
    print(f"  downloading {member} (this is the slow part)", flush=True)
    zp = hf_hub_download(repo, member, repo_type="dataset")
    print("  extracting", flush=True)
    got = extract(zp, want, WORK / modality)
    print(f"  extracted {len(got)}", flush=True)
    blobs = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, (name, data) in enumerate(ex.map(worker, got.items()), 1):
            if data:
                blobs[name] = data
            if i % 200 == 0:
                print(f"  {i}/{len(got)} ok={len(blobs)}", flush=True)
    pack(blobs, stem)
    return len(blobs), len(want)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("audio", "both"):
        run("audio", "confit/audiocaps", "data/val.zip", to_mp3, "audio")
    if which in ("video", "both"):
        run("video", "friedrichor/MSR-VTT", "MSRVTT_Videos.zip", to_frame, "video")
    print("MEDIA DONE")
