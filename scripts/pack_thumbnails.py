"""Pack thumbnails into one file addressed by byte range.

Uploading 123,287 individual files took three hours to move 10,338 of them:
per-file HTTP overhead dominates completely and the job would have run 36
hours. One packed file uploads once, and a browser fetches a single thumbnail
with a Range request, which is what it wanted to do anyway since it only ever
shows a dozen at a time.

Packed in gallery order, so the offset table is indexed by row and needs no
key lookup: 8 bytes per image instead of a 6MB JSON of filenames.

    python scripts/pack_thumbnails.py --thumbs /root/thumbs \
        --index /root/gallery_123k_v2.srtidx --out /root/packed
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--thumbs", type=Path, required=True, help="dir of <split>/<file>.jpg")
    p.add_argument("--index", type=Path, required=True, help="srtidx, for gallery order")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def index_keys(path: Path) -> list[str]:
    b = path.read_bytes()
    magic = b[:8]
    dim, count = struct.unpack("<II", b[8:16])
    off = 16
    if magic == b"SRTIDX02":
        off += count * 4 + dim * count
    elif magic == b"SRTIDX01":
        off += dim * count * 2
    else:
        raise SystemExit(f"bad magic {magic!r}")
    keys = []
    for _ in range(count):
        (n,) = struct.unpack("<I", b[off:off + 4])
        off += 4
        keys.append(b[off:off + n].decode())
        off += n
    return keys


def main() -> None:
    a = parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    keys = index_keys(a.index)
    print(f"{len(keys)} keys in gallery order", flush=True)

    blob = a.out / "thumbs.bin"
    table = bytearray()
    missing = 0
    with blob.open("wb") as f:
        pos = 0
        for i, k in enumerate(keys):
            p = a.thumbs / k
            if p.exists():
                data = p.read_bytes()
            else:
                data = b""
                missing += 1
            f.write(data)
            table += struct.pack("<II", pos, len(data))
            pos += len(data)
            if i % 20000 == 0:
                print(f"  {i}/{len(keys)}  {pos / 1e6:.0f} MB", flush=True)

    (a.out / "thumbs_offsets.bin").write_bytes(bytes(table))
    print(f"wrote {blob} {blob.stat().st_size / 1e6:.0f} MB", flush=True)
    print(f"wrote offsets {len(table) / 1e6:.1f} MB for {len(keys)} rows, "
          f"{missing} missing", flush=True)


if __name__ == "__main__":
    main()
