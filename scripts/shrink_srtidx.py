"""Cut an existing .srtidx down to its first N rows.

The browser tier holds the whole gallery resident, so gallery size is a memory
decision, not just a download one. This makes a smaller one without re-encoding
anything: the rows are already projected and normalised, so a prefix of them is
a valid index over a smaller pool.

Recall falls as the pool grows, so a smaller gallery scores *better* on paper
while searching less. Report the pool size next to any number from it.

    python scripts/shrink_srtidx.py --in gallery_123k_v3.srtidx \
        --limit 20000 --out gallery_20k_v3.srtidx
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

MAGIC_F16 = b"SRTIDX01"
MAGIC_I8 = b"SRTIDX02"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="src", type=Path, required=True)
    p.add_argument("--limit", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    raw = a.src.read_bytes()
    magic = raw[:8]
    if magic not in (MAGIC_F16, MAGIC_I8):
        raise SystemExit(f"{a.src}: not an srtidx (magic {magic!r})")
    dim, count = struct.unpack_from("<II", raw, 8)
    if a.limit >= count:
        raise SystemExit(f"--limit {a.limit} is not smaller than the {count} rows present")

    off = 16
    if magic == MAGIC_I8:
        scales = raw[off : off + 4 * count]
        off += 4 * count
        row_bytes = dim
    else:
        scales = b""
        row_bytes = dim * 2
    data = raw[off : off + row_bytes * count]
    off += row_bytes * count

    keys = []
    for _ in range(count):
        (n,) = struct.unpack_from("<I", raw, off)
        off += 4
        keys.append(raw[off : off + n])
        off += n
    if off != len(raw):
        raise SystemExit(f"{a.src}: {len(raw) - off} trailing bytes, refusing to guess")

    k = a.limit
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("wb") as f:
        f.write(magic)
        f.write(struct.pack("<II", dim, k))
        if scales:
            f.write(scales[: 4 * k])
        f.write(data[: row_bytes * k])
        for key in keys[:k]:
            f.write(struct.pack("<I", len(key)))
            f.write(key)

    mb = a.out.stat().st_size / 1e6
    print(f"wrote {a.out}: {k} x {dim} ({magic.decode()}), {mb:.1f} MB")
    print(f"  from {count} rows, {a.src.stat().st_size / 1e6:.1f} MB")
    print(f"  first key {keys[0].decode()}, last kept {keys[k - 1].decode()}")


if __name__ == "__main__":
    main()
