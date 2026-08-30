#!/usr/bin/env python3
"""Cut the same 300 demo films from all three backbones, so the Space can show them disagreeing.

The Space already ships a 300-film test-split subset of gemma-4's states. The
pooled result is only legible if a visitor can see the three backbones score the
same film differently and the average land better than any of them, so the
subset needs the other two backbones at the identical keys.

Keys rather than row offsets. All three vendors happen to share an `ok` mask, so
positional indexing would work today, and would break silently the first time
one backbone fails on an image the others handle. This asserts the overlap
instead.

gemma-4 is taken from the subset the Space already ships rather than re-sliced.
The local `cxr14_gemma4.npz` is the 34,999-image pilot, not the full 112,120,
and re-slicing from it would have quietly produced the wrong films.

    python scripts/build_pooled_demo_subset.py
"""
import argparse
import json

import numpy as np
from huggingface_hub import HfApi, hf_hub_download

DATA = "RiverRider/srt-cxr14-frozen-probe"
VENDORS = ["aria", "qwen3omni"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--states-dir", default="artifacts/nla/omni/states")
    p.add_argument("--out", default="/tmp/test_subset_3vendor.npz")
    p.add_argument("--upload", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    man = hf_hub_download(DATA, "manifests/cxr14_manifest.json", repo_type="dataset")
    rows_all = json.load(open(man))["rows"]
    sub = np.load(hf_hub_download(DATA, "demo/test_subset.npz", repo_type="dataset"),
                  allow_pickle=True)
    keys = [str(k) for k in sub["keys"]]
    print(f"{len(keys)} demo films, {len(rows_all)} manifest rows")

    out = {"labels": sub["labels"], "keys": sub["keys"], "view": sub["view"],
           "states_gemma4": sub["states"].astype(np.float32)}
    print(f"  {'gemma4':10} {out['states_gemma4'].shape}  (from the shipped subset)")
    for v in VENDORS:
        z = np.load(f"{a.states_dir}/cxr14_{v}.npz")
        ok = z["ok"]
        if len(rows_all) != len(ok):
            raise SystemExit(f"{v}: manifest {len(rows_all)} rows, states {len(ok)}")
        idx = {r["key"]: i for i, (r, k) in
               enumerate((r, k) for r, k in zip(rows_all, ok) if k)}
        missing = [k for k in keys if k not in idx]
        if missing:
            raise SystemExit(f"{v}: {len(missing)} demo keys absent, e.g. {missing[:3]}")
        X = z["item"][ok].astype(np.float32)
        out[f"states_{v}"] = X[np.array([idx[k] for k in keys])]
        print(f"  {v:10} {out[f'states_{v}'].shape}")

    np.savez_compressed(a.out, **out)
    import os
    print(f"wrote {a.out}  {os.path.getsize(a.out) / 1e6:.1f} MB")
    if a.upload:
        HfApi().upload_file(path_or_fileobj=a.out, repo_id=DATA, repo_type="dataset",
                            path_in_repo="demo/test_subset_3vendor.npz")
        print("  -> demo/test_subset_3vendor.npz")


if __name__ == "__main__":
    main()
