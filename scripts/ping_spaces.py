#!/usr/bin/env python3
"""Keep the Spaces warm, and notice when one is actually broken.

A cpu-basic Space that gets no traffic is reaped. The next visitor gets HF's
cold start: the edge serves the Gradio shell, the browser then blocks on
/config while the container boots, and Gradio renders nothing rather than an
error. A white page, with no way to tell it from a crash.

srt-omni-demo did exactly that: last container start 2026-08-28 19:51, reported
white on 2026-08-30, healthy the moment it was woken (/config 200 in 0.10s).

So this pings every Space on a schedule. It also checks /config rather than
only /, because / is the part that keeps working when the app is down, which
makes it the wrong thing to monitor.

    python scripts/ping_spaces.py --author RiverRider
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import sys
import time

import requests
from huggingface_hub import HfApi


def host(space_id: str) -> str:
    owner, name = space_id.split("/")
    slug = f"{owner}-{name}".lower().replace(".", "-").replace("_", "-")
    return f"https://{slug}.hf.space"


def probe(space_id: str, stage: str, sdk: str, timeout: int) -> dict:
    url = host(space_id)
    t0 = time.time()
    out = {"id": space_id, "stage": stage, "sdk": sdk, "url": url,
           "code": 0, "live": 0}
    # A static Space has no /config; asking for one reports a broken Space that
    # is fine. Gradio cannot render without it, so for gradio it is the only
    # check worth making: / keeps returning 200 when the app is down.
    probe_path = "/" if sdk == "static" else "/config"
    try:
        out["code"] = requests.get(url, timeout=timeout).status_code
        out["live"] = requests.get(f"{url}{probe_path}", timeout=timeout).status_code
    except Exception as e:
        out["err"] = type(e).__name__
    out["secs"] = round(time.time() - t0, 1)
    out["checked"] = probe_path
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--author", default="RiverRider")
    ap.add_argument("--timeout", type=int, default=180,
                    help="a cold container can take minutes to answer")
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()

    api = HfApi()
    spaces = sorted(api.list_spaces(author=a.author), key=lambda s: s.id)
    meta = {}
    for s in spaces:
        try:
            i = api.space_info(s.id)
            meta[s.id] = (i.runtime.stage, getattr(i, "sdk", None) or "?")
        except Exception:
            meta[s.id] = ("?", "?")
    print(f"{len(spaces)} spaces for {a.author}\n")

    with cf.ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(lambda s: probe(s.id, *meta[s.id], a.timeout), spaces))

    print(f"{'space':40s}{'sdk':8s}{'stage':10s}{'/':>5}{'live':>6}{'secs':>7}  checked")
    bad, slow = [], []
    for r in sorted(rows, key=lambda r: r["id"]):
        if r["live"] != 200:
            bad.append(r)
        elif r["secs"] > 10:
            slow.append(r)
        mark = "  <-- not serving" if r["live"] != 200 else (
            "  <-- was cold" if r["secs"] > 10 else "")
        print(f"{r['id']:40s}{r['sdk']:8s}{r['stage']:10s}{r['code']:>5}"
              f"{r['live']:>6}{r['secs']:>7}  {r['checked']}{mark}")

    if slow:
        print(f"\n{len(slow)} were asleep and had to be woken. Every one of those "
              f"is a white page for whoever arrives first:")
        for r in slow:
            print(f"  {r['id']:40s} {r['secs']}s")
    if bad:
        print(f"\n{len(bad)} not serving:")
        for r in bad:
            print(f"  {r['id']}  stage={r['stage']} sdk={r['sdk']}  "
                  f"{r.get('err', 'http ' + str(r['live']))}")
        return 1
    print("\nall serving")
    return 0


if __name__ == "__main__":
    sys.exit(main())
