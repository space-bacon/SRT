"""Why do candidates fail to produce a signature?

The chat-native selector only gets a reading from 58% of candidates, and a
candidate it cannot run is a candidate it cannot count. That is the single
biggest weakness in the method, and nobody has looked at the failures.

This replays banked candidates through the same probe the selector uses, but
captures the child's stderr instead of discarding it, and tabulates what
actually goes wrong.

Execution belongs on a box you are willing to lose.

    python probe_autopsy.py --data consensus_demo.json --n 60
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/root/consensus")
sys.path.insert(0, "/root/consensus/scripts")

from chat_consensus import code_of, infer_kinds, pick_entry  # noqa: E402
from consensus_select import _value, probe_program  # noqa: E402
from srt_select.sandbox import ENV, GUARD  # noqa: E402

import subprocess  # noqa: E402

EXC = re.compile(r"^(\w+Error|\w+Exception|SystemExit|KeyboardInterrupt)\b", re.M)


def run_verbose(program: str, timeout: float = 8.0):
    """The sandbox's run(), but keeping why the child died."""
    try:
        p = subprocess.run([sys.executable, "-I", "-c", GUARD + program],
                           capture_output=True, text=True, timeout=timeout,
                           cwd="/tmp", env=ENV)
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    if p.returncode == 0:
        return p.stdout, None
    m = EXC.findall(p.stderr or "")
    return None, (m[-1] if m else f"exit{p.returncode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/root/consensus/consensus_demo.json")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--rung", default="7B")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    B = json.load(open(a.data))
    probs = B["problems"]
    ents = B["rungs"][a.rung]["entries"]
    idx = [i for i, e in enumerate(ents) if e.get("ok")][: a.n]

    reasons: Counter[str] = Counter()
    per_pool = []

    def one(i: int):
        e, p = ents[i], probs[i]
        codes = [code_of(c) for c in e["candidates"]]
        entry = pick_entry(codes, p["prompt"])
        kinds = infer_kinds(codes, entry, p["prompt"])
        if not (entry and kinds):
            return None
        rng = random.Random(0)
        cases = [tuple(_value(k, rng) for k in kinds) for _ in range(6)]
        local = Counter()
        ok = 0
        for c in codes:
            out, why = run_verbose(probe_program(c, entry, cases))
            if out is not None:
                ok += 1
            else:
                local[why] += 1
        return ok, local, kinds

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(one, idx):
            if r is None:
                continue
            ok, local, kinds = r
            reasons.update(local)
            per_pool.append((ok, kinds))

    ran = sum(o for o, _ in per_pool)
    tot = len(per_pool) * len(ents[idx[0]]["candidates"])
    print(f"rung {a.rung}: {len(per_pool)} pools, run rate {ran / tot:.3f} ({ran}/{tot})")
    print("\nwhy candidates failed to produce a signature:")
    fails = sum(reasons.values())
    for why, n in reasons.most_common(14):
        print(f"  {why:24} {n:5d}  {n / fails:6.1%}")

    print("\nrun rate by inferred signature:")
    byk: dict[str, list[int]] = {}
    for ok, kinds in per_pool:
        key = ",".join(kinds)
        byk.setdefault(key, [0, 0])
        byk[key][0] += ok
        byk[key][1] += len(ents[idx[0]]["candidates"])
    for key, (o, t) in sorted(byk.items(), key=lambda kv: -kv[1][1])[:10]:
        print(f"  {key:28} {o / t:6.3f}  (n={t})")


if __name__ == "__main__":
    main()
