"""Compare code extractors on RAW replies, scored against the real pass matrix.

An earlier version of this ran on candidates that had already been through the
current extractor, so the arms were indistinguishable and the result meant
nothing. This reads the raw generations, applies each extractor from scratch,
and scores the selector's pick against HumanEval's own tests.

    python extract_ab2.py --arm coder7B_inst__chat --k 8
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, "/root/consensus")
sys.path.insert(0, "/root/consensus/scripts")

from chat_consensus import code_of, infer_kinds, pick_entry  # noqa: E402
from consensus_select import _value, probe_program  # noqa: E402
from srt_select.sandbox import ENV, GUARD  # noqa: E402

FENCE = re.compile(r"```(?:[a-zA-Z]*)\n(.*?)(?:```|\Z)", re.S)
EXC = re.compile(r"^(\w+Error|\w+Exception)\b", re.M)


def parses(src: str) -> bool:
    try:
        ast.parse(src)
        return True
    except (SyntaxError, ValueError):
        return False


def trunc(src: str) -> str:
    if parses(src):
        return src
    lines = src.split("\n")
    for cut in range(len(lines), 0, -1):
        head = "\n".join(lines[:cut])
        if head.strip() and parses(head):
            return head
    return src


def v_trunc(reply: str) -> str:
    """Longest block, truncated until it parses."""
    b = FENCE.findall(reply)
    return trunc(max(b, key=len) if b else reply)


def v_concat(reply: str) -> str:
    """All blocks joined so imports survive, then truncated."""
    b = FENCE.findall(reply)
    if not b:
        return trunc(reply)
    joined = "\n\n".join(x.strip() for x in b)
    return joined if parses(joined) else trunc(max(b, key=len))


ARMS = {"current": code_of, "trunc": v_trunc, "concat": v_concat}


def run_v(program: str, timeout: float = 8.0):
    try:
        p = subprocess.run([sys.executable, "-I", "-c", GUARD + program],
                           capture_output=True, text=True, timeout=timeout,
                           cwd="/tmp", env=ENV)
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    if p.returncode == 0:
        return p.stdout, None
    m = EXC.findall(p.stderr or "")
    return None, (m[-1] if m else "other")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="coder7B_inst__chat")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--raw", default="/root/consensus/raw")
    ap.add_argument("--passmat", default="/root/consensus/passmat")
    ap.add_argument("--probs", default="/root/consensus/data/humaneval.json")
    a = ap.parse_args()

    probs = json.load(open(a.probs))
    rows = json.load(open(os.path.join(a.raw, f"{a.arm}.json")))
    ok = np.load(os.path.join(a.passmat, f"{a.arm}.npz"))["ok"][:, : a.k]

    def one(i: int):
        raw = rows[i][: a.k]
        res = {}
        for tag, fn in ARMS.items():
            codes = [fn(c) for c in raw]
            entry = pick_entry(codes, probs[i]["prompt"])
            kinds = infer_kinds(codes, entry, probs[i]["prompt"]) if entry else None
            if not (entry and kinds):
                res[tag] = (0, None, Counter({"no-entry": len(raw)}))
                continue
            rng = random.Random(0)
            cases = [tuple(_value(k, rng) for k in kinds) for _ in range(6)]
            sigs, why = [], Counter()
            for c in codes:
                s, w = run_v(probe_program(c, entry, cases))
                sigs.append(s)
                if s is None:
                    why[w] += 1
            valid = [s for s in sigs if s is not None]
            if not valid:
                res[tag] = (0, None, why)
                continue
            top = Counter(valid).most_common(1)[0][0]
            res[tag] = (len(valid), next(j for j, s in enumerate(sigs) if s == top), why)
        return i, res

    agg = {t: [0, 0, 0, 0, Counter()] for t in ARMS}  # ran, tot, resolved, passed, why
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, res in ex.map(one, range(len(rows))):
            for t, (ran, pick, why) in res.items():
                agg[t][0] += ran
                agg[t][1] += a.k
                agg[t][4].update(why)
                if pick is not None:
                    agg[t][2] += 1
                    agg[t][3] += bool(ok[i, pick])

    print(f"\n{a.arm}  K={a.k}  {len(rows)} problems")
    print(f"{'extractor':12} {'run rate':>9} {'resolved':>9} {'pick passes':>12} "
          f"{'pick passes (of all)':>21}")
    for t in ARMS:
        ran, tot, res, ps, _ = agg[t]
        print(f"{t:12} {ran / tot:>9.3f} {res:>9d} {ps / res if res else 0:>12.3f} "
              f"{ps / len(rows):>21.3f}")
    print(f"\nsingle-sample baseline: {ok.mean():.3f}   oracle: {ok.any(1).mean():.3f}")


if __name__ == "__main__":
    main()
