"""A/B a stricter code extractor against the one the selector currently uses.

WITHDRAWN. This script reads candidates from `consensus_demo.json`, whose
`candidates` field was already passed through `code_of()` when the bank was
built. Both arms therefore operate on pre-extracted text and differ on 0 of
2552 replies, so its numbers compare an extractor against itself. Superseded by
`extract_ab2.py`, which reads the raw replies in `artifacts/nla/he_k32/` and
scores against the committed pass matrices. Kept because the failure mode
(comparing two readers of an already-transformed input) is easy to repeat.

The autopsy says 95% of unreadable candidates are SyntaxError, NameError or
IndentationError, which are extraction faults rather than the candidate being
wrong. This measures whether parsing the reply properly recovers them.

    python extract_ab.py --rung 7B --n 60
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/root/consensus")
sys.path.insert(0, "/root/consensus/scripts")

from chat_consensus import code_of, infer_kinds, pick_entry  # noqa: E402
from consensus_select import _value, probe_program  # noqa: E402
from srt_select.sandbox import ENV, GUARD  # noqa: E402

FENCE = re.compile(r"```(?:[a-zA-Z]*)\n(.*?)(?:```|\Z)", re.S)


def parses(src: str) -> bool:
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False


def code_of_v3(reply: str) -> str:
    """Truncate-to-parseable only. Never concatenates, so semantics cannot shift."""
    blocks = FENCE.findall(reply)
    src = max(blocks, key=len) if blocks else reply
    if parses(src):
        return src
    lines = src.split("\n")
    for cut in range(len(lines), 0, -1):
        head = "\n".join(lines[:cut])
        if head.strip() and parses(head):
            return head
    return src


def code_of_v2(reply: str) -> str:
    """Prefer a parseable program, and keep every block so imports survive."""
    blocks = FENCE.findall(reply)
    if blocks:
        joined = "\n\n".join(b.strip() for b in blocks)
        if parses(joined):
            return joined
        longest = max(blocks, key=len)
        if parses(longest):
            return longest
        # An unclosed fence trails prose; drop lines from the end until it parses.
        lines = longest.split("\n")
        for cut in range(len(lines), 0, -1):
            head = "\n".join(lines[:cut])
            if head.strip() and parses(head):
                return head
        return longest
    if parses(reply):
        return reply
    lines = reply.split("\n")
    for cut in range(len(lines), 0, -1):
        head = "\n".join(lines[:cut])
        if head.strip() and parses(head):
            return head
    return reply


EXC = re.compile(r"^(\w+Error|\w+Exception)\b", re.M)


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
    ap.add_argument("--data", default="/root/consensus/consensus_demo.json")
    ap.add_argument("--rung", default="7B")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    B = json.load(open(a.data))
    probs, ents = B["problems"], B["rungs"][a.rung]["entries"]
    idx = [i for i, e in enumerate(ents) if e.get("ok")][: a.n]

    def one(i: int):
        e, p = ents[i], probs[i]
        raw = e["candidates"]
        out = {}
        for tag, fn in (("v1", code_of), ("v2", code_of_v2), ("v3", code_of_v3)):
            codes = [fn(c) for c in raw]
            entry = pick_entry(codes, p["prompt"])
            kinds = infer_kinds(codes, entry, p["prompt"]) if entry else None
            if not (entry and kinds):
                out[tag] = (0, len(raw), Counter({"no-entry": len(raw)}), None)
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
            top = Counter(valid).most_common(1)[0] if valid else (None, 0)
            pick = next((j for j, s in enumerate(sigs) if s == top[0]), None)
            out[tag] = (len(valid), len(raw), why, pick)
        return out, e

    agg = {t: [0, 0, Counter(), 0, 0] for t in ("v1", "v2", "v3")}
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for out, e in ex.map(one, idx):
            for tag in ("v1", "v2", "v3"):
                ran, tot, why, pick = out[tag]
                agg[tag][0] += ran
                agg[tag][1] += tot
                agg[tag][2].update(why)
                if pick is not None:
                    agg[tag][3] += 1
                    agg[tag][4] += bool(e["passed"][pick])

    print(f"rung {a.rung}, {len(idx)} pools\n")
    print(f"{'':6} {'run rate':>10} {'resolved':>10} {'pick passes':>12}")
    names = {"v1": "current", "v2": "concat+trunc", "v3": "trunc only"}
    for tag in ("v1", "v2", "v3"):
        ran, tot, why, res, ok = agg[tag]
        name = names[tag]
        print(f"{name:8} {ran / tot:>10.3f} {res:>10d} {ok / res if res else 0:>12.3f}")
    print("\nremaining failures, stricter extractor:")
    f2 = agg["v2"][2]
    tot2 = sum(f2.values()) or 1
    for w, n in f2.most_common(8):
        print(f"  {w:20} {n:5d}  {n / tot2:6.1%}")


if __name__ == "__main__":
    main()
