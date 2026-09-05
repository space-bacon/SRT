"""Consensus selection from a chat turn alone.

The benchmark version of this is given entry_point, a test suite and a canonical
solution. A chat assistant has none of that. It has the user's message and K replies,
and it has to choose one before anyone knows which is right.

Everything the selector needs is recovered from the candidates themselves:

  which function is the answer   the name most candidates define
  what arguments it takes        the majority signature, or a call in the user's text
  what counts as agreement       identical outputs on the same synthesised inputs

Nothing here reads a test suite or a reference solution. `--validate` scores the
choice against held-out tests afterwards, which is measurement, not input.

    python scripts/chat_consensus.py --validate humaneval
"""
import argparse
import ast
import json
import os
import random
import re
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
from consensus_select import POOL, _kind_of, _value, probe_program, run_capture  # noqa: E402

FENCE = re.compile(r"```(?:[a-zA-Z]*)\n(.*?)(?:```|\Z)", re.S)


def code_of(reply):
    """The runnable part of a chat reply."""
    blocks = FENCE.findall(reply)
    if blocks:
        return max(blocks, key=len)
    return reply


def _defs(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    return [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def pick_entry(codes, user_text=""):
    """The function name the pool agrees it is writing."""
    asked = [n.name for n in _defs(user_text)]
    names = Counter()
    for c in codes:
        d = _defs(c)
        if not d:
            continue
        for n in d:
            names[n.name] += 1
    if not names:
        return None
    if asked:
        for a in asked:
            if a in names:
                return a
    return names.most_common(1)[0][0]


def _ann_kind(node):
    if node is None:
        return None
    try:
        t = ast.unparse(node).lower()
    except Exception:
        return None
    for k in ("list", "tuple", "dict", "set"):
        if t.startswith(k):
            inner = next((x for x in ("str", "float", "int") if x in t[len(k):]), "int")
            return f"{k}[{inner}]"
    return next((x for x in ("int", "float", "str", "bool") if x in t), None)


def infer_kinds(codes, entry, user_text=""):
    """Argument kinds, from an example call if the user gave one, else the signatures."""
    for m in re.finditer(rf"{re.escape(entry)}\s*\(([^\n)]*)\)", user_text):
        try:
            args = ast.literal_eval("(" + m.group(1) + ",)")
        except Exception:
            continue
        if args:
            return [_kind_of(v) for v in args]
    sigs = Counter()
    ann = {}
    for c in codes:
        for n in _defs(c):
            if n.name != entry:
                continue
            names = [a.arg for a in n.args.args]
            sigs[len(names)] += 1
            for a in n.args.args:
                k = _ann_kind(a.annotation)
                if k:
                    ann.setdefault(a.arg, k)
    if not sigs:
        return []
    arity = sigs.most_common(1)[0][0]
    kinds = []
    for c in codes:
        for n in _defs(c):
            if n.name == entry and len(n.args.args) == arity:
                kinds = [ann.get(a.arg) or _ann_kind(a.annotation) or "any"
                         for a in n.args.args]
                break
        if kinds:
            break
    return kinds


def choose(user_text, replies, n_cases=6, timeout=8.0, seed=0):
    """Index of the selected reply, or None if the pool cannot be run at all."""
    codes = [code_of(r) for r in replies]
    entry = pick_entry(codes, user_text)
    if entry is None:
        return None, {"reason": "no function defined by any candidate"}
    kinds = infer_kinds(codes, entry, user_text)
    if not kinds:
        return None, {"reason": "could not infer arguments", "entry": entry}
    rng = random.Random(seed)
    cases = [tuple(_value(k, rng) for k in kinds) for _ in range(n_cases)]
    sigs = [run_capture(probe_program(c, entry, cases), timeout) for c in codes]
    valid = [i for i, s in enumerate(sigs) if s is not None]
    if not valid:
        return None, {"reason": "no candidate ran", "entry": entry}
    top, votes = Counter(sigs[i] for i in valid).most_common(1)[0]
    pick = next(i for i in valid if sigs[i] == top)
    return pick, {"entry": entry, "kinds": kinds, "ran": len(valid),
                  "cluster_size": votes, "clusters": len(set(sigs[i] for i in valid))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", choices=["humaneval", "mbpp"], default="humaneval")
    ap.add_argument("--gen-dir", default="artifacts/nla/coder_ladder")
    ap.add_argument("--cache", default="artifacts/nla/verifier/passmat")
    ap.add_argument("--out", default="artifacts/nla/verifier/chat_consensus.json")
    ap.add_argument("--cases", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--arms", nargs="*", default=None)
    a = ap.parse_args()

    import glob
    from concurrent.futures import ThreadPoolExecutor
    from verifier_select import pass_matrix

    probs = json.load(open(os.path.join(
        HERE, "data", "humaneval.json" if a.validate == "humaneval" else "mbpp.json")))
    files = sorted(f for f in glob.glob(os.path.join(HERE, a.gen_dir, "*.json"))
                   if "__" in os.path.basename(f))
    if a.arms:
        files = [f for f in files if os.path.basename(f)[:-5] in a.arms]

    res = {"bench": a.validate, "n_problems": len(probs), "arms": {},
           "note": "selector sees only the prompt text and the candidate replies"}
    agg = {"floor": [], "chat_consensus": [], "oracle": [], "resolved": []}
    for f in files:
        arm = os.path.basename(f)[:-5]
        rows = json.load(open(f))
        if len(rows) != len(probs):
            continue
        kmin = min(len(c) for c in rows)
        rows = [c[:kmin] for c in rows]
        ok = pass_matrix(arm, rows, probs, os.path.join(HERE, a.cache),
                         a.timeout, a.workers)
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            picks = list(ex.map(
                lambda i: choose(probs[i]["prompt"], rows[i], a.cases, a.timeout)[0],
                range(len(rows))))
        sel = [bool(ok[i, picks[i]]) for i in range(len(rows)) if picks[i] is not None]
        resolved = sum(1 for p in picks if p is not None)
        # Unresolved problems fall back to an arbitrary candidate, as a deployment would.
        full = [bool(ok[i, picks[i] if picks[i] is not None else 0])
                for i in range(len(rows))]
        res["arms"][arm] = {
            "chat_consensus": round(float(np.mean(full)), 4),
            "on_resolved_only": round(float(np.mean(sel)), 4) if sel else None,
            "resolved": resolved, "n": len(rows),
            "floor": round(float(ok.mean()), 4),
            "oracle": round(float(ok.any(1).mean()), 4)}
        v = res["arms"][arm]
        # Scored over every problem, so bounded by the oracle; `on_resolved_only` is a declared subset rate.
        if v["chat_consensus"] > v["oracle"] + 1e-9:
            raise SystemExit(f"{arm}: chat_consensus {v['chat_consensus']} exceeds oracle {v['oracle']}")
        agg["floor"].append(ok.mean())
        agg["chat_consensus"].append(np.mean(full))
        agg["oracle"].append(ok.any(1).mean())
        agg["resolved"].append(resolved / len(rows))
        print(f"  {arm:34s} chat {v['chat_consensus']:.4f}  resolved {resolved}/{len(rows)}"
              f"  floor {v['floor']:.4f}  oracle {v['oracle']:.4f}", flush=True)

    res["summary"] = {k: round(float(np.mean(v)), 4) for k, v in agg.items()}
    res["summary"]["gain_over_floor"] = round(
        res["summary"]["chat_consensus"] - res["summary"]["floor"], 4)
    s = res["summary"]
    print(f"\n  chat-only consensus {s['chat_consensus']:.4f}  floor {s['floor']:.4f}  "
          f"oracle {s['oracle']:.4f}  gain {s['gain_over_floor']:+.4f}  "
          f"resolved {s['resolved']*100:.1f}%", flush=True)

    op = os.path.join(HERE, a.out)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    json.dump(res, open(op, "w"), indent=2)
    print(f"\nwrote {op}", flush=True)


if __name__ == "__main__":
    main()
