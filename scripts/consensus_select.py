"""Consensus execution: a check for the 36 problems that state no example.

Visible-test filtering tops out at 128 of 164 HumanEval problems, because the rest
describe the task in prose only. No extractor reaches them; the information is not
in the prompt.

What is still available is agreement. Synthesise inputs from the signature, run every
candidate on them, and group candidates by the outputs they produce. Candidates that
compute the same function land in the same cluster, and the largest cluster is the
majority opinion of the pool. This needs no ground truth, no stated examples, and no
training, so it extends coverage to every problem.

Exceptions are recorded as outcomes rather than discarded: agreeing on which inputs
raise is itself evidence of computing the same function.
"""
import argparse
import ast
import json
import os
import random
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
sys.path.insert(0, HERE)
from code_select import extract  # noqa: E402
from srt_select.sandbox import run_json  # noqa: E402

POOL = {
    "int": [0, 1, 2, 3, 5, 7, 10, -1, -4, 100],
    "float": [0.0, 0.5, 1.0, 2.5, -1.5, 3.14],
    "str": ["", "a", "abc", "Hello", "abcdef", "a b c", "AaBb", "xyzxyz"],
    "bool": [True, False],
}


def _ann(node):
    if node is None:
        return "any"
    try:
        text = ast.unparse(node).lower()
    except Exception:
        return "any"
    for k in ("list", "tuple", "dict", "set"):
        if text.startswith(k):
            inner = "int"
            for t in ("str", "float", "int"):
                if t in text[len(k):]:
                    inner = t
                    break
            return f"{k}[{inner}]"
    for t in ("int", "float", "str", "bool"):
        if t in text:
            return t
    return "any"


def _value(kind, rng):
    if kind.startswith(("list", "tuple", "set")):
        inner = kind[kind.index("[") + 1:-1]
        n = rng.randint(0, 5)
        vals = [rng.choice(POOL.get(inner, POOL["int"])) for _ in range(n)]
        return tuple(vals) if kind.startswith("tuple") else vals
    if kind.startswith("dict"):
        return {rng.choice(POOL["str"]): rng.choice(POOL["int"])
                for _ in range(rng.randint(0, 3))}
    if kind in POOL:
        return rng.choice(POOL[kind])
    return rng.choice(POOL["int"] + POOL["str"])


def _kind_of(v):
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, (list, tuple, set)):
        name = type(v).__name__
        inner = _kind_of(next(iter(v))) if v else "int"
        return f"{name}[{inner}]"
    if isinstance(v, dict):
        return "dict[str]"
    return "any"


def inputs_from_example(visible_test, entry, n_cases, seed=0):
    """Inputs for prompts that show an example call but no signature, as MBPP does.

    The example's own arguments are case one; the rest are fresh values of the same
    shapes, so agreement is tested beyond the single case the model was shown.
    """
    if not visible_test:
        return []
    try:
        tree = ast.parse(visible_test)
    except SyntaxError:
        return []
    call = next((n for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == entry), None)
    if call is None:
        return []
    try:
        base = tuple(ast.literal_eval(x) for x in call.args)
    except Exception:
        return []
    rng = random.Random(seed)
    kinds = [_kind_of(v) for v in base]
    out = [base]
    for _ in range(max(0, n_cases - 1)):
        out.append(tuple(_value(k, rng) for k in kinds))
    return out


def synth_inputs(prompt, entry, n_cases, seed=0):
    """Argument tuples derived from the signature's annotations."""
    rng = random.Random(seed)
    fn = None
    for suffix in ("", "\n    pass\n", "    pass\n"):
        try:
            tree = ast.parse(prompt + suffix)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry:
                fn = node
                break
        if fn:
            break
    if fn is None:
        return []
    kinds = [_ann(a.annotation) for a in fn.args.args]
    if not kinds:
        return []
    return [tuple(_value(k, rng) for k in kinds) for _ in range(n_cases)]


def probe_program(body, entry, cases):
    return (body + "\n\nimport json as _j\n_o = []\n"
            f"for _a in {cases!r}:\n"
            "    try:\n"
            f"        _o.append(repr({entry}(*_a)))\n"
            "    except Exception as _e:\n"
            "        _o.append('ERR:' + type(_e).__name__)\n"
            "print(_j.dumps(_o))\n")


def run_capture(program, timeout):
    # The wall-clock timeout here is enforced by this parent, so it buys nothing once
    # the parent is killed: the child orphans and spins forever. srt_select.sandbox
    # applies RLIMIT_CPU inside the child, which is the only limit that survives us.
    return run_json(program, timeout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", default="artifacts/nla/coder_ladder")
    ap.add_argument("--probs", default="data/humaneval.json")
    ap.add_argument("--cache", default="artifacts/nla/verifier/passmat")
    ap.add_argument("--out", default="artifacts/nla/verifier/consensus.json")
    ap.add_argument("--cases", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--arms", nargs="*", default=None)
    a = ap.parse_args()

    from concurrent.futures import ThreadPoolExecutor
    import glob
    from verifier_select import pass_matrix

    probs = json.load(open(os.path.join(HERE, a.probs)))
    cases = []
    for p in probs:
        c = synth_inputs(p["prompt"], p["entry_point"], a.cases)
        if not c:
            c = inputs_from_example(p.get("visible_test"), p["entry_point"], a.cases)
        cases.append(c)
    print(f"{sum(1 for c in cases if c)} / {len(probs)} problems got synthesised inputs",
          flush=True)

    files = sorted(f for f in glob.glob(os.path.join(HERE, a.gen_dir, "*.json"))
                   if "__" in os.path.basename(f))
    if a.arms:
        files = [f for f in files if os.path.basename(f)[:-5] in a.arms]

    res = {"n_cases": a.cases, "arms": {},
           "coverage": int(sum(1 for c in cases if c))}
    agg = {"floor": [], "consensus": [], "consensus_strict": [],
           "consensus_on_covered": [], "oracle": []}
    for f in files:
        arm = os.path.basename(f)[:-5]
        rows = json.load(open(f))
        if len(rows) != len(probs):
            continue
        kmin = min(len(c) for c in rows)
        rows = [c[:kmin] for c in rows]
        ok = pass_matrix(arm, rows, probs, os.path.join(HERE, a.cache),
                         a.timeout, a.workers)
        jobs, idx = [], []
        for i, cands in enumerate(rows):
            if not cases[i]:
                continue
            for j, c in enumerate(cands):
                body = extract(probs[i]["prompt"], c, probs[i]["entry_point"])
                jobs.append(probe_program(body, probs[i]["entry_point"], cases[i]))
                idx.append((i, j))
        sigs = {}
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for (i, j), s in zip(idx, ex.map(lambda g: run_capture(g, a.timeout), jobs)):
                sigs[(i, j)] = s
        picks = []
        for i in range(len(rows)):
            k = ok.shape[1]
            got = [sigs.get((i, j)) for j in range(k)]
            valid = [j for j in range(k) if got[j] is not None]
            if not valid:
                picks.append(None)
                continue
            top = Counter(got[j] for j in valid).most_common(1)[0][0]
            picks.append(next(j for j in valid if got[j] == top))
        sel = [bool(ok[i, picks[i]]) for i in range(len(rows)) if picks[i] is not None]
        covered = sum(1 for p in picks if p is not None)
        # Score every problem. Uncovered pools fall back to candidate 0, which is
        # what srt_select.select() and chat_consensus.py do; `strict` charges them
        # as failures. Floor and oracle are over all rows, so `consensus` must be too.
        full = [bool(ok[i, picks[i] if picks[i] is not None else 0])
                for i in range(len(rows))]
        strict = [bool(ok[i, picks[i]]) if picks[i] is not None else False
                  for i in range(len(rows))]
        res["arms"][arm] = {
            "consensus": round(float(np.mean(full)), 4),
            "consensus_strict": round(float(np.mean(strict)), 4),
            "consensus_on_covered": round(float(np.mean(sel)), 4) if sel else None,
            "covered_problems": covered,
            "floor": round(float(ok.mean()), 4),
            "oracle": round(float(ok.any(1).mean()), 4)}
        # No arm may exceed its own oracle on a column scored over all problems. `consensus_on_covered`
        # is a declared subset rate and is exempt; scripts/check_oracle_bound.py applies the same rule repo-wide.
        for col in ("consensus", "consensus_strict"):
            if res["arms"][arm][col] > res["arms"][arm]["oracle"] + 1e-9:
                raise SystemExit(f"{arm}: {col} {res['arms'][arm][col]} exceeds oracle {res['arms'][arm]['oracle']}; a subset was averaged against a full denominator")
        agg["floor"].append(ok.mean())
        agg["oracle"].append(ok.any(1).mean())
        agg["consensus"].append(np.mean(full))
        agg["consensus_strict"].append(np.mean(strict))
        agg["consensus_on_covered"].append(np.mean(sel) if sel else 0.0)
        print(f"  {arm:34s} consensus {res['arms'][arm]['consensus']:.4f}  "
              f"strict {res['arms'][arm]['consensus_strict']:.4f}  "
              f"covered {covered}/{len(rows)}  floor {ok.mean():.4f}  "
              f"oracle {ok.any(1).mean():.4f}", flush=True)

    res["summary"] = {k: round(float(np.mean(v)), 4) for k, v in agg.items()}
    res["summary"]["consensus_minus_floor"] = round(
        res["summary"]["consensus"] - res["summary"]["floor"], 4)
    res["summary"]["consensus_strict_minus_floor"] = round(
        res["summary"]["consensus_strict"] - res["summary"]["floor"], 4)
    print(f"\n  consensus {res['summary']['consensus']:.4f}  "
          f"strict {res['summary']['consensus_strict']:.4f}  "
          f"covered-only {res['summary']['consensus_on_covered']:.4f}  "
          f"floor {res['summary']['floor']:.4f}  oracle {res['summary']['oracle']:.4f}  "
          f"gain {res['summary']['consensus_minus_floor']:+.4f}", flush=True)

    op = os.path.join(HERE, a.out)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    json.dump(res, open(op, "w"), indent=2)
    print(f"\nwrote {op}", flush=True)


if __name__ == "__main__":
    main()
