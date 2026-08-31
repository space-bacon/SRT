"""Selection over a POOLED multi-lab candidate set.

Every read so far was measured inside one model's pool: eight samples, one lab, one
tokenizer. The union ceiling then reported what a five-lab pool *could* reach with a
perfect chooser. Neither answers the question the pooling idea actually poses, which
is whether a real selector over the pooled set beats simply deploying the best single
model.

That gap matters because the two settings are not alike. A within-model pool sits near
0.86 intra-similarity, where a selector has little to separate. A pooled set spans
different tokenizers, different post-training and different failure modes, which is the
regime agreement-based selection is supposed to suit.

Compared here, all on the same problems:

  best_single_pass1     deploy the strongest member, one sample
  best_single_consensus that member's own eight, chosen by agreement
  pooled_consensus      all members' candidates, chosen by agreement
  pooled_chat_only      same, but recovering the target from the replies alone
  union_oracle          any candidate from any member passes

If pooled_consensus does not beat best_single_consensus, pooling buys nothing that a
single good model plus a selector does not already give, and the ensemble framing is
dead regardless of what the oracle ceiling says.

    python scripts/pooled_select.py --workers 48
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
from chat_consensus import choose  # noqa: E402
from code_select import execute_file, extract  # noqa: E402
from consensus_select import (inputs_from_example, probe_program,  # noqa: E402
                              run_capture, synth_inputs)

GEN = "artifacts/nla/ensemble"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", default=GEN)
    ap.add_argument("--probs", default="data/humaneval.json")
    ap.add_argument("--out", default="artifacts/nla/ensemble/pooled_select.json")
    ap.add_argument("--cases", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--exclude", nargs="*", default=[])
    a = ap.parse_args()

    from concurrent.futures import ThreadPoolExecutor

    probs = json.load(open(os.path.join(HERE, a.probs)))
    files = sorted(f for f in glob.glob(os.path.join(HERE, a.gen_dir, "*__chat.json"))
                   if os.path.basename(f)[:-len("__chat.json")] not in a.exclude)
    members = [os.path.basename(f)[:-len("__chat.json")] for f in files]
    print(f"{len(members)} members: {', '.join(members)}", flush=True)

    rows, ok = {}, {}
    for f, m in zip(files, members):
        g = json.load(open(f))
        k = min(len(c) for c in g)
        rows[m] = [c[:k] for c in g]
        ok[m] = execute_file(rows[m], probs, a.timeout, a.workers)
        print(f"  {m:18s} pass@1 {ok[m].mean():.4f}  pass@k {ok[m].any(1).mean():.4f}",
              flush=True)

    n = len(probs)
    best = max(members, key=lambda m: ok[m].mean())
    print(f"\nstrongest member by pass@1: {best}", flush=True)

    # Pool: every member's candidates for a problem, tagged by origin.
    pool = [[(m, c) for m in members for c in rows[m][i]] for i in range(n)]
    pool_ok = [np.array([ok[m][i, j] for m in members for j in range(ok[m].shape[1])])
               for i in range(n)]

    cases = []
    for p in probs:
        c = synth_inputs(p["prompt"], p["entry_point"], a.cases)
        if not c:
            c = inputs_from_example(p.get("visible_test"), p["entry_point"], a.cases)
        cases.append(c)

    def agree_pick(i):
        """Largest output-agreement cluster over the pooled candidates."""
        if not cases[i]:
            return None
        bodies = [extract(probs[i]["prompt"], c, probs[i]["entry_point"])
                  for _, c in pool[i]]
        sigs = [run_capture(probe_program(b, probs[i]["entry_point"], cases[i]), a.timeout)
                for b in bodies]
        valid = [j for j, s in enumerate(sigs) if s is not None]
        if not valid:
            return None
        top = Counter(sigs[j] for j in valid).most_common(1)[0][0]
        return next(j for j in valid if sigs[j] == top)

    with ThreadPoolExecutor(max_workers=max(4, a.workers // 8)) as ex:
        picks = list(ex.map(agree_pick, range(n)))
    pooled = float(np.mean([bool(pool_ok[i][picks[i]]) if picks[i] is not None
                            else bool(pool_ok[i][0]) for i in range(n)]))

    with ThreadPoolExecutor(max_workers=max(4, a.workers // 8)) as ex:
        cpicks = list(ex.map(
            lambda i: choose(probs[i]["prompt"], [c for _, c in pool[i]],
                             a.cases, a.timeout)[0], range(n)))
    pooled_chat = float(np.mean([bool(pool_ok[i][cpicks[i]]) if cpicks[i] is not None
                                 else bool(pool_ok[i][0]) for i in range(n)]))

    def single_agree(m):
        def pick(i):
            if not cases[i]:
                return None
            bodies = [extract(probs[i]["prompt"], c, probs[i]["entry_point"])
                      for c in rows[m][i]]
            sigs = [run_capture(probe_program(b, probs[i]["entry_point"], cases[i]),
                                a.timeout) for b in bodies]
            valid = [j for j, s in enumerate(sigs) if s is not None]
            if not valid:
                return None
            top = Counter(sigs[j] for j in valid).most_common(1)[0][0]
            return next(j for j in valid if sigs[j] == top)
        with ThreadPoolExecutor(max_workers=max(4, a.workers // 8)) as ex2:
            ps = list(ex2.map(pick, range(n)))
        return float(np.mean([bool(ok[m][i, ps[i]]) if ps[i] is not None else bool(ok[m][i, 0])
                              for i in range(n)]))

    bs_cons = single_agree(best)
    union = float(np.mean([pool_ok[i].any() for i in range(n)]))

    res = {
        "members": {m: {"pass_at_1": round(float(ok[m].mean()), 4),
                        "pass_at_k": round(float(ok[m].any(1).mean()), 4)} for m in members},
        "best_member": best,
        "best_single_pass1": round(float(ok[best].mean()), 4),
        "best_single_consensus": round(bs_cons, 4),
        "pooled_consensus": round(pooled, 4),
        "pooled_chat_only": round(pooled_chat, 4),
        "union_oracle": round(union, 4),
        "pool_size": len(pool[0]),
    }
    res["pooled_minus_best_single_consensus"] = round(
        res["pooled_consensus"] - res["best_single_consensus"], 4)
    res["pooled_minus_best_single_pass1"] = round(
        res["pooled_consensus"] - res["best_single_pass1"], 4)

    print(f"\n  best member one sample        {res['best_single_pass1']:.4f}")
    print(f"  best member + own agreement   {res['best_single_consensus']:.4f}")
    print(f"  POOLED agreement ({res['pool_size']} cands)   {res['pooled_consensus']:.4f}")
    print(f"  pooled, chat-only recovery    {res['pooled_chat_only']:.4f}")
    print(f"  union oracle                  {res['union_oracle']:.4f}")
    print(f"\n  pooling vs best-single-with-selector  "
          f"{res['pooled_minus_best_single_consensus']:+.4f}")

    op = os.path.join(HERE, a.out)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    json.dump(res, open(op, "w"), indent=2)
    print(f"\nwrote {op}", flush=True)


if __name__ == "__main__":
    main()
