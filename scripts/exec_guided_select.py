"""Execution-guided selection: the baseline the verifier actually has to beat.

Whenever a prompt ships example tests, anyone can filter candidates by running them.
That is free, needs no training, and is what a serious system would do first. A
learned verifier is only interesting if it adds something on top.

Visible tests come from the prompt itself and never from the held-out suite:
HumanEval doctest examples are parsed out of the docstring, MBPP carries the single
assert that was shown to the model. Held-out tests are used only for scoring.

Strategies compared on identical candidate pools:

  floor            one candidate at random
  verifier         argmax learned score
  exec             random among candidates passing the visible tests
  exec+verifier    argmax learned score among those passing
  oracle           any candidate passes

    python scripts/exec_guided_select.py --bench humaneval
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
from code_select import extract, run_one  # noqa: E402
from verifier_select import pass_matrix  # noqa: E402
from visible_tests import HELPER, validated  # noqa: E402

SCORER = "sentence-transformers/all-MiniLM-L6-v2"


def visible_tests(p):
    """Validated assertions derivable from the prompt alone.

    A weak extractor here silently handicaps this baseline, which would bias the
    comparison toward the verifier. An earlier doctest-only version covered 54 of
    164 HumanEval problems at 0.60 tests each; this covers 116 at 2.78.
    """
    if p.get("visible_test"):
        return [p["visible_test"]]
    return validated(p["prompt"], p["entry_point"], p["canonical_solution"], run_one)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["humaneval", "mbpp"], default="humaneval")
    ap.add_argument("--gen-dir", default="artifacts/nla/coder_ladder")
    ap.add_argument("--cache", default="artifacts/nla/verifier/passmat")
    ap.add_argument("--vis-cache", default="artifacts/nla/verifier/vismat")
    ap.add_argument("--out", default="artifacts/nla/verifier/exec_guided.json")
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from concurrent.futures import ThreadPoolExecutor
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression

    probs = json.load(open(os.path.join(
        HERE, "data", "humaneval.json" if a.bench == "humaneval" else "mbpp.json")))
    vis = [visible_tests(p) for p in probs]
    print(f"{a.bench}: {len(probs)} problems, "
          f"{sum(1 for v in vis if v)} with visible tests, "
          f"{np.mean([len(v) for v in vis]):.2f} asserts each", flush=True)

    files = sorted(f for f in glob.glob(os.path.join(HERE, a.gen_dir, "*.json"))
                   if "__" in os.path.basename(f))
    st = SentenceTransformer(SCORER)
    os.makedirs(os.path.join(HERE, a.vis_cache), exist_ok=True)

    arms, X, Y, V, P = [], [], [], [], []
    for f in files:
        arm = os.path.basename(f)[:-5]
        rows = json.load(open(f))
        if len(rows) != len(probs):
            continue
        kmin = min(len(c) for c in rows)
        rows = [c[:kmin] for c in rows]
        ok = pass_matrix(arm, rows, probs, os.path.join(HERE, a.cache),
                         a.timeout, a.workers)
        vp = os.path.join(HERE, a.vis_cache, f"{arm}.npz")
        if os.path.isfile(vp):
            vm = np.load(vp)["vis"]
        else:
            jobs, idx = [], []
            for i, cands in enumerate(rows):
                for j, c in enumerate(cands):
                    if not vis[i]:
                        continue
                    body = extract(probs[i]["prompt"], c, probs[i]["entry_point"])
                    jobs.append(body + "\n\n" + HELPER + "\n" + "\n".join(vis[i]) + "\n")
                    idx.append((i, j))
            vm = np.zeros_like(ok)
            with ThreadPoolExecutor(max_workers=a.workers) as ex:
                for (i, j), r in zip(idx, ex.map(lambda g: run_one(g, a.timeout), jobs)):
                    vm[i, j] = r
            np.savez_compressed(vp, vis=vm)
        flat = [c if c.strip() else " " for r in rows for c in r]
        emb = st.encode(flat, batch_size=256, show_progress_bar=False).astype(np.float32)
        n, k = ok.shape
        arms.append((arm, n, k))
        X.append(emb), Y.append(ok.ravel()), V.append(vm.ravel())
        P.append(np.repeat(np.arange(n), k))
        print(f"  {arm:34s} pass {ok.mean():.4f}  visible-pass {vm.mean():.4f}", flush=True)

    X, Y, V, P = (np.concatenate(z) for z in (X, Y, V, P))
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(probs))
    fold_of = {p: i % a.folds for i, p in enumerate(order)}
    pred = np.zeros(len(X))
    for fold in range(a.folds):
        te = np.array([fold_of[p] == fold for p in P])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X[~te], Y[~te])
        pred[te] = clf.predict_proba(X[te])[:, 1]

    agg = {kk: [] for kk in ["floor", "verifier", "exec", "exec_verifier", "oracle"]}
    per_arm, off = {}, 0
    for arm, n, k in arms:
        sl = slice(off, off + n * k)
        off += n * k
        y = Y[sl].reshape(n, k)
        v = V[sl].reshape(n, k).astype(bool)
        pr = pred[sl].reshape(n, k)
        row = {}
        row["floor"] = float(y.mean())
        row["oracle"] = float(y.any(1).mean())
        row["verifier"] = float(np.mean([y[i, int(pr[i].argmax())] for i in range(n)]))
        ex_pick, exv_pick = [], []
        for i in range(n):
            cand = np.flatnonzero(v[i])
            if len(cand) == 0:
                cand = np.arange(k)          # nothing passes, fall back to the pool
            ex_pick.append(bool(y[i, rng.choice(cand)]))
            exv_pick.append(bool(y[i, cand[int(pr[i][cand].argmax())]]))
        row["exec"] = float(np.mean(ex_pick))
        row["exec_verifier"] = float(np.mean(exv_pick))
        for kk in agg:
            agg[kk].append(row[kk])
        per_arm[arm] = {kk: round(row[kk], 4) for kk in row}

    res = {"bench": a.bench, "n_arms": len(arms), "n_problems": len(probs),
           "visible_asserts_mean": round(float(np.mean([len(v) for v in vis])), 2),
           "overall": {kk: round(float(np.mean(vv)), 4) for kk, vv in agg.items()},
           "per_arm": per_arm}
    o = res["overall"]
    res["overall"]["exec_minus_floor"] = round(o["exec"] - o["floor"], 4)
    res["overall"]["verifier_minus_exec"] = round(o["verifier"] - o["exec"], 4)
    res["overall"]["execverifier_minus_exec"] = round(o["exec_verifier"] - o["exec"], 4)
    for kk in ["floor", "verifier", "exec", "exec_verifier", "oracle"]:
        print(f"  {kk:16s} {o[kk]:.4f}", flush=True)
    print(f"\n  exec - floor          {res['overall']['exec_minus_floor']:+.4f}", flush=True)
    print(f"  verifier - exec       {res['overall']['verifier_minus_exec']:+.4f}", flush=True)
    print(f"  exec+verifier - exec  {res['overall']['execverifier_minus_exec']:+.4f}", flush=True)

    op = os.path.join(HERE, a.out)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    json.dump(res, open(op, "w"), indent=2)
    print(f"\nwrote {op}", flush=True)


if __name__ == "__main__":
    main()
