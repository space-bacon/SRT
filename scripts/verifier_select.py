"""A learned verifier as the read, trained on execution labels.

Every read we have tried was an unsupervised geometric quantity hoping to correlate
with correctness, and both died: centered medoid captured 2.0% of the headroom, and
the transport selector did not survive refitting. This fits directly on correctness
instead, using the execution labels we already own.

Discipline that the retraction bought us:

  splits are by PROBLEM, never by candidate. Eight samples of one problem are near
  duplicates, so a candidate-level split leaks the answer.

  every arbitrary knob is varied before any number is quoted. Here that is the fold
  seed and the feature set. Per-arm results are reported so stability across 30 arms
  is visible rather than averaged away.

Baselines are the ones that actually compete: the random-single floor, the centered
medoid, and the oracle ceiling. Greedy is not a baseline.

    python scripts/verifier_select.py --workers 96
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
from code_select import execute_file, medoid  # noqa: E402

SCORER = "sentence-transformers/all-MiniLM-L6-v2"


def pass_matrix(arm, rows, probs, cache, timeout, workers):
    p = os.path.join(cache, f"{arm}.npz")
    if os.path.isfile(p):
        return np.load(p)["ok"]
    ok = execute_file(rows, probs, timeout, workers)
    os.makedirs(cache, exist_ok=True)
    np.savez_compressed(p, ok=ok)
    return ok


def auroc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, bool)
    if label.all() or not label.any():
        return float("nan")
    r = score.argsort().argsort().astype(float) + 1
    n1, n0 = label.sum(), (~label).sum()
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", default="artifacts/nla/coder_ladder")
    ap.add_argument("--probs", default="data/humaneval.json")
    ap.add_argument("--cache", default="artifacts/nla/verifier/passmat")
    ap.add_argument("--out", default="artifacts/nla/verifier/results.json")
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--regs", type=float, nargs="*", default=[0.1, 1.0])
    a = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression

    probs = json.load(open(os.path.join(HERE, a.probs)))
    files = sorted(f for f in glob.glob(os.path.join(HERE, a.gen_dir, "*.json"))
                   if "__" in os.path.basename(f))
    st = SentenceTransformer(SCORER)

    arms, X, Y, P, L = [], [], [], [], []
    for f in files:
        arm = os.path.basename(f)[:-5]
        rows = json.load(open(f))
        if len(rows) != len(probs):
            continue
        kmin = min(len(c) for c in rows)
        rows = [c[:kmin] for c in rows]
        ok = pass_matrix(arm, rows, probs, os.path.join(HERE, a.cache),
                         a.timeout, a.workers)
        flat = [c if c.strip() else " " for r in rows for c in r]
        emb = st.encode(flat, batch_size=256, show_progress_bar=False).astype(np.float32)
        n, k = ok.shape
        arms.append((arm, n, k))
        X.append(emb)
        L.append(np.array([[len(c), c.count(chr(10)), c.count("def ")] for c in flat],
                          dtype=np.float32))
        Y.append(ok.ravel())
        P.append(np.repeat(np.arange(n), k))
        print(f"  {arm:34s} n={n} k={k} pass={ok.mean():.4f}", flush=True)

    X = np.concatenate(X)
    Y = np.concatenate(Y)
    P = np.concatenate(P)
    L = np.concatenate(L)
    print(f"\n{len(X):,} candidates over {len(arms)} arms, {len(probs)} problems",
          flush=True)

    res = {"n_candidates": int(len(X)), "n_arms": len(arms), "n_problems": len(probs),
           "scorer": SCORER, "split": "GroupKFold by problem id",
           "folds": a.folds, "seeds": a.seeds, "runs": {}}

    for C in a.regs:
        for seed in a.seeds:
            rng = np.random.default_rng(seed)
            order = rng.permutation(len(probs))
            fold_of = {p: i % a.folds for i, p in enumerate(order)}
            pred = np.zeros(len(X))
            for fold in range(a.folds):
                te = np.array([fold_of[p] == fold for p in P])
                clf = LogisticRegression(max_iter=2000, C=C)
                clf.fit(X[~te], Y[~te])
                pred[te] = clf.predict_proba(X[te])[:, 1]

            sel, floor, orc, med = [], [], [], []
            per_arm = {}
            off = 0
            for ai, (arm, n, k) in enumerate(arms):
                sl = slice(off, off + n * k)
                off += n * k
                y = Y[sl].reshape(n, k)
                pr = pred[sl].reshape(n, k)
                e = X[sl].reshape(n, k, -1)
                s = [bool(y[i, int(pr[i].argmax())]) for i in range(n)]
                m = [bool(y[i, medoid(e[i] - e.reshape(-1, e.shape[-1]).mean(0))[0]])
                     for i in range(n)]
                per_arm[arm] = {"verifier": round(float(np.mean(s)), 4),
                                "floor": round(float(y.mean()), 4),
                                "medoid_centered": round(float(np.mean(m)), 4),
                                "oracle": round(float(y.any(1).mean()), 4)}
                sel.extend(s), med.extend(m)
                floor.append(y.mean()), orc.append(y.any(1).mean())
            key = f"C{C}_seed{seed}"
            r = {"verifier": round(float(np.mean(sel)), 4),
                 "medoid_centered": round(float(np.mean(med)), 4),
                 "floor": round(float(np.mean(floor)), 4),
                 "oracle": round(float(np.mean(orc)), 4),
                 "auroc": round(auroc(pred, Y), 4)}
            r["verifier_minus_floor"] = round(r["verifier"] - r["floor"], 4)
            r["headroom_captured"] = round(
                (r["verifier"] - r["floor"]) / max(r["oracle"] - r["floor"], 1e-9), 4)
            r["per_arm"] = per_arm
            res["runs"][key] = r
            print(f"  {key:14s} verifier {r['verifier']:.4f}  medoid {r['medoid_centered']:.4f}  "
                  f"floor {r['floor']:.4f}  oracle {r['oracle']:.4f}  auroc {r['auroc']:.4f}  "
                  f"captured {r['headroom_captured']*100:.1f}%", flush=True)


    vals = [v["verifier_minus_floor"] for v in res["runs"].values()]
    res["stability_across_seeds"] = {"min": min(vals), "max": max(vals),
                                     "spread": round(max(vals) - min(vals), 4)}
    print(f"\n  gain over floor across seeds: {min(vals):+.4f} to {max(vals):+.4f} "
          f"(spread {max(vals)-min(vals):.4f})", flush=True)

    # Control: surface features only. If this matches the verifier, the embedding
    # is contributing nothing and we are ranking by verbosity.
    rng = np.random.default_rng(0)
    order = rng.permutation(len(probs))
    fold_of = {p: i % a.folds for i, p in enumerate(order)}
    predL = np.zeros(len(L))
    Ls = (L - L.mean(0)) / (L.std(0) + 1e-9)
    for fold in range(a.folds):
        te = np.array([fold_of[p] == fold for p in P])
        clf = LogisticRegression(max_iter=2000)
        clf.fit(Ls[~te], Y[~te])
        predL[te] = clf.predict_proba(Ls[te])[:, 1]
    selL, off = [], 0
    for arm, n, k in arms:
        y = Y[off:off + n * k].reshape(n, k)
        pr = predL[off:off + n * k].reshape(n, k)
        off += n * k
        selL.extend(bool(y[i, int(pr[i].argmax())]) for i in range(n))
    res["length_control"] = {"select": round(float(np.mean(selL)), 4),
                             "auroc": round(auroc(predL, Y), 4)}

    # Null: pick one candidate uniformly at random per problem per arm.
    rng = np.random.default_rng(0)
    null = []
    for _ in range(2000):
        tot, off = [], 0
        for arm, n, k in arms:
            y = Y[off:off + n * k].reshape(n, k)
            off += n * k
            tot.extend(y[np.arange(n), rng.integers(0, k, n)])
        null.append(np.mean(tot))
    null = np.array(null)
    best = max(v["verifier"] for v in res["runs"].values())
    res["null"] = {"mean": round(float(null.mean()), 4),
                   "p95": round(float(np.percentile(null, 95)), 4),
                   "p_value_best_verifier": float((null >= best).mean())}
    print(f"  length control  select {res['length_control']['select']:.4f}  "
          f"auroc {res['length_control']['auroc']:.4f}", flush=True)
    print(f"  null mean {res['null']['mean']:.4f}  p95 {res['null']['p95']:.4f}  "
          f"p={res['null']['p_value_best_verifier']:.5f}", flush=True)

    op = os.path.join(HERE, a.out)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    json.dump(res, open(op, "w"), indent=2)
    print(f"\nwrote {op}", flush=True)


if __name__ == "__main__":
    main()
