#!/usr/bin/env python
"""Probe analysis: do prefix hidden states predict whether a reasoning trace ends in a correct answer?

Per prefix position and layer: centre by the training-fold mean, project to PCA components fitted on the training fold, fit a ridge-penalised
logistic read-out, and score held-out folds grouped by task. Beside every AUROC: the confidence-feature baseline (mean logprob and entropy
of the prefix), a label-permutation floor, and the raw anisotropy of the state pool. Intervals come from a task-level bootstrap.
"""
import argparse, json, os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


def cv_scores(X, y, groups, comps=64, C=0.05, seed=0):
    s = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        mu = X[tr].mean(0, keepdims=True)
        pca = PCA(n_components=min(comps, len(tr) - 1), random_state=seed).fit(X[tr] - mu)
        Ztr, Zte = pca.transform(X[tr] - mu), pca.transform(X[te] - mu)
        sd = Ztr.std(0, keepdims=True) + 1e-6
        clf = LogisticRegression(C=C, max_iter=2000).fit(Ztr / sd, y[tr])
        s[te] = clf.decision_function(Zte / sd)
    return s


def cluster_boot_auc(y, s, groups, n=500, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    out = []
    for _ in range(n):
        pick = rng.choice(ug, len(ug))
        ii = np.concatenate([idx[g] for g in pick])
        if len(set(y[ii])) == 2:
            out.append(roc_auc_score(y[ii], s[ii]))
    return [round(float(np.percentile(out, 2.5)), 3), round(float(np.percentile(out, 97.5)), 3)]


def anisotropy(X, n=300, seed=0):
    rng = np.random.default_rng(seed)
    Z = X[rng.choice(len(X), min(n, len(X)), replace=False)]
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    S = Z @ Z.T
    return float((S.sum() - len(Z)) / (len(Z) * (len(Z) - 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", nargs="+", required=True)
    ap.add_argument("--labels", required=True, help="JSON {key|sample: 0/1} with keys formatted 'key__sample'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--perms", type=int, default=50)
    a = ap.parse_args()
    labels = json.load(open(a.labels))
    keys, feats, conf = [], [], []
    for p in a.states:
        z = np.load(p)
        if z["feats"].ndim != 4:
            continue
        keys += [f"{k}__{s}" for k, s in z["keys"]]
        feats.append(z["feats"]); conf.append(z["conf"])
        positions, layers = z["positions"], z["layers"]
    feats, conf = np.concatenate(feats), np.concatenate(conf)
    keep = [i for i, k in enumerate(keys) if k in labels]
    feats, conf, keys = feats[keep], conf[keep], [keys[i] for i in keep]
    y_all = np.array([labels[k] for k in keys])
    groups_all = np.array([k.split("_")[0] for k in keys])
    res = {"n_samples": len(keys), "positives": int(y_all.sum()), "positions": positions.tolist(), "layers": layers.tolist(), "by_position": {}}
    rng = np.random.default_rng(0)
    for pi, p in enumerate(positions):
        have = np.array([np.isfinite(conf[i, pi]).all() for i in range(len(keys))])
        y, g = y_all[have], groups_all[have]
        if have.sum() < 40 or len(set(y)) < 2:
            continue
        rec = {"n": int(have.sum()), "positives": int(y.sum()), "layers": {}}
        Xc = conf[have, pi]
        sc = cv_scores(np.nan_to_num(Xc), y, g, comps=min(4, Xc.shape[1]))
        rec["confidence_baseline_auc"] = round(float(roc_auc_score(y, sc)), 3)
        for li, l in enumerate(layers):
            X = feats[have, pi, li].astype(np.float32)
            s = cv_scores(X, y, g)
            auc = float(roc_auc_score(y, s))
            perm = []
            for _ in range(a.perms):
                yp = y.copy()
                for gg in np.unique(g):
                    pass
                yp = rng.permutation(y)
                perm.append(float(roc_auc_score(yp, cv_scores(X, yp, g))))
            rec["layers"][str(l)] = {"auc": round(auc, 3), "ci95_task_bootstrap": cluster_boot_auc(y, s, g),
                                     "permutation_floor_mean": round(float(np.mean(perm)), 3), "permutation_floor_sd": round(float(np.std(perm)), 3),
                                     "raw_anisotropy": round(anisotropy(X), 3)}
        res["by_position"][str(int(p))] = rec
    tmp = a.out + ".tmp"
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, a.out)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
