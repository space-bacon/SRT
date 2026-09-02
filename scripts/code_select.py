"""Candidate selection on HumanEval: does centered semantic voting beat the floor?

Reuses the coder-ladder generations (164 problems x K candidates) as candidate
pools, executes every candidate to get ground truth, then compares selection
strategies against the random-single floor and the oracle pass@K ceiling.

The centered-vs-raw contrast is the point: embedding anisotropy makes raw cosine
crowd every candidate together, so raw-cosine voting is expected to underperform.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOPS = ["\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint(", "\n@", "\nassert "]

sys.path.insert(0, HERE)
# Setting the hard limit too raises on macOS whenever it exceeds the inherited one,
# which killed every child before it ran and scored the whole pool as failing.
# srt_select.sandbox clamps each limit to the inherited hard limit instead.
from srt_select.sandbox import GUARD  # noqa: E402


def extract(prompt, completion, entry):
    """Turn a raw completion into a runnable program body."""
    if "```" in completion:
        blocks = re.findall(r"```(?:[a-zA-Z]*)\n(.*?)(?:```|\Z)", completion, re.S)
        body = blocks[0] if blocks else completion
        if f"def {entry}" in body:
            return body
        return prompt + body
    cut = completion
    for s in STOPS:
        i = cut.find(s)
        if i > 0:
            cut = cut[:i]
    return prompt + cut


def run_one(program, timeout):
    try:
        p = subprocess.run(
            [sys.executable, "-I", "-c", GUARD + program],
            capture_output=True, timeout=timeout, cwd="/tmp",
            env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1",
                 "HOME": "/tmp", "OMP_NUM_THREADS": "1"},
        )
        return p.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def execute_file(rows, probs, timeout, workers):
    """rows[i][k] -> bool pass matrix."""
    jobs, index = [], []
    for i, cands in enumerate(rows):
        pr = probs[i]
        for k, c in enumerate(cands):
            body = extract(pr["prompt"], c, pr["entry_point"])
            jobs.append(body + "\n\n" + pr["test"] + f"\ncheck({pr['entry_point']})\n")
            index.append((i, k))
    out = np.zeros((len(rows), len(rows[0])), dtype=bool)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for (i, k), ok in zip(index, ex.map(lambda j: run_one(j, timeout), jobs)):
            out[i, k] = ok
    return out


def medoid(emb):
    """Index of the candidate with highest mean similarity to the rest."""
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    sim = e @ e.T
    np.fill_diagonal(sim, 0.0)
    return int(sim.sum(1).argmax()), float(sim[np.triu_indices(len(e), 1)].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", default="artifacts/nla/coder_ladder")
    ap.add_argument("--probs", default="data/humaneval.json")
    ap.add_argument("--out", default="artifacts/nla/code_select/results.json")
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--only", default="")
    a = ap.parse_args()

    probs = json.load(open(os.path.join(HERE, a.probs)))
    op = os.path.join(HERE, a.out)
    # The generation directory also holds bookkeeping files (task_ids.json,
    # scaling_curve.json). task_ids.json has one entry per problem, so it passes
    # the length check below, and slicing its strings yields k = 11 "candidates"
    # of single characters. Arm files are the ones tagged `<rung>_<variant>`.
    files = sorted(f for f in glob.glob(os.path.join(HERE, a.gen_dir, "*.json"))
                   if "__" in os.path.basename(f))
    if a.only:
        files = [f for f in files if a.only in os.path.basename(f)]
    print(f"{len(files)} generation files, {len(probs)} problems", flush=True)

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    results = {}
    for f in files:
        tag = os.path.basename(f)[:-5]
        rows = json.load(open(f))
        if len(rows) != len(probs):
            print(f"  {tag:34s} SKIP {len(rows)} rows", flush=True)
            continue
        kmin = min(len(c) for c in rows)
        if kmin != max(len(c) for c in rows):
            print(f"  {tag:34s} ragged K, truncating to {kmin}", flush=True)
        rows = [c[:kmin] for c in rows]
        ok = execute_file(rows, probs, a.timeout, a.workers)
        n, k = ok.shape

        flat = [c for cands in rows for c in cands]
        emb = st.encode(flat, batch_size=256, show_progress_bar=False).astype(np.float32)
        emb = emb.reshape(n, k, -1)
        mu = emb.reshape(n * k, -1).mean(0)  # pool mean over this arm's candidates

        raw_pick, cen_pick, aniso = [], [], []
        for i in range(n):
            r, ar = medoid(emb[i])
            c, _ = medoid(emb[i] - mu)
            raw_pick.append(ok[i, r])
            cen_pick.append(ok[i, c])
            aniso.append(ar)

        res = {
            "n": n, "k": k,
            "floor_pass1": float(ok.mean()),
            "oracle_passk": float(ok.any(1).mean()),
            "vote_raw": float(np.mean(raw_pick)),
            "vote_centered": float(np.mean(cen_pick)),
            "anisotropy_raw_cos": float(np.mean(aniso)),
        }
        res["headroom"] = res["oracle_passk"] - res["floor_pass1"]
        res["cen_minus_raw"] = res["vote_centered"] - res["vote_raw"]
        res["cen_minus_floor"] = res["vote_centered"] - res["floor_pass1"]
        results[tag] = res
        print(f"  {tag:34s} floor {res['floor_pass1']:.4f}  raw {res['vote_raw']:.4f}  "
              f"cen {res['vote_centered']:.4f}  oracle {res['oracle_passk']:.4f}  "
              f"aniso {res['anisotropy_raw_cos']:.4f}", flush=True)
        os.makedirs(os.path.dirname(op), exist_ok=True)
        json.dump(results, open(op, "w"), indent=2)

    print(f"\nwrote {op}", flush=True)


if __name__ == "__main__":
    main()
