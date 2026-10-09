#!/usr/bin/env python
"""Execute the Python candidates found in a code-mode run and attach demo-pass counts and test predictions.

Writes <out> as JSON lines (key, sample, n_train, n_pass, pred, error) through a temp file and rename.
"""
import argparse, concurrent.futures as cf, json, os, re, subprocess, sys


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", text or "", re.S)
    return blocks[-1] if blocks else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenges", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    data = json.load(open(a.challenges))
    here = os.path.dirname(os.path.abspath(__file__))
    jobs = []
    for p in a.runs:
        for line in open(p):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("status") != "ok":
                continue
            tid, ti = r["key"].rsplit("_", 1)
            jobs.append((r["key"], r["sample"], extract_code(r["content"]), data[tid]["train"], data[tid]["test"][int(ti)]["input"], [t["input"] for t in data[tid]["test"]], int(ti)))

    def one(j):
        key, s, code, train, test_in, all_tests, ti = j
        if not code:
            return {"key": key, "sample": s, "n_train": len(train), "n_pass": 0, "pred": None, "error": "no code block"}
        try:
            p = subprocess.run([sys.executable, os.path.join(here, "exec_one.py")], input=json.dumps({"code": code, "train": train, "test_input": test_in, "test_inputs": all_tests, "ti": ti}),
                               capture_output=True, text=True, timeout=60)
            r = json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout.strip() else {"n_train": len(train), "n_pass": 0, "pred": None, "error": "no output " + p.stderr[-120:]}
        except Exception as e:
            r = {"n_train": len(train), "n_pass": 0, "pred": None, "error": "exec: " + repr(e)[:120]}
        return {"key": key, "sample": s, **r}

    tmp = a.out + ".tmp"
    with cf.ThreadPoolExecutor(a.workers) as ex, open(tmp, "w") as f:
        for r in ex.map(one, jobs):
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, a.out)
    print("executed", len(jobs))


if __name__ == "__main__":
    main()
