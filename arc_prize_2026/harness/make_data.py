"""Build Kaggle-format ARC-AGI-2 challenge and solution files from the public GitHub data."""
import glob, json, os, sys
root, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
for split in ("training", "evaluation"):
    ch, sol = {}, {}
    for p in sorted(glob.glob(f"{root}/data/{split}/*.json")):
        k = os.path.basename(p)[:-5]
        t = json.load(open(p))
        ch[k] = {"train": t["train"], "test": [{"input": x["input"]} for x in t["test"]]}
        sol[k] = [x["output"] for x in t["test"]]
    for name, obj in (("challenges", ch), ("solutions", sol)):
        tmp = f"{out}/arc-agi_{split}_{name}.json.tmp"
        json.dump(obj, open(tmp, "w"))
        os.replace(tmp, f"{out}/arc-agi_{split}_{name}.json")
    n_out = sum(len(v) for v in sol.values())
    print(split, len(ch), "tasks", n_out, "test outputs")
