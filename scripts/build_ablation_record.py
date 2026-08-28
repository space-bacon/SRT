"""Consolidate the MTEB-STS ablation history into one comparable table.

Nineteen adapter configurations were evaluated, most of them negative. That
record is the useful artifact: almost nobody publishes the runs that failed.

Two things this script refuses to paper over.

The six `*_seed{0,1,2}` directories are byte-identical copies, not replicates.
MTEB evaluation of a fixed checkpoint is deterministic, so re-running it changes
nothing. They therefore give NO variance estimate, and the headline gap between
v22c_a050 and v18 has no error bar. The directory names imply a seed study that
was never run.

`v22c_a050_beyond` was scored on 14 subsets while everything else used the same
39, so its mean is not comparable and it is excluded from the ranking rather
than left to top a sorted list.

    python scripts/build_ablation_record.py
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import statistics as st

ROOT = "artifacts/mteb"
OUT = "artifacts/nla/q4/adapter_ablation_record.json"

WHAT_IT_TESTED = {
    "v12": "early contrastive baseline",
    "v14": "NLI + MSMARCO + Quora corpus mix",
    "v15a": "NLI-only, warm from v14",
    "v15b": "E5-style query/passage prefixes bolted onto a warm start",
    "v16": "second epoch of NLI-only InfoNCE at lower lr",
    "v17": "supervised cosine-MSE regression on STSB",
    "v18": "CoSENT rank loss on STSB, lr 1e-5, 5 epochs",
    "v19a": "gentler CoSENT schedule chosen on dev Spearman",
    "v19b": "CoSENT over 200K binary NLI pairs plus graded STSB",
    "v20": "NLI labels mapped to graded similarity {1.0, 0.5, 0.0}",
    "v21a": "teacher distillation, mxbai cosines as continuous labels",
    "v21b_a050": "weight soup of v18 and v20 at alpha 0.50",
    "v21b_a070": "weight soup of v18 and v20 at alpha 0.70",
    "v22a": "STSB upsampled against the distilled pool",
    "v22b": "multi-teacher distillation",
    "v22c_a030": "soup of v18 and v21a at alpha 0.30",
    "v22c_a050": "soup of v18 and v21a at alpha 0.50",
    "v22c_a070": "soup of v18 and v21a at alpha 0.70",
}

VERDICT = {
    "v15b": "catastrophic: prefixes destroy a warm-started encoder",
    "v17": "negative: regression on absolute cosines overfits one domain",
    "v16": "negative: the recipe was already saturated, more steps is noise",
    "v19a": "null: barely moved off the warm start",
    "v19b": "mixed: cross-lingual up, graded English down",
    "v20": "negative: NLI labels are not similarity labels",
    "v22a": "negative: upsampling did not recover English",
    "v22b": "negative: more teachers did not help",
    "v21b_a050": "negative: even mix over-tilts toward the weaker parent",
}


def subsets(path):
    d = json.load(open(path))
    out = {}
    for task, v in d.items():
        if not isinstance(v, dict):
            continue
        for split, rows in v.items():
            for r in rows:
                if isinstance(r, dict) and "main_score" in r:
                    out[(task, r.get("hf_subset", ""))] = r["main_score"]
    return out


def main() -> None:
    runs = {}
    for p in sorted(glob.glob(f"{ROOT}/*/summary.json")):
        name = os.path.basename(os.path.dirname(p))
        runs[name] = {"scores": subsets(p),
                      "sha": hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]}

    base = set(runs["v18"]["scores"])
    comparable = {k: v for k, v in runs.items() if set(v["scores"]) == base}
    excluded = {k: len(v["scores"]) for k, v in runs.items()
                if set(v["scores"]) != base}

    # Group byte-identical files: these are copies, not repeat measurements.
    by_sha = {}
    for k, v in comparable.items():
        by_sha.setdefault(v["sha"], []).append(k)
    dupes = {s: sorted(ks) for s, ks in by_sha.items() if len(ks) > 1}

    seen, table = set(), []
    for name in sorted(comparable):
        sha = comparable[name]["sha"]
        if sha in seen:
            continue
        seen.add(sha)
        canonical = sorted(by_sha[sha], key=len)[0]
        table.append({
            "config": canonical,
            "mean_sts": round(st.mean(comparable[name]["scores"].values()), 4),
            "tested": WHAT_IT_TESTED.get(canonical, ""),
            "verdict": VERDICT.get(canonical, ""),
            "duplicate_dirs": [d for d in by_sha[sha] if d != canonical],
        })
    table.sort(key=lambda r: -r["mean_sts"])

    v18 = next(r for r in table if r["config"] == "v18")["mean_sts"]
    for r in table:
        r["delta_vs_v18"] = round(r["mean_sts"] - v18, 4)

    res = {
        "question": "what actually moved MTEB-STS across 19 adapter configurations",
        "backbone": "frozen Qwen2.5-7B, 12.7M trainable adapter",
        "n_subsets": len(base),
        "n_comparable_configs": len(table),
        "table": table,
        "excluded_not_comparable": {
            k: f"{n} subsets, not the same {len(base)} as every other run"
            for k, n in excluded.items()},
        "no_variance_estimate": {
            "claim": "v22c_a050 beats v18 by +0.0037 mean MTEB-STS",
            "status": "UNQUANTIFIED",
            "why": "the six *_seed{0,1,2} directories are byte-identical copies, "
                   "not training-seed replicates. MTEB evaluation of a fixed "
                   "checkpoint is deterministic, so repeating it cannot estimate "
                   "variance. No adapter was ever retrained with a different seed, "
                   "so this gap has no error bar and may not be a real difference.",
            "byte_identical_groups": dupes,
            "to_fix": "retrain v18 from v15a with >=3 training seeds and evaluate "
                      "each; that gives the first genuine error bar for this line",
        },
        "reading": "the negatives are the point. Prefix injection onto a warm "
                   "start is catastrophic, regression on absolute cosines "
                   "overfits, NLI labels are not similarity labels, and weight "
                   "souping beat three separate training attempts at recovering "
                   "English calibration.",
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)

    print(f"{len(table)} comparable configs over {len(base)} subsets")
    for r in table:
        d = f"{r['delta_vs_v18']:+.4f}"
        print(f"  {r['config']:<12} {r['mean_sts']:.4f}  {d:>8}  {r['verdict'][:44]}")
    print(f"\nexcluded: {excluded}")
    print(f"byte-identical groups: {dupes}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
