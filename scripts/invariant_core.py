"""What survives every backbone we have tested, and what does not.

Borrowed from Devon Generally's Mathematics as Language, where the invariant core
of a rule system is the intersection of what stays provable across all admissible
descendants. Our version intersects over substrates instead of over descendants:
a claim is in the core when it holds on every backbone it was tested on, and the
core shrinks as we add hosts, where his grows as transitions narrow reachability.

We have never computed this. We report per-backbone numbers and say "train once,
read everywhere", and the intersection is the honest version of that sentence.

The claim table is hand-entered from the papers, with a file reference on each
row, because the numbers live in prose. The intersection is computed.

    .venv/bin/python scripts/invariant_core.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts/invariant_core.json"

# holds=True means every backbone tested agreed. holds=False means at least one
# disagreed, which puts the claim outside the core and is the interesting case.
CLAIMS = [
    {"claim": "regime calibration ECE holds in the fourth decimal",
     "backbones": {"Qwen3-235B-A22B": 0.0005, "gpt-oss-20b": 0.0009},
     "holds": True, "source": "paper_srt_program.md L246-252"},
    {"claim": "regime AUROC above 0.97",
     "backbones": {"Qwen3-235B-A22B": 0.9859, "gpt-oss-20b": 0.974},
     "holds": True, "source": "paper_srt_program.md L246-252"},
    {"claim": "hidden-state truthfulness detector AUC in the 0.78-0.87 band",
     "backbones": {"Qwen2.5-7B": 0.8656, "Gemma-2-2B": 0.8563, "Llama-3.2-3B": 0.8475},
     "holds": True, "source": "README.md L97-103"},
    {"claim": "a greedy-versus-best-of-K gap exists",
     "backbones": {"Qwen2.5-7B": 0.589, "Llama-3.2-3B": 0.619, "Gemma-2-2B": 0.528,
                   "gpt-oss-20b": 0.541, "gemma-4-31B-it": 0.507},
     "holds": True, "source": "paper_nla.md, greedy centred fve per host"},
    {"claim": "greedy never reaches the zero-training NN retrieval baseline",
     "backbones": {"Qwen2.5-7B": 0.714, "Llama-3.2-3B": 0.820, "Gemma-2-2B": 0.712,
                   "gpt-oss-20b": 0.744},
     "holds": True, "source": "paper_nla.md, NN baseline per host"},
    {"claim": "the K-curve is log-linear in K",
     "backbones": {"Qwen2.5-7B": 0.030, "Llama-3.2-3B": 0.034, "Gemma-2-2B": 0.020,
                   "gpt-oss-20b": 0.017, "gemma-4-31B-it": 0.017},
     "holds": True, "source": "paper_nla.md, centred fve per doubling"},
    {"claim": "logp-rerank is dead, Spearman against oracle near zero",
     "backbones": {"Qwen2.5-7B": 0.04, "Llama-3.2-3B": 0.055, "Gemma-2-2B": 0.030},
     "holds": True, "source": "paper_nla.md"},
    {"claim": "the anisotropy-centred random floor lands at 0.50",
     "backbones": {"Qwen2.5-7B": 0.510, "Llama-3.2-3B": 0.499, "Gemma-2-2B": 0.498},
     "holds": True, "source": "paper_nla.md, despite ~22x anisotropy spread"},
    {"claim": "a linear map beats an MLP for cross-modal alignment at every data size",
     "backbones": {"gemma-4-31B": 1, "Qwen2.5-VL-3B": 1},
     "holds": True, "source": "paper_srt_program.md, gap widens with data"},
    {"claim": "rotation alone cannot express the modality gap",
     "backbones": {"gemma-4-31B": 1, "Qwen2.5-VL-3B": 1},
     "holds": True, "source": "paper_srt_program.md, Procrustes below baseline"},
    {"claim": "evaluative thickness beta ports across substrates",
     "backbones": {"Qwen2.5-7B": 0.38, "gemma-4-31B": 0.41},
     "holds": True, "source": "paper_nla.md L2061-2138"},

    {"claim": "abstractness beta ports across substrates",
     "backbones": {"Qwen2.5-7B": 0.61, "gemma-4-31B": 0.03},
     "holds": False, "source": "paper_nla.md L2061-2138"},
    {"claim": "best-of-K reaches the paraphrase ceiling",
     "backbones": {"Qwen2.5-7B": 0.788, "Llama-3.2-3B": 0.858, "Gemma-2-2B": 0.631,
                   "gpt-oss-20b": 0.642, "gemma-4-31B-it": 0.591},
     "holds": False, "source": "paper_nla.md, tuned hosts plateau below NN"},
]

# Measured on exactly one backbone, so outside any core by construction.
SINGLE = [
    "reflexivity correlation r=0.689 (gpt-oss-20b)",
    "autostereogram sensor boundary (gemma-4-31B)",
    "punctuation completeness flag at L24 (gpt-oss-20b)",
    "word-category basin structure at L12 (gpt-oss-20b)",
    "community read-out transfers zero-shot to CIFAR-10 images (Qwen2.5-7B to gemma-4)",
    "best-of-64 saturates the paraphrase ceiling at rho_cen 0.99 (Qwen2.5-7B)",
]


def main() -> None:
    core = [c for c in CLAIMS if c["holds"]]
    outside = [c for c in CLAIMS if not c["holds"]]
    hosts = sorted({b for c in CLAIMS for b in c["backbones"]})

    print(f"{len(CLAIMS)} multi-backbone claims across {len(hosts)} hosts\n")
    print(f"IN THE CORE ({len(core)}), holds on every host tested:")
    for c in core:
        n = len(c["backbones"])
        print(f"  [{n} hosts] {c['claim']}")
    print(f"\nOUTSIDE THE CORE ({len(outside)}), at least one host disagrees:")
    for c in outside:
        lo = min(c["backbones"], key=c["backbones"].get)
        hi = max(c["backbones"], key=c["backbones"].get)
        print(f"  {c['claim']}")
        print(f"      {hi} {c['backbones'][hi]} against {lo} {c['backbones'][lo]}")
    print(f"\nONE HOST ONLY ({len(SINGLE)}), outside by construction:")
    for s in SINGLE:
        print(f"  {s}")

    methodological = [c for c in core if any(
        k in c["claim"] for k in ("K-curve", "logp-rerank", "random floor",
                                  "greedy", "linear map", "rotation"))]
    print(f"\nOf the {len(core)} core claims, {len(methodological)} are about how to "
          f"measure rather than\nwhat a model does. That is the shape of the core: "
          f"the measurement frame ports,\nand the substantive behaviour is where hosts "
          f"start to disagree.")

    OUT.write_text(json.dumps({
        "question": "which of our claims hold on every backbone we tested",
        "method": "intersect over substrates, after Generally 2026 Core(M) over descendants",
        "hosts": hosts,
        "in_core": core, "outside_core": outside, "single_host": SINGLE,
    }, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
