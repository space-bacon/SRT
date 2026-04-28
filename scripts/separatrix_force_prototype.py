#!/usr/bin/env python3
"""Counterfactual prototype-forcing test for Haylett prediction (2).

Tests whether forcing the v8a SRT-Adapter's community vector to the
'mystical' / 'bedrock' / 'technical' centroid (estimated from the
separatrix probe) selectively increases the log-probability of that
branch's actual continuation tokens.

Procedure
---------
1. Read separatrix readouts and the original separatrix probe.
2. From readouts.jsonl (artifacts/separatrix/v8a/readouts.jsonl), pull the
   per-item per-branch community_vector. Compute the centroid for each
   branch type (technical, mystical, bedrock).
3. For each item, for each branch (technical, mystical, bedrock):
     ce_natural = mean cross-entropy of branch continuation tokens given
       prompt, with no forcing (community discovered normally).
     ce_force[X] = mean CE under forced_community = centroid[X]
       for X in {technical, mystical, bedrock}.
   For each branch, the diagonal contrast is:
     delta_self = ce_force[other] - ce_force[self]      (expect > 0)
   averaged over the two off-diagonal alternatives.
4. Wilcoxon signed-rank across items.

Pred (2) holds iff delta_self > 0 with non-trivial frac+ for at least the
mystical and bedrock branches (technical may not be a coherent prototype
since it is the model's default mode).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from srt.adapter import SRTAdapter
from srt.config import SRTConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("force_proto")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    p.add_argument("--adapter", required=True)
    p.add_argument("--probe", default="data/probes/separatrix_illusion_v1.jsonl")
    p.add_argument("--readouts", default="artifacts/separatrix/v8a/readouts.jsonl",
                   help="JSONL with community_vector per branch (from separatrix_eval.py)")
    p.add_argument("--output", required=True)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def wilcoxon(deltas):
    nz = [d for d in deltas if d != 0.0]
    n = len(nz)
    if n == 0:
        return None
    s = sorted(enumerate(nz), key=lambda kv: abs(kv[1]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(s[j + 1][1]) == abs(s[i][1]):
            j += 1
        ar = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[s[k][0]] = ar
        i = j + 1
    wp = sum(r for r, d in zip(ranks, nz) if d > 0)
    wn = sum(r for r, d in zip(ranks, nz) if d < 0)
    w = min(wp, wn)
    mu = n * (n + 1) / 4
    sg = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w - mu) / sg if sg > 0 else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))
    return dict(n=n, mean=sum(deltas) / len(deltas),
                fracpos=sum(1 for d in deltas if d > 0) / len(deltas),
                p=p)


def compute_centroids(readouts_path: Path) -> dict[str, torch.Tensor]:
    rows = [json.loads(l) for l in readouts_path.read_text().splitlines() if l.strip()]
    by_branch: dict[str, list[list[float]]] = {"technical": [], "mystical": [], "bedrock": []}
    for r in rows:
        for b in ("technical", "mystical", "bedrock"):
            v = r["branches"].get(b, {}).get("community_vector")
            if v:
                by_branch[b].append(v)
    centroids = {}
    for b, vecs in by_branch.items():
        if not vecs:
            log.warning("no community vectors for branch %s", b)
            continue
        t = torch.tensor(vecs)
        centroids[b] = t.mean(dim=0)
        log.info("centroid[%s]: n=%d d=%d norm=%.3f",
                 b, t.shape[0], t.shape[1], float(centroids[b].norm()))
    return centroids


@torch.no_grad()
def branch_ce(model: SRTAdapter, ids_full: torch.Tensor,
              att_full: torch.Tensor, n_prompt: int,
              forced: torch.Tensor | None) -> float:
    """Mean CE over the continuation tokens (positions n_prompt..T-1)."""
    fc = None
    if forced is not None:
        fc = forced.to(ids_full.device).to(model._lm_head.weight.dtype).unsqueeze(0)
    o = model(input_ids=ids_full, attention_mask=att_full, forced_community=fc)
    # next-token CE on continuation: logits[i] predicts token[i+1]
    # so logits[n_prompt-1 : T-1] predict tokens[n_prompt : T]
    T = int(att_full.sum())
    logits = o.logits[0, n_prompt - 1: T - 1, :].float()  # (Tc, V)
    targets = ids_full[0, n_prompt: T]                    # (Tc,)
    if logits.shape[0] == 0:
        return float("nan")
    ce = F.cross_entropy(logits, targets, reduction="mean")
    return float(ce)


@torch.no_grad()
def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    cfg = SRTConfig(backbone_id=args.backbone, backbone_dtype=args.dtype)
    cfg.community.use_prototypes = False
    model = SRTAdapter(cfg).to(device)
    model.load_adapter(args.adapter)
    model.eval()

    tok = AutoTokenizer.from_pretrained(args.backbone)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    centroids = compute_centroids(Path(args.readouts))
    if set(centroids) != {"technical", "mystical", "bedrock"}:
        log.error("missing centroids: have %s", list(centroids))
        return 2

    items = [json.loads(l) for l in Path(args.probe).read_text().splitlines() if l.strip()]
    log.info("loaded %d items", len(items))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    BRANCHES = ("technical", "mystical", "bedrock")
    with out_path.open("w") as fout:
        for item in items:
            prompt = item["prompt"]
            n_prompt = int(tok(prompt, add_special_tokens=False,
                               return_tensors="pt")["input_ids"].shape[1])
            row = {"id": item["id"], "concept": item["concept"],
                   "domain": item["domain"], "n_prompt_tokens": n_prompt,
                   "branches": {}}
            for b in BRANCHES:
                text = prompt + item[f"{b}_continuation"]
                enc = tok(text, return_tensors="pt", truncation=True,
                          max_length=args.max_seq_len, padding=False)
                ids = enc["input_ids"].to(device)
                att = enc["attention_mask"].to(device)
                T = int(att.sum())
                if T <= n_prompt:
                    continue
                ces = {}
                ces["natural"] = branch_ce(model, ids, att, n_prompt, None)
                for forced_b in BRANCHES:
                    ces[f"force_{forced_b}"] = branch_ce(
                        model, ids, att, n_prompt, centroids[forced_b]
                    )
                row["branches"][b] = ces
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            rows.append(row)

    # --- Aggregate ---
    log.info("=== Counterfactual prototype-forcing (n=%d items) ===", len(rows))
    log.info("Diagonal effect: CE under force_self should be LOWER than under force_other")
    log.info("(reported delta = CE_other - CE_self; expect > 0)")
    summary = {"n_items": len(rows), "results": {}}
    for b in BRANCHES:
        deltas = []
        for r in rows:
            if b not in r["branches"]:
                continue
            ce = r["branches"][b]
            self_ce = ce[f"force_{b}"]
            other_ces = [ce[f"force_{ob}"] for ob in BRANCHES if ob != b]
            mean_other = sum(other_ces) / len(other_ces)
            deltas.append(mean_other - self_ce)
        w = wilcoxon(deltas)
        summary["results"][b] = w
        if w:
            log.info("branch=%-9s mean_delta=%+.4f frac+=%.2f n=%d p=%.3g",
                     b, w["mean"], w["fracpos"], w["n"], w["p"])
    # Pairwise diagonal contrasts (force_X better for branch X than for
    # the other two specific branches?)
    log.info("Pairwise (force_self vs each force_other):")
    pairwise = {}
    for b in BRANCHES:
        for ob in BRANCHES:
            if ob == b:
                continue
            deltas = [r["branches"][b][f"force_{ob}"] - r["branches"][b][f"force_{b}"]
                      for r in rows if b in r["branches"]]
            w = wilcoxon(deltas)
            key = f"branch={b}, force_{ob} - force_{b}"
            pairwise[key] = w
            if w:
                log.info("  %s: mean=%+.4f frac+=%.2f p=%.3g",
                         key, w["mean"], w["fracpos"], w["p"])
    summary["pairwise"] = pairwise
    Path(args.output).with_name("force_proto_summary.json").write_text(
        json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
