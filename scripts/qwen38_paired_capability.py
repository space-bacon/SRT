"""Paired capability evaluation: base vs abliterated, same batch, both resident.

SugarCrepe is a seven-split proxy that a blind bigram prior largely solves, so it
cannot tell us what an abliteration edit costs. Cross-entropy is the model's own
training objective, so it is the capability measure that matters, and with both
models on separate GPUs the comparison is exact rather than floor-limited.

Groups evaluated, each answering one of the registered predictions:

  generic      data/val_200.jsonl            capability retention on held-out text
  phenomena    probe_battery_v1.jsonl        uniformity across linguistic phenomena
  contested    contestedness_battery_v1.jsonl  does the edit bite harder on
                                              contested material? (tier 0 = physics
                                              and arithmetic, up to contested
                                              political signs)
  refusal      nla_demo_probe_refusal.json   refusal-style vs compliance-style
                                              continuations, the direct mechanism

Prediction 1 (surgical): dCE ~ 0 everywhere, flat across groups.
Prediction 2 (broad damage): dCE > 0 everywhere.
Prediction 3 (refusal-local): dCE ~ 0 on generic, POSITIVE on refusal-style text
                              (less likely = higher cross-entropy; we first wrote this
                              with the sign backwards).

    python qwen38_paired_capability.py --repo-root /root/srt-adapter
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BASE = "Qwen/Qwen3.8-27B"
ABL = "JonathanColetti/Qwen3.8-27B-Uncensored"
MAX_SEQ = 256
READOUT_LAYER = 48  # peak-rho layer from the paired sweep; 0.75 fractional depth


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--base", default=BASE)
    p.add_argument("--abl", default=ABL)
    p.add_argument("--layer", type=int, default=READOUT_LAYER)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--out", type=Path, default=Path("qwen38_capability.json"))
    return p.parse_args()


def load_groups(root: Path) -> dict[str, list[dict]]:
    g: dict[str, list[dict]] = {}

    rows = [json.loads(l) for l in (root / "data/val_200.jsonl").read_text().splitlines() if l.strip()]
    g["generic"] = [{"text": r["text"], "tag": str(r.get("community_label", "?"))} for r in rows]

    rows = [json.loads(l) for l in (root / "data/probes/probe_battery_v1.jsonl").read_text().splitlines() if l.strip()]
    g["phenomena"] = [{"text": r["text"], "tag": r["label"]} for r in rows]

    rows = [json.loads(l) for l in (root / "data/probes/contestedness_battery_v1.jsonl").read_text().splitlines() if l.strip()]
    g["contested"] = [{"text": r["prompt"], "tag": f"tier{r['tier']}",
                       "contest_score": r.get("contest_score")} for r in rows]

    pairs = json.loads((root / "artifacts/nla_demo_probe_refusal.json").read_text())
    ref = []
    for p in pairs:
        for side in ("a", "b"):
            t = p.get(f"text_{side}")
            if t:
                ref.append({"text": t, "tag": "refusal_style" if side == "a" else "compliance_style"})
    g["refusal"] = ref
    return g


def load(name: str, device: str):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    proc = AutoProcessor.from_pretrained(name)
    model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16).to(device)
    model.eval()
    return proc, model


def encode_batch(chunk: list[str], tok):
    ids_list = []
    for t in chunk:
        ids = tok(t, truncation=True, max_length=MAX_SEQ, add_special_tokens=True).input_ids
        bos = tok.bos_token_id
        if bos is not None and ids[0] != bos:
            ids = [bos] + ids[: MAX_SEQ - 1]
        ids_list.append(ids)
    T = max(len(x) for x in ids_list)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.bos_token_id
    input_ids = torch.full((len(ids_list), T), pad, dtype=torch.long)
    attn = torch.zeros((len(ids_list), T), dtype=torch.long)
    for j, ids in enumerate(ids_list):
        input_ids[j, : len(ids)] = torch.tensor(ids)
        attn[j, : len(ids)] = 1
    return input_ids, attn


@torch.no_grad()
def ce_and_state(model, input_ids, attn, device, layer):
    """Mean per-token CE over real (non-pad) positions, plus the last-token state."""
    ids, am = input_ids.to(device), attn.to(device)
    out = model(input_ids=ids, attention_mask=am, output_hidden_states=True, use_cache=False)
    logits = out.logits[:, :-1].float()
    tgt = ids[:, 1:]
    mask = am[:, 1:].bool()
    ce = F.cross_entropy(logits.transpose(1, 2), tgt, reduction="none")
    per_seq = (ce * mask).sum(1) / mask.sum(1).clamp(min=1)
    last = (am.sum(-1) - 1)
    rows = torch.arange(ids.size(0), device=device)
    state = out.hidden_states[layer][rows, last].float().cpu()
    return per_seq.cpu(), state


def summarize(name: str, recs: list[dict]) -> dict:
    ce_b = np.array([r["ce_base"] for r in recs])
    ce_a = np.array([r["ce_abl"] for r in recs])
    d = ce_a - ce_b
    n = len(d)
    out = {
        "n": n,
        "ce_base": float(ce_b.mean()),
        "ce_abl": float(ce_a.mean()),
        "d_ce": float(d.mean()),
        "d_ce_sd": float(d.std(ddof=1)) if n > 1 else 0.0,
        "d_ce_ci95": float(1.96 * d.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        "frac_worse": float((d > 0).mean()),
    }
    by_tag: dict[str, list[float]] = {}
    for r in recs:
        by_tag.setdefault(r["tag"], []).append(r["ce_abl"] - r["ce_base"])
    out["by_tag"] = {k: {"n": len(v), "d_ce": float(np.mean(v))}
                     for k, v in sorted(by_tag.items())}
    return out


def main() -> None:
    args = parse_args()
    groups = load_groups(args.repo_root)
    print({k: len(v) for k, v in groups.items()}, flush=True)

    proc_b, model_b = load(args.base, "cuda:0")
    proc_a, model_a = load(args.abl, "cuda:1")
    tok_b = getattr(proc_b, "tokenizer", proc_b)
    tok_a = getattr(proc_a, "tokenizer", proc_a)
    probe = [r["text"] for v in groups.values() for r in v][:128]
    assert all(tok_b(t).input_ids == tok_a(t).input_ids for t in probe), \
        "tokenizers disagree; the pairing would be confounded"

    results: dict[str, dict] = {}
    disp: dict[str, dict] = {}
    for gname, items in groups.items():
        recs, Sb, Sa = [], [], []
        for i in range(0, len(items), args.batch):
            chunk = items[i:i + args.batch]
            ids, attn = encode_batch([c["text"] for c in chunk], tok_b)
            cb, sb = ce_and_state(model_b, ids, attn, "cuda:0", args.layer)
            ca, sa = ce_and_state(model_a, ids, attn, "cuda:1", args.layer)
            Sb.append(sb)
            Sa.append(sa)
            for c, x, y in zip(chunk, cb.tolist(), ca.tolist()):
                recs.append({"tag": c["tag"], "ce_base": x, "ce_abl": y})
        results[gname] = summarize(gname, recs)

        B, A = torch.cat(Sb), torch.cat(Sa)
        d = (A.mean(0) - B.mean(0)).norm().item()
        disp[gname] = {
            "disp_norm": d,
            "rho": d / max(B.norm(dim=-1).median().item(), 1e-9),
            "mean_cos": float(F.cosine_similarity(B, A, dim=-1).mean()),
        }
        r = results[gname]
        print(f"{gname:11s} n={r['n']:4d}  CE {r['ce_base']:.4f} -> {r['ce_abl']:.4f}  "
              f"dCE {r['d_ce']:+.4f} +/-{r['d_ce_ci95']:.4f}  "
              f"rho {disp[gname]['rho']:.4f}  cos {disp[gname]['mean_cos']:.4f}", flush=True)

    payload = {"base": args.base, "abl": args.abl, "layer": args.layer,
               "max_seq": MAX_SEQ, "capability": results, "displacement": disp}
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}", flush=True)

    print("\nrefusal-style vs compliance-style dCE:")
    for k, v in results["refusal"]["by_tag"].items():
        print(f"  {k:18s} n={v['n']:3d}  dCE {v['d_ce']:+.4f}")
    print("\ncontestedness tiers:")
    for k, v in results["contested"]["by_tag"].items():
        print(f"  {k:18s} n={v['n']:3d}  dCE {v['d_ce']:+.4f}")


if __name__ == "__main__":
    main()
