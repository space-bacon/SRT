"""Can we steer with a direction someone else proved works?

Ledger entry 4 of the SRT program is an open negative: the instruments read
reliably but every injection we tried moved behaviour by less than 0.002 nats.
The reason is that we were guessing directions.

Tonight we measured one. `mu_base - mu_abl` is the mean displacement an
abliteration edit produces: zero through layer 14, onset at 15, peak rho 0.074
at 48, register-local, capability-preserving. Somebody performed a successful
causal intervention and left its geometry on the table.

So add it back. If refusal returns to the abliterated model when we inject
+alpha*delta, and disappears from the base model when we inject -alpha*delta,
then a behavioural modification is a linear, transferable, measurable direction
and the read/steer gap closes.

Measured in cross-entropy on register-matched text, so nothing depends on
judging generations:

  RESTORE   abliterated + alpha*delta  -> refusal CE should FALL back toward base
  REMOVE    base       - alpha*delta   -> refusal CE should RISE toward abliterated
  NULL      same norm, random direction -> should do neither

The null is the whole experiment. A large enough perturbation in any direction
degrades everything, so "refusal moved" only counts if it moves further than a
random direction of identical norm, and if generic text moves less.

    python qwen38_steer.py --alphas 0 0.5 1 2 4
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
ONSET = 15          # first layer the abliteration edit touches at all
MAX_SEQ = 256


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path("/root/srt-adapter"))
    p.add_argument("--sweep", type=Path, default=Path("/root/qwen38_layer_sweep.pt"),
                   help="paired encode holding every layer, for the delta")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 4.0])
    p.add_argument("--mode", choices=["increment", "single", "cumulative"],
                   default="increment",
                   help="increment: inject delta_l - delta_{l-1} at each layer, which "
                        "reproduces the observed cumulative profile. cumulative: inject "
                        "delta_l at every layer, which compounds it ~50x and destroys "
                        "the model (measured). single: inject only at --inject-layer.")
    p.add_argument("--inject-layer", type=int, default=48)
    p.add_argument("--onset", type=int, default=ONSET)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("/root/qwen38_steer.json"))
    return p.parse_args()


def load_groups(root: Path) -> dict[str, list[str]]:
    pairs = json.loads((root / "artifacts/nla_demo_probe_refusal.json").read_text())
    g = {"refusal_style": [], "compliance_style": []}
    for p in pairs:
        if p.get("text_a"):
            g["refusal_style"].append(p["text_a"])
        if p.get("text_b"):
            g["compliance_style"].append(p["text_b"])
    rows = [json.loads(l) for l in (root / "data/val_200.jsonl").read_text().splitlines() if l.strip()]
    g["generic"] = [r["text"] for r in rows][:100]
    return g


def load(name: str, device: str):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    proc = AutoProcessor.from_pretrained(name)
    model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16).to(device)
    model.eval()
    return proc, model


def decoder_layers(model):
    m = model
    for attr in ("model", "language_model", "text_model"):
        if hasattr(m, attr):
            m = getattr(m, attr)
    assert hasattr(m, "layers"), "could not locate decoder layers"
    return m.layers


class Injector:
    """Adds a fixed vector to the residual stream from `onset` onward.

    Registered on the decoder layers so the perturbation propagates exactly the
    way the weight edit's own effect would, rather than being applied once at
    the read-out point.
    """

    def __init__(self, model, vec_by_layer, alpha, onset):
        self.handles = []
        layers = decoder_layers(model)
        for i, layer in enumerate(layers):
            if (i + 1) not in vec_by_layer:
                continue
            v = vec_by_layer[i + 1] * alpha

            def hook(_mod, _inp, out, v=v):
                if isinstance(out, tuple):
                    return (out[0] + v.to(out[0].dtype).to(out[0].device),) + out[1:]
                return out + v.to(out.dtype).to(out.device)

            self.handles.append(layer.register_forward_hook(hook))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def mean_ce(model, texts, tok, device, batch):
    tot, n = 0.0, 0
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
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
        ids_d, am = input_ids.to(device), attn.to(device)
        out = model(input_ids=ids_d, attention_mask=am, use_cache=False)
        logits = out.logits[:, :-1].float()
        tgt, mask = ids_d[:, 1:], am[:, 1:].bool()
        ce = F.cross_entropy(logits.transpose(1, 2), tgt, reduction="none")
        tot += float(((ce * mask).sum(1) / mask.sum(1).clamp(min=1)).sum())
        n += len(chunk)
    return tot / max(n, 1)


def main() -> None:
    args = parse_args()
    groups = load_groups(args.repo_root)
    print({k: len(v) for k, v in groups.items()}, flush=True)

    S = torch.load(args.sweep, map_location="cpu", weights_only=True)
    layers = S["layers"]
    cum = {l: (S["base"][:, k].mean(0) - S["abl"][:, k].mean(0))
           for k, l in enumerate(layers)}
    if args.mode == "cumulative":
        delta = {l: v for l, v in cum.items() if l >= args.onset}
    elif args.mode == "single":
        delta = {args.inject_layer: cum[args.inject_layer]}
    else:
        # delta_l - delta_{l-1}: what each layer ADDS, so the running sum
        # reproduces the measured profile instead of compounding it.
        delta = {}
        for l in sorted(cum):
            if l < args.onset:
                continue
            prev = cum.get(l - 1, torch.zeros_like(cum[l]))
            delta[l] = cum[l] - prev
    norms = {l: float(v.norm()) for l, v in delta.items()}
    print(f"mode={args.mode} layers {sorted(delta)} | norms "
          f"{min(norms.values()):.3f} to {max(norms.values()):.3f} "
          f"(cumulative at L48 = {float(cum[48].norm()):.3f})", flush=True)

    g = torch.Generator().manual_seed(args.seed)
    rand = {}
    for l, v in delta.items():
        r = torch.randn(v.shape, generator=g)
        rand[l] = r / r.norm() * v.norm()          # matched norm, random direction

    proc_b, model_b = load(BASE, "cuda:0")
    proc_a, model_a = load(ABL, "cuda:1")
    tok = getattr(proc_b, "tokenizer", proc_b)

    res: dict = {"alphas": args.alphas, "onset": args.onset, "mode": args.mode,
                 "inject_layer": args.inject_layer,
                 "delta_norms": norms, "arms": {}}

    def run(tag, model, device, vecs, alpha):
        with Injector(model, vecs, alpha, args.onset):
            row = {k: round(mean_ce(model, v, tok, device, args.batch), 4)
                   for k, v in groups.items()}
        row["refusal_minus_generic"] = round(row["refusal_style"] - row["generic"], 4)
        res["arms"][tag] = row
        print(f"{tag:34s} refusal {row['refusal_style']:.4f}  "
              f"compliance {row['compliance_style']:.4f}  generic {row['generic']:.4f}",
              flush=True)

    run("base_unperturbed", model_b, "cuda:0", {}, 0.0)
    run("abl_unperturbed", model_a, "cuda:1", {}, 0.0)

    for a in args.alphas:
        if a == 0:
            continue
        run(f"RESTORE abl +{a}*delta", model_a, "cuda:1", delta, a)
        run(f"NULL    abl +{a}*rand", model_a, "cuda:1", rand, a)
        run(f"REMOVE  base -{a}*delta", model_b, "cuda:0", delta, -a)
        run(f"NULL    base -{a}*rand", model_b, "cuda:0", rand, -a)

    args.out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}", flush=True)

    b, ab = res["arms"]["base_unperturbed"], res["arms"]["abl_unperturbed"]
    gap = ab["refusal_style"] - b["refusal_style"]
    print(f"\nrefusal CE gap to close: {gap:+.4f} nats")
    for a in args.alphas:
        if a == 0:
            continue
        r = res["arms"][f"RESTORE abl +{a}*delta"]
        n = res["arms"][f"NULL    abl +{a}*rand"]
        rec = (ab["refusal_style"] - r["refusal_style"]) / gap * 100 if gap else 0
        recn = (ab["refusal_style"] - n["refusal_style"]) / gap * 100 if gap else 0
        print(f"  alpha {a:>4}: delta recovers {rec:6.1f}% of the gap, "
              f"random {recn:6.1f}%  (generic dCE {r['generic'] - ab['generic']:+.4f})")


if __name__ == "__main__":
    main()
