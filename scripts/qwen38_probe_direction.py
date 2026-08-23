"""Could our read-only instruments have found the steering direction themselves?

Tonight's steering direction came from diffing an abliterated model against its
base. That is a lucky accident: it needs someone to have already performed the
edit. The method question is whether a probe on the BASE MODEL ALONE recovers
the same direction.

If it does, reading and steering were never separate capabilities in this
program, and a steering vector can be built for any behaviour we can probe,
with no edited model required. The artifact is one direction per layer.

Three things measured here:

  1. AGREEMENT  cos(probe direction, abliteration delta) per layer. The probe is
                the difference of class means, the standard steering-vector
                recipe, fit on base-model states only.
  2. CONTROL    cos(random direction, delta), and a label-shuffled probe, so
                "the probe found it" has a floor to clear.
  3. TRANSFER   inject the PROBE direction into the abliterated model and run
                the same cross-entropy curve the measured delta produced.

    python qwen38_probe_direction.py --alphas 0 2 4 6
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

BASE = "Qwen/Qwen3.8-27B"
ABL = "JonathanColetti/Qwen3.8-27B-Uncensored"
ONSET = 15
MAX_SEQ = 256


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path("/root/srt-adapter"))
    p.add_argument("--sweep", type=Path, default=Path("/root/qwen38_layer_sweep.pt"))
    p.add_argument("--alphas", type=float, nargs="+", default=[0.0, 2.0, 4.0, 6.0])
    p.add_argument("--onset", type=int, default=ONSET)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("/root/qwen38_probe_direction.json"))
    return p.parse_args()


def load_groups(root: Path):
    pairs = json.loads((root / "artifacts/nla_demo_probe_refusal.json").read_text())
    ref = [p["text_a"] for p in pairs if p.get("text_a")]
    com = [p["text_b"] for p in pairs if p.get("text_b")]
    rows = [json.loads(l) for l in (root / "data/val_200.jsonl").read_text().splitlines() if l.strip()]
    return ref, com, [r["text"] for r in rows][:100]


def load(name: str, device: str):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    proc = AutoProcessor.from_pretrained(name)
    model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16).to(device)
    model.eval()
    return proc, model


def batch_ids(chunk, tok):
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
def all_layer_states(model, texts, tok, device, batch, n_layers):
    out = []
    for i in range(0, len(texts), batch):
        ids, attn = batch_ids(texts[i:i + batch], tok)
        res = model(input_ids=ids.to(device), attention_mask=attn.to(device),
                    output_hidden_states=True, use_cache=False)
        last = (attn.sum(-1) - 1).to(device)
        rows = torch.arange(ids.size(0), device=device)
        out.append(torch.stack([res.hidden_states[l][rows, last].float().cpu()
                                for l in range(n_layers)], dim=1))
    return torch.cat(out)


def decoder_layers(model):
    m = model
    for attr in ("model", "language_model", "text_model"):
        if hasattr(m, attr):
            m = getattr(m, attr)
    return m.layers


class Injector:
    def __init__(self, model, vec_by_layer, alpha):
        self.handles = []
        for i, layer in enumerate(decoder_layers(model)):
            if (i + 1) not in vec_by_layer:
                continue
            v = vec_by_layer[i + 1] * alpha

            def hook(_m, _i, out, v=v):
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
        ids, attn = batch_ids(texts[i:i + batch], tok)
        ids_d, am = ids.to(device), attn.to(device)
        out = model(input_ids=ids_d, attention_mask=am, use_cache=False)
        logits = out.logits[:, :-1].float()
        tgt, mask = ids_d[:, 1:], am[:, 1:].bool()
        ce = F.cross_entropy(logits.transpose(1, 2), tgt, reduction="none")
        tot += float(((ce * mask).sum(1) / mask.sum(1).clamp(min=1)).sum())
        n += len(texts[i:i + batch])
    return tot / max(n, 1)


def increments(cum: dict, onset: int) -> dict:
    out = {}
    for l in sorted(cum):
        if l < onset:
            continue
        prev = cum.get(l - 1, torch.zeros_like(cum[l]))
        out[l] = cum[l] - prev
    return out


def main() -> None:
    args = parse_args()
    ref, com, gen = load_groups(args.repo_root)
    print(f"refusal {len(ref)}  compliance {len(com)}  generic {len(gen)}", flush=True)

    S = torch.load(args.sweep, map_location="cpu", weights_only=True)
    layers = S["layers"]
    delta_cum = {l: (S["base"][:, k].mean(0) - S["abl"][:, k].mean(0))
                 for k, l in enumerate(layers)}

    proc_b, model_b = load(BASE, "cuda:0")
    proc_a, model_a = load(ABL, "cuda:1")
    tok = getattr(proc_b, "tokenizer", proc_b)
    n_layers = len(layers)

    # Probe fit on BASE states only: difference of class means, the standard
    # steering-vector recipe. Refusal minus compliance points the same way the
    # abliteration delta should (base is MORE refusing than abliterated).
    R = all_layer_states(model_b, ref, tok, "cuda:0", args.batch, n_layers)
    C = all_layer_states(model_b, com, tok, "cuda:0", args.batch, n_layers)
    probe_cum = {l: (R[:, l].mean(0) - C[:, l].mean(0)) for l in range(n_layers)}

    # The delta from the sweep was measured on CAPTION states while the probe is
    # fit on refusal-register text. Comparing those is not a fair test, so also
    # measure the delta on the refusal texts themselves.
    Ra = all_layer_states(model_a, ref, tok, "cuda:1", args.batch, n_layers)
    delta_ref = {l: (R[:, l].mean(0) - Ra[:, l].mean(0)) for l in range(n_layers)}

    g = torch.Generator().manual_seed(args.seed)
    both = torch.cat([R, C])
    perm = torch.randperm(len(both), generator=g)
    half = len(R)
    shuf_cum = {l: (both[perm[:half], l].mean(0) - both[perm[half:], l].mean(0))
                for l in range(n_layers)}

    print(f"\n{'layer':>5s} {'probe~dCap':>11s} {'probe~dRef':>11s} {'shuf~dRef':>10s} "
          f"{'rand~dRef':>10s} {'dCap~dRef':>10s}", flush=True)
    agree = {}
    for l in sorted(delta_cum):
        if l < args.onset:
            continue
        d, dr = delta_cum[l], delta_ref[l]
        r = torch.randn(d.shape, generator=g)
        cs = lambda x, y: round(float(F.cosine_similarity(x, y, dim=0)), 4)
        agree[l] = {"probe_vs_deltaCaption": cs(probe_cum[l], d),
                    "probe_vs_deltaRefusal": cs(probe_cum[l], dr),
                    "shuffled_vs_deltaRefusal": cs(shuf_cum[l], dr),
                    "random_vs_deltaRefusal": cs(r, dr),
                    "deltaCaption_vs_deltaRefusal": cs(d, dr)}
        if l % 8 == 0 or l == args.onset:
            a = agree[l]
            print(f"{l:5d} {a['probe_vs_deltaCaption']:11.4f} "
                  f"{a['probe_vs_deltaRefusal']:11.4f} "
                  f"{a['shuffled_vs_deltaRefusal']:10.4f} "
                  f"{a['random_vs_deltaRefusal']:10.4f} "
                  f"{a['deltaCaption_vs_deltaRefusal']:10.4f}", flush=True)

    res = {"agreement": agree, "arms": {}, "alphas": args.alphas, "onset": args.onset}

    # Scale the probe direction to the delta's per-layer norm so alpha means the
    # same thing in both experiments.
    probe_scaled = {}
    for l, d in delta_cum.items():
        if l < args.onset:
            continue
        p = probe_cum[l]
        probe_scaled[l] = p / (p.norm() + 1e-9) * d.norm()
    probe_inc = increments(probe_scaled, args.onset)
    delta_inc = increments({l: v for l, v in delta_cum.items()}, args.onset)
    dref_scaled = {l: delta_ref[l] / (delta_ref[l].norm() + 1e-9) * delta_cum[l].norm()
                   for l in delta_cum if l >= args.onset}
    dref_inc = increments(dref_scaled, args.onset)

    def run(tag, model, device, vecs, alpha):
        with Injector(model, vecs, alpha):
            row = {"refusal": round(mean_ce(model, ref, tok, device, args.batch), 4),
                   "compliance": round(mean_ce(model, com, tok, device, args.batch), 4),
                   "generic": round(mean_ce(model, gen, tok, device, args.batch), 4)}
        res["arms"][tag] = row
        print(f"{tag:32s} refusal {row['refusal']:.4f}  generic {row['generic']:.4f}",
              flush=True)
        return row

    print("", flush=True)
    b = run("base_unperturbed", model_b, "cuda:0", {}, 0.0)
    a0 = run("abl_unperturbed", model_a, "cuda:1", {}, 0.0)
    gap = a0["refusal"] - b["refusal"]

    for al in args.alphas:
        if al == 0:
            continue
        run(f"abl +{al}*PROBE", model_a, "cuda:1", probe_inc, al)
        run(f"abl +{al}*dRef", model_a, "cuda:1", dref_inc, al)
        run(f"abl +{al}*delta", model_a, "cuda:1", delta_inc, al)

    args.out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}", flush=True)
    print(f"\nrefusal gap to close: {gap:+.4f} nats")
    for al in args.alphas:
        if al == 0:
            continue
        p = res["arms"][f"abl +{al}*PROBE"]
        d = res["arms"][f"abl +{al}*delta"]
        dr = res["arms"][f"abl +{al}*dRef"]
        print(f"  alpha {al:>4}: probe {(a0['refusal'] - p['refusal']) / gap * 100:6.1f}%  "
              f"dRef {(a0['refusal'] - dr['refusal']) / gap * 100:6.1f}%  "
              f"delta {(a0['refusal'] - d['refusal']) / gap * 100:6.1f}%")


if __name__ == "__main__":
    main()
