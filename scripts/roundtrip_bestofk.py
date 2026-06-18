#!/usr/bin/env python3
"""Best-of-K round-trip fidelity test for the NLA Activation Verbalizer.

For a set of real L20 hidden states, generate K candidate verbalizations from the
AV and re-encode each through the backbone, measuring cosine to the original
state. Reports:

  * floor       : a good verbalization scored against a MISMATCHED target
                  (the anisotropy floor — the "challenge" baseline).
  * greedy      : single deterministic verbalization (the low end).
  * best-of-{K} : max round-trip over K samples (legitimate — the target hidden
                  state is available at inference, so picking the best candidate
                  is deployable, not cheating). This is where ~0.9 appears.

Both raw cosine and fve_nrm = 0.5*(1+cos) are reported, plus a centered variant
(subtract the pool mean from both sides) since raw cosine on this backbone is
anisotropy-dominated (~0.24 for unrelated states).

Run on a CUDA box (transformers==4.53.3), loads the AV backbone (~15 GB):
    python scripts/roundtrip_bestofk.py --k 64 --positions 24
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from srt_introspect import Trace  # noqa: E402

PASSAGES = [
    "The P versus NP problem asks whether every problem whose solution can be quickly verified can also be quickly solved.",
    "Prime factorization underlies RSA encryption: multiplying two large primes is easy, but recovering them is believed hard.",
    "She watched the rain trace slow rivers down the window while the kettle came to a boil in the quiet kitchen.",
    "Quantum computers exploit superposition and entanglement to evaluate many computational paths at once.",
    "The committee debated the amendment for hours before adjourning without reaching any consensus on the budget.",
    "A nondeterministic Turing machine may branch into many configurations, accepting if any branch reaches an accepting state.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen using energy absorbed from sunlight.",
    "He laughed at the absurd joke, then immediately regretted it when he saw the look on her face.",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--positions", type=int, default=24)
    p.add_argument("--verbalize-tokens", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="artifacts/nla/roundtrip_bestofk.json")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    trace = Trace.load(device=args.device)
    av = trace.av
    L = trace.av_extract_layer
    dev = trace.device
    pool_mu = None

    # Collect real L20 hidden states from diverse passages.
    targets = []  # (v float32 on cpu)
    for passage in PASSAGES:
        enc = trace.tok(passage, return_tensors="pt").to(dev)
        out = av.backbone(input_ids=enc.input_ids, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[L][0].float()  # (T, d)
        # sample a few interior positions per passage
        T = h.shape[0]
        for pos in range(max(1, T // 5), T, max(1, T // 4)):
            targets.append(h[pos].cpu())
            if len(targets) >= args.positions:
                break
        if len(targets) >= args.positions:
            break

    pool = torch.stack(targets).to(dev)          # (M, d)
    pool_mu = pool.mean(dim=0, keepdim=True)      # (1, d)
    M = pool.shape[0]
    print(f"Collected {M} target hidden states (L{L}). K={args.k}\n")

    def cos(a, b):
        return float(torch.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1))

    def roundtrip_h(text):
        enc = trace.tok(text, return_tensors="pt").to(dev)
        if enc.input_ids.shape[1] == 0:
            return None
        o = av.backbone(input_ids=enc.input_ids, output_hidden_states=True, use_cache=False)
        return o.hidden_states[L][0, -1].float()

    ks = [1, 4, 8, 16, 32, args.k]
    ks = sorted(set(k for k in ks if k <= args.k))
    rows = []
    for i in range(M):
        v = pool[i]
        # greedy single verbalization
        gtext = av.verbalize(v.to(av.proj.weight.dtype).unsqueeze(0),
                             max_new_tokens=args.verbalize_tokens, do_sample=False)[0]
        gh = roundtrip_h(gtext)
        greedy_cos = cos(gh, v) if gh is not None else 0.0
        # K sampled candidates
        vb = v.to(av.proj.weight.dtype).unsqueeze(0).repeat(args.k, 1).contiguous()
        cand = av.verbalize(vb, max_new_tokens=args.verbalize_tokens,
                           do_sample=True, temperature=1.0, top_p=0.95)
        cand_h = [roundtrip_h(t) for t in cand]
        raw = [cos(h, v) if h is not None else -1.0 for h in cand_h]
        # centered: subtract pool mean from both sides
        cen = [cos(h - pool_mu[0], v - pool_mu[0]) if h is not None else -1.0 for h in cand_h]
        best_at = {k: max(raw[:k]) for k in ks}
        best_at_cen = {k: max(cen[:k]) for k in ks}
        # challenge floor: best candidate scored vs a MISMATCHED target
        j = (i + M // 2) % M
        bi = max(range(args.k), key=lambda x: raw[x])
        floor_cos = cos(cand_h[bi], pool[j]) if cand_h[bi] is not None else 0.0
        rows.append({"greedy": greedy_cos, "mean_k": sum(raw) / len(raw),
                     "best": best_at, "best_cen": best_at_cen, "floor": floor_cos})
        print(f"  pos {i+1}/{M}: greedy={greedy_cos:.3f}  "
              f"best-of-{args.k}={best_at[args.k]:.3f} (cen {best_at_cen[args.k]:.3f})  "
              f"floor={floor_cos:.3f}")

    def fve(c):
        return 0.5 * (1 + c)

    agg = {"n": M, "k": args.k,
           "floor_raw": sum(r["floor"] for r in rows) / M,
           "greedy_raw": sum(r["greedy"] for r in rows) / M,
           "mean_k_raw": sum(r["mean_k"] for r in rows) / M,
           "best_raw": {k: sum(r["best"][k] for r in rows) / M for k in ks},
           "best_cen": {k: sum(r["best_cen"][k] for r in rows) / M for k in ks}}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**agg, "rows": rows}, indent=2))

    print("\n" + "=" * 64)
    print(f"ROUND-TRIP FIDELITY — NLA AV (n={M} states, K={args.k})")
    print("=" * 64)
    print(f"  {'condition':<22} {'raw cos':>9} {'fve_nrm':>9} {'cen cos':>9}")
    print(f"  {'floor (mismatched)':<22} {agg['floor_raw']:9.3f} {fve(agg['floor_raw']):9.3f}")
    print(f"  {'greedy (K=1)':<22} {agg['greedy_raw']:9.3f} {fve(agg['greedy_raw']):9.3f}")
    print(f"  {'mean of K':<22} {agg['mean_k_raw']:9.3f} {fve(agg['mean_k_raw']):9.3f}")
    for k in ks:
        print(f"  {'best-of-' + str(k):<22} {agg['best_raw'][k]:9.3f} "
              f"{fve(agg['best_raw'][k]):9.3f} {agg['best_cen'][k]:9.3f}")
    print("=" * 64)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
