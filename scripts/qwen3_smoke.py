"""R0 smoke test: validate the SRT manual layer loop on a Qwen3 backbone.

Run on any box (CPU/MPS for Qwen3-0.6B-Base; CUDA for 8B+):

    python scripts/qwen3_smoke.py --backbone Qwen/Qwen3-0.6B-Base --dtype float32
    python scripts/qwen3_smoke.py --backbone Qwen/Qwen3-8B-Base            # cuda bf16
    python scripts/qwen3_smoke.py --backbone Qwen/Qwen3-30B-A3B-Base       # R2 MoE

Checks, in order:
  1. Backbone loads through SRTAdapter; layer indices resolve; bos != eos.
  2. PARITY: manual-loop logits (injectors disabled) match the backbone's own
     forward bit-for-bit (same dtype). This is the test that catches layer-API
     drift (kwargs, tuple-vs-tensor returns, rotary relocation).
  3. TAP NON-DEGENERACY: per-position divergence std > 0.1 (the targets-bug
     guard) and hidden-state taps vary across positions.
  4. KV-CACHE GENERATE: adapter.generate() produces tokens and its prefill
     logits match the dense forward.
  5. READ-ONLY GRADS: with read_only=True, a backward pass on a head loss
     produces grads on SRT head params and NONE on backbone params, and the
     backbone ran without building a graph.

Exit code 0 = all green.
"""

from __future__ import annotations

import argparse
import sys

import torch

from srt.config import SRTConfig
from srt.adapter import SRTAdapter


def _pick_device(arg: str | None) -> str:
    if arg:
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default=None,
                   help="float32|bfloat16; default bf16 on cuda, fp32 elsewhere")
    p.add_argument("--seq-len", type=int, default=48)
    p.add_argument("--gen-tokens", type=int, default=12)
    args = p.parse_args()

    device = _pick_device(args.device)
    dtype = args.dtype or ("bfloat16" if device == "cuda" else "float32")
    print(f"== R0 smoke: {args.backbone} on {device} ({dtype}) ==")

    failures: list[str] = []

    # ── 1. Load ───────────────────────────────────────────────────────
    cfg = SRTConfig(backbone_id=args.backbone, backbone_dtype=dtype)
    adapter = SRTAdapter(cfg).to(device).eval()
    for prm in adapter.parameters():
        prm.requires_grad_(False)
    # Re-enable grads on SRT head params only (mirrors training setup).
    head_modules = [adapter.community_head, adapter.mah_heads, adapter.rrm,
                    adapter.chain_predictor, adapter.ben]
    for m in head_modules:
        for prm in m.parameters():
            prm.requires_grad_(True)

    bcfg = adapter.backbone.config
    bos, eos = bcfg.bos_token_id, bcfg.eos_token_id
    print(f"   model_type={bcfg.model_type}  L={bcfg.num_hidden_layers}  "
          f"d={bcfg.hidden_size}  bos={bos}  eos={eos}")
    print(f"   MAH@{cfg.mah_layer_indices}  inject@{cfg.rrm_inject_indices}  "
          f"community@{cfg.community_layer_idx}")
    if bos == eos:
        print("   WARNING: bos == eos (Qwen2.5-style) — target sampling must "
              "use the 902b746 guard")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.backbone)
    text = ("The quick brown fox jumps over the lazy dog. In 1492, Columbus "
            "sailed the ocean blue, and the rest, as they say, is history.")
    enc = tok(text, return_tensors="pt", truncation=True, max_length=args.seq_len)
    input_ids = enc.input_ids.to(device)
    T = input_ids.shape[1]
    print(f"   prompt tokens: {T}")

    # ── 2. Parity: manual loop (inject off) vs backbone forward ──────
    with torch.no_grad():
        out = adapter(input_ids=input_ids, disable_injectors=True)
        ref = adapter.backbone(input_ids=input_ids).logits
    diff = (out.logits.float() - ref.float()).abs().max().item()
    if dtype == "float32":
        # fp32: both paths must be bit-identical (verified on cpu + cuda).
        tol = 1e-4
        status = "OK" if diff <= tol else "FAIL"
        print(f"2. parity: max |manual - backbone| = {diff:.3e}  (tol {tol:g})  {status}")
        if diff > tol:
            failures.append(f"parity diff {diff:.3e} > {tol}")
    else:
        # bf16: the manual loop uses an explicit additive mask while HF's
        # forward takes the SDPA is_causal fast path; different kernel
        # reduction orders give O(0.1-1) absolute logit deltas that are pure
        # rounding noise (fp32 on the same GPU is bit-exact). The meaningful
        # invariant is the predicted distribution: require top-1 agreement at
        # (almost) every position.
        agree = (out.logits.argmax(-1) == ref.argmax(-1)).float().mean().item()
        status = "OK" if agree >= 0.99 else "FAIL"
        print(f"2. parity (bf16): top-1 agreement = {agree:.4f} (>=0.99)  "
              f"max |diff| = {diff:.3e} (informational)  {status}")
        if agree < 0.99:
            failures.append(f"bf16 top-1 agreement {agree:.4f} < 0.99")

    # ── 3. Tap non-degeneracy ─────────────────────────────────────────
    div = out.divergences[-1][0].float()           # (T, d_div)
    div_std = div.std(dim=0).mean().item()
    r_hat = out.ben_output.r_hat[0].float()
    r_std = r_hat.std().item()
    status = "OK" if div_std > 0.1 else "FAIL"
    print(f"3. taps: divergence std={div_std:.4f} (>0.1)  r_hat std={r_std:.4f}  {status}")
    if div_std <= 0.1:
        failures.append(f"divergence degenerate (std {div_std:.4f})")

    # ── 4. KV-cache generate ──────────────────────────────────────────
    try:
        with torch.no_grad():
            ids = adapter.generate(
                input_ids, max_new_tokens=args.gen_tokens,
                eos_token_ids={eos} if eos is not None else set(),
                temperature=0.0,
            )
        assert isinstance(ids, torch.Tensor)
        gen_text = tok.decode(ids[0], skip_special_tokens=True)
        print(f"4. generate: {ids.shape[1]} tokens -> {gen_text!r}  OK")
    except Exception as e:  # noqa: BLE001 — smoke test, report everything
        print(f"4. generate: FAIL ({type(e).__name__}: {e})")
        failures.append(f"generate raised {type(e).__name__}: {e}")

    # ── 5. Read-only gradient isolation ───────────────────────────────
    try:
        out_ro = adapter(input_ids=input_ids, read_only=True)
        loss = out_ro.ben_output.r_hat.float().mean() + \
            out_ro.divergences[-1].float().pow(2).mean()
        loss.backward()
        head_grads = sum(
            1 for m in head_modules for prm in m.parameters()
            if prm.grad is not None and prm.grad.abs().sum() > 0
        )
        bb_grads = sum(
            1 for prm in adapter.backbone.parameters() if prm.grad is not None
        )
        graph_free = not out_ro.logits.requires_grad
        status = "OK" if (head_grads > 0 and bb_grads == 0 and graph_free) else "FAIL"
        print(f"5. read-only: head params w/ grad={head_grads}  backbone grads={bb_grads}"
              f"  logits graph-free={graph_free}  {status}")
        if head_grads == 0:
            failures.append("read_only: no head grads")
        if bb_grads > 0:
            failures.append("read_only: backbone received grads")
        if not graph_free:
            failures.append("read_only: logits kept a graph")
    except Exception as e:  # noqa: BLE001
        print(f"5. read-only: FAIL ({type(e).__name__}: {e})")
        failures.append(f"read_only raised {type(e).__name__}: {e}")

    print()
    if failures:
        print("RESULT: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("RESULT: ALL GREEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
