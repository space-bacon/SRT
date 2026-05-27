"""Phase 0 smoke test for `srt_introspect` integration.

Verifies that, on the local machine (Mac M2 Ultra / MPS or CUDA), we can:

  1. Load the Stage 4 ActivationVerbalizer (AV) from release config +
     local best_av.pt checkpoint.
  2. Encode a short prompt through the AV's frozen backbone, extract
     the last-token L20 hidden state v.
  3. Verbalize v back to text at K samples and measure latency.
  4. (Optionally) load Stage 3 SRTAdapter and run forward on the same
     prompt, extracting per-token divergence / regime signals.

Outputs a one-screen latency/feasibility report. No artifacts written.

Usage:
    source .venv-adapter/bin/activate
    python scripts/probes/smoke_introspect.py
    python scripts/probes/smoke_introspect.py --skip-adapter   # AV only
    python scripts/probes/smoke_introspect.py --device cpu     # force CPU
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from srt.nla import ActivationVerbalizer, NLAConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
AV_CONFIG_LOCAL = REPO_ROOT / "release" / "nla_v1" / "config.json"
AV_CKPT_LOCAL = REPO_ROOT / "artifacts" / "nla" / "ce_seq64_np16_30k" / "best_av.pt"
AV_HF_REPO = "RiverRider/srt-nla-av-v1"

PROMPT = "What killed the dinosaurs? The leading hypothesis is the"


def _resolve_av_artifacts() -> tuple[Path, Path]:
    """Return (config_path, ckpt_path); fall back to HF if local missing."""
    if AV_CONFIG_LOCAL.is_file() and AV_CKPT_LOCAL.is_file():
        return AV_CONFIG_LOCAL, AV_CKPT_LOCAL
    print(f"  local AV artifacts not found, pulling from HF: {AV_HF_REPO}")
    from huggingface_hub import hf_hub_download
    cfg = AV_CONFIG_LOCAL if AV_CONFIG_LOCAL.is_file() else Path(
        hf_hub_download(AV_HF_REPO, "config.json")
    )
    ckpt = AV_CKPT_LOCAL if AV_CKPT_LOCAL.is_file() else Path(
        hf_hub_download(AV_HF_REPO, "best_av.pt")
    )
    return cfg, ckpt


def _pick_device(arg: str | None) -> str:
    if arg:
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _hr(s: str) -> None:
    print(f"\n--- {s} ---")


def run_av_smoke(device: str, k: int, max_new_tokens: int) -> dict:
    _hr("AV: load")
    cfg_path, ckpt_path = _resolve_av_artifacts()
    cfg = NLAConfig.from_json(cfg_path)
    print(f"  cfg={cfg_path}")
    print(f"  ckpt={ckpt_path}")
    print(f"  backbone={cfg.backbone_id} L={cfg.extraction_layer} dtype={cfg.backbone_dtype}")
    print(f"  prefix_mode={cfg.prefix_mode} num_prefix_tokens={cfg.num_prefix_tokens} num_inject_slots={cfg.num_inject_slots}")

    t0 = time.perf_counter()
    av = ActivationVerbalizer(cfg)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict):
        # Released best_av.pt is wrapped: {'trainable': {...}, 'step': ..., 'cfg': ...}
        # Older naming: {'state_dict': {...}}
        for _wkey in ("trainable", "state_dict"):
            if _wkey in sd and isinstance(sd[_wkey], dict):
                sd = sd[_wkey]
                break
    missing, unexpected = av.load_state_dict(sd, strict=False)
    # The backbone params are not in best_av.pt; they were just loaded fresh
    # from HF via AutoModelForCausalLM. Filter them out of the missing-keys
    # report so we only see real surprises.
    real_missing = [k for k in missing if not k.startswith("backbone.")]
    print(f"  loaded: {len(real_missing)} adapter-missing, {len(unexpected)} unexpected")
    if real_missing:
        print(f"  WARNING missing adapter keys: {real_missing[:5]}{'...' if len(real_missing)>5 else ''}")
    if unexpected:
        print(f"  WARNING unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    av = av.to(device).eval()
    print(f"  load_time={time.perf_counter()-t0:.1f}s  device={device}")

    _hr("AV: extract L20 hidden state from prompt")
    tok = av.tokenizer
    enc = tok(PROMPT, return_tensors="pt").to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = av.backbone(**enc, output_hidden_states=True, use_cache=False)
    extract_s = time.perf_counter() - t0
    # hidden_states is len num_hidden_layers + 1 (index 0 = embeddings)
    h_L = out.hidden_states[cfg.extraction_layer]  # (1, T, d)
    v = h_L[0, -1].to(torch.float32)  # last token's L20 state
    print(f"  prompt={PROMPT!r}")
    print(f"  T={enc.input_ids.shape[1]}  hidden_size={v.shape[0]}  ||v||={v.norm().item():.2f}  extract_time={extract_s*1000:.0f}ms")

    _hr(f"AV: verbalize K={k} samples (max_new_tokens={max_new_tokens})")
    k = int(k)
    v_batch = v.to(av.proj.weight.dtype).unsqueeze(0).repeat(k, 1).contiguous().to(device)
    t0 = time.perf_counter()
    texts = av.verbalize(
        v_batch,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
    )
    verbalize_s = time.perf_counter() - t0
    print(f"  total={verbalize_s:.2f}s  per_sample={verbalize_s/k*1000:.0f}ms")
    for i, t in enumerate(texts):
        snippet = t.replace("\n", " ").strip()[:120]
        print(f"  [{i}] {snippet}")

    return {
        "device": device,
        "extract_ms": extract_s * 1000,
        "verbalize_total_s": verbalize_s,
        "verbalize_per_sample_ms": verbalize_s / k * 1000,
        "samples": texts,
        "v_norm": v.norm().item(),
    }


def run_adapter_smoke(device: str) -> dict | None:
    _hr("SRTAdapter (Stage 3): load")
    try:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file as load_safetensors
        from srt.adapter import SRTAdapter
        from srt.config import SRTConfig
    except Exception as e:
        print(f"  SKIP — import failed: {e}")
        return None

    # Pull v1.0 (stable) adapter + config from HF
    try:
        cfg_path = hf_hub_download("RiverRider/srt-adapter-v1.0", "config.json")
        weights_path = hf_hub_download("RiverRider/srt-adapter-v1.0", "adapter.safetensors")
    except Exception as e:
        print(f"  SKIP — HF download failed: {e}")
        return None

    import json
    raw_cfg = json.loads(Path(cfg_path).read_text())
    print(f"  config keys: {sorted(raw_cfg.keys())[:8]}...")

    t0 = time.perf_counter()
    try:
        cfg = SRTConfig.from_json(cfg_path)
    except Exception as e:
        print(f"  SKIP — SRTConfig construction failed: {e}")
        return None

    try:
        model = SRTAdapter(cfg)
        sd = load_safetensors(weights_path, device="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        real_missing = [k for k in missing if not k.startswith("backbone.")]
        print(f"  loaded: {len(real_missing)} adapter-missing, {len(unexpected)} unexpected (load_time={time.perf_counter()-t0:.1f}s)")
        if real_missing[:3]:
            print(f"  sample missing: {real_missing[:3]}")
    except Exception as e:
        print(f"  SKIP — SRTAdapter init/load failed: {e}")
        return None

    model = model.to(device).eval()

    _hr("SRTAdapter: forward on prompt")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.backbone_id if hasattr(cfg, "backbone_id") else "Qwen/Qwen2.5-7B")
    enc = tok(PROMPT, return_tensors="pt").to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
    fwd_s = time.perf_counter() - t0
    print(f"  forward_time={fwd_s*1000:.0f}ms")
    print(f"  output attrs: {[a for a in dir(out) if not a.startswith('_')][:15]}")

    # Try to read divergences and BEN output
    if hasattr(out, "divergences") and out.divergences is not None:
        divs = out.divergences
        if isinstance(divs, dict):
            for k_, v_ in list(divs.items())[:3]:
                if hasattr(v_, "shape"):
                    print(f"  divergences[{k_}]: shape={tuple(v_.shape)}  norm_per_tok={v_.float().norm(dim=-1).mean().item():.3f}")
    if hasattr(out, "ben_output") and out.ben_output is not None:
        ben = out.ben_output
        for attr in ("r_hat", "regime_logits"):
            x = getattr(ben, attr, None)
            if x is not None and hasattr(x, "shape"):
                print(f"  ben.{attr}: shape={tuple(x.shape)}  sample={x.flatten()[:3].tolist()}")

    return {"forward_ms": fwd_s * 1000}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default=None, help="cuda|mps|cpu (auto if omitted)")
    p.add_argument("-k", type=int, default=4, help="K samples for verbalize timing")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--skip-adapter", action="store_true")
    args = p.parse_args()

    device = _pick_device(args.device)
    print(f"device={device}  torch={torch.__version__}")

    av_res = run_av_smoke(device, args.k, args.max_new_tokens)

    adapter_res = None
    if not args.skip_adapter:
        try:
            adapter_res = run_adapter_smoke(device)
        except Exception as e:
            print(f"\nadapter smoke failed (non-fatal): {type(e).__name__}: {e}")

    _hr("Phase 0 verdict")
    print(f"AV extract:    {av_res['extract_ms']:.0f}ms / prompt")
    print(f"AV verbalize:  {av_res['verbalize_per_sample_ms']:.0f}ms / sample @ {args.max_new_tokens} tok")
    print(f"  → K=8 budget: {av_res['verbalize_per_sample_ms']*8/1000:.1f}s per verbalization")
    print(f"  → K=64 budget: {av_res['verbalize_per_sample_ms']*64/1000:.1f}s per verbalization")
    if adapter_res:
        print(f"Adapter fwd:   {adapter_res['forward_ms']:.0f}ms / prompt")
    else:
        print("Adapter:       skipped/failed (see above)")


if __name__ == "__main__":
    main()
