"""Shared NLA pipeline used by demo/app.py.

Wraps the trained Activation Verbalizer (AV) for any of three backbones
behind a single class with three operations:

    1. extract_hidden(prompt, token_index) -> torch.Tensor
       Take a prompt string, run the frozen backbone forward, return the
       hidden state at the configured extraction layer at the given token
       position (default: last token).

    2. verbalize(v, K, temperature) -> list[(text, raw_fve, centred_fve)]
       Generate K candidate verbalisations of vector v, score each one
       round-trip, return them sorted by centred fidelity.

    3. steer_with_text(prompt, replacement_text, alpha) -> str
       Re-encode `replacement_text` to a layer-L hidden vector v_new, then
       run the frozen backbone forward on `prompt` with v_new added (scaled
       by alpha) to the L20 last-token hidden state at every position. Return
       the greedy continuation.

Loaded weights come from the public HF release pack; nothing in this
file is repo-specific past the import path of `srt.nla`.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

# Make `srt` importable when running from repo root or from demo/.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from srt.nla.config import NLAConfig  # noqa: E402
from srt.nla.verbalizer import ActivationVerbalizer  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

log = logging.getLogger("nla_demo")


# ---------------------------------------------------------------------------
# Backbone registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BackboneSpec:
    label: str                # display name in UI
    backbone_id: str          # HF model ID for the frozen LM
    extraction_layer: int     # layer to read hidden states from
    av_repo: str              # HF repo with best_av.pt
    av_filename: str = "best_av.pt"
    targets_repo: str | None = None   # HF dataset for centring pool
    targets_filename: str | None = None
    num_prefix_tokens: int = 1
    num_inject_slots: int = 1


BACKBONES: dict[str, BackboneSpec] = {
    "qwen2.5-7b": BackboneSpec(
        label="Qwen 2.5-7B (L20)",
        backbone_id="Qwen/Qwen2.5-7B",
        extraction_layer=20,
        av_repo="RiverRider/srt-nla-av-v1",
        av_filename="best_av.pt",
        targets_repo="RiverRider/srt-nla-targets-v1",
        targets_filename="targets_q7b_L20_seq64_30k_seed1.pt",
    ),
    "llama-3.2-3b": BackboneSpec(
        label="Llama 3.2-3B (L20)",
        backbone_id="meta-llama/Llama-3.2-3B",
        extraction_layer=20,
        av_repo="RiverRider/srt-nla-av-llama32-3b",
        av_filename="best_av.pt",
        targets_repo="RiverRider/srt-nla-targets-llama32-3b-v1",
        targets_filename="targets_L20_seq64_30k_seed1.pt",
    ),
    "gemma-2-2b": BackboneSpec(
        label="Gemma 2-2B (L19)",
        backbone_id="google/gemma-2-2b",
        extraction_layer=19,
        av_repo="RiverRider/srt-nla-av-gemma2-2b-v1",
        av_filename="best_av.pt",
        targets_repo="RiverRider/srt-nla-targets-gemma2-2b-v1",
        targets_filename="targets_L19_seq64_30k_seed1.pt",
    ),
}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class NLAPipeline:
    """One backbone + AV + centring pool, kept resident on a single device."""

    def __init__(
        self,
        spec: BackboneSpec,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        pool_size: int = 2000,
    ) -> None:
        self.spec = spec
        self.device = torch.device(device)
        log.info("Loading frozen backbone %s on %s", spec.backbone_id, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(spec.backbone_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.backbone = AutoModelForCausalLM.from_pretrained(
            spec.backbone_id, torch_dtype=dtype
        ).to(self.device)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        cfg = NLAConfig(
            backbone_id=spec.backbone_id,
            extraction_layer=spec.extraction_layer,
            num_prefix_tokens=spec.num_prefix_tokens,
            num_inject_slots=spec.num_inject_slots,
            max_new_tokens=64,
        )
        self.av = ActivationVerbalizer(cfg, backbone=self.backbone, tokenizer=self.tokenizer)

        log.info("Downloading AV checkpoint from %s", spec.av_repo)
        ckpt_path = hf_hub_download(spec.av_repo, spec.av_filename)
        sd = torch.load(ckpt_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        # Filter to AV-only keys (proj, prefix_embeds, etc.)
        own = self.av.state_dict()
        filtered = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        missing = set(own) - set(filtered)
        if missing:
            log.warning("AV checkpoint missing keys: %s", sorted(missing)[:10])
        self.av.load_state_dict(filtered, strict=False)
        self.av.to(self.device)
        self.av.eval()

        # Centring pool
        self.mu: torch.Tensor | None = None
        if spec.targets_repo and spec.targets_filename:
            try:
                log.info("Downloading centring pool %s", spec.targets_repo)
                pool_path = hf_hub_download(
                    spec.targets_repo, spec.targets_filename, repo_type="dataset"
                )
                pool = torch.load(pool_path, map_location="cpu")
                if isinstance(pool, dict):
                    pool = pool.get("hiddens", pool.get("targets", next(iter(pool.values()))))
                pool = pool[: pool_size]
                self.mu = pool.float().mean(0).to(self.device)
                log.info("Centring pool ||mu|| = %.2f", self.mu.norm().item())
            except Exception as e:  # noqa: BLE001
                log.warning("Could not load centring pool (%s); centred fve disabled", e)

        self.dtype = dtype

    # ------------------------------------------------------------------
    # Hidden-state extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_hidden(
        self,
        prompt: str,
        token_index: int = -1,
    ) -> torch.Tensor:
        """Run the frozen backbone on `prompt`; return the layer-L hidden
        state at `token_index` (default: last input token), shape (d,)."""
        ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.backbone(
            input_ids=ids.input_ids,
            attention_mask=ids.attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        h = out.hidden_states[self.spec.extraction_layer][0]  # (T, d)
        if token_index < 0:
            token_index = h.size(0) + token_index
        return h[token_index].float().detach()

    @torch.no_grad()
    def extract_all_hidden(self, prompt: str) -> tuple[list[str], torch.Tensor]:
        """Return (per-token decoded strings, hiddens of shape (T, d))."""
        ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.backbone(
            input_ids=ids.input_ids,
            attention_mask=ids.attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        h = out.hidden_states[self.spec.extraction_layer][0].float().detach()
        toks = [self.tokenizer.decode([t]) for t in ids.input_ids[0].tolist()]
        return toks, h

    # ------------------------------------------------------------------
    # Verbalisation + scoring
    # ------------------------------------------------------------------

    @torch.no_grad()
    def verbalize(
        self,
        v: torch.Tensor,
        K: int = 8,
        temperature: float = 1.0,
        max_new_tokens: int = 64,
    ) -> list[tuple[str, float, float]]:
        """Sample K candidate verbalisations, round-trip score each, return
        list of (text, raw_fve, centred_fve) sorted by centred_fve desc.

        If centring pool is unavailable, centred_fve == raw_fve."""
        v = v.to(self.device).float()
        v_rep = v.unsqueeze(0).expand(K, -1)
        ids = self.av.generate(
            v_rep,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-6),
            top_p=1.0,
        )
        texts = self.tokenizer.batch_decode(ids, skip_special_tokens=True)

        # Round-trip: feed each text into the same backbone, extract layer-L
        # last-token hidden, score against v.
        scored: list[tuple[str, float, float]] = []
        for txt in texts:
            if not txt.strip():
                scored.append((txt, 0.0, 0.0))
                continue
            h = self.extract_hidden(txt, token_index=-1)
            raw = 0.5 * (1.0 + F.cosine_similarity(h.unsqueeze(0), v.unsqueeze(0)).item())
            if self.mu is not None:
                hc, vc = h - self.mu, v - self.mu
                cen = 0.5 * (1.0 + F.cosine_similarity(hc.unsqueeze(0), vc.unsqueeze(0)).item())
            else:
                cen = raw
            scored.append((txt, raw, cen))
        scored.sort(key=lambda r: r[2], reverse=True)
        return scored

    # ------------------------------------------------------------------
    # Steering by edited verbalisation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def steer_with_text(
        self,
        prompt: str,
        replacement_text: str,
        alpha: float = 1.0,
        max_new_tokens: int = 64,
    ) -> str:
        """Re-encode `replacement_text` to a layer-L hidden vector v_new,
        then run greedy generation on `prompt` with a forward hook that
        adds `alpha * (v_new - v_orig)` to layer-L's output at every
        position. Returns the generated continuation only.
        """
        v_orig = self.extract_hidden(prompt, token_index=-1)
        v_new = self.extract_hidden(replacement_text, token_index=-1)
        delta = (v_new - v_orig) * alpha

        # Find the residual stream block at extraction_layer.
        # Both LLaMA-like and Gemma-like architectures expose this as
        # backbone.model.layers[L].
        try:
            target_block = self.backbone.model.layers[self.spec.extraction_layer]
        except AttributeError:
            target_block = self.backbone.transformer.h[self.spec.extraction_layer]

        delta_b = delta.to(self.dtype)

        def _hook(_mod, _inp, out):
            if isinstance(out, tuple):
                h = out[0]
                h = h + delta_b
                return (h,) + out[1:]
            return out + delta_b

        handle = target_block.register_forward_hook(_hook)
        try:
            ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            gen = self.backbone.generate(
                **ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        finally:
            handle.remove()

        new_ids = gen[0, ids.input_ids.shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Live thought-trace
    # ------------------------------------------------------------------

    @torch.no_grad()
    def thought_trace(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        every: int = 4,
        K: int = 4,
    ) -> Iterable[tuple[str, str | None]]:
        """Greedy-generate `max_new_tokens` tokens after `prompt`. Every
        `every` tokens, run the AV on the running last-token hidden state
        and yield (token_text, verbalisation) pairs. Other steps yield
        (token_text, None)."""
        ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        cur = ids.input_ids
        attn = ids.attention_mask
        for step in range(max_new_tokens):
            out = self.backbone(
                input_ids=cur,
                attention_mask=attn,
                output_hidden_states=True,
                use_cache=False,
            )
            next_id = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            tok_text = self.tokenizer.decode(next_id[0])
            cur = torch.cat([cur, next_id], dim=1)
            attn = torch.cat([attn, torch.ones_like(next_id)], dim=1)
            if step % every == 0:
                v = out.hidden_states[self.spec.extraction_layer][0, -1].float().detach()
                ranked = self.verbalize(v, K=K, temperature=1.0, max_new_tokens=24)
                yield tok_text, ranked[0][0] if ranked else None
            else:
                yield tok_text, None
            if next_id.item() == (self.tokenizer.eos_token_id or -1):
                break
