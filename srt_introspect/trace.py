"""`Trace` — the public sidecar wrapper.

Loads:
  - SRTAdapter (Stage 3) from a HF repo: provides per-token divergence /
    regime / r_hat / logits.
  - ActivationVerbalizer (Stage 4) from a HF repo: turns an L20 hidden state
    into natural-language descriptions.

Generates one token at a time so we can record the SRT signals as the
sequence grows, then runs the adaptive-density scheduler post-hoc to pick
which token sites get verbalized.

This is read-only at inference: we don't modify the host model's weights.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors
from transformers import AutoTokenizer

from srt.adapter import SRTAdapter
from srt.config import SRTConfig
from srt.nla import ActivationVerbalizer, NLAConfig

from .scheduler import quantile_by_density
from .types import Result, Step


DEFAULT_ADAPTER_REPO = "RiverRider/srt-adapter-v1.0"
DEFAULT_AV_REPO = "RiverRider/srt-nla-av-v1"


@dataclass
class _SignalsAtStep:
    divergence: float
    regime: int
    r_hat: float


def _pick_device(arg: str | None) -> str:
    if arg is not None:
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Trace:
    """Adaptive-density reasoning-trace sidecar."""

    def __init__(
        self,
        adapter: SRTAdapter,
        av: ActivationVerbalizer,
        tokenizer,
        device: str,
        av_extract_layer: int,
    ) -> None:
        self.adapter = adapter
        self.av = av
        self.tok = tokenizer
        self.device = device
        self.av_extract_layer = av_extract_layer

    # ------------------------------------------------------------------ load
    @classmethod
    def load(
        cls,
        backbone_id: str = "Qwen/Qwen2.5-7B",  # noqa: ARG003 — informational; pinned in adapter cfg
        *,
        adapter_repo: str = DEFAULT_ADAPTER_REPO,
        av_repo: str = DEFAULT_AV_REPO,
        device: str | None = None,
    ) -> "Trace":
        device = _pick_device(device)

        # Stage 3 adapter
        cfg_path = hf_hub_download(adapter_repo, "config.json")
        w_path = hf_hub_download(adapter_repo, "adapter.safetensors")
        adapter_cfg = SRTConfig.from_json(cfg_path)
        adapter = SRTAdapter(adapter_cfg)
        adapter.load_state_dict(load_safetensors(w_path, device="cpu"), strict=False)
        adapter = adapter.to(device).eval()
        for p in adapter.parameters():
            p.requires_grad_(False)

        # Stage 4 AV (separate backbone instance — same weights, ~14GB extra in bf16)
        av_cfg_path = hf_hub_download(av_repo, "config.json")
        av_ckpt_path = hf_hub_download(av_repo, "best_av.pt")
        av_cfg = NLAConfig.from_json(av_cfg_path)
        av = ActivationVerbalizer(av_cfg)
        sd = torch.load(av_ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict):
            for wkey in ("trainable", "state_dict"):
                if wkey in sd and isinstance(sd[wkey], dict):
                    sd = sd[wkey]
                    break
        av.load_state_dict(sd, strict=False)
        av = av.to(device).eval()
        for p in av.parameters():
            p.requires_grad_(False)

        tok = AutoTokenizer.from_pretrained(adapter_cfg.backbone_id)
        return cls(
            adapter=adapter,
            av=av,
            tokenizer=tok,
            device=device,
            av_extract_layer=av_cfg.extraction_layer,
        )

    # -------------------------------------------------------------- generate
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 200,
        budget: int = 12,
        k: int = 8,
        temperature: float = 0.7,
        top_p: float = 0.95,
        repetition_penalty: float = 1.15,
        verbalize_max_new_tokens: int = 32,
    ) -> Result:
        """Generate a continuation and an adaptive-density reasoning trace.

        Args:
            prompt: input text.
            max_new_tokens: cap on continuation length.
            budget: number of verbalization slots (adaptive density target).
            k: number of verbalization samples per slot (consensus signal).
            temperature/top_p: generation sampling.
            repetition_penalty: >1.0 suppresses already-generated tokens to
                discourage degenerate loops on base (non-instruct) backbones.
            verbalize_max_new_tokens: AV decoder length for each slot.
        """
        device = self.device
        enc = self.tok(prompt, return_tensors="pt").to(device)
        input_ids = enc.input_ids
        attn = enc.attention_mask
        prompt_len = input_ids.shape[1]

        per_token: list[_SignalsAtStep] = []
        generated_tokens: list[int] = []
        eos = self.tok.eos_token_id

        for _ in range(max_new_tokens):
            out = self.adapter(input_ids=input_ids, attention_mask=attn)
            logits = out.logits[:, -1, :].float()
            # repetition penalty over the whole generated context
            if repetition_penalty != 1.0:
                seen = input_ids[0].tolist()
                _apply_repetition_penalty(logits, seen, repetition_penalty)
            # sample
            next_id = _sample(logits, temperature=temperature, top_p=top_p)

            # record signals at the just-produced position (the *last* input
            # token before we appended — i.e. the position whose hidden state
            # produced these logits)
            last_pos = input_ids.shape[1] - 1
            div_norm = _div_norm_at(out.divergences, last_pos)
            regime = int(out.ben_output.regime_logits[0, last_pos].argmax().item())
            r_hat = float(out.ben_output.r_hat[0, last_pos].item())
            per_token.append(_SignalsAtStep(divergence=div_norm, regime=regime, r_hat=r_hat))

            generated_tokens.append(int(next_id))
            if eos is not None and int(next_id) == eos:
                break
            input_ids = torch.cat([input_ids, next_id.view(1, 1)], dim=1)
            attn = torch.cat([attn, torch.ones((1, 1), dtype=attn.dtype, device=device)], dim=1)

        text = self.tok.decode(generated_tokens, skip_special_tokens=True)

        # Build Step list (one per generated token). Note: signals[i] are the
        # divergences at the position whose logits *produced* generated_tokens[i],
        # which is the natural "what was the model thinking when it emitted
        # this?" alignment.
        steps: list[Step] = []
        for i, tid in enumerate(generated_tokens):
            sig = per_token[i] if i < len(per_token) else _SignalsAtStep(0.0, 0, 0.0)
            steps.append(
                Step(
                    token_idx=i,
                    token=self.tok.decode([tid], skip_special_tokens=False),
                    divergence=sig.divergence,
                    regime=sig.regime,
                    r_hat=sig.r_hat,
                )
            )

        # Adaptive-density scheduling over the divergence series, masking
        # tokens that decode to pure whitespace / empty — verbalizing those
        # is not informative for the UI.
        div_series = [s.divergence for s in steps]
        skip = {i for i, s in enumerate(steps) if not s.token.strip()}
        picked = quantile_by_density(div_series, budget=budget, skip_indices=skip)

        if picked:
            verbs = self._verbalize_positions(
                full_input_ids=input_ids,  # already grown to prompt + generated
                prompt_len=prompt_len,
                gen_token_indices=picked,
                k=k,
                max_new_tokens=verbalize_max_new_tokens,
            )
            for idx, v in zip(picked, verbs):
                steps[idx].verbalization = v

        return Result(text=text, steps=steps)

    # ---------------------------------------------------------- verbalize ops
    @torch.no_grad()
    def _verbalize_positions(
        self,
        *,
        full_input_ids: torch.Tensor,  # (1, prompt_len + gen_len)
        prompt_len: int,
        gen_token_indices: list[int],
        k: int,
        max_new_tokens: int,
    ) -> list[str]:
        """Extract L20 hidden states at the requested generated-token positions
        using the AV's own backbone (separate from the adapter's backbone), then
        run AV.verbalize on the batch and return one consensus string per
        position."""
        # AV.backbone is the frozen verbalizer-backbone copy; one forward over
        # the full sequence captures all L20 states we need.
        out = self.av.backbone(
            input_ids=full_input_ids.to(self.device),
            output_hidden_states=True,
            use_cache=False,
        )
        h_L = out.hidden_states[self.av_extract_layer][0]  # (T, d)

        # absolute positions: prompt_len + gen_idx maps to the position whose
        # hidden state was used to produce gen_token_indices[i]. Per the
        # standard causal-LM convention, the hidden state at position p
        # produces the token at position p+1. So to capture "what was the
        # model thinking when it produced gen_token[i]" we want hidden state
        # at absolute position (prompt_len - 1 + i).
        abs_positions = [prompt_len - 1 + g for g in gen_token_indices]

        out_strs: list[str] = []
        for abs_pos in abs_positions:
            v = h_L[abs_pos].to(torch.float32)
            v_batch = (
                v.to(self.av.proj.weight.dtype)
                .unsqueeze(0)
                .repeat(int(k), 1)
                .contiguous()
                .to(self.device)
            )
            texts = self.av.verbalize(
                v_batch,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
            )
            out_strs.append(_consensus(texts))
        return out_strs


# ----------------------------------------------------------------- helpers


def _div_norm_at(divergences: list[torch.Tensor] | None, pos: int) -> float:
    """Sum of L2 norms of MAH divergence vectors at token position `pos`."""
    if not divergences:
        return 0.0
    total = 0.0
    for d in divergences:
        # d: (1, T, d_div)
        v = d[0, pos].float()
        total += float(v.norm().item())
    return total


def _apply_repetition_penalty(logits: torch.Tensor, seen_ids: list[int], penalty: float) -> None:
    """In-place repetition penalty (HF-style): for already-seen tokens, divide
    positive logits and multiply negative logits by `penalty`."""
    if not seen_ids or penalty == 1.0:
        return
    ids = torch.tensor(list(set(seen_ids)), device=logits.device, dtype=torch.long)
    scores = logits[0, ids]
    scores = torch.where(scores > 0, scores / penalty, scores * penalty)
    logits[0, ids] = scores


def _sample(logits: torch.Tensor, *, temperature: float, top_p: float) -> torch.Tensor:
    """Standard nucleus sampling. logits: (1, V) -> token id tensor shape (1,)."""
    if temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cum = sorted_probs.cumsum(dim=-1)
    mask = cum - sorted_probs > top_p
    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    choice = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_idx.gather(-1, choice).squeeze(0)


def _consensus(samples: Iterable[str]) -> str:
    """Pick the longest non-empty sample after stripping. Simple consensus —
    can be replaced with embedding-similarity centroid later."""
    cleaned = [s.strip() for s in samples if s and s.strip()]
    if not cleaned:
        return ""
    cleaned.sort(key=len, reverse=True)
    return cleaned[0]
