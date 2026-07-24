"""Deployable decoding methods for the SRT-NLA verbalizer.

The headline result of `scripts/rerank_eval.py` is that **oracle-scored
best-of-K** closes the greedy → ceiling gap on every checkpoint we have:
greedy ρ_norm ≈ 0.26, K=64 oracle ρ_norm ≈ 0.92 against a paraphrase
ceiling of 1.00. Because the target activation ``v`` is, by construction,
available at deploy time in the SRT-NLA setup (the system was given ``v``
in order to verbalize it), oracle scoring is a *free* deploy-time signal,
not an unfair eval-time cheat.

This module packages that decoding method into a clean reusable function
so it can be called from train scripts, eval scripts, demos, or product
code. The cost is K-way sampling + one batched scoring forward pass.

Two scoring options:
- **centered** (recommended): ``0.5 * (1 + cos(h_last - mu, v - mu))``
  with ``mu`` the per-coordinate mean of a held-out pool of real
  backbone-L activations. Anisotropy-corrected.
- **raw**: ``0.5 * (1 + cos(h_last, v))``. Inflated by Qwen L20's
  cos≈0.24 anisotropy floor; reported only for back-compatibility with
  the pre-`centered_eval.py` literature.

ρ_norm conversion (the paper's units) is provided as a helper:
``rho_norm(cen) = (cen - random_floor) / (paraphrase_ceiling - random_floor)``
with the calibrated constants ``random_floor = 0.510`` and
``paraphrase_ceiling = 0.799``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

# Anchors + rho_norm live in srt.nla.metrics (single source of truth);
# re-exported here (see __all__) for backward compatibility.
from srt.nla.metrics import PARAPHRASE_CEILING_CEN, RANDOM_FLOOR_CEN, rho_norm
from srt.nla.verbalizer import ActivationVerbalizer


def _fve_from_cos(c: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + c)


def _cos_pairs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    an = F.normalize(a.float(), dim=-1)
    bn = F.normalize(b.float(), dim=-1)
    return (an * bn).sum(-1)


@torch.no_grad()
def _rollout(
    av: ActivationVerbalizer,
    v: torch.Tensor,
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    eos_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate (B, T) token ids and (B, T) attention mask truncated at first EOS.

    Convention note: the attention mask *includes* the first EOS position
    (``pos <= first_eos``), so ``_h_last_prefix_free`` reads the hidden
    state at the EOS token. This deliberately matches
    ``scripts/centered_eval.py`` (which produced the RANDOM_FLOOR_CEN /
    PARAPHRASE_CEILING_CEN anchors) — do not change one without
    recalibrating the other.
    """
    gen_ids = av.generate(
        v,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=max(temperature, 1e-6),
        top_p=1.0,
    )
    B, T = gen_ids.shape
    device = gen_ids.device
    is_eos = gen_ids == eos_id
    pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    first_eos = torch.where(
        is_eos.any(dim=-1, keepdim=True),
        is_eos.float().argmax(dim=-1, keepdim=True).long(),
        torch.full((B, 1), T - 1, device=device, dtype=torch.long),
    )
    gen_attn = (pos <= first_eos).long()
    return gen_ids, gen_attn


@torch.no_grad()
def _h_last_prefix_free(
    backbone,
    gen_ids: torch.Tensor,
    gen_attn: torch.Tensor,
    layer: int,
) -> torch.Tensor:
    """Read h_last from a *prefix-free* re-forward of the generated tokens.

    Critical: scoring must NOT include the proj(v) prefix in the forward
    used to extract h_last, because the prefix would leak v directly into
    the activation and inflate the metric. This matches centered_eval.py
    and the deployment-realistic measurement (text → embed → read activation).
    """
    out = backbone(
        input_ids=gen_ids,
        attention_mask=gen_attn,
        output_hidden_states=True,
        use_cache=False,
    )
    full_h = out.hidden_states[layer]
    B = gen_ids.size(0)
    last_idx = (gen_attn.sum(-1) - 1).clamp(min=0).long()
    rows = torch.arange(B, device=full_h.device)
    return full_h[rows, last_idx].float()


@dataclass
class OracleRerankResult:
    """Result of oracle-rerank decoding for a batch of B target vectors."""

    best_ids: torch.Tensor          # (B, T) chosen rollout token ids
    best_attn: torch.Tensor         # (B, T) attention mask
    best_h_last: torch.Tensor       # (B, d) chosen rollout h_last
    best_cen: torch.Tensor          # (B,) centered_fve scores of the chosen rollout
    best_idx: torch.Tensor          # (B,) which of the K candidates was chosen
    all_cen: torch.Tensor | None    # (B, K) scores for every candidate (if return_all)
    all_h_last: torch.Tensor | None # (B, K, d) (if return_all)
    all_ids: torch.Tensor | None    # (B, K, T) (if return_all)


@torch.no_grad()
def oracle_rerank_decode(
    av: ActivationVerbalizer,
    backbone,
    v: torch.Tensor,
    *,
    layer: int,
    K: int = 64,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    mu: torch.Tensor | None = None,
    score_batch_size: int = 64,
    eos_id: int | None = None,
    return_all: bool = False,
) -> OracleRerankResult:
    """Sample K rollouts per target, score with centered_fve, return the argmax.

    This is the recommended deploy-time decoding method for SRT-NLA when
    ``v`` is available (which it is, by construction).

    Args:
        av: an ``ActivationVerbalizer`` already loaded with weights and on
            the same device as ``v``.
        backbone: the frozen causal-LM backbone (same one wrapped by ``av``).
        v: target activation vectors, shape ``(B, d)``.
        layer: which hidden-state index to read for h_last. The AR/AV
            extraction layer (typically 20 for Qwen2.5-7B).
        K: number of samples per target.
        max_new_tokens: rollout length cap.
        temperature: sampling temperature (top-p is fixed to 1.0 here; the
            K-curve is dominated by K, not nucleus).
        mu: anisotropy mean ``(1, d)``. If ``None``, falls back to **raw**
            scoring (cos to v with no centering); strongly recommended to
            pass an actual pool-mean for anisotropy-correct comparison
            against published numbers.
        score_batch_size: cap concurrent (candidate, prefix) pairs in the
            scoring forward to bound memory.
        eos_id: token id used to truncate rollouts. Defaults to the AV
            tokenizer's eos.
        return_all: if True, also return per-candidate scores + h_last + ids.

    Returns:
        ``OracleRerankResult``. ``best_text`` is not auto-decoded — call
        ``av.tokenizer.batch_decode(result.best_ids, skip_special_tokens=True)``.
    """
    if v.dim() != 2:
        raise ValueError(f"v must be (B, d); got shape {tuple(v.shape)}")
    B, d = v.shape
    device = v.device
    if eos_id is None:
        eos_id = int(av.tokenizer.eos_token_id)

    # 1) K rollouts per target — flatten to (B*K, ...) for one big sampling call.
    v_rep = v.unsqueeze(1).expand(B, K, d).reshape(B * K, d)
    s_ids, s_attn = _rollout(
        av, v_rep,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        eos_id=eos_id,
    )

    # 2) Score in chunks to bound memory.
    h_chunks: list[torch.Tensor] = []
    for j in range(0, s_ids.size(0), score_batch_size):
        h = _h_last_prefix_free(
            backbone,
            s_ids[j : j + score_batch_size],
            s_attn[j : j + score_batch_size],
            layer,
        )
        h_chunks.append(h)
    h_all = torch.cat(h_chunks, dim=0)                              # (B*K, d)

    # 3) Score: centered if mu given, raw otherwise.
    if mu is not None:
        if mu.dim() == 1:
            mu = mu.unsqueeze(0)
        mu_dev = mu.to(device=device, dtype=torch.float32)
        cen = _fve_from_cos(_cos_pairs(h_all - mu_dev, v_rep - mu_dev))
    else:
        cen = _fve_from_cos(_cos_pairs(h_all, v_rep))

    cen = cen.view(B, K)
    h_BK = h_all.view(B, K, d)
    ids_BK = s_ids.view(B, K, -1)
    attn_BK = s_attn.view(B, K, -1)

    # 4) Argmax over K per target.
    best_idx = cen.argmax(dim=1)                                    # (B,)
    rows = torch.arange(B, device=device)
    best_cen = cen[rows, best_idx]
    best_h = h_BK[rows, best_idx]
    best_ids = ids_BK[rows, best_idx]
    best_attn = attn_BK[rows, best_idx]

    return OracleRerankResult(
        best_ids=best_ids,
        best_attn=best_attn,
        best_h_last=best_h,
        best_cen=best_cen,
        best_idx=best_idx,
        all_cen=cen if return_all else None,
        all_h_last=h_BK if return_all else None,
        all_ids=ids_BK if return_all else None,
    )


__all__ = [
    "RANDOM_FLOOR_CEN",
    "PARAPHRASE_CEILING_CEN",
    "rho_norm",
    "OracleRerankResult",
    "oracle_rerank_decode",
]
