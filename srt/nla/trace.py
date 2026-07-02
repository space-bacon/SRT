"""Full input→output activation trace over the frozen backbone.

The base NLA loop verbalizes a *single* hidden state (the last-valid-token
activation at one layer). That answers "what is this one vector about?" but
not the question this module targets: **how does the model get from an input
prompt to its answer?**

``NLATracer`` closes that gap. Given a prompt it:

1. lets the frozen backbone *generate its own continuation* (the output),
2. runs one forward pass over ``prompt + output`` and reads hidden states at
   every requested layer and token position (input tokens *and* generated
   output tokens),
3. verbalizes each of those hidden states with the trained
   ``ActivationVerbalizer`` (best-of-``K`` with round-trip scoring),

and returns a structured :class:`NLATrace`: a ``(layer, position)`` grid of
natural-language descriptions of what the residual stream holds at every
step of the input→output loop.

Caveats worth stating plainly:

- The AV is trained on a *single* extraction layer (``cfg.extraction_layer``,
  e.g. L20 on Qwen). Verbalizing activations from *other* layers is
  out-of-distribution for the adapter; the text is still produced and
  round-trip-scored honestly, but off-layer fidelity is expected to be lower.
  Read the round-trip ``fve_centered`` per cell before trusting a cell.
- Anisotropy centring (``mu``) is layer-specific. Pass a per-layer ``mu`` for
  meaningful centred scores on non-extraction layers; a single ``mu`` is only
  strictly valid at the layer it was computed on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn.functional as F

from srt.nla.state_index import StateIndex
from srt.nla.verbalizer import ActivationVerbalizer

logger = logging.getLogger(__name__)


@dataclass
class TraceCell:
    """One verbalized hidden state at a single ``(layer, position)``."""

    layer: int
    position: int
    token: str
    is_output: bool
    text: str
    fve: float
    fve_centered: float
    code: int | None = None
    candidates: list[tuple[str, float, float]] | None = None

    def to_dict(self) -> dict:
        d = {
            "layer": self.layer,
            "position": self.position,
            "token": self.token,
            "is_output": self.is_output,
            "text": self.text,
            "fve": self.fve,
            "fve_centered": self.fve_centered,
            "code": self.code,
        }
        if self.candidates is not None:
            d["candidates"] = [
                {"text": t, "fve": r, "fve_centered": c} for t, r, c in self.candidates
            ]
        return d


@dataclass
class NLATrace:
    """A full ``(layer, position)`` verbalization grid for one prompt."""

    prompt: str
    generated: str
    tokens: list[str]
    n_input_tokens: int
    layers: list[int]
    positions: list[int]
    cells: list[TraceCell]
    backbone_id: str
    extraction_layer: int
    _index: dict[tuple[int, int], TraceCell] = field(
        default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        self._index = {(c.layer, c.position): c for c in self.cells}

    def cell(self, layer: int, position: int) -> TraceCell | None:
        """Return the cell at ``(layer, position)`` or ``None``."""
        return self._index.get((layer, position))

    def grid(self) -> dict[int, dict[int, TraceCell]]:
        """Return ``{layer: {position: cell}}`` for row/column iteration."""
        out: dict[int, dict[int, TraceCell]] = {L: {} for L in self.layers}
        for c in self.cells:
            out.setdefault(c.layer, {})[c.position] = c
        return out

    def code_groups(self) -> dict[int, list[TraceCell]]:
        """Group cells by magic-number ``code`` (recurring internal states).

        Only populated when the trace was produced with a ``state_index``.
        ``{code: [cells sharing it]}``; buckets with >1 cell are states the
        model re-visits across positions/layers.
        """
        out: dict[int, list[TraceCell]] = {}
        for c in self.cells:
            if c.code is not None:
                out.setdefault(c.code, []).append(c)
        return out

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "generated": self.generated,
            "tokens": self.tokens,
            "n_input_tokens": self.n_input_tokens,
            "layers": self.layers,
            "positions": self.positions,
            "backbone_id": self.backbone_id,
            "extraction_layer": self.extraction_layer,
            "cells": [c.to_dict() for c in self.cells],
        }


def _resolve_layers(
    layers: int | Sequence[int] | str | None, extraction_layer: int, num_layers: int
) -> list[int]:
    if layers is None:
        return [extraction_layer]
    if isinstance(layers, str):
        if layers == "all":
            return list(range(1, num_layers + 1))
        raise ValueError(f"unknown layers spec {layers!r}")
    if isinstance(layers, int):
        layers = [layers]
    out = sorted({int(L) for L in layers})
    for L in out:
        if not (0 <= L <= num_layers):
            raise ValueError(f"layer {L} out of range [0, {num_layers}]")
    return out


def _resolve_positions(
    positions: Sequence[int] | str | None, n_tokens: int, stride: int
) -> list[int]:
    if positions is None or positions == "all":
        return list(range(0, n_tokens, max(1, stride)))
    out = []
    for p in positions:
        p = int(p)
        if p < 0:
            p += n_tokens
        if not (0 <= p < n_tokens):
            raise ValueError(f"position {p} out of range [0, {n_tokens})")
        out.append(p)
    return sorted(set(out))


class NLATracer:
    """Verbalize the full input→output residual stream of a frozen backbone.

    Wraps a trained :class:`ActivationVerbalizer`. The backbone and tokenizer
    are shared with the AV, so this adds no extra GPU memory beyond the trace
    activations themselves.

    Parameters
    ----------
    av:
        A loaded :class:`ActivationVerbalizer` (carries the frozen backbone,
        tokenizer, and NLA config).
    mu:
        Anisotropy centring mean(s) for the centred round-trip score. Either
        a single ``(d,)`` tensor (applied to every layer), a mapping
        ``{layer: (d,)}`` for per-layer centring, or ``None`` to disable
        centring (``fve_centered == fve``).
    """

    def __init__(
        self,
        av: ActivationVerbalizer,
        *,
        mu: torch.Tensor | dict[int, torch.Tensor] | None = None,
    ) -> None:
        self.av = av
        self.backbone = av.backbone
        self.tokenizer = av.tokenizer
        self.cfg = av.cfg
        self.device = next(av.parameters()).device
        self._num_layers = self.backbone.config.num_hidden_layers
        self._d = self.backbone.config.hidden_size
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._mu = mu

    # ------------------------------------------------------------------
    def _mu_for(self, layer: int) -> torch.Tensor | None:
        if self._mu is None:
            return None
        if isinstance(self._mu, dict):
            m = self._mu.get(layer)
        else:
            m = self._mu
        return None if m is None else m.to(self.device).float()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def trace(
        self,
        prompt: str,
        *,
        layers: int | Sequence[int] | str | None = None,
        positions: Sequence[int] | str | None = None,
        position_stride: int = 1,
        K: int = 4,
        temperature: float = 1.0,
        verbalize_max_new_tokens: int | None = None,
        generate_output: bool = True,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        gen_batch_size: int = 16,
        keep_candidates: bool = False,
        state_index: StateIndex | None = None,
        register_codes: bool = True,
    ) -> NLATrace:
        """Produce a full ``(layer, position)`` verbalization trace.

        Parameters
        ----------
        prompt:
            The input text. Its tokens are the "input" positions.
        layers:
            ``None`` -> the trained extraction layer only; ``"all"`` -> every
            hidden layer (1..num_layers); or an int / list of layer indices.
        positions:
            ``None`` / ``"all"`` -> every token position (subject to
            ``position_stride``); or an explicit list (negative indices count
            from the end of the full sequence).
        position_stride:
            Keep every ``stride``-th position when ``positions`` is ``None``.
        K:
            Best-of-``K`` candidate verbalizations per cell.
        generate_output:
            If ``True``, let the backbone generate its own continuation first
            so the trace spans input *and* output tokens. If ``False``, only
            the prompt tokens are traced.
        max_new_tokens / do_sample:
            Controls the *output generation* step (not the verbalizer).
        verbalize_max_new_tokens:
            Max tokens for each verbalization (defaults to ``cfg.max_new_tokens``).
        gen_batch_size:
            Micro-batch size for verbalization + re-encoding.
        keep_candidates:
            Store all ``K`` scored candidates per cell (else only the best).
        state_index:
            Optional :class:`~srt.nla.state_index.StateIndex`. When given, each
            cell's hidden state is encoded to a magic-number ``code`` (set on
            the cell). If ``register_codes`` is also ``True``, each state is
            added to the index's codebook with the cell's best verbalization,
            building a reusable dictionary of internal states.
        """
        # ---- 1. build the full input→output token sequence ----------
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = enc.input_ids
        n_input = input_ids.shape[1]

        if generate_output:
            gen = self.backbone.generate(
                input_ids=input_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            full_ids = gen[0]
        else:
            full_ids = input_ids[0]

        n_tokens = full_ids.shape[0]
        generated_text = (
            self.tokenizer.decode(full_ids[n_input:], skip_special_tokens=True)
            if n_tokens > n_input
            else ""
        )
        tok_strings = [self.tokenizer.decode([t]) for t in full_ids.tolist()]

        # ---- 2. one forward pass → all hidden states ----------------
        out = self.backbone(
            input_ids=full_ids.unsqueeze(0),
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = out.hidden_states  # tuple len num_layers+1, each (1,T,d)

        layer_list = _resolve_layers(layers, self.cfg.extraction_layer, self._num_layers)
        pos_list = _resolve_positions(positions, n_tokens, position_stride)

        off_layer = [L for L in layer_list if L != self.cfg.extraction_layer]
        if off_layer:
            logger.warning(
                "Tracing layers %s that differ from the AV's trained "
                "extraction layer %d — off-layer verbalizations are "
                "out-of-distribution; trust the per-cell fve_centered.",
                off_layer,
                self.cfg.extraction_layer,
            )

        # ---- 3. collect one target vector per (layer, position) -----
        targets: list[torch.Tensor] = []
        cell_layer: list[int] = []
        cell_pos: list[int] = []
        for L in layer_list:
            hL = hidden_states[L][0]  # (T, d)
            for p in pos_list:
                targets.append(hL[p].float())
                cell_layer.append(L)
                cell_pos.append(p)

        if not targets:
            return NLATrace(
                prompt=prompt,
                generated=generated_text,
                tokens=tok_strings,
                n_input_tokens=n_input,
                layers=layer_list,
                positions=pos_list,
                cells=[],
                backbone_id=self.cfg.backbone_id,
                extraction_layer=self.cfg.extraction_layer,
            )

        target_mat = torch.stack(targets).to(self.device)  # (N, d)
        n_cells = target_mat.shape[0]

        # ---- 4. verbalize every target, best-of-K ------------------
        vmax = verbalize_max_new_tokens or self.cfg.max_new_tokens
        # repeat each target K times, contiguously: text i belongs to cell i//K
        v_expanded = target_mat.repeat_interleave(K, dim=0)  # (N*K, d)
        # per-expanded-row layer id, so a layer-conditioned AV knows the source
        # layer of each vector (ignored by AVs built without use_layer_embed).
        layer_ids = torch.tensor(cell_layer, dtype=torch.long).repeat_interleave(K)
        texts: list[str] = []
        for start in range(0, v_expanded.shape[0], gen_batch_size):
            chunk = v_expanded[start : start + gen_batch_size]
            ids = self.av.generate(
                chunk,
                max_new_tokens=vmax,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-6),
                top_p=1.0,
                layer=layer_ids[start : start + gen_batch_size].to(self.device),
            )
            texts.extend(self.tokenizer.batch_decode(ids, skip_special_tokens=True))

        # ---- 5. round-trip re-encode every candidate ---------------
        raw_scores = [0.0] * len(texts)
        cen_scores = [0.0] * len(texts)
        for start in range(0, len(texts), gen_batch_size):
            batch_texts = texts[start : start + gen_batch_size]
            idxs = list(range(start, start + len(batch_texts)))
            # blank verbalizations score 0 and are excluded from the forward
            live = [(j, t) for j, t in zip(idxs, batch_texts) if t.strip()]
            if not live:
                continue
            live_idx = [j for j, _ in live]
            live_txt = [t for _, t in live]
            re_enc = self.tokenizer(
                live_txt, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            re_out = self.backbone(
                input_ids=re_enc.input_ids,
                attention_mask=re_enc.attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            last_idx = (re_enc.attention_mask.sum(dim=1).long() - 1).clamp(min=0)
            row = torch.arange(len(live_idx), device=self.device)
            for b, j in enumerate(live_idx):
                L = cell_layer[j // K]
                h = re_out.hidden_states[L][b, last_idx[b]].float()
                v = target_mat[j // K]
                raw = 0.5 * (1.0 + F.cosine_similarity(h.unsqueeze(0), v.unsqueeze(0)).item())
                mu = self._mu_for(L)
                if mu is not None:
                    cen = 0.5 * (
                        1.0
                        + F.cosine_similarity(
                            (h - mu).unsqueeze(0), (v - mu).unsqueeze(0)
                        ).item()
                    )
                else:
                    cen = raw
                raw_scores[j] = raw
                cen_scores[j] = cen

        # ---- 6. pick best-of-K and assemble cells ------------------
        cells: list[TraceCell] = []
        for c in range(n_cells):
            lo = c * K
            hi = lo + K
            cand = [
                (texts[i], raw_scores[i], cen_scores[i]) for i in range(lo, hi)
            ]
            cand.sort(key=lambda r: r[2], reverse=True)
            best_text, best_raw, best_cen = cand[0]
            pos = cell_pos[c]
            cells.append(
                TraceCell(
                    layer=cell_layer[c],
                    position=pos,
                    token=tok_strings[pos],
                    is_output=pos >= n_input,
                    text=best_text,
                    fve=best_raw,
                    fve_centered=best_cen,
                    candidates=cand if keep_candidates else None,
                )
            )

        # ---- 7. optional magic-number state indexing ---------------
        if state_index is not None:
            codes = state_index.encode(target_mat)  # (N,) int64
            assert isinstance(codes, torch.Tensor)
            code_list = codes.tolist()
            for c in range(n_cells):
                cells[c].code = int(code_list[c])
            if register_codes:
                for c in range(n_cells):
                    state_index.add(
                        target_mat[c], text=cells[c].text, cen=cells[c].fve_centered
                    )

        return NLATrace(
            prompt=prompt,
            generated=generated_text,
            tokens=tok_strings,
            n_input_tokens=n_input,
            layers=layer_list,
            positions=pos_list,
            cells=cells,
            backbone_id=self.cfg.backbone_id,
            extraction_layer=self.cfg.extraction_layer,
        )
