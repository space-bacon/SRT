"""Build multi-position, multi-layer ``(v, prefix)`` training pairs.

The base NLA training pair is ``(v = last-token hidden of a full sequence at
one layer, gold text = that whole sequence)``. That teaches the AV to
verbalize only the *final* state of a sequence at a *single* layer.

To train an AV that verbalizes the whole **input → output** loop at **every
layer**, we emit one pair per ``(sequence, position, layer)`` instead::

    v         = hidden state at layer L, token position t
    gold_ids  = the prefix tokens [0 .. t] that produced it

Because the backbone is causal, the hidden state at position ``t`` in a full
forward pass depends only on tokens ``0..t``. It is therefore *exactly* the
last-token hidden state you would get by running the frozen backbone on the
prefix ``ids[:t+1]`` alone. Every prefix is thus an exact gold verbalization
of ``v`` at every layer simultaneously — no re-tokenization drift, and both
short (input-like) and long (output-like) positions are covered.

This module is pure ``torch`` / Python: it consumes a ``sample_targets``
object in memory and returns ``(targets, records)``, so it is unit-testable
without loading a backbone.
"""

from __future__ import annotations

from typing import Sequence

import torch

__all__ = ["normalize_layer_activations", "resolve_saved_layers", "build_trace_pairs"]


def resolve_saved_layers(obj: dict) -> list[int]:
    """Return the sorted list of layers present in a sample_targets object.

    Supports the multi-layer format (``activations_by_layer``) and the legacy
    single-layer format (``activations`` + ``meta.extraction_layer``).
    """
    abl = obj.get("activations_by_layer")
    if isinstance(abl, dict) and abl:
        return sorted(int(k) for k in abl.keys())
    meta = obj.get("meta", {})
    layer = meta.get("extraction_layer")
    if layer is None:
        raise KeyError(
            "sample object has neither 'activations_by_layer' nor "
            "meta.extraction_layer; cannot resolve layers"
        )
    return [int(layer)]


def normalize_layer_activations(obj: dict, layer: int) -> list[torch.Tensor]:
    """Return the per-sequence ``(T_i, d)`` activation list for one layer."""
    abl = obj.get("activations_by_layer")
    if isinstance(abl, dict) and abl:
        if layer in abl:
            return abl[layer]
        if str(layer) in abl:  # tolerate JSON-stringified keys
            return abl[str(layer)]
        raise KeyError(f"layer {layer} not saved; available: {sorted(abl)}")
    # legacy single-layer format
    meta = obj.get("meta", {})
    if int(meta.get("extraction_layer", -1)) != layer:
        raise KeyError(
            f"legacy sample object only holds layer "
            f"{meta.get('extraction_layer')}, not {layer}"
        )
    return obj["activations"]


def _token_ids(obj: dict) -> list[list[int]]:
    """Return per-sequence token-id lists, preferring exact saved ids."""
    if "token_ids" in obj and obj["token_ids"] is not None:
        out = []
        for t in obj["token_ids"]:
            out.append(t.tolist() if isinstance(t, torch.Tensor) else list(t))
        return out
    raise KeyError(
        "sample object has no 'token_ids'; re-run sample_targets.py with the "
        "updated script so exact prefix ids are saved (decoded 'sequences' "
        "cannot be sliced per position without re-tokenization drift)"
    )


def build_trace_pairs(
    obj: dict,
    *,
    layers: Sequence[int] | str | None = None,
    position_stride: int = 1,
    min_prefix_len: int = 1,
    max_prefix_len: int | None = None,
    max_sequences: int | None = None,
    max_pairs: int | None = None,
    seed: int = 0,
) -> tuple[torch.Tensor, list[dict]]:
    """Flatten a sample_targets object into ``(targets, records)``.

    Parameters
    ----------
    obj:
        A dict loaded from a ``sample_targets.py`` ``.pt`` file. Must contain
        ``token_ids`` and either ``activations_by_layer`` (multi-layer) or the
        legacy ``activations`` + ``meta.extraction_layer``.
    layers:
        ``None`` -> every saved layer; ``"all"`` -> same; or an explicit list.
    position_stride:
        Keep every ``stride``-th token position within each sequence.
    min_prefix_len / max_prefix_len:
        Bound the prefix length (``= position + 1``). Short prefixes are
        input-like; long prefixes are output-like.
    max_sequences:
        Cap how many sequences are used.
    max_pairs:
        Randomly subsample to at most this many pairs (seeded, ``target_idx``
        re-indexed to stay aligned with the returned ``targets`` tensor).

    Returns
    -------
    targets:
        ``(N, d)`` float32 CPU tensor. ``targets[record["target_idx"]]`` is the
        vector for that record.
    records:
        List of ``{"target_idx", "gold_ids", "seq_idx", "pos", "layer",
        "n_tokens"}`` dicts.
    """
    saved = resolve_saved_layers(obj)
    if layers is None or layers == "all":
        layer_list = saved
    elif isinstance(layers, int):
        layer_list = [layers]
    else:
        layer_list = sorted({int(L) for L in layers})
    missing = [L for L in layer_list if L not in saved]
    if missing:
        raise KeyError(f"requested layers {missing} not in saved layers {saved}")

    ids_all = _token_ids(obj)
    acts_by_layer = {L: normalize_layer_activations(obj, L) for L in layer_list}

    n_seq = len(ids_all)
    if max_sequences is not None:
        n_seq = min(n_seq, max_sequences)
    stride = max(1, position_stride)

    targets: list[torch.Tensor] = []
    records: list[dict] = []
    for seq_idx in range(n_seq):
        ids = ids_all[seq_idx]
        T = len(ids)
        for t in range(0, T, stride):
            plen = t + 1
            if plen < min_prefix_len:
                continue
            if max_prefix_len is not None and plen > max_prefix_len:
                break
            gold = ids[: t + 1]
            for L in layer_list:
                acts_L = acts_by_layer[L]
                if seq_idx >= len(acts_L) or t >= acts_L[seq_idx].shape[0]:
                    continue
                records.append(
                    {
                        "target_idx": len(targets),
                        "gold_ids": gold,
                        "seq_idx": seq_idx,
                        "pos": t,
                        "layer": L,
                        "n_tokens": plen,
                    }
                )
                targets.append(acts_L[seq_idx][t].float())

    if not targets:
        return torch.empty(0), []

    if max_pairs is not None and len(records) > max_pairs:
        g = torch.Generator().manual_seed(seed)
        keep = torch.randperm(len(records), generator=g)[:max_pairs].tolist()
        keep.sort()
        new_targets = [targets[i] for i in keep]
        new_records = []
        for new_idx, old_idx in enumerate(keep):
            r = dict(records[old_idx])
            r["target_idx"] = new_idx
            new_records.append(r)
        targets, records = new_targets, new_records

    return torch.stack(targets), records
