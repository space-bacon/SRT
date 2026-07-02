"""Unit tests for ``srt.nla.trace_data`` — multi-position/-layer pair builder.

Pure torch; no backbone needed.
"""

from __future__ import annotations

import torch

from srt.nla.trace_data import (
    build_trace_pairs,
    normalize_layer_activations,
    resolve_saved_layers,
)


def _fake_obj():
    # Two sequences, lengths 3 and 2, hidden dim d=4, two saved layers (10, 20).
    d = 4
    token_ids = [torch.tensor([5, 6, 7]), torch.tensor([8, 9])]
    acts = {
        10: [torch.arange(3 * d).float().view(3, d), torch.arange(2 * d).float().view(2, d)],
        20: [torch.ones(3, d) * 2, torch.ones(2, d) * 3],
    }
    return {
        "token_ids": token_ids,
        "activations_by_layer": acts,
        "meta": {"extraction_layer": 20, "d": d},
    }


def test_resolve_and_normalize():
    obj = _fake_obj()
    assert resolve_saved_layers(obj) == [10, 20]
    a10 = normalize_layer_activations(obj, 10)
    assert len(a10) == 2 and a10[0].shape == (3, 4)


def test_build_pairs_all_layers_all_positions():
    obj = _fake_obj()
    targets, records = build_trace_pairs(obj)  # all layers, every position
    # seq0 has 3 positions, seq1 has 2 → 5 positions × 2 layers = 10 pairs.
    assert len(records) == 10
    assert targets.shape == (10, 4)
    # target_idx aligns with the targets tensor.
    for r in records:
        assert 0 <= r["target_idx"] < targets.shape[0]
    # gold prefix length equals pos+1 and is a strict prefix of the sequence.
    seqs = [t.tolist() for t in obj["token_ids"]]
    for r in records:
        assert r["gold_ids"] == seqs[r["seq_idx"]][: r["pos"] + 1]
        assert r["n_tokens"] == r["pos"] + 1
    # the layer-20 target for seq0 pos0 must be exactly [2,2,2,2]
    match = [
        r for r in records if r["layer"] == 20 and r["seq_idx"] == 0 and r["pos"] == 0
    ]
    assert match
    v = targets[match[0]["target_idx"]]
    assert torch.allclose(v, torch.ones(4) * 2)


def test_build_pairs_single_layer_and_stride_and_bounds():
    obj = _fake_obj()
    targets, records = build_trace_pairs(
        obj, layers=[20], position_stride=2, min_prefix_len=1, max_prefix_len=3
    )
    # seq0 positions {0,2}; seq1 positions {0} → 3 pairs, layer 20 only.
    assert all(r["layer"] == 20 for r in records)
    assert len(records) == 3
    assert targets.shape[0] == 3


def test_build_pairs_max_pairs_reindexes():
    obj = _fake_obj()
    targets, records = build_trace_pairs(obj, max_pairs=4, seed=1)
    assert len(records) == 4
    assert targets.shape[0] == 4
    assert sorted(r["target_idx"] for r in records) == [0, 1, 2, 3]


def test_build_pairs_legacy_single_layer_format():
    d = 4
    obj = {
        "token_ids": [torch.tensor([1, 2])],
        "activations": [torch.arange(2 * d).float().view(2, d)],
        "meta": {"extraction_layer": 20, "d": d},
    }
    assert resolve_saved_layers(obj) == [20]
    targets, records = build_trace_pairs(obj)
    assert len(records) == 2 and targets.shape == (2, 4)
    assert all(r["layer"] == 20 for r in records)
