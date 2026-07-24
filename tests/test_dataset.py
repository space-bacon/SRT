"""Unit tests for srt/data/dataset.py.

Pure-CPU, no model download. A fake BPE-ish tokenizer provides offset
mappings so the label-alignment and community-id logic (source of the
v3-v5 SupCon collapse) is exercised without HF downloads.
"""

from __future__ import annotations

import json

import torch

from srt.data.dataset import (
    SRTAdapterDataset,
    _align_word_labels_to_bpe,
    _stable_hash,
    make_collate_fn,
)


# ------------------------------------------------------------------ helpers
class FakeTokenizer:
    """Whitespace tokenizer that mimics the HF fast-tokenizer call contract
    used by SRTAdapterDataset (return_tensors='pt', offset mapping)."""

    def __call__(self, text, truncation=True, max_length=512,
                 return_tensors="pt", return_offsets_mapping=True):
        ids, offsets = [], []
        pos = 0
        for w in text.split():
            start = text.find(w, pos)
            end = start + len(w)
            ids.append(len(ids) + 10)  # arbitrary distinct ids
            offsets.append((start, end))
            pos = end
        ids = ids[:max_length]
        offsets = offsets[:max_length]
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
            "offset_mapping": torch.tensor([offsets]),
        }


def _write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# -------------------------------------------------------------- _stable_hash
class TestStableHash:
    def test_deterministic(self):
        assert _stable_hash("askphilosophy") == _stable_hash("askphilosophy")

    def test_distinct_strings_differ(self):
        assert _stable_hash("a") != _stable_hash("b")

    def test_in_range(self):
        for s in ("", "x", "community_42"):
            assert 0 <= _stable_hash(s) < 100003


# ------------------------------------------------- _align_word_labels_to_bpe
class TestAlignWordLabels:
    def test_basic_alignment(self):
        text = "hello world"
        words = text.split()
        offsets = torch.tensor([[0, 5], [6, 11]])
        labels, mask = _align_word_labels_to_bpe(words, [1.0, 2.0], offsets, text)
        assert mask.all()
        assert labels.tolist() == [1.0, 2.0]

    def test_multi_token_word_shares_label(self):
        text = "unbelievable"
        offsets = torch.tensor([[0, 4], [4, 12]])  # two BPE pieces, one word
        labels, mask = _align_word_labels_to_bpe(["unbelievable"], [3.0], offsets, text)
        assert mask.all()
        assert labels.tolist() == [3.0, 3.0]

    def test_length_mismatch_returns_all_false(self):
        text = "hello world"
        offsets = torch.tensor([[0, 5], [6, 11]])
        labels, mask = _align_word_labels_to_bpe(["hello"], [1.0, 2.0], offsets, text)
        assert not mask.any()
        assert labels.sum().item() == 0.0

    def test_special_token_zero_offset_skipped(self):
        text = "hi"
        offsets = torch.tensor([[0, 0], [0, 2]])  # first is a special token
        labels, mask = _align_word_labels_to_bpe(["hi"], [5.0], offsets, text)
        assert mask.tolist() == [False, True]


# -------------------------------------------------------------- dataset core
class TestDatasetCommunityId:
    def test_explicit_community_id_wins(self, tmp_path):
        p = tmp_path / "d.jsonl"
        _write_jsonl(p, [{"text": "a b", "community_id": 7, "community_label": "x"}])
        ds = SRTAdapterDataset(p, FakeTokenizer())
        assert ds[0]["community_id"].item() == 7
        assert ds.n_empty_community_fallback == 0

    def test_label_fallback_used_when_id_missing(self, tmp_path):
        p = tmp_path / "d.jsonl"
        _write_jsonl(p, [{"text": "a b", "community_label": "askphilosophy"}])
        ds = SRTAdapterDataset(p, FakeTokenizer())
        assert ds[0]["community_id"].item() == _stable_hash("askphilosophy")
        assert ds.n_empty_community_fallback == 0

    def test_empty_fallback_is_counted(self, tmp_path):
        """The v3-v5 bug class: rows with no usable community field all hash
        to one id. The dataset must count these loudly."""
        p = tmp_path / "d.jsonl"
        _write_jsonl(p, [
            {"text": "a b"},
            {"text": "c d", "community_label": ""},
        ])
        ds = SRTAdapterDataset(p, FakeTokenizer())
        id0 = ds[0]["community_id"].item()
        id1 = ds[1]["community_id"].item()
        assert id0 == id1 == _stable_hash("")  # degenerate by construction
        assert ds.n_empty_community_fallback == 2

    def test_alignment_failure_is_counted(self, tmp_path):
        p = tmp_path / "d.jsonl"
        # 2 words but 3 labels → alignment returns all-False mask
        _write_jsonl(p, [{"text": "a b", "r_true": [1.0, 2.0, 3.0], "community_id": 0}])
        ds = SRTAdapterDataset(p, FakeTokenizer())
        item = ds[0]
        assert not item["r_mask"].any()
        assert ds.n_r_true_alignment_failures == 1

    def test_archetype_id_default_minus_one(self, tmp_path):
        p = tmp_path / "d.jsonl"
        _write_jsonl(p, [{"text": "a b", "community_id": 0}])
        ds = SRTAdapterDataset(p, FakeTokenizer())
        assert ds[0]["archetype_id"].item() == -1


# ------------------------------------------------------------------- collate
class TestCollate:
    def _item(self, T, community=0):
        return {
            "input_ids": torch.arange(T),
            "attention_mask": torch.ones(T, dtype=torch.long),
            "labels": torch.arange(T),
            "r_true": torch.ones(T),
            "r_mask": torch.ones(T, dtype=torch.bool),
            "community_id": torch.tensor(community),
            "archetype_id": torch.tensor(-1),
        }

    def test_pads_to_longest(self):
        collate = make_collate_fn(pad_token_id=99)
        batch = collate([self._item(2), self._item(4)])
        assert batch["input_ids"].shape == (2, 4)
        assert batch["input_ids"][0, 2:].tolist() == [99, 99]
        assert batch["attention_mask"][0, 2:].tolist() == [0, 0]
        assert batch["labels"][0, 2:].tolist() == [-100, -100]
        assert batch["r_mask"][0, 2:].tolist() == [False, False]

    def test_scalar_fields_stacked(self):
        collate = make_collate_fn(pad_token_id=0)
        batch = collate([self._item(2, community=1), self._item(2, community=5)])
        assert batch["community_id"].tolist() == [1, 5]
        assert batch["archetype_id"].tolist() == [-1, -1]
