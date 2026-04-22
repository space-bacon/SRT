"""Dataset for SRT Adapter training.

Loads JSONL files produced by the Reddit corpus pipeline and aligns word-level
labels (r_true) to BPE token boundaries using the backbone's own tokenizer.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class SRTAdapterDataset(Dataset):
    """Load JSONL samples and tokenize with a BPE tokenizer.

    Expected JSONL format (all fields except ``text`` are optional):
        {"text": "...", "r_true": [0.1, -0.2, ...], "community": "some_label"}
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int = 512,
        max_samples: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples: list[dict[str, Any]] = []

        path = Path(path)
        logger.info("Loading dataset from %s", path)
        with open(path) as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                row = json.loads(line)
                if "text" in row and row["text"].strip():
                    self.samples.append(row)

        logger.info("Loaded %d samples from %s", len(self.samples), path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.samples[idx]
        text = row["text"]

        # Tokenize
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        input_ids = enc["input_ids"].squeeze(0)  # (T,)
        attention_mask = enc["attention_mask"].squeeze(0)  # (T,)
        offsets = enc["offset_mapping"].squeeze(0)  # (T, 2)
        T = input_ids.size(0)

        # Align word-level r_true → token-level
        r_true = torch.zeros(T)
        r_mask = torch.zeros(T, dtype=torch.bool)

        if "r_true" in row and row["r_true"]:
            word_r = row["r_true"]
            words = text.split()
            token_r, token_mask = _align_word_labels_to_bpe(
                words, word_r, offsets, text
            )
            r_true[:len(token_r)] = token_r[:T]
            r_mask[:len(token_mask)] = token_mask[:T]

        # Community id: stable int hash of the source-community string.
        # Used by the supervised contrastive loss to push prototypes apart.
        # Two samples from the same source share an id; samples from different
        # sources almost certainly differ (mod a large prime to avoid collisions).
        community_str = row.get("community", "") or ""
        community_id = _stable_hash(community_str)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),  # next-token prediction
            "r_true": r_true,
            "r_mask": r_mask,
            "community_id": torch.tensor(community_id, dtype=torch.long),
        }


def _stable_hash(s: str, modulus: int = 100003) -> int:
    """Stable non-cryptographic hash → int in [0, modulus). FNV-1a 32-bit.

    We use FNV-1a rather than the built-in `hash()` because the latter is
    randomized per-process under PYTHONHASHSEED=random, which would give
    different ids each run.
    """
    h = 2166136261
    for ch in s.encode("utf-8"):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    return h % modulus


def _align_word_labels_to_bpe(
    words: list[str],
    word_labels: list[float],
    offsets: torch.Tensor,
    text: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align word-level labels to BPE token positions.

    For each BPE token, find which word it belongs to (by character offset)
    and assign that word's label.

    Args:
        words: list of whitespace-split words.
        word_labels: per-word label values.
        offsets: (T, 2) character offsets from tokenizer.
        text: original text string.

    Returns:
        (token_labels, token_mask) both of shape (T,).
    """
    T = offsets.size(0)
    token_labels = torch.zeros(T)
    token_mask = torch.zeros(T, dtype=torch.bool)

    if len(words) != len(word_labels):
        # Mismatched lengths — skip alignment
        return token_labels, token_mask

    # Build word → character span mapping
    word_spans: list[tuple[int, int]] = []
    pos = 0
    for word in words:
        start = text.find(word, pos)
        if start == -1:
            break
        end = start + len(word)
        word_spans.append((start, end))
        pos = end

    if len(word_spans) != len(words):
        return token_labels, token_mask

    # For each token, find its word
    for tok_idx in range(T):
        tok_start, tok_end = offsets[tok_idx].tolist()
        if tok_start == 0 and tok_end == 0:
            continue  # special token

        tok_mid = (tok_start + tok_end) / 2
        for word_idx, (ws, we) in enumerate(word_spans):
            if ws <= tok_mid < we:
                token_labels[tok_idx] = word_labels[word_idx]
                token_mask[tok_idx] = True
                break

    return token_labels, token_mask


def make_collate_fn(pad_token_id: int = 0):
    """Create a collate function with the correct pad token id."""

    def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Collate with dynamic padding to the longest sequence in the batch."""
        max_len = max(item["input_ids"].size(0) for item in batch)

        padded: dict[str, list[torch.Tensor]] = {key: [] for key in batch[0]}
        for item in batch:
            T = item["input_ids"].size(0)
            pad_len = max_len - T
            padded["input_ids"].append(F.pad(item["input_ids"], (0, pad_len), value=pad_token_id))
            padded["attention_mask"].append(
                F.pad(item["attention_mask"], (0, pad_len), value=0)
            )
            padded["labels"].append(F.pad(item["labels"], (0, pad_len), value=-100))
            padded["r_true"].append(F.pad(item["r_true"], (0, pad_len), value=0.0))
            padded["r_mask"].append(F.pad(item["r_mask"], (0, pad_len), value=False))
            # community_id is a scalar — stack without padding
            if "community_id" in item:
                padded.setdefault("community_id", []).append(item["community_id"])

        return {k: torch.stack(v) for k, v in padded.items()}

    return collate_fn
