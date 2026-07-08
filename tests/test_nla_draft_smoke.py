"""Smoke tests for draft-conditioned decoding (retrieval-then-edit).

Validates the two new pieces on a tiny backbone, CPU-only:
  1. ``ActivationVerbalizer.generate(context_ids=...)`` — draft tokens are
     appended after the injection prefix and generation still returns only
     new tokens.
  2. ``scripts/train_nla_draft.py`` — one full optimizer step plus one val
     pass runs end-to-end on synthetic pairs/targets.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

TINY = os.environ.get(
    "SRT_NLA_TINY_BACKBONE",
    "hf-internal-testing/tiny-random-LlamaForCausalLM",
)

REPO = Path(__file__).resolve().parent.parent


def _tiny():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        backbone = AutoModelForCausalLM.from_pretrained(TINY, torch_dtype=torch.float32)
        tok = AutoTokenizer.from_pretrained(TINY)
    except Exception as e:  # offline CI
        pytest.skip(f"tiny backbone unavailable: {e}")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id or 0
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()
    return backbone, tok


def test_generate_with_context_ids():
    from srt.nla import ActivationVerbalizer, NLAConfig

    backbone, tok = _tiny()
    cfg = NLAConfig(
        backbone_id=TINY,
        backbone_dtype="float32",
        extraction_layer=min(2, backbone.config.num_hidden_layers),
        max_new_tokens=4,
        num_prefix_tokens=2,
    )
    av = ActivationVerbalizer(cfg, backbone=backbone, tokenizer=tok)
    d = backbone.config.hidden_size
    v = F.normalize(torch.randn(2, d), dim=-1)
    ctx = torch.randint(0, backbone.config.vocab_size, (2, 5))

    ids_plain = av.generate(v, max_new_tokens=4, do_sample=False)
    ids_ctx = av.generate(v, max_new_tokens=4, do_sample=False, context_ids=ctx)
    assert ids_ctx.dim() == 2 and ids_ctx.size(0) == 2
    # Only new tokens are returned (HF inputs_embeds semantics), so length
    # matches the plain call's, not prefix+context.
    assert ids_ctx.size(1) <= 4 and ids_plain.size(1) <= 4

    # Ragged context via attention mask must also run.
    attn = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.long)
    ids_ragged = av.generate(v, max_new_tokens=4, do_sample=False,
                             context_ids=ctx, context_attn=attn)
    assert ids_ragged.size(0) == 2


@pytest.mark.slow
def test_train_nla_draft_one_step(tmp_path: Path):
    backbone, tok = _tiny()
    d = backbone.config.hidden_size

    n = 24
    targets = torch.randn(n, d)
    torch.save({"targets": targets}, tmp_path / "targets.pt")
    with (tmp_path / "pairs.jsonl").open("w") as f:
        for i in range(n):
            ids = tok(f"synthetic pair number {i} about topic {i % 5}",
                      add_special_tokens=False)["input_ids"][:12]
            f.write(json.dumps({"target_idx": i, "gold_ids": ids}) + "\n")

    env = dict(os.environ, PYTHONPATH=str(REPO))
    proc = subprocess.run(
        [
            sys.executable, str(REPO / "scripts" / "train_nla_draft.py"),
            "--targets", str(tmp_path / "targets.pt"),
            "--pairs", str(tmp_path / "pairs.jsonl"),
            "--backbone", TINY,
            "--dtype", "float32",
            "--layer", str(min(2, backbone.config.num_hidden_layers)),
            "--num-prefix-tokens", "2",
            "--max-seq-len", "12",
            "--max-draft-len", "8",
            "--epochs", "1",
            "--batch-size", "4",
            "--val-fraction", "0.2",
            "--val-every", "2",
            "--val-vectors", "2",
            "--val-bestof", "2",
            "--patience", "0",
            "--warmup-steps", "0",
            "--out", str(tmp_path / "out"),
        ],
        capture_output=True, text=True, timeout=600, env=env,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert (tmp_path / "out" / "best_av.pt").exists(), proc.stderr[-3000:]
    ck = torch.load(tmp_path / "out" / "best_av.pt", weights_only=False)
    assert "trainable" in ck and "val" in ck
    assert "draft_cen" in ck["val"] and "greedy_cen" in ck["val"]
