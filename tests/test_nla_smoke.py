"""N0 smoke for SRT-NLA — exercises the AV→AR plumbing on a tiny backbone.

The bench in ``scripts/bench_nla_n0.py`` is the user-facing entry point;
this test is the CI-friendly slice that just checks the round-trip runs
end-to-end without crashing on a tiny test model.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F


TINY = os.environ.get(
    "SRT_NLA_TINY_BACKBONE",
    "hf-internal-testing/tiny-random-LlamaForCausalLM",
)


@pytest.mark.slow
def test_nla_n0_roundtrip_tiny():
    """AV.generate → AR.reconstruct must run end-to-end and return ``(B, d)``."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from srt.nla import ActivationReconstructor, ActivationVerbalizer, NLAConfig
    from srt.nla.loss import fraction_variance_explained, mse_nrm

    try:
        backbone = AutoModelForCausalLM.from_pretrained(TINY, torch_dtype=torch.float32)
        tok = AutoTokenizer.from_pretrained(TINY)
    except Exception as e:  # network/model unavailable in offline CI
        pytest.skip(f"tiny backbone unavailable: {e}")

    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id or 0
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()

    cfg = NLAConfig(
        backbone_id=TINY,
        backbone_dtype="float32",
        extraction_layer=min(2, backbone.config.num_hidden_layers),
        max_new_tokens=4,
        num_prefix_tokens=1,
    )
    av = ActivationVerbalizer(cfg, backbone=backbone, tokenizer=tok)
    ar = ActivationReconstructor(cfg, backbone=backbone)

    d = backbone.config.hidden_size
    v = F.normalize(torch.randn(2, d), dim=-1)
    ids = av.generate(v, max_new_tokens=4, do_sample=False)
    assert ids.dim() == 2 and ids.size(0) == 2
    attn = (ids != tok.pad_token_id).long()
    if attn.sum() == 0:  # all pad — fall back to a unit mask so AR can run
        attn = torch.ones_like(ids)

    v_hat = ar.reconstruct(ids, attention_mask=attn)
    assert v_hat.shape == (2, d)

    # Untrained: fve should be finite (likely near zero), mse near 2.
    fve = fraction_variance_explained(v_hat.float(), v.float()).item()
    m = mse_nrm(v_hat.float(), v.float()).item()
    assert torch.isfinite(torch.tensor(fve))
    assert torch.isfinite(torch.tensor(m))


def test_nla_loss_shapes_cpu():
    """Loss helpers compute on CPU without the backbone."""
    from srt.nla.loss import (
        cosine_similarity,
        fraction_variance_explained,
        mse_nrm,
        nla_reward,
    )

    v_hat = torch.randn(8, 16)
    v_target = torch.randn(8, 16)
    assert mse_nrm(v_hat, v_target).ndim == 0
    assert cosine_similarity(v_hat, v_target).ndim == 0
    assert fraction_variance_explained(v_hat, v_target).ndim == 0
    r = nla_reward(v_hat, v_target)
    assert r.total.ndim == 0
    # Identity round-trip is perfect.
    perfect = nla_reward(v_target, v_target)
    assert perfect.mse.item() < 1e-6


def test_nla_config_defaults():
    from srt.nla import NLAConfig

    cfg = NLAConfig()
    assert cfg.extraction_layer == 20
    assert cfg.pool == "last"
    assert cfg.num_prefix_tokens >= 1
