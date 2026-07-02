"""Smoke test for the full input→output NLA trace (``srt.nla.trace``).

Exercises ``NLATracer`` end-to-end on a tiny backbone: generate a short
continuation, verbalize hidden states at multiple layers over both input
and output token positions, and check the returned ``NLATrace`` structure.
"""

from __future__ import annotations

import os

import pytest
import torch


TINY = os.environ.get(
    "SRT_NLA_TINY_BACKBONE",
    "hf-internal-testing/tiny-random-LlamaForCausalLM",
)


@pytest.mark.slow
def test_nla_trace_input_and_output():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from srt.nla import ActivationVerbalizer, NLAConfig, NLATrace, NLATracer

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

    n_layers = backbone.config.num_hidden_layers
    cfg = NLAConfig(
        backbone_id=TINY,
        backbone_dtype="float32",
        extraction_layer=min(2, n_layers),
        max_new_tokens=4,
        num_prefix_tokens=1,
    )
    av = ActivationVerbalizer(cfg, backbone=backbone, tokenizer=tok)
    tracer = NLATracer(av)

    trace = tracer.trace(
        "Hello world",
        layers=[1, min(2, n_layers)],
        K=2,
        generate_output=True,
        max_new_tokens=3,
        verbalize_max_new_tokens=3,
        keep_candidates=True,
    )

    assert isinstance(trace, NLATrace)
    # Output tokens were generated → full sequence longer than the prompt.
    assert len(trace.tokens) >= trace.n_input_tokens
    assert trace.cells, "trace produced no cells"

    # Grid is layer × position; every cell has a verbalization and finite score.
    for c in trace.cells:
        assert c.layer in trace.layers
        assert 0.0 <= c.fve <= 1.0
        assert torch.isfinite(torch.tensor(c.fve_centered))
        assert c.candidates is not None and len(c.candidates) == 2

    # At least one cell should be flagged as an output (generated) position
    # when the continuation is non-empty.
    if len(trace.tokens) > trace.n_input_tokens:
        assert any(c.is_output for c in trace.cells)

    # to_dict round-trips the structure.
    d = trace.to_dict()
    assert d["n_input_tokens"] == trace.n_input_tokens
    assert len(d["cells"]) == len(trace.cells)


@pytest.mark.slow
def test_nla_trace_input_only():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from srt.nla import ActivationVerbalizer, NLAConfig, NLATracer

    try:
        backbone = AutoModelForCausalLM.from_pretrained(TINY, torch_dtype=torch.float32)
        tok = AutoTokenizer.from_pretrained(TINY)
    except Exception as e:
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
    tracer = NLATracer(av)

    trace = tracer.trace("Hello world", generate_output=False, K=1)
    # No generation → every cell is an input position.
    assert trace.generated == ""
    assert all(not c.is_output for c in trace.cells)
    assert trace.layers == [cfg.extraction_layer]


@pytest.mark.slow
def test_nla_trace_assigns_state_index_codes():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from srt.nla import ActivationVerbalizer, NLAConfig, NLATracer, StateIndex

    try:
        backbone = AutoModelForCausalLM.from_pretrained(TINY, torch_dtype=torch.float32)
        tok = AutoTokenizer.from_pretrained(TINY)
    except Exception as e:
        pytest.skip(f"tiny backbone unavailable: {e}")

    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id or 0
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()

    cfg = NLAConfig(
        backbone_id=TINY, backbone_dtype="float32",
        extraction_layer=min(2, backbone.config.num_hidden_layers),
        max_new_tokens=4, num_prefix_tokens=1,
    )
    av = ActivationVerbalizer(cfg, backbone=backbone, tokenizer=tok)
    tracer = NLATracer(av)

    si = StateIndex(backbone.config.hidden_size, n_bits=24, seed=0)
    trace = tracer.trace(
        "Hello world", layers="all", K=1, generate_output=True, max_new_tokens=3,
        state_index=si,
    )
    # every cell got a magic number, and the codebook was populated
    assert all(c.code is not None for c in trace.cells)
    assert len(si) >= 1
    # code_groups partitions the cells by code, covering all of them
    groups = trace.code_groups()
    assert sum(len(v) for v in groups.values()) == len(trace.cells)
    # a cell's code round-trips through the index
    c0 = trace.cells[0]
    assert si.decode is not None  # index is usable for retrieval

