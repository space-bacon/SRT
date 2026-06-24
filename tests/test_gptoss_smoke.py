"""gpt-oss port smoke test: validate the manual layer loop on a gpt-oss-style
backbone with alternating sliding-window / full attention.

This builds a TINY random `gpt_oss` model (no download, no GPU, fp32 on CPU)
and checks that the SRT manual loop reproduces the backbone's own forward
bit-for-bit. The point is to catch sliding-window mask-routing bugs locally
before spending anything on gpt-oss-20b / 120b.

Skips automatically if the installed transformers has no `gpt_oss` support
(needs transformers >= 4.55).
"""

from __future__ import annotations

import tempfile

import pytest

torch = pytest.importorskip("torch")


def _build_tiny_gptoss(tmpdir: str):
    """Create and save a tiny random gpt_oss model. Returns its path."""
    transformers = pytest.importorskip("transformers")
    try:
        from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM
    except Exception:  # pragma: no cover - version without gpt_oss
        pytest.skip("transformers build has no gpt_oss support (need >= 4.55)")

    # Small but with BOTH attention types present and a window short enough
    # that a >window sequence actually exercises the banded mask.
    cfg = GptOssConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_local_experts=4,
        num_experts_per_tok=2,
        sliding_window=4,
        max_position_embeddings=64,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    model = GptOssForCausalLM(cfg).to(torch.float32).eval()
    model.save_pretrained(tmpdir)
    return tmpdir


def _adapter_for(path: str):
    from srt.adapter import SRTAdapter
    from srt.config import SRTConfig

    cfg = SRTConfig(backbone_id=path, backbone_dtype="float32")
    adapter = SRTAdapter(cfg).to("cpu").eval()
    return adapter


def test_gptoss_sliding_detected_and_parity():
    with tempfile.TemporaryDirectory() as tmp:
        path = _build_tiny_gptoss(tmp)
        adapter = _adapter_for(path)

        # 1. Sliding-window attention must be detected on a gpt_oss backbone.
        assert adapter._has_sliding, "sliding-window attention not detected"
        assert adapter._sliding_window > 0
        assert any(lt == "sliding_attention" for lt in adapter._layer_types)
        assert any(lt == "full_attention" for lt in adapter._layer_types)

        # 2. Parity vs the backbone's own forward, fp32 bit-exact, on a
        #    sequence LONGER than the window so sliding layers actually band.
        seq_len = adapter._sliding_window * 3 + 1
        input_ids = torch.randint(0, 256, (1, seq_len))
        with torch.no_grad():
            man = adapter(input_ids=input_ids, disable_injectors=True).logits
            ref = adapter.backbone(input_ids=input_ids).logits
        max_diff = (man.float() - ref.float()).abs().max().item()
        assert max_diff <= 1e-4, f"manual loop diverged from HF forward: {max_diff}"


def test_gptoss_kv_generate_parity():
    """The cached-decode path (prefill + decode) must match a dense forward on
    the prefill logits, including the sliding-window mask over the cache."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _build_tiny_gptoss(tmp)
        adapter = _adapter_for(path)

        seq_len = adapter._sliding_window * 3 + 1
        input_ids = torch.randint(0, 256, (1, seq_len))
        # Prefill through the cached path, compare its last-token logits to a
        # fresh dense forward's last-token logits.
        from transformers import DynamicCache

        with torch.no_grad():
            cache = DynamicCache()
            logits, _, _ = adapter._cached_step(
                input_ids=input_ids,
                backbone_cache=cache,
                mah_kv={},
                community_vec=None,
                start_pos=0,
                disable_injectors=True,
            )
            ref = adapter.backbone(input_ids=input_ids).logits
        diff = (logits[:, -1].float() - ref[:, -1].float()).abs().max().item()
        assert diff <= 1e-4, f"cached prefill diverged from HF forward: {diff}"
