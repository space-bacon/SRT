"""Tests for sunstone_sidecar. Fast tests run anywhere; tests marked
`integration` need a GPU/backbone and verify the acceptance criterion
(reproduce banked state vectors through the new API).
"""
import numpy as np
import pytest

from sunstone_sidecar.heads import HEAD_REGISTRY, load_head, project
from sunstone_sidecar.index import Index


def test_registry_has_both_tiers():
    assert "google/gemma-4-31B-it" in HEAD_REGISTRY
    assert "Qwen/Qwen2.5-VL-3B-Instruct" in HEAD_REGISTRY


def test_project_normalizes():
    rng = np.random.default_rng(0)
    d, p, n = 64, 16, 8
    W, b = rng.normal(size=(p, d)), rng.normal(size=p)
    mu = rng.normal(size=d)
    z = project(rng.normal(size=(n, d)), W, b, mu)
    assert z.shape == (n, p)
    assert np.allclose(np.linalg.norm(z, axis=-1), 1.0, atol=1e-6)
    z1 = project(rng.normal(size=d), W, b, mu)
    assert z1.shape == (p,)


def test_index_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    idx = Index(proj_dim=16)
    vecs = rng.normal(size=(20, 16)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=-1, keepdims=True)
    for i, v in enumerate(vecs):
        idx.add(f"img_{i}", v)
    hits = idx.search(vecs[7], k=3)
    assert hits[0][0] == "img_7" and hits[0][1] > 0.999
    p = tmp_path / "idx.npz"
    idx.save(str(p))
    idx2 = Index.load(str(p))
    hits2 = idx2.search(vecs[7], k=3)
    assert hits2[0][0] == "img_7"


@pytest.mark.integration
def test_head_download_and_shapes():
    head = load_head("google/gemma-4-31B-it")
    assert head["W_img"].shape == (1024, 5376)
    assert head["mu_img"].shape == (5376,)
    assert head["layer"] == 47
    head3b = load_head("Qwen/Qwen2.5-VL-3B-Instruct")
    assert head3b["W_img"].shape[1] == 2048
    assert head3b["layer"] == 29


@pytest.mark.integration
def test_reproduces_banked_vectors():
    """Acceptance: the new tap reproduces the banked encoding conventions.

    Uses the qwen3b tier (fits on most machines). Downloads the banked
    eval encodings from the HF artifacts repo and re-encodes a handful of
    val2017 captions; vectors must match to float tolerance.
    """
    import torch
    from huggingface_hub import hf_hub_download
    from sunstone_sidecar.taps import TransformersTap

    banked = torch.load(
        hf_hub_download("RiverRider/srt-nla-gemma4-artifacts",
                        "scalefloor/encoded_L29_n5000.pt"),
        map_location="cpu", weights_only=True)
    caps = [c[0] if isinstance(c, list) else c
            for c in banked["captions5"][:4]]
    tap = TransformersTap.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", layer=29)
    got = tap.text_vectors(caps)
    ref = banked["cap5"].float().numpy()
    # banked cap5 rows are the eval tail's 5-per-image captions; row 0 of
    # each group is captions5[i][0]
    ref_rows = np.stack([ref[5 * i] for i in range(4)])
    cos = (got * ref_rows).sum(-1) / (
        np.linalg.norm(got, axis=-1) * np.linalg.norm(ref_rows, axis=-1))
    assert (cos > 0.999).all(), f"convention drift: cos={cos}"
