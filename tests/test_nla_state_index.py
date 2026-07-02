"""Unit tests for ``srt.nla.state_index`` — magic-number state index.

Pure torch; no backbone needed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from srt.nla.state_index import StateIndex, dedup_by_state_index


def _pool(n=64, d=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g)


def test_encode_is_deterministic_and_reproducible():
    d = 32
    v = torch.randn(d, generator=torch.Generator().manual_seed(1))
    a = StateIndex(d, n_bits=24, seed=7)
    b = StateIndex(d, n_bits=24, seed=7)
    assert a.encode(v) == b.encode(v)  # same seed → same magic number
    assert a.encode(v) == a.encode(v)  # deterministic
    # batched encode agrees with per-item encode
    vs = _pool(8, d)
    codes = a.encode(vs)
    assert codes.shape == (8,)
    for i in range(8):
        assert int(codes[i]) == a.encode(vs[i])


def test_locality_similar_states_get_small_hamming():
    d = 64
    idx = StateIndex(d, n_bits=48, seed=0)
    v = torch.randn(d)
    near = v + 0.01 * torch.randn(d)        # tiny perturbation → tiny angle
    far = torch.randn(d)                     # unrelated direction
    cv, cn, cf = idx.encode(v), idx.encode(near), idx.encode(far)
    h_near = idx.hamming(cv, cn)
    h_far = idx.hamming(cv, cf)
    assert h_near < h_far
    assert h_near <= 3  # near-identical vectors share almost all bits


def test_codebook_add_lookup_decode():
    d = 32
    idx = StateIndex(d, n_bits=20, seed=3)
    v = torch.randn(d)
    code = idx.add(v, text="a spinning quicksort", cen=0.9)
    assert len(idx) == 1
    entry = idx.lookup(v)
    assert entry is not None and entry.code == code and entry.count == 1
    assert idx.decode(v) == "a spinning quicksort"
    # a better verbalization replaces the canonical one; a worse one does not
    idx.add(v, text="better desc", cen=0.95)
    idx.add(v, text="worse desc", cen=0.10)
    assert idx.decode(v) == "better desc"
    assert idx.lookup(v).count == 3


def test_multi_probe_lookup_within_hamming():
    d = 48
    idx = StateIndex(d, n_bits=40, seed=1)
    v = torch.randn(d)
    idx.add(v, text="stored")
    near = v + 0.02 * torch.randn(d)
    # exact bucket may differ by a bit or two; multi-probe should still find it
    assert idx.decode(near, max_hamming=8) == "stored"


def test_dedup_finds_recurring_states():
    d = 16
    base = torch.randn(d)
    # three copies of the same state interleaved with two distinct states
    vs = torch.stack([base, torch.randn(d), base.clone(), torch.randn(d), base.clone()])
    codes, groups = dedup_by_state_index(vs, n_bits=32, seed=0)
    # the three identical states must share one code
    assert codes[0] == codes[2] == codes[4]
    assert sorted(groups[int(codes[0])]) == [0, 2, 4]
    # at most 3 distinct codes for 3 distinct states
    assert len(groups) <= 3


def test_centering_changes_codes():
    d = 32
    vs = _pool(16, d)
    mu = vs.mean(0)
    raw = StateIndex(d, n_bits=24, seed=0, mu=None)
    cen = StateIndex(d, n_bits=24, seed=0, mu=mu)
    # centring shifts the hashing origin, so at least some codes differ
    assert any(raw.encode(vs[i]) != cen.encode(vs[i]) for i in range(16))


def test_save_load_roundtrip(tmp_path):
    d = 32
    mu = torch.randn(d)
    idx = StateIndex(d, n_bits=24, seed=5, mu=mu)
    vs = _pool(20, d)
    for i in range(20):
        idx.add(vs[i], text=f"state-{i}")
    p = tmp_path / "index.pt"
    idx.save(p)
    idx2 = StateIndex.load(p)
    assert idx2.n_bits == idx.n_bits and idx2.seed == idx.seed
    assert len(idx2) == len(idx)
    # codes and decodings survive the round-trip
    for i in range(20):
        assert idx2.encode(vs[i]) == idx.encode(vs[i])
        assert idx2.decode(vs[i]) == idx.decode(vs[i])


def test_stats_reports_buckets():
    d = 16
    idx = StateIndex(d, n_bits=16, seed=0)
    base = torch.randn(d)
    idx.add(base, text="x")
    idx.add(base.clone())
    idx.add(torch.randn(d))
    s = idx.stats()
    assert s["n_states"] == 3
    assert s["n_codes"] == len(idx)
    assert s["with_text"] >= 1


def _clustered(k=4, per=20, d=16, spread=0.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(k, d, generator=g) * 5.0
    pts, labels = [], []
    for j in range(k):
        pts.append(centers[j] + spread * torch.randn(per, d, generator=g))
        labels += [j] * per
    return torch.cat(pts), torch.tensor(labels), centers


def test_vq_fit_and_encode_recovers_clusters():
    vs, labels, _ = _clustered(k=4, per=25, d=32, seed=1)
    idx = StateIndex.fit_vq(vs, k=4, seed=1)
    assert idx.mode == "vq"
    codes = idx.encode(vs)
    # points from the same true cluster must share a code (up to label permutation)
    for j in range(4):
        cj = codes[labels == j]
        assert (cj == cj[0]).all()
    assert idx.stats()["k"] == 4


def test_vq_decode_is_nearest_centroid():
    vs, labels, centers = _clustered(k=3, per=30, d=24, seed=2)
    idx = StateIndex.fit_vq(vs, k=3, seed=2)
    idx.add_many(vs, texts=[f"cluster-{int(l)}" for l in labels])
    # a fresh point near cluster 1's center decodes to a stored cluster-1 text
    near = centers[1] + 0.01 * torch.randn(24)
    dec = idx.decode(near)
    assert dec is not None and dec.startswith("cluster-")
    # its code equals the code of the training points from that cluster
    assert idx.encode(near) == idx.encode(vs[labels == 1][0])


def test_vq_save_load_roundtrip(tmp_path):
    vs, labels, _ = _clustered(k=3, per=15, d=20, seed=3)
    idx = StateIndex.fit_vq(vs, k=3, seed=3)
    idx.add_many(vs, texts=[f"c{int(l)}" for l in labels])
    p = tmp_path / "vq.pt"
    idx.save(p)
    idx2 = StateIndex.load(p)
    assert idx2.mode == "vq"
    for i in range(len(vs)):
        assert idx2.encode(vs[i]) == idx.encode(vs[i])
        assert idx2.decode(vs[i]) == idx.decode(vs[i])

