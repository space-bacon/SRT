"""Magic-number state index — discrete addressable codes for hidden states.

The full trace (``srt.nla.trace``) hands us a hidden state
``v \\in \\mathbb{R}^{d}`` at every (layer, position). Those vectors are the
model's internal states, but a 3584-float vector is not a *handle*: you
cannot say "the model is in state #4711 again", cache a verbalization for it,
or notice that two positions hold the same state.

This module assigns each internal state a compact integer — the "magic
number" — via a **locality-sensitive SimHash** (signed random projections),
and keeps a **codebook** mapping ``code -> canonical verbalization``.

Why SimHash:

- **Addressable.** ``encode(v)`` is a deterministic integer you can print,
  store, diff, and use as a dict key.
- **Locality-sensitive.** For centred vectors, ``P(bit_i(a) == bit_i(b)) =
  1 - theta/pi`` where ``theta`` is the angle between ``a`` and ``b``. So
  Hamming distance between two magic numbers is a monotone proxy for
  ``1 - cos``: nearby states get nearby codes. This matches the paper's
  finding that nearest-neighbour retrieval in centred activation space is a
  strong (greedy-beating) decoder — the code *is* an approximate NN address.
- **Training-free & anisotropy-aware.** Centre by the pool mean ``mu`` (the
  same correction the metric uses) before projecting, so the code is not
  dominated by the backbone's mean direction.

The codebook turns the index into a decoder: ``decode(v)`` is an O(1) hash
lookup that returns a stored verbalization, instead of the O(pool) nearest-
neighbour scan the paper's retrieval baseline runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

__all__ = ["CodebookEntry", "StateIndex", "dedup_by_state_index"]

_MAX_BITS = 62  # keep codes inside a signed int64 with head-room


@dataclass
class CodebookEntry:
    """One bucket of the codebook: a code and its canonical verbalization."""

    code: int
    centroid: torch.Tensor  # (d,) running mean of member states (centred space)
    count: int = 0
    text: str | None = None  # canonical verbalization for this code
    best_cen: float = -1.0   # centred fve of `text` against the centroid


class StateIndex:
    """Locality-sensitive integer index + verbalization codebook for states.

    Two encoders:

    - ``mode="simhash"`` (default) — training-free signed random projections.
      Codes are locality-sensitive (Hamming distance ∝ angle) and span up to
      ``2**n_bits`` buckets.
    - ``mode="vq"`` — nearest of ``k`` fitted centroids (vector quantization).
      Codes are cluster ids ``0..k-1``; each is a semantic bucket. Build with
      :meth:`fit_vq`. Retrieval decoding is then an exact NN over ``k`` centroids.

    Parameters
    ----------
    d:
        Hidden-state dimensionality.
    mode:
        ``"simhash"`` or ``"vq"``.
    n_bits:
        SimHash bit width (<= 62). Ignored for ``vq``.
    centroids:
        ``(k, d)`` fitted centroids in *centred* space. Required for ``vq``
        (usually supplied by :meth:`fit_vq`, not by hand).
    mu:
        Anisotropy mean ``(d,)`` subtracted before hashing/quantizing.
    seed:
        Seeds the random hyperplanes, so codes are reproducible across runs.
    device:
        Device for the projection matrix / centroids.
    """

    def __init__(
        self,
        d: int,
        *,
        mode: str = "simhash",
        n_bits: int = 24,
        centroids: torch.Tensor | None = None,
        mu: torch.Tensor | None = None,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ) -> None:
        if mode not in ("simhash", "vq"):
            raise ValueError(f"mode must be 'simhash' or 'vq'; got {mode!r}")
        self.d = int(d)
        self.mode = mode
        self.n_bits = int(n_bits)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.mu = None if mu is None else mu.detach().float().to(self.device)
        self.planes: torch.Tensor | None = None
        self._weights: torch.Tensor | None = None
        self.centroids: torch.Tensor | None = None
        if mode == "simhash":
            if not (1 <= n_bits <= _MAX_BITS):
                raise ValueError(f"n_bits must be in [1, {_MAX_BITS}]; got {n_bits}")
            g = torch.Generator(device="cpu").manual_seed(seed)
            # (d, n_bits) random hyperplane normals. Fixed → reproducible codes.
            self.planes = torch.randn(self.d, self.n_bits, generator=g).to(self.device)
            self._weights = torch.tensor(
                [1 << i for i in range(self.n_bits)], dtype=torch.long, device=self.device
            )
            self.centroids = None
        else:  # vq
            if centroids is None:
                raise ValueError("mode='vq' requires centroids (use StateIndex.fit_vq)")
            self.planes = None
            self._weights = None
            self.centroids = centroids.detach().float().to(self.device)  # (k, d) centred
        self.codebook: dict[int, CodebookEntry] = {}

    # ------------------------------------------------------------------
    @staticmethod
    def _kmeans(x: torch.Tensor, k: int, *, iters: int = 25, seed: int = 0) -> torch.Tensor:
        """Lloyd's k-means with k-means++ init on ``x`` (N, d) -> ``(k, d)``."""
        n = x.size(0)
        k = min(k, n)
        g = torch.Generator(device="cpu").manual_seed(seed)
        xc = x.cpu()
        # k-means++ seeding: pick spread-out initial centroids.
        first = int(torch.randint(n, (1,), generator=g).item())
        chosen = [first]
        d2 = ((xc - xc[first]) ** 2).sum(1)
        for _ in range(1, k):
            total = float(d2.sum().item())
            if total <= 0:
                chosen.append(int(torch.randint(n, (1,), generator=g).item()))
            else:
                nxt = int(torch.multinomial(d2 / d2.sum(), 1, generator=g).item())
                chosen.append(nxt)
            d2 = torch.minimum(d2, ((xc - xc[chosen[-1]]) ** 2).sum(1))
        centroids = xc[torch.tensor(chosen)].clone()
        for _ in range(iters):
            assign = torch.cdist(xc, centroids).argmin(dim=1)
            new = centroids.clone()
            for j in range(k):
                m = assign == j
                if m.any():
                    new[j] = xc[m].mean(0)
            if torch.allclose(new, centroids):
                centroids = new
                break
            centroids = new
        return centroids.to(x.device)

    @classmethod
    def fit_vq(
        cls,
        pool: torch.Tensor,
        k: int,
        *,
        mu: torch.Tensor | None = None,
        iters: int = 25,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ) -> "StateIndex":
        """Fit a ``k``-centroid VQ index on a pool of states ``(N, d)``."""
        pool = pool.float()
        mu_t = pool.mean(0) if mu is None else mu.float()
        centroids = cls._kmeans(pool - mu_t, k, iters=iters, seed=seed)
        return cls(
            pool.size(-1), mode="vq", centroids=centroids, mu=mu_t, seed=seed, device=device
        )

    # ------------------------------------------------------------------
    def _centered(self, v: torch.Tensor) -> torch.Tensor:
        v = v.float().to(self.device)
        return v if self.mu is None else v - self.mu

    def _codes(self, vs: torch.Tensor) -> torch.Tensor:
        """Always-batched encode: ``(N, d)`` -> ``(N,)`` int64 codes."""
        vc = self._centered(vs)
        if self.mode == "simhash":
            assert self.planes is not None and self._weights is not None
            bits = (vc @ self.planes) > 0  # (N, n_bits) bool
            return (bits.long() * self._weights).sum(dim=-1)  # (N,) int64
        # vq: nearest centroid id
        assert self.centroids is not None
        return torch.cdist(vc, self.centroids).argmin(dim=-1).long()  # (N,)

    def encode(self, v: torch.Tensor) -> int | torch.Tensor:
        """Map state(s) to magic number(s).

        ``v`` shape ``(d,)`` -> python ``int``; ``(N, d)`` -> ``(N,)`` int64.
        """
        if v.dim() == 1:
            return int(self._codes(v.unsqueeze(0))[0].item())
        return self._codes(v)

    def hamming(self, a: int, b: int) -> int:
        """Bit distance between two magic numbers (angular-distance proxy)."""
        return int((a ^ b).bit_count() if hasattr(int, "bit_count") else bin(a ^ b).count("1"))

    # ------------------------------------------------------------------
    def add(
        self,
        v: torch.Tensor,
        *,
        text: str | None = None,
        cen: float | None = None,
    ) -> int:
        """Register a state, updating its codebook bucket. Returns the code.

        The bucket keeps a running centroid (centred space) and, if ``text``
        is supplied, the verbalization with the highest centred fidelity
        ``cen`` seen so far as the bucket's canonical decoding. When ``cen``
        is not given it is computed as the centred fve of ``v`` against the
        current centroid (a self-consistency score).
        """
        if v.dim() != 1:
            raise ValueError("add() takes a single (d,) state; use add_many for batches")
        code = int(self._codes(v.unsqueeze(0))[0].item())
        vc = self._centered(v)
        entry = self.codebook.get(code)
        if entry is None:
            entry = CodebookEntry(code=code, centroid=vc.clone(), count=1)
            self.codebook[code] = entry
        else:
            entry.count += 1
            entry.centroid += (vc - entry.centroid) / entry.count  # running mean
        if text is not None:
            score = cen if cen is not None else float(
                0.5 * (1.0 + F.cosine_similarity(vc.unsqueeze(0), entry.centroid.unsqueeze(0)).item())
            )
            if score > entry.best_cen:
                entry.best_cen = score
                entry.text = text
        return code

    def add_many(
        self,
        vs: torch.Tensor,
        *,
        texts: list[str | None] | None = None,
        cens: list[float] | None = None,
    ) -> torch.Tensor:
        """Register a batch ``(N, d)``; returns the ``(N,)`` codes."""
        codes = self._codes(vs)
        for i in range(vs.size(0)):
            self.add(
                vs[i],
                text=None if texts is None else texts[i],
                cen=None if cens is None else cens[i],
            )
        return codes  # type: ignore[return-value]

    # ------------------------------------------------------------------
    def lookup(self, v: torch.Tensor, *, max_hamming: int = 0) -> CodebookEntry | None:
        """Return the codebook entry for ``v``.

        Exact code match first. If absent:

        - ``simhash``: return the stored entry whose code is closest in Hamming
          distance within ``max_hamming`` (multi-probe LSH), else ``None``.
        - ``vq``: return the nearest populated bucket by centroid L2 (an exact
          NN decoder over the fitted centroids); ``max_hamming`` is ignored.
        """
        vv = v.unsqueeze(0) if v.dim() == 1 else v
        code = int(self._codes(vv)[0].item())
        entry = self.codebook.get(code)
        if entry is not None or not self.codebook:
            return entry
        if self.mode == "vq":
            vc = self._centered(vv).squeeze(0)
            best: CodebookEntry | None = None
            best_d = float("inf")
            for e in self.codebook.values():
                dd = float((vc - e.centroid).pow(2).sum().item())
                if dd < best_d:
                    best_d, best = dd, e
            return best
        if max_hamming <= 0:
            return None
        best = None
        best_h = max_hamming + 1
        for c, e in self.codebook.items():
            h = self.hamming(code, c)
            if h < best_h:
                best_h, best = h, e
        return best

    def decode(self, v: torch.Tensor, *, max_hamming: int = 2) -> str | None:
        """Retrieval decoding: canonical verbalization for ``v`` via its code."""
        entry = self.lookup(v, max_hamming=max_hamming)
        return None if entry is None else entry.text

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.codebook)

    @property
    def codes(self) -> list[int]:
        return sorted(self.codebook.keys())

    def stats(self) -> dict:
        counts = [e.count for e in self.codebook.values()]
        return {
            "mode": self.mode,
            "n_codes": len(self.codebook),
            "n_states": int(sum(counts)),
            "n_bits": self.n_bits if self.mode == "simhash" else None,
            "k": None if self.centroids is None else self.centroids.size(0),
            "load_factor": (sum(counts) / len(counts)) if counts else 0.0,
            "with_text": sum(1 for e in self.codebook.values() if e.text is not None),
        }

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Persist the index + codebook (simhash codes regenerate from seed)."""
        obj = {
            "d": self.d,
            "mode": self.mode,
            "n_bits": self.n_bits,
            "seed": self.seed,
            "mu": None if self.mu is None else self.mu.cpu(),
            "centroids": None if self.centroids is None else self.centroids.cpu(),
            "entries": [
                {
                    "code": e.code,
                    "centroid": e.centroid.cpu(),
                    "count": e.count,
                    "text": e.text,
                    "best_cen": e.best_cen,
                }
                for e in self.codebook.values()
            ],
        }
        torch.save(obj, path)

    @classmethod
    def load(cls, path: str | Path, *, device: str | torch.device = "cpu") -> "StateIndex":
        # The saved object is pure tensors/ints/floats/strs/None, so
        # weights_only=True loads it without pickle-deserialization risk.
        obj = torch.load(path, map_location="cpu", weights_only=True)
        idx = cls(
            obj["d"],
            mode=obj.get("mode", "simhash"),
            n_bits=obj["n_bits"],
            centroids=obj.get("centroids"),
            mu=obj["mu"],
            seed=obj["seed"],
            device=device,
        )
        for e in obj["entries"]:
            idx.codebook[int(e["code"])] = CodebookEntry(
                code=int(e["code"]),
                centroid=e["centroid"].to(idx.device),
                count=int(e["count"]),
                text=e["text"],
                best_cen=float(e["best_cen"]),
            )
        return idx


def dedup_by_state_index(
    vs: torch.Tensor,
    *,
    n_bits: int = 24,
    mu: torch.Tensor | None = None,
    seed: int = 0,
    texts: list[str] | None = None,
) -> tuple[torch.Tensor, dict[int, list[int]]]:
    """Collapse a set of states into their magic numbers.

    Returns ``(codes, groups)`` where ``codes[i]`` is the magic number of
    state ``i`` and ``groups[code]`` lists the indices sharing that code.
    Useful for finding *recurring internal states* across a full trace (e.g.
    the same "state" being re-visited at many token positions/layers).
    """
    idx = StateIndex(vs.size(-1), n_bits=n_bits, mu=mu, seed=seed, device=vs.device)
    codes = idx._codes(vs)  # (N,)
    groups: dict[int, list[int]] = {}
    for i, c in enumerate(codes.tolist()):
        groups.setdefault(int(c), []).append(i)
    return codes, groups
