"""In-memory vector index with npz persistence.

Vectors stored here are already head-projected and L2-normalized, so
search is a dot product. 123K images fit in ~250MB fp16 and search in
well under a second on CPU.
"""
from __future__ import annotations

import numpy as np


class Index:
    def __init__(self, proj_dim: int):
        self.proj_dim = proj_dim
        self._keys: list[str] = []
        self._pending: list[np.ndarray] = []
        self._mat: np.ndarray | None = None   # materialized rows

    def __len__(self) -> int:
        return len(self._keys)

    def add(self, key: str, vec: np.ndarray) -> None:
        self._keys.append(key)
        self._pending.append(vec.astype(np.float32))

    def _materialize(self) -> np.ndarray:
        if self._pending:
            new = np.stack(self._pending)
            self._mat = new if self._mat is None else np.concatenate(
                [self._mat, new])
            self._pending = []
        if self._mat is None:
            raise ValueError("index is empty")
        return self._mat

    def search(self, qvec: np.ndarray, k: int = 8) -> list[tuple[str, float]]:
        mat = self._materialize()
        sims = mat @ qvec.astype(np.float32)
        order = np.argsort(-sims)[:k]
        return [(self._keys[i], float(sims[i])) for i in order]

    def save(self, path: str) -> None:
        mat = self._materialize()
        np.savez_compressed(path, vecs=mat.astype(np.float16),
                            keys=np.array(self._keys, dtype=object),
                            proj_dim=self.proj_dim)

    @classmethod
    def load(cls, path: str) -> "Index":
        z = np.load(path, allow_pickle=True)
        idx = cls(int(z["proj_dim"]))
        idx._mat = z["vecs"].astype(np.float32)
        idx._keys = [str(k) for k in z["keys"]]
        return idx
