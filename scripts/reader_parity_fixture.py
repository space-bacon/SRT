"""Write the parity fixture for a swapped reader, so the Rust port is checked, not trusted.

For a handful of probe sentences this records what PyTorch/sentence-transformers
produces at each stage the browser reproduces: the pooled encoder embedding
(the "state"), the head projection, and the top-5 gallery rows. The native
example `runtime/examples/reader_parity.rs` recomputes all three with candle and
reports the gap.

    .venv-tools/bin/python scripts/reader_parity_fixture.py --reader bge-small
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
from reader_ladder import ENCODERS  # noqa: E402
from reader_swap import POOLING, read_srtidx  # noqa: E402

PROBES = [
    "A young woman in pink leggings playing tennis",
    "two dogs running on a beach at sunset",
    "A plate of pasta with tomato sauce and basil on a wooden table.",
    "a red double decker bus driving down a city street",
    "A man riding a wave on top of a surfboard.",
    "kitchen with stainless steel appliances and white cabinets",
    "A giraffe standing next to a tree in a field.",
    "A group of people sitting around a table with laptops",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reader", default="bge-small")
    p.add_argument("--heads", type=Path, default=HERE / "artifacts/local/reader_swap/heads")
    p.add_argument("--gallery", type=Path, default=HERE / "artifacts/local/browser/gallery_123k_v3.srtidx")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    model_id, prefix = ENCODERS[a.reader]
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(model_id, device="cpu")
    st.max_seq_length = 64
    pool_cfg = [m for m in st if m.__class__.__name__ == "Pooling"][0].pooling_mode
    assert pool_cfg == POOLING[a.reader], (pool_cfg, POOLING[a.reader])
    has_norm = any(m.__class__.__name__ == "Normalize" for m in st)
    E = st.encode([prefix + t for t in PROBES], convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)
    assert has_norm and np.allclose(np.linalg.norm(E, axis=1), 1.0, atol=1e-4)

    h = torch.load(a.heads / f"text_head_{a.reader}_v3gallery.pt", map_location="cpu", weights_only=False)
    W, b, mu = h["txt"]["weight"].numpy(), h["txt"]["bias"].numpy(), h["mu_txt"].numpy()
    Z = (E - mu) @ W.T + b
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)

    Zg, keys = read_srtidx(a.gallery)
    S = torch.from_numpy(Z) @ Zg.T
    top = S.topk(5, dim=1).indices.numpy()

    fx = {
        "reader": model_id, "prefix": prefix, "pooling": pool_cfg, "normalize": True, "max_len": 64,
        "head": f"text_head_{a.reader}_v3gallery.safetensors", "gallery": a.gallery.name,
        "probes": [{"text": t, "embedding": E[i].tolist(), "projection": Z[i].tolist(),
                    "top5_rows": top[i].tolist(), "top5_keys": [keys[j] for j in top[i]]}
                   for i, t in enumerate(PROBES)],
    }
    out = a.out or a.heads / f"parity_{a.reader}.json"
    out.write_text(json.dumps(fx))
    print(f"wrote {out}: {len(PROBES)} probes, d={E.shape[1]}, pooling {pool_cfg}")
    for i, t in enumerate(PROBES[:3]):
        print(f"  {t!r} -> {fx['probes'][i]['top5_keys'][0]}")


if __name__ == "__main__":
    main()
