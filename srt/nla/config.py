"""Configuration for SRT-NLA."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class NLAConfig:
    """Natural Language Autoencoder configuration.

    Defaults target Qwen2.5-7B (28 hidden layers, hidden_size=3584). The
    NLA reference paper extracts at L=20 (~2/3 depth); we keep the same.
    """

    backbone_id: str = "Qwen/Qwen2.5-7B"
    backbone_dtype: str = "bfloat16"

    # Layer to read activations from (1-indexed into output_hidden_states,
    # which is length num_hidden_layers + 1 with index 0 = embeddings).
    extraction_layer: int = 20

    # Vector dimensionality. None = use backbone hidden_size.
    d_vector: int | None = None

    # AV generation
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 1.0

    # Pooling for AR
    pool: str = "last"  # "last" | "mean" | "first"

    # Reward weights (see docs/SRT_NLA_PLAN.md §5)
    lambda_mag: float = 0.1  # magnitude-match penalty
    beta_kl: float = 0.05  # legibility KL vs base
    gamma_entropy: float = 0.05  # entropy hinge coefficient (both sides)
    delta_community: float = 0.2  # community-consistency bonus
    h_min: float = 1.5  # minimum acceptable token-level entropy (nats)
    h_max: float = 3.0  # maximum acceptable token-level entropy (nats); above this the upper hinge pulls H back down to prevent uniform-noise runaway

    # Injection layout — number of learned prefix tokens after the injected
    # vector. The injected vector occupies slot 0; this many extra learned
    # embeddings (BOS-like) follow before the model autoregresses.
    num_prefix_tokens: int = 1

    # Prefix mode:
    #   "static" — `prefix_embeds` is a (P, d_embed) learned tensor shared
    #              across all inputs (original design).
    #   "mlp"    — a 2-layer MLP maps v → (P, d_embed) per-input, so the
    #              prefix is conditioned on the target vector. Strictly
    #              more expressive; same trainable shape elsewhere.
    prefix_mode: str = "static"
    prefix_mlp_hidden: int = 256

    # Multi-position injection: number of slots at which proj(v) is injected
    # (with independent linear projections per slot). 1 = original single-slot
    # design. >1 gives the backbone several independent views of v before the
    # learned prefix; intended to break the single-slot information bottleneck.
    num_inject_slots: int = 1

    # ---- (de)serialization helpers --------------------------------------

    def to_json(self, path: str | Path) -> None:
        """Write this config to a JSON file (used by HF model cards)."""
        Path(path).write_text(json.dumps(asdict(self), indent=2, sort_keys=True))

    @classmethod
    def from_json(cls, path: str | Path) -> "NLAConfig":
        """Load a config from JSON. Unknown keys are ignored for
        forward-compatibility with newer checkpoints."""
        data = json.loads(Path(path).read_text())
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
