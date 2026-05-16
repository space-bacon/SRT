"""Configuration for SRT-NLA."""

from __future__ import annotations

from dataclasses import dataclass


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
    gamma_entropy: float = 0.05  # entropy floor hinge
    delta_community: float = 0.2  # community-consistency bonus
    h_min: float = 1.5  # minimum acceptable token-level entropy (nats)

    # Injection layout — number of learned prefix tokens after the injected
    # vector. The injected vector occupies slot 0; this many extra learned
    # embeddings (BOS-like) follow before the model autoregresses.
    num_prefix_tokens: int = 1
