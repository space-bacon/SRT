"""Data types returned by `srt_introspect.Trace.generate`."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Step:
    """A single observed step in the generated trace.

    `token_idx` is the position within the generated continuation (0-based).
    `divergence` is the aggregate metapragmatic-divergence norm at that token
    (sum over MAH layers, L2 norm of the divergence vector). Larger = the
    model's internal state is moving fast at this token.

    `regime` is the BEN regime classification (0=smooth, 1=bifurcating).

    `r_hat` is the BEN bifurcation probability in [0, 1].

    `verbalization` is the natural-language description of the L20 hidden
    state at this token, populated only for steps the adaptive-density
    scheduler selected. None for unselected steps.
    """

    token_idx: int
    token: str
    divergence: float
    regime: int
    r_hat: float
    verbalization: str | None = None
    # Per-MAH-layer L2 divergence norms, in the same order as
    # `SRTConfig.mah_layer_indices`. The aggregate `divergence` field above
    # is the sum of these. Empty if the adapter has no MAH layers.
    per_layer_divergence: list[float] = field(default_factory=list)
    # Shannon entropy (nats) of the next-token distribution produced by
    # the backbone for this position. Useful for comparing the adapter's
    # divergence signal against raw predictive uncertainty.
    entropy: float = 0.0
    # Round-trip fidelity: cosine between the original L20 hidden state at this
    # token and the L20 hidden state obtained by re-encoding the generated
    # verbalization text. Populated only for verbalized steps. Self-validates
    # the "this is what the model was thinking" claim. None if not verbalized.
    # NOTE: raw cosine on this backbone is anisotropy-dominated (~0.24 for
    # unrelated text); interpret against the paraphrase ceiling, not 0.
    roundtrip_cos: float | None = None


@dataclass
class Result:
    text: str
    steps: list[Step] = field(default_factory=list)

    def selected(self) -> list[Step]:
        """Just the steps that received a verbalization."""
        return [s for s in self.steps if s.verbalization is not None]
