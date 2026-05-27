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


@dataclass
class Result:
    text: str
    steps: list[Step] = field(default_factory=list)

    def selected(self) -> list[Step]:
        """Just the steps that received a verbalization."""
        return [s for s in self.steps if s.verbalization is not None]
