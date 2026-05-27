"""Adaptive-density reasoning-trace sidecar for SRT-instrumented backbones.

Public API:

    from srt_introspect import Trace
    t = Trace.load("Qwen/Qwen2.5-7B")
    result = t.generate("What killed the dinosaurs?", max_new_tokens=200, budget=12, k=8)
    print(result.text)
    for step in result.steps:
        print(step.token_idx, step.token, step.divergence, step.regime, step.verbalization)

Read-only at inference: never modifies the host model.
"""

from .types import Step, Result
from .trace import Trace

__all__ = ["Trace", "Step", "Result"]
