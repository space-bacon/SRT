"""Adaptive-density scheduler: pick which tokens to verbalize.

Given a per-token divergence series d[0..T-1] and a budget N of verbalization
slots, return N token indices weighted by where divergence is *moving fast*.
The intuition: dense narration where the model's internal state is changing
rapidly, sparse where it's coasting.

Algorithm (equal-mass cumulative quantiles):

    1. mass[t] = max(d[t] - floor, eps)            # strictly positive weight
    2. cum[t]  = cumsum(mass)
    3. for i in range(N):
           target = (i + 0.5) / N * cum[-1]
           idx_i  = argmin_t |cum[t] - target|

This places one verbalization at each of N equal-mass quantiles of the
divergence trace, so dense regions get more samples than flat ones.

`floor` defaults to a fraction of the median, which keeps very-low-divergence
runs from being completely starved of samples (you still want at least one
verbalization in a long boring stretch).
"""

from __future__ import annotations

import math


def quantile_by_density(
    divergence: list[float],
    budget: int,
    *,
    floor_frac: float = 0.1,
    min_spacing: int = 1,
    skip_indices: set[int] | None = None,
) -> list[int]:
    """Return `budget` distinct token indices weighted by divergence mass.

    `min_spacing` rejects consecutive duplicate picks (which can happen on
    very peaky traces) by nudging to the next free index.
    `skip_indices` are excluded from being picked (use for whitespace /
    trivial tokens that carry no semantic state worth narrating).
    """
    T = len(divergence)
    if T == 0 or budget <= 0:
        return []
    skip = skip_indices or set()
    eligible = [t for t in range(T) if t not in skip]
    if budget >= len(eligible):
        return eligible

    # build positive mass with a small floor so flat regions still get hit;
    # skipped indices get zero mass so the cumsum naturally walks past them
    if T > 0:
        sorted_d = sorted(divergence)
        median = sorted_d[T // 2]
        floor = max(1e-6, floor_frac * (median if median > 0 else 1e-3))
    else:
        floor = 1e-6
    mass = [0.0 if t in skip else max(d, floor) for t, d in enumerate(divergence)]

    # cumulative
    cum = []
    s = 0.0
    for m in mass:
        s += m
        cum.append(s)
    total = cum[-1]

    picked: list[int] = []
    used: set[int] = set()
    eligible_set = set(eligible)
    for i in range(budget):
        target = (i + 0.5) / budget * total
        # binary search would be nicer, but T is small (hundreds)
        idx = min(eligible, key=lambda t: abs(cum[t] - target))
        # enforce min_spacing on eligible-only
        attempts = 0
        while (idx in used or any(abs(idx - u) < min_spacing for u in used)) and attempts < T:
            idx = (idx + 1) % T
            if idx not in eligible_set:
                attempts += 1
                continue
            attempts += 1
        if idx in eligible_set and idx not in used:
            picked.append(idx)
            used.add(idx)
    picked.sort()
    return picked
