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
) -> list[int]:
    """Return `budget` distinct token indices weighted by divergence mass.

    `min_spacing` rejects consecutive duplicate picks (which can happen on
    very peaky traces) by nudging to the next free index.
    """
    T = len(divergence)
    if T == 0 or budget <= 0:
        return []
    if budget >= T:
        return list(range(T))

    # build positive mass with a small floor so flat regions still get hit
    if T > 0:
        sorted_d = sorted(divergence)
        median = sorted_d[T // 2]
        floor = max(1e-6, floor_frac * (median if median > 0 else 1e-3))
    else:
        floor = 1e-6
    mass = [max(d, floor) for d in divergence]

    # cumulative
    cum = []
    s = 0.0
    for m in mass:
        s += m
        cum.append(s)
    total = cum[-1]

    picked: list[int] = []
    used: set[int] = set()
    for i in range(budget):
        target = (i + 0.5) / budget * total
        # binary search would be nicer, but T is small (hundreds)
        idx = min(range(T), key=lambda t: abs(cum[t] - target))
        # enforce min_spacing
        while idx in used or any(abs(idx - u) < min_spacing for u in used):
            idx += 1
            if idx >= T:
                idx = 0
                while idx in used and idx < T:
                    idx += 1
                if idx >= T:
                    break
        if idx < T:
            picked.append(idx)
            used.add(idx)
    picked.sort()
    return picked
