#!/usr/bin/env python3
"""What does clipping a noisy gain at zero do to a leaderboard ranking?

A composite of the form (1 - trap) + max(0, delta) looks harmless, but the clip
is a nonlinearity applied to an estimate. By Jensen, E[max(0, d_hat)] exceeds
max(0, E[d_hat]), so a model whose adapter does nothing at all still earns a
positive contribution as long as its gain was measured with error. The noisier
the measurement, the larger the reward, which inverts the incentive the metric
is trying to create.

For a zero-true-gain model the bias is exactly sigma/sqrt(2*pi).

    python scripts/clip_bias_sim.py
"""
import argparse
import json

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=400_000)
    p.add_argument("--sigmas", default="0.005,0.01,0.02,0.05")
    p.add_argument("--true-gain", type=float, default=0.010)
    p.add_argument("--out", default="artifacts/nla/clip_bias_sim.json")
    return p.parse_args()


def main():
    a = parse_args()
    rng = np.random.default_rng(0)
    rows = []
    for s in [float(x) for x in a.sigmas.split(",")]:
        d = rng.normal(0.0, s, a.n)
        empirical = float(np.maximum(0, d).mean())
        # A genuine small gain, against an adapter that does nothing.
        zero = np.maximum(0, rng.normal(0.0, s, a.n))
        real = np.maximum(0, rng.normal(a.true_gain, s, a.n))
        rows.append({
            "sigma": s,
            "bias_at_zero_gain": round(empirical, 5),
            "analytic_sigma_over_sqrt_2pi": round(s / np.sqrt(2 * np.pi), 5),
            "p_zero_gain_outranks_true_gain": round(float((zero > real).mean()), 4),
            "pct_zero_gain_outranks_true_gain": round(100 * float((zero > real).mean()), 1),
        })
        print(f"sigma {s:.3f}  bias {empirical:.4f}  "
              f"analytic {s/np.sqrt(2*np.pi):.4f}  "
              f"zero-gain wins {100*(zero > real).mean():.1f}%")

    out = {
        "metric": "(1 - trap_rate) + max(0, delta)",
        "true_gain_compared_against": a.true_gain,
        "n_draws": a.n,
        "finding": "clipping a noisy gain at zero gives a model with no real "
                   "adapter benefit a positive expected contribution of "
                   "sigma/sqrt(2*pi), so noisier measurement scores higher",
        "rows": rows,
    }
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
