"""Generate diagnostic plots from training + benchmark artifacts.

Plots:
  1. Training loss curves (train + val total + components)
  2. Per-layer divergence and injection norm trajectories
  3. r̂ envelope over training (mean, ±std, min, max)
  4. r̂ vs r_true scatter (from traces.json — limited to traced samples)

Outputs PNGs into artifacts/plots/
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ART = Path("/Users/burtron/development/srt-adapter/artifacts")
OUT = ART / "plots"
OUT.mkdir(exist_ok=True)

LOG = ART / "checkpoints/step94k/adapter_v3_stdout.log"
TRAIN_JSONL = ART / "checkpoints/step94k/train_log.jsonl"
BENCH_VAL = ART / "benchmark_step94k/metrics.json"
BENCH_CURATED = ART / "benchmark_curated_regen/metrics.json"
TRACES_VAL = ART / "benchmark_step94k/traces.json"
TRACES_CURATED = ART / "benchmark_curated_regen/traces.json"


# ───────── Parse train log + jsonl ─────────
train_rows = []
with open(TRAIN_JSONL) as f:
    for line in f:
        train_rows.append(json.loads(line))

# Parse VAL records from stdout log
val_pattern = re.compile(
    r"VAL step=(\d+)\s+total=([\d.]+)\s+ce=([\d.]+)\s+bif=([\d.]+)"
)
val_rows = []
for line in open(LOG):
    m = val_pattern.search(line)
    if m:
        val_rows.append({
            "step": int(m.group(1)),
            "total": float(m.group(2)),
            "ce": float(m.group(3)),
            "bif": float(m.group(4)),
        })


# ───────── Plot 1: loss curves ─────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Total loss
ax = axes[0, 0]
steps = [r["step"] for r in train_rows]
ax.plot(steps, [r["total"] for r in train_rows], color="#888", lw=0.5, alpha=0.5,
        label="train (raw)")
# Smoothed train
window = 20
if len(train_rows) > window:
    sm = [sum(r["total"] for r in train_rows[max(0, i-window):i+1]) / min(i+1, window+1)
          for i in range(len(train_rows))]
    ax.plot(steps, sm, color="C0", lw=1.2, label=f"train (MA{window})")
val_steps = [r["step"] for r in val_rows]
ax.plot(val_steps, [r["total"] for r in val_rows], color="C3", marker="o", ms=3,
        label="val")
ax.set_xlabel("step")
ax.set_ylabel("total loss")
ax.set_title("Total loss")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# CE
ax = axes[0, 1]
ax.plot(steps, [r["ce"] for r in train_rows], color="#888", lw=0.5, alpha=0.5)
if len(train_rows) > window:
    sm = [sum(r["ce"] for r in train_rows[max(0, i-window):i+1]) / min(i+1, window+1)
          for i in range(len(train_rows))]
    ax.plot(steps, sm, color="C0", lw=1.2, label="train")
ax.plot(val_steps, [r["ce"] for r in val_rows], color="C3", marker="o", ms=3,
        label="val")
ax.axhline(2.79, color="k", ls="--", lw=0.8, alpha=0.5, label="backbone floor")
ax.set_xlabel("step")
ax.set_ylabel("CE")
ax.set_title("Cross-entropy (backbone LM head)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Bif loss
ax = axes[1, 0]
ax.plot(steps, [r["bif"] for r in train_rows], color="#888", lw=0.4, alpha=0.4)
if len(train_rows) > window:
    sm = [sum(r["bif"] for r in train_rows[max(0, i-window):i+1]) / min(i+1, window+1)
          for i in range(len(train_rows))]
    ax.plot(steps, sm, color="C2", lw=1.2, label="train")
ax.plot(val_steps, [r["bif"] for r in val_rows], color="C3", marker="o", ms=3,
        label="val")
ax.set_xlabel("step")
ax.set_ylabel("bif loss")
ax.set_title("Bifurcation loss (r̂ smooth-L1)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Chain loss
ax = axes[1, 1]
chain_vals = [r.get("chain", 0) for r in train_rows]
ax.plot(steps, chain_vals, color="#888", lw=0.4, alpha=0.4)
if len(train_rows) > window:
    sm = [sum(r.get("chain", 0) for r in train_rows[max(0, i-window):i+1]) / min(i+1, window+1)
          for i in range(len(train_rows))]
    ax.plot(steps, sm, color="C4", lw=1.2)
ax.set_xlabel("step")
ax.set_ylabel("chain loss")
ax.set_title("Chain loss (interpretant-prediction)")
ax.grid(True, alpha=0.3)

fig.suptitle("SRT-Adapter v3 — training trajectory", fontsize=12, y=1.00)
fig.tight_layout()
fig.savefig(OUT / "01_loss_curves.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {OUT / '01_loss_curves.png'}")


# ───────── Plot 2: per-layer divergence + injection norms ─────────
# divergence_norms_per_layer / injection_norms_per_layer are summary stats only
# in metrics.json. The richer data is in train log: div_norms and inj_norms per row
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

ax = axes[0]
# train_rows have "div_norms" as list of 3 floats
div_arrays = defaultdict(list)
for r in train_rows:
    if "div_norms" in r and isinstance(r["div_norms"], list):
        for i, v in enumerate(r["div_norms"]):
            div_arrays[i].append((r["step"], v))
for li, pts in sorted(div_arrays.items()):
    if not pts:
        continue
    s, v = zip(*pts)
    ax.plot(s, v, lw=0.5, alpha=0.4, color=f"C{li}")
    if len(v) > window:
        sm = [sum(v[max(0, i-window):i+1]) / min(i+1, window+1) for i in range(len(v))]
        ax.plot(s, sm, lw=1.5, color=f"C{li}", label=f"layer {li} (MAH@{[7,14,21][li]})")
ax.axhline(1.0, color="k", ls="--", lw=0.6, alpha=0.5, label="target = 1.0")
ax.set_xlabel("step")
ax.set_ylabel("‖divergence‖")
ax.set_title("MAH divergence norms per layer")
ax.legend(fontsize=8, loc="lower right")
ax.grid(True, alpha=0.3)

ax = axes[1]
inj_arrays = defaultdict(list)
for r in train_rows:
    if "inj_norms" in r and isinstance(r["inj_norms"], list):
        for i, v in enumerate(r["inj_norms"]):
            inj_arrays[i].append((r["step"], v))
for li, pts in sorted(inj_arrays.items()):
    if not pts:
        continue
    s, v = zip(*pts)
    ax.plot(s, v, lw=0.5, alpha=0.4, color=f"C{li}")
    if len(v) > window:
        sm = [sum(v[max(0, i-window):i+1]) / min(i+1, window+1) for i in range(len(v))]
        ax.plot(s, sm, lw=1.5, color=f"C{li}", label=f"inject layer {li} (idx {[14,21][li]})")
ax.axhline(1.0, color="k", ls="--", lw=0.6, alpha=0.5, label="target = 1.0")
ax.set_xlabel("step")
ax.set_ylabel("‖injection‖")
ax.set_title("RRM injection norms per layer  [these turn out to be ineffective]")
ax.legend(fontsize=8, loc="lower right")
ax.grid(True, alpha=0.3)

fig.suptitle("SRT-Adapter v3 — internal signal norms", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "02_internal_norms.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {OUT / '02_internal_norms.png'}")


# ───────── Plot 3: r̂ envelope over training ─────────
fig, ax = plt.subplots(figsize=(11, 5))
steps3 = [r["step"] for r in train_rows if "r_hat_mean" in r]
mean = [r["r_hat_mean"] for r in train_rows if "r_hat_mean" in r]
std = [r["r_hat_std"] for r in train_rows if "r_hat_std" in r]
mn = [r["r_hat_min"] for r in train_rows if "r_hat_min" in r]
mx = [r["r_hat_max"] for r in train_rows if "r_hat_max" in r]

ax.fill_between(steps3, [m - s for m, s in zip(mean, std)],
                [m + s for m, s in zip(mean, std)],
                color="C0", alpha=0.25, label="±1 std")
ax.plot(steps3, mean, color="C0", lw=1.2, label="mean")
ax.plot(steps3, mn, color="C2", lw=0.6, alpha=0.6, label="batch min")
ax.plot(steps3, mx, color="C3", lw=0.6, alpha=0.6, label="batch max")
ax.axhline(1.0, color="k", ls="--", lw=0.5, alpha=0.4)
ax.axhline(-1.0, color="k", ls="--", lw=0.5, alpha=0.4)
ax.axhline(0, color="k", ls=":", lw=0.4, alpha=0.4)
ax.set_xlabel("step")
ax.set_ylabel("r̂")
ax.set_title("r̂ distribution envelope over training (per-batch)")
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "03_r_hat_envelope.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {OUT / '03_r_hat_envelope.png'}")


# ───────── Plot 4: r̂ vs r_true scatter from traces ─────────
def collect_pairs(traces_path):
    """Return (r_hat, r_true_compressed) pairs from masked tokens in traces."""
    rh, rt = [], []
    for tr in json.load(open(traces_path)):
        for h, t, m in zip(tr["r_hat"], tr["r_true"], tr["r_mask"]):
            if m:
                rh.append(h)
                # log-compress to match BEN training target
                sgn = 1 if t > 0 else (-1 if t < 0 else 0)
                rt.append(sgn * math.log1p(abs(t)))
    return rh, rt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, name, traces_path, bench_path in [
    (axes[0], "Validation", TRACES_VAL, BENCH_VAL),
    (axes[1], "Curated (regen)", TRACES_CURATED, BENCH_CURATED),
]:
    rh, rt = collect_pairs(traces_path)
    ax.scatter(rt, rh, s=3, alpha=0.25, color="C0")
    lim = [-0.5, 3.0]
    ax.plot(lim, lim, ls="--", color="k", alpha=0.4, lw=0.8, label="y=x")
    ax.set_xlim(lim)
    ax.set_ylim([-1.05, 1.05])
    ax.set_xlabel("r_true (log-compressed)")
    ax.set_ylabel("r̂  (BEN tanh output)")
    bench = json.load(open(bench_path))
    pearson = bench.get("pearson_r_hat_vs_r_true_compressed", float("nan"))
    n_traced = len(rh)
    ax.set_title(f"{name}\nPearson (full set) = {pearson:.3f}, scatter n = {n_traced}")
    ax.axhline(1.0, color="r", ls=":", lw=0.6, alpha=0.5, label="tanh saturation")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

fig.suptitle("r̂ vs r_true on traced subset of each benchmark", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "04_r_hat_vs_r_true.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {OUT / '04_r_hat_vs_r_true.png'}")

print("\nAll plots in:", OUT)
