# Natural-Language Activation (NLA) Verbalization:
## Probing the Decodability of Frozen Hidden States via Prefix-Tuned Generation

*Draft — May 2026*

---

## Abstract

We train a small prefix-tuning adapter (12.7M params) over a frozen Qwen-2.5-7B
to **verbalize** a single hidden activation $v \in \mathbb{R}^{3584}$ extracted at
layer 20: given $v$, the adapter generates a short text whose own layer-20
last-token hidden $h$ maximizes $\mathrm{fve\_nrm}(h, v) = \tfrac{1}{2}(1 + \cos(h, v))$.
On the raw metric the trained adapter appeared stuck at $\approx 0.689$ across
four architectural levers (multi-inject, MLP prefix, PG+KL, more data). We show
that this number is not a model ceiling but a **measurement artifact** of an
uncentered metric over an anisotropic representation. Under a held-out
*anisotropy-corrected* metric and against four oracle/baseline anchors (replay,
paraphrase, NN-retrieval, random-floor), the same adapter at best-of-64
**saturates the Qwen paraphrase ceiling**, $\rho_{\text{cen}} \approx 0.99$. The
real open problem is the **greedy gap**: deterministic decoding closes only
$\approx 28\%$ of the centered ceiling and is *beaten* by zero-training nearest-
neighbour retrieval ($\rho_{\text{cen}} \approx 0.70$). We argue this reframes
hidden-state verbalization as a *decoding* problem, not a capacity problem, and
that any fve_nrm-style evaluation must report (i) an anisotropy-centered metric
and (ii) a retrieval baseline to be interpretable.

---

## 1. Setting

- **Backbone.** Qwen/Qwen2.5-7B, bf16, fully frozen.
- **Probe layer.** $\ell=20$; targets $v$ = last-valid-token hidden state at
  layer 20 of a 64-token Qwen continuation.
- **Adapter.** `ActivationVerbalizer` with 16 static prefix tokens and one
  inject slot; 12.7M trainable parameters. Trained with token CE on
  $(v, \text{text})$ pairs.
- **Metric.** $\mathrm{fve\_nrm}(h, v) = \tfrac{1}{2}(1 + \cos(h, v))$. This is
  the per-target score throughout training, evaluation, and best-of-K.
- **Checkpoints.** `ce_seq64_np16/best_av.pt` (10k pairs), `ce_seq64_np16_30k/best_av.pt`
  (30k pairs), both warm-started identically.

## 2. The 0.689 puzzle

The legacy `probe_bestofn.py` script reported a best-of-64 score of
$0.689$ on the 10k checkpoint and $0.675$ on the 30k checkpoint. We tested
four levers, expecting at least one to move the number:

| lever | result |
|---|---|
| $M=4$ multi-inject slots | no change |
| MLP-conditioned prefix | no change |
| PG + KL fine-tune | no change |
| +30k training pairs | regression to 0.675 |

The natural conclusion was an architectural ceiling. **It was a measurement
artifact.** Two issues:

1. **Anisotropy.** Two unrelated Qwen-7B last-token L20 hiddens already have
   $\cos \approx 0.24$, i.e. $\mathrm{fve\_nrm} \approx 0.62$, driven entirely
   by a shared anisotropic mean $\mu$ with $\|\mu\| \approx 55$
   (`oracle_ceiling.py`). The "0.689" was only $\sim 0.07$ above the
   anisotropy floor.
2. **Scoring choice in `probe_bestofn.py`.** A re-implementation
   (`centered_eval.py`) that scores each of $K=64$ samples by
   $\mathrm{fve\_nrm}(h_k, v)$ and takes the per-target max recovers
   $\mathbf{0.85}$, not $0.689$, on the same checkpoint, same $K$, larger
   $M=200$. The legacy script was effectively measuring a different
   per-target statistic.

## 3. A clean reference frame

We anchor the metric with four reference points computed on a 200-target
held-out slice (`oracle_ceiling.py`):

| anchor | raw fve_nrm | centered fve_nrm | meaning |
|---|---|---|---|
| replay (oracle re-encode) | 0.973 | **0.968** | same text → upper bound |
| paraphrase best-of-8 (Qwen) | 0.848 | **0.799** | semantic ceiling |
| NN in-pool (pool=200) | 0.750 | 0.663 | retrieval at small pool |
| NN-retrieval (pool=2000) | 0.795 | 0.714 | retrieval at scale (`centered_eval.py`) |
| random floor (off-diagonal cos) | 0.622 | **0.510** | unrelated samples |

The centered metric subtracts $\mu$ from both sides before cosine. It
drops the random floor by $\sim 0.11$ and the paraphrase ceiling by
$\sim 0.05$, preserving ordering but stretching the dynamic range.

We then define normalized progress:

$$
\rho_{\text{cen}}(s) = \frac{s - \mathrm{rand}_{\text{cen}}}{\mathrm{para}_{\text{cen}} - \mathrm{rand}_{\text{cen}}}
= \frac{s - 0.510}{0.799 - 0.510} = \frac{s - 0.510}{0.289}.
$$

## 4. Adapter results, properly anchored

`centered_eval.py` on the M=200 target slice, K=64 samples, pool=2000:

| condition | raw fve_nrm | centered fve_nrm | $\rho_{\text{cen}}$ |
|---|---|---|---|
| **10k ckpt, greedy** | 0.694 | 0.589 | 0.27 |
| **10k ckpt, sampled (mean)** | 0.691 | 0.585 | 0.26 |
| **10k ckpt, best-of-64** | **0.846** | **0.788** | **0.96** |
| **30k ckpt, greedy** | 0.688 | 0.591 | 0.28 |
| **30k ckpt, sampled (mean)** | 0.684 | 0.582 | 0.25 |
| **30k ckpt, best-of-64** | **0.847** | **0.797** | **0.99** |
| NN-retrieval (pool=2000) | 0.795 | 0.714 | 0.71 |
| random floor | 0.622 | 0.510 | 0.00 |
| paraphrase ceiling | 0.848 | 0.799 | 1.00 |

Three observations:

1. **The adapter saturates the paraphrase ceiling at best-of-64.** Raw
   $\rho \approx 0.996$, centered $\rho \approx 0.99$. The four lever
   experiments weren't failing to break a wall; they were already at the wall.
2. **Greedy decoding is the real bottleneck.** Centered $\rho \approx 0.28$.
   Whatever the prefix encodes is far more decodable under stochastic search
   than under argmax.
3. **A zero-training nearest-neighbour lookup beats the greedy adapter.**
   $\rho_{\text{cen}}(\text{NN}) \approx 0.71 \gg 0.28$. The trained model
   does not even reach the retrieval baseline on its deterministic decode.

## 5. Implications

- **Verbalization is sampling-bound, not capacity-bound on this backbone.**
  Under a meaningful metric, 12.7M trainable parameters suffice to make the
  *space of paraphrases* reachable; the per-roll-out probability mass at the
  argmax mode is what fails.
- **Any future fve_nrm-style evaluation needs (i) anisotropy centering and
  (ii) a retrieval baseline.** The raw metric over an anisotropic backbone is
  not a faithful indicator of progress: a 0.05 gain on raw fve_nrm collapses
  to nothing once $\mu$ is subtracted, and may correspond to a trained model
  that still loses to a 1-line numpy NN lookup.
- **The interesting open question is the greedy gap**: closing
  $\rho_{\text{cen}}^{\text{greedy}}$ from 0.28 toward 0.99 without paying
  $K=64$ inference cost. Plausible directions: temperature distillation from
  best-of-K into greedy, length-conditioned decoding, or contrastive
  fine-tuning against retrieved hard negatives.

## 6. Artifacts

- `scripts/oracle_ceiling.py` — replay / random / NN / paraphrase, raw + centered.
  Output: `artifacts/nla/oracle_ceiling_30k_v2.json`.
- `scripts/centered_eval.py` — adapter greedy / sampled / best-of-K and
  NN-retrieval, raw + centered, on M target vectors with a held-out pool.
  Outputs: `artifacts/nla/centered_eval_{10k,30k}_M200.json`.
- Checkpoints: `artifacts/nla/ce_seq64_np16/best_av.pt`,
  `artifacts/nla/ce_seq64_np16_30k/best_av.pt`.
- Targets: `artifacts/nla/targets_q7b_L20_seq64_{10k,30k_seed1}.pt`.

## 7. Limitations

- Single backbone (Qwen-2.5-7B), single layer ($\ell=20$), single target
  type (64-token continuations). The anisotropy magnitude $\|\mu\| \approx 55$
  is backbone-specific; centering is required, but the size of the correction
  is not universal.
- Paraphrase ceiling is itself stochastic ($k=8$ samples per source); centered
  $\rho > 1$ would not be surprising at much higher $k$ — the ceiling is a
  *practical* upper bound on what a Qwen-shaped model can say differently.
- We did not re-run the four lever experiments under the centered metric; we
  hypothesize they would each show small but non-zero centered improvements
  that were invisible on the raw metric.
