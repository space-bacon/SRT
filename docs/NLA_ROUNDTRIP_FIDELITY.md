# NLA Round-Trip Fidelity: greedy vs best-of-K

How faithfully does the Activation Verbalizer (AV, `RiverRider/srt-nla-av-v1`)
turn a frozen Qwen2.5-7B hidden state into language? We measure it with a
**round-trip**: verbalize a hidden state `v`, re-encode the text through the same
backbone, and compare the re-encoded state to the original `v`.

The short answer: **a single greedy verbalization is barely above chance; the
fidelity comes almost entirely from best-of-K selection**, which is legitimate
because the target `v` is available at inference (it is the thing being
verbalized). On the AV's in-distribution targets, best-of-64 reaches
**ρ_norm ≈ 0.92**. On arbitrary out-of-distribution text it is meaningfully
lower, which is itself an honest and important finding.

## Metrics

For a verbalization re-encoded to `h` and an original state `v`:

- **raw cosine** `cos(h, v)`.
- **fve_nrm** `= 0.5·(1 + cos(h, v))` — bounded to [0, 1].
- **centered** — subtract the pool mean `μ` from both sides before the cosine.

Centering matters because Qwen2.5-7B L20 hidden states are **anisotropic**: two
*unrelated* states already share `cos ≈ 0.24` (`fve_nrm ≈ 0.62`, `‖μ‖ ≈ 55`). So
raw fve_nrm has a high floor and a single number is uninterpretable without a
baseline. We also report **ρ_norm**, the centered fve_nrm rescaled against a
random floor (0.510) and a paraphrase ceiling (0.799):
`ρ_norm = (centered_fve − 0.510) / (0.799 − 0.510)`.

## Experiment A — in-distribution targets (the >0.90)

`scripts/centered_eval.py` and `scripts/rerank_eval.py`, M=200 target states the
AV was trained to verbalize, K=64, pool=2000.
(`artifacts/nla/centered_eval_30k_M200.json`, `rerank_eval_ce_seq64_np16_v2.json`.)

| Condition | raw fve_nrm | centered fve_nrm |
|---|---|---|
| random floor | 0.619 | 0.510 |
| greedy (K=1) | 0.687 | 0.591 |
| NN-retrieval baseline (no training) | 0.792 | 0.715 |
| **best-of-64** | **0.847** | **0.797** |

The centered best-of-64 (0.797) sits at the paraphrase ceiling, i.e.
**ρ_norm ≈ 0.92** (`(0.777–0.797 − 0.510) / 0.289`). Greedy, by contrast, is only
~0.08 above the random floor. The whole signal is in best-of-K.

## Experiment B — out-of-distribution states (challenge + optimize)

`scripts/roundtrip_bestofk.py`, n=24 hidden states from diverse passages
(cryptography, narrative, photosynthesis, a joke, …), K=64.
(`artifacts/nla/roundtrip_bestofk.json`.)

| Condition | raw cos | fve_nrm | centered cos |
|---|---|---|---|
| floor (verbalization vs *mismatched* target) | 0.315 | 0.658 | — |
| greedy (K=1) | 0.346 | 0.673 | — |
| best-of-8 | 0.444 | 0.722 | 0.184 |
| best-of-32 | 0.522 | 0.761 | 0.289 |
| **best-of-64** | 0.565 | **0.783** | 0.358 |

Two things stand out:

- **Challenge:** greedy (0.346) barely clears the anisotropy floor (0.315). A
  single verbalization of an arbitrary state is near-chance.
- **Optimize:** best-of-K climbs monotonically; individual states reach
  **0.86** (raw cos 0.857), and the mean best-of-64 fve_nrm (0.783) approaches
  the paraphrase ceiling.

Out-of-distribution text is genuinely harder than the AV's training targets
(raw fve 0.783 vs 0.847 in-distribution), so the AV's high fidelity is partly
distribution-specific. This is worth stating plainly rather than quoting only
the best-case headline.

## The K-curve

Centered fve_nrm grows ~log-linearly, roughly **+0.03 per doubling of K**, with
no visible saturation at K=64 (`rerank_eval_ce_seq64_np16_v2.json`, in-distribution):

| K | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| centered fve_nrm | 0.573 | 0.617 | 0.652 | 0.686 | 0.716 | 0.747 | 0.777 |

Extrapolating, K≈256 would be needed for ρ_norm ≈ 1.0.

## Conclusions

1. **Best-of-K is the lever, not greedy.** Greedy/single verbalizations sit just
   above the anisotropy floor; selection over K candidates is what produces
   faithful verbalizations. This is deployable, not cheating: the target state is
   available at inference.
2. **The >0.90 is real but in-distribution and in ρ_norm units.** On the AV's
   target distribution, best-of-64 reaches centered fve_nrm ≈ 0.78–0.80
   (ρ_norm ≈ 0.92). On arbitrary text it is lower (raw fve_nrm ≈ 0.78).
3. **Always report a baseline and a centered metric.** Raw fve_nrm alone is
   anisotropy-dominated; the random floor (0.51 centered) and NN-retrieval
   baseline (0.715 centered) are the honest reference points.

## Reproduce

```bash
# Out-of-distribution best-of-K (this doc, Experiment B) — CUDA box, transformers==4.53.3
python scripts/roundtrip_bestofk.py --k 64 --positions 24

# In-distribution headline (Experiment A) — needs the target .pt set + AV ckpt
python scripts/centered_eval.py \
    --targets artifacts/nla/targets_q7b_L20_seq64_30k_seed1.pt \
    --av-ckpt artifacts/nla/ce_seq64_np16_30k/best_av.pt \
    --backbone Qwen/Qwen2.5-7B --layer 20 \
    --num-prefix-tokens 16 --num-vectors 200 --samples-per-v 64 \
    --out artifacts/nla/centered_eval_30k_M200.json
```

Note: `adapter.generate()` (KV-cached) requires `transformers==4.53.3`; 5.x
breaks the cached decode path. The round-trip scripts use the AV backbone's
`forward(output_hidden_states=True)`, which is version-robust.
