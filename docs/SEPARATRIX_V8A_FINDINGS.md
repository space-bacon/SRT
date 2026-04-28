# Separatrix-illusion probe · v8a findings

**Date:** 2026-04-28
**Setup:** SRT adapter v8a (14.5M params, `use_prototypes=False`) on frozen
Qwen2.5-7B (28 layers). MAH taps at L7, L14, L21. Community discovery at L4.
**Probe:** [data/probes/separatrix_illusion_v1.jsonl](../data/probes/separatrix_illusion_v1.jsonl)
(61 items × 4 branches = 244 forwards).
**Hardware:** vast.ai A6000 (49 GB), bf16 forwards, max_seq_len=128.
**Wall-clock:** 13 s for the full sweep.
**Artifacts:** [readouts.jsonl](../artifacts/separatrix/v8a/readouts.jsonl),
[summary.json](../artifacts/separatrix/v8a/summary.json).
**Predictions tested:** see
[data/probes/SEPARATRIX_ILLUSION_README.md](../data/probes/SEPARATRIX_ILLUSION_README.md).

## Headline

| prediction | direction | result | p (two-sided Wilcoxon) | frac+ |
|---|---|---|---:|---:|
| (1) Community separation: cos(tech,bedrock) > cos(tech,mystic) | predicted ↑ | **CONFIRMED** | 0.031 | 0.62 |
| (3) BEN r_hat(prompt) > 0 | predicted ↑ | **CONFIRMED (unanimous)** | — | 61/61 |
| (4) MAH peak(tech+mystic) > MAH peak(tech+bedrock) | predicted ↑ | **FALSIFIED (strong opposite)** | 3.8 × 10⁻¹¹ | 0.05 |

## (1) Community separation — confirmed, weakly

- mean cos(tech, mystic)   = **0.9052**
- mean cos(tech, bedrock)  = **0.9167**
- mean cos(mystic, bedrock) = 0.8835

The community vectors are *tightly clustered* (all pairwise means > 0.88,
consistent with v8a's continuous community-encoder geometry — no quantisation
to discrete prototypes). Within that tight cluster, the predicted ordering
holds: bedrock continuations land closer to the technical pole than mystical
ones do, by a small but real margin. 62% of the 61 items obey the ordering;
a two-sided signed-rank test rejects the null (p=0.031). This is consistent
with the §6.5 reframing: the community head is the *alignment axis*, and on
material that hovers near a separatrix between technical and mystical
register, that axis nudges in the predicted direction.

## (3) BEN r_hat on prompts — confirmed unanimously

- mean r_hat over prompt tokens = **1.121** (single-precision floats; ref=0)
- 61 / 61 prompts exceed reference

Every prompt in the separatrix battery activates the BEN regime above the
neutral reference. This is the "activity axis" reading from §6.5: prompts
that *invite* a tech-vs-mystic-vs-bedrock disambiguation are interpretively
loaded by the activity gauge regardless of which continuation is later
attached. Note the ref=0 baseline is conservative (BEN's mean output on
generic prose was not measured here); a stronger statement requires
calibrating ref_rhat on a generic-text reference set in v2.

## (4) MAH peak on continuations — falsified, strongly

- mean MAH peak (continuation = mystical) = **1.493**
- mean MAH peak (continuation = bedrock)  = **2.072**

The MAH layer fires *more* on (technical-prompt + bedrock-physics
continuation) than on (technical-prompt + mystical continuation), in 95% of
the 61 items, p = 3.8 × 10⁻¹¹.

This is the cleanest falsification in the SRT-adapter results so far, and it
inverts the naïve "MAH should peak when interpretive worlds collide" reading.
Three readings to test:

1. **Bedrock IS the harder collision.** The mystical continuations in this
   probe are mostly low-pressure stylistic drift ("flows in the great
   fabric of being…"). They don't actually *commit* to an alternate
   ontology — they hand-wave. Bedrock continuations, by contrast, do commit:
   they pull in real Wittgensteinian / Quinean / Cavellian moves that
   require the model to *coordinate* technical vocabulary with a meta-level
   register. MAH may be detecting that coordination work, not the
   technical/mystical mismatch we hypothesised.

2. **The mystical branch is in-distribution slop.** Qwen-7B is heavily
   exposed to mysticism-as-aesthetic on the open web; the pattern is so
   well-rehearsed that it produces little divergence between attention-head
   "what was likely" and the actual continuation. Bedrock-philosophy text
   is rarer and harder to predict in context, so divergence runs higher.

3. **Probe construction effect.** All bedrock continuations were authored
   by the same hand and may be systematically longer / denser / more
   technical-jargon-rich than mystical ones, biasing the divergence norm.
   v2 should length-balance and re-author from a different source.

Whichever reading wins, **MAH is not measuring "register collision" the way
we wrote it down.** That's a load-bearing finding for the §6 interpretation
of MAH and needs to flow back into v9 or the paper draft.

## What this changes

- **§6.5 stands.** The activity / alignment decomposition predicts (1) and
  (3); both held. Reasonable to keep the framing.
- **§6 MAH section needs revision.** The "interpretive divergence" reading
  is at least incomplete and probably wrong on this probe. Replace with a
  weaker claim ("MAH measures local prediction surprise modulated by attention
  geometry") and cite the separatrix result as a constraint on stronger
  semantic readings.
- **Pred (2) (counterfactual decoding under forced prototype) still
  outstanding.** v8a does not expose a prototype-forcing codepath; v9 should
  add one as a first-class API so we can test the strongest version of the
  separatrix claim.

## Reproducibility

```bash
HF_HOME=/path/to/hf_cache HF_HUB_OFFLINE=1 \
PYTHONPATH=/path/to/srt-adapter \
python3 scripts/separatrix_eval.py \
  --adapter /path/to/srt-adapter-v8a/adapter.pt \
  --probe data/probes/separatrix_illusion_v1.jsonl \
  --output artifacts/separatrix/v8a/readouts.jsonl \
  --max-seq-len 128
```

Self-test (no GPU): `python3 scripts/separatrix_eval.py --self-test`.

## Caveats

- n=61 items, single hand-authored battery (one author, one pass).
- Bedrock continuations not length-matched to mystical.
- Reference r_hat=0 is conservative; calibrating against generic prose
  would strengthen pred (3).
- v8a community head is continuous (no discrete prototypes); cosine
  separations are necessarily small in absolute magnitude.
- bf16 forwards on a single A6000; deterministic enough for these
  population-level statistics but per-item readouts may shift in fp32.
