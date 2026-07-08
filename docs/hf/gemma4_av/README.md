---
license: apache-2.0
base_model: google/gemma-4-31B-it
tags:
  - interpretability
  - activation-verbalization
  - srt
  - gemma-4
  - negative-result
---

# SRT-NLA Activation Verbalizer — gemma-4-31B-it (L47)

Two Activation Verbalizer checkpoints trained to decode layer-47 hidden
states of a frozen `google/gemma-4-31B-it` into text, released primarily
as a documented **negative result** for the cross-backbone decoding-gap
comparison (paper §11.7, [github.com/space-bacon/SRT](https://github.com/space-bacon/SRT),
`paper_nla.md`).

## The honest headline

On this chat-tuned host the CE-trained verbalizer **mode-collapses under
greedy decoding**: it emits one degenerate loop for every target and
scores at the random floor (centered fve 0.500 vs floor 0.494), even
though the injected vector demonstrably carries the content (it halves
the gold text's cross-entropy, 8.70 → 4.13 nats/token). Best-of-K oracle
rerank climbs only +0.017 per doubling of K (0.591 at K=32), the same
slope as gpt-oss-20b and half of Qwen2.5-7B's, and never reaches the
zero-training NN-retrieval baseline (0.695).

**Use retrieval instead.** The recommended decoder for this backbone is
nearest-neighbour retrieval against the L47 indexes in
[`RiverRider/srt-nla-gemma4-artifacts`](https://huggingface.co/datasets/RiverRider/srt-nla-gemma4-artifacts),
which also powers the open-vocabulary caption retrieval in the
[SRT-Sunstone demo](https://huggingface.co/spaces/RiverRider/srt-sunstone).

## Files

| file | recipe | val (centered fve) |
|---|---|---|
| `ce/best_av.pt` | CE on gold tokens, np=16, corpus targets, 2 ep | greedy 0.50 (=floor), best-of-32 0.59 |
| `draft/best_av.pt` | + NN text draft in context (retrieval-then-edit) | greedy 0.51, copy baseline 0.71 (ignored the draft) |

The draft recipe is refuted with a mechanism: a four-way CE decomposition
shows the activation-space neighbour's text contributes 0.03 nats of
predictive value for the gold text, so CE training has no gradient toward
using it. Activation-space similarity is not token-space predictive
utility.

## Protocol notes (required to reproduce)

- gemma-4-it degenerates under bare-BOS self-sampling; targets must be
  encoded corpus text (`scripts/sample_targets.py --corpus`).
- The backbone is BOS-sensitive: prefix-free re-encodes must prepend BOS
  (replay 0.615 without, 0.9986 with).
- Anchors (L47): replay 0.994 / NN 0.695 / floor 0.494.
- Load with `ActivationVerbalizer` (`srt/nla/`), `num_prefix_tokens=16`,
  `extraction_layer=47`, backbone via `srt.nla.load_frozen_backbone`
  (Gemma4ForConditionalGeneration; AutoModelForCausalLM silently loads
  random weights for this architecture).
