---
license: apache-2.0
base_model: google/gemma-4-31B-it
tags:
  - interpretability
  - activation-verbalization
  - srt
  - gemma-4
  - multimodal
  - vision-language
---

# SRT-NLA Activation Verbalizer — gemma-4-31B-it (L47)

**Read the hidden states of a frozen gemma-4-31B-it in natural language —
whether the state came from text or from an image.**

These checkpoints are the generative component of the gemma-4 SRT
verbalization stack. Inject a layer-47 activation vector as a soft
prefix and the frozen backbone decodes candidate text for it. Layer 47
is the backbone's **cross-modal alignment peak**: image soft-tokens and
word tokens share one interpretant space there, so the same machinery
reads visual states and text states alike.

## What the stack does (zero training on any of it)

- **Image to caption.** An image's mean L47 state retrieves on-topic
  full-sentence captions from an open pool of 10,000 COCO captions:
  **5/5 CIFAR natural images at rank 1** (horse → "an old black and
  white photo with a person riding a horse", 0.680; dog 0.651; ship
  0.631). Live in the
  [SRT-Sunstone demo](https://huggingface.co/spaces/RiverRider/srt-sunstone).
- **Reports the texture that is present.** Given a random-dot autostereogram
  whose figure exists only in binocular disparity, the stack retrieves
  "An abstract mosaic of tiny colored squares", a faithful sentence-level
  report of what a flat encoder actually receives (paper §11.6.2–.3).
- **Reads text states at ceiling.** Centered replay 0.994 inside a fully
  calibrated anchor frame (NN retrieval 0.695, random floor 0.494), so
  every reported number is interpretable.
- **Generates verbalization candidates.** Sampled best-of-K from these
  checkpoints climbs monotonically with K (0.51 → 0.59 centered fve at
  K=32); the target vector is available at inference, so oracle scoring
  of candidates is free.

Companion artifacts (L47 retrieval indexes for captions and text,
anchors, eval results, source pools):
[`RiverRider/srt-nla-gemma4-artifacts`](https://huggingface.co/datasets/RiverRider/srt-nla-gemma4-artifacts).
Program: [github.com/space-bacon/SRT](https://github.com/space-bacon/SRT)
(`paper_nla.md` §11.6–§11.7).

## Which decode to use

| you want | use |
|---|---|
| the best text for a state (text or image query) | **NN retrieval** against the L47 indexes — fast, calibrated, powers the demo |
| generative candidates / paraphrase diversity | **this AV, sampled best-of-K** with oracle rerank (K=8–32) |
| greedy one-shot decoding | not a recommended mode on this backbone (see notes) |

This mirrors the program-wide finding: on every backbone studied, the
deployable decode is retrieval or best-of-K, never argmax.

## Files

| file | recipe |
|---|---|
| `ce/best_av.pt` | CE on gold tokens, np=16, corpus targets, 2 epochs |
| `draft/best_av.pt` | as above, plus in-context NN draft (kept for the paper's §11.7 ablation) |

## Measurement notes (reported openly; they shape the decode table)

- Argmax decoding mode-collapses on this chat-tuned host even though the
  injected vector demonstrably carries the content — it halves the gold
  text's cross-entropy (8.70 → 4.13 nats/token). The information is in
  the distribution; sampling surfaces it, argmax does not.
- The best-of-K slope (+0.017/doubling) matches gpt-oss-20b and is half
  of base-model Qwen2.5-7B's, supporting a base-vs-instruction-tuned
  verbalization hypothesis (testable on the gemma-4 base checkpoint).
- In-context drafts do not help decoding: an activation-space
  neighbour's text adds ~0.03 nats of predictive value for the gold
  text. Activation similarity is not token-space predictive utility.

## Reproduction protocol

- Targets must be encoded corpus text (`scripts/sample_targets.py
  --corpus`); chat-tuned gemma-4 degenerates under bare-BOS
  self-sampling.
- The backbone is BOS-sensitive: prefix-free re-encodes must prepend BOS
  (centered replay 0.615 without, 0.9986 with).
- Load with `ActivationVerbalizer` (`srt/nla/`), `num_prefix_tokens=16`,
  `extraction_layer=47`, backbone via `srt.nla.load_frozen_backbone`
  (`Gemma4ForConditionalGeneration`; `AutoModelForCausalLM` silently
  loads random weights for this architecture).
