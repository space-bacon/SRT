# Data directory

This directory contains a small public slice used for reproducing the
benchmarks reported in the paper:

- `val_200.jsonl` — 200-sample validation slice for smoke tests and
  `scripts/benchmark.py` quick runs.
- `archetypes.json`, `archetype_topics_heldout_v1.json` — out-of-distribution
  archetype taxonomy used in §5.8 / §5.9 evaluation.
- `probes/` — separatrix and TakensHead probe batteries (§6.9).

## Held-back assets

The following are referenced in the paper but **not** included in the public
release. They constitute the data and pipeline IP behind the next-generation
checkpoint and are available only through commercial licensing:

- The Reddit Discourse Corpus (1M train / 100K val, 35 communities,
  per-token reflexivity / chain / community annotations) — paper §4.1,
  cited as Lancaster (2026c).
- The per-token annotation pipeline that produces the above from raw
  Reddit text.
- The C1 scholarly corpus (`data/corpus_c1/`) targeted for v9 — sources,
  manifests, builder, and scored splits.
- The teacher-distillation labelling pipeline used for v21a / v22b
  (`mixedbread-ai/mxbai-embed-large-v1` cosine scoring of NLI + STSB
  pair pools).
- Next-generation training recipes beyond v22c\_a050.

The released checkpoints (v8a, v15a / `srt-adapter-v1.0`, v22c\_a050) are
sufficient to reproduce every numerical claim in the paper using the
included evaluation harness; no held-back asset is required at evaluation
time.
