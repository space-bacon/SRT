---
title: SRT Showcase
emoji: 🧭
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: "6.17.3"
app_file: app.py
python_version: "3.10"
pinned: false
hardware: zerogpu
short_description: Watch a frozen Qwen-2.5-7B think — live SRT introspection
---

# SRT Showcase — watch a frozen model think

Live, token-by-token introspection of **Qwen-2.5-7B + the SRT adapter**.

As the model generates, every token is tinted by its predictive **entropy** (the
validated uncertainty signal). At the highest-effort token positions, chosen by
an adaptive-density scheduler, the **Activation Verbalizer** decodes the model's
internal hidden state into natural language, and each verbalization carries a
**round-trip fidelity badge**: it is re-encoded and compared back to the original
hidden state, so the "this is what the model was thinking" claim is visibly
self-validating.

Features:

- Live token stream tinted by entropy or SRT divergence, with per-token hover
  rollovers (entropy, divergence, reflexivity `r̂`, regime).
- Running entropy meter and entropy / divergence charts.
- Expand/collapse verbalization cards with round-trip fidelity badges.
- A/B panel: the same prompt with SRT injection on vs off (bare backbone),
  seeded identically.
- A curated example gallery covering confident recall, false premises,
  misconceptions, reasoning pivots, genuine uncertainty, and safety boundaries.

## Honest scope

Entropy is the load-bearing uncertainty signal. The SRT side-channels
(divergence, `r̂`, regime) and the verbalizations are shown as **observational
readouts** of internal state. This is a window into the model, not a validated
hallucination detector.

## Notes

- First request is cold (~60–90 s) while ZeroGPU acquires a GPU and the ~16 GB
  backbone weights load; subsequent requests are warm.
- A second backbone copy is loaded for the Activation Verbalizer.

Source: <https://github.com/space-bacon/SRT>
