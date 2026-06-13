# Changelog

All notable changes to this project will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **SRT adapter ported to a frozen Qwen3-235B-A22B (22B-active MoE) backbone.**
  First SRT read-out adapter on a frontier-scale host. Phase-A (read-only)
  checkpoint released as
  [`RiverRider/srt-adapter-qwen3-235b`](https://huggingface.co/RiverRider/srt-adapter-qwen3-235b).
  Held-out probe (3000 rows): regime ECE 0.0005 / AUROC 0.9859, community
  NMI 0.6247, r̂ Pearson 0.751.
- `read_only` Phase-A training mode in `srt/adapter.py` and `scripts/train.py`:
  the backbone runs forward-only under `no_grad`; only the ~15.9M SRT head
  params train on detached taps. Cost is nearly independent of backbone size.
- Device-aware sharded backbone support (`device_map="auto"`,
  `set_head_device`) so the manual layer loop runs across multiple GPUs; SRT
  heads pinned to one device, taps/injections routed across layer boundaries.
- `scripts/phaseA_probe.py`: single-pass, sharding-aware held-out probe
  (regime ECE/AUROC/Brier, r̂ MAE/Pearson, community NMI/ARI, divergence norms).
  Persists raw arrays + partial JSON before clustering so an expensive forward
  pass is never lost to a cheap post-processing failure.

### Fixed
- SDPA `is_causal` parity on deep MoE backbones: passing an explicit additive
  causal mask diverged from the backbone's own `is_causal` fast path, and the
  bf16 epsilon was amplified by 94-layer discrete expert routing into real
  logit flips. Now pass `attention_mask=None` when there is no padding mask,
  making the manual loop byte-identical to the HF forward.
- bs=128 validation OOM on the sharded 235B: chunked cross-entropy (16-row
  chunks) plus freeing the previous batch's logits per validation iteration.
- Two moderate Dependabot alerts: `starlette>=0.40.0` (CVE-2024-47874 multipart
  DoS) and `jinja2>=3.1.6` (CVE-2024-56326 sandbox breakout).

## [1.0.0] - 2025-04-28

### Added
- Initial public release of the SRT-Adapter source.
- `srt/` package: `adapter`, `config`, modules (`mah`, `rrm`, `ben`, `community`),
  data loaders, training losses.
- Reference training script (`scripts/train.py`) and benchmarking utilities.
- `examples/` for loading the released adapter and scoring / encoding text.
- Smoke tests under `tests/`.
- Released checkpoints on Hugging Face:
  - `RiverRider/srt-adapter-v1.0` (first versioned release)
  - `RiverRider/srt-adapter-v8a` (encoder-as-community headline run)
- Live demos:
  - `RiverRider/srt-adapter-v1.0-demo`
  - `RiverRider/srt-adapter-v8a-demo`
- Apache-2.0 license; CITATION.cff.
