# Changelog

All notable changes to this project will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Open-vocabulary caption retrieval on the gemma-4 visual channel**
  (`paper_nla.md` §11.6.3): an image's mean L47 state retrieves full sentences
  from 10k COCO captions with zero training (5/5 CIFAR natural images on-topic
  at rank 1; per-category up to 0.778). Live in the Sunstone Space as a third
  read-out panel. Scripts: `gemma4_vision_retrieval.py`,
  `augment_gallery_captions.py`; L47 retrieval indexes released in
  [`RiverRider/srt-nla-gemma4-artifacts`](https://huggingface.co/datasets/RiverRider/srt-nla-gemma4-artifacts).
- **Greedy-gap campaign on gemma-4-31B-it** (`paper_nla.md` §11.7): fourth
  backbone for the decoding-gap comparison. CE verbalizer mode-collapses under
  argmax despite the injected vector halving gold CE; draft-conditioned
  decoding refuted with a four-way CE decomposition (activation similarity ≠
  predictive utility); K-curve slope +0.017/doubling patterns with gpt-oss,
  not Qwen, supporting a base-vs-instruction-tuned verbalization hypothesis.
  New tooling: `train_nla_draft.py`, `nla_ce_decomp.py`, `nla_decode_probe.py`,
  `nla_anchors.py` (backbone-generic), multimodal-aware
  `srt/nla/backbones.py` loader, corpus-encode mode in `sample_targets.py`.
  Checkpoints: [`RiverRider/srt-nla-av-gemma4`](https://huggingface.co/RiverRider/srt-nla-av-gemma4).
- **SRT-Sunstone: cross-modal read-out on a frozen gemma-4-31B-it.** A 12.3M
  community read-out head trained on text alone reads images zero-shot
  (CIFAR-10 image→word retrieval@1 = 0.93, chance 0.10; alignment peaks ~80%
  depth, collapses at the final layer). Model:
  [`RiverRider/Gemma-4-31B-it-SRT-Sunstone`](https://huggingface.co/RiverRider/Gemma-4-31B-it-SRT-Sunstone).
  Live Space: [`RiverRider/srt-sunstone`](https://huggingface.co/spaces/RiverRider/srt-sunstone)
  (source under `demo/cross_modal_space/`). Paper: `paper_nla.md` §11.6–§11.6.1.
- **Autostereogram boundary-condition study** (`paper_nla.md` §11.6.2): the
  read-out honestly reports texture on a random-dot stereogram (figure exists
  only in binocular disparity); a simulated binocular-fusion front-end
  (`scripts/stereo_decode.py`) recovers the hidden figure, after which both
  the generative caption and the read-out name it. New scripts:
  `make_stereogram.py`, `stereo_decode.py`, `stereogram_readout.py`,
  `gemma_caption.py`, `make_stereo_figure.py`; artifacts under
  `artifacts/nla/gemma4/stereo/`.
- **SRT adapter ported to frozen gpt-oss-20b (MXFP4 MoE), Phase A+B.**
  Held-out probe: regime ECE 0.0009 / AUROC 0.9742, r̂ Pearson 0.689,
  community NMI 0.4226. Full NLA pipeline run (targets → pairs → AV → 4096-code
  VQ codebook); the AV is an honest negative on this backbone (best-of-64
  centered fve 0.642 < NN-retrieval 0.744), so the codebook / NN retrieval is
  the recommended decoder. Releases:
  [`RiverRider/srt-adapter-gptoss20b`](https://huggingface.co/RiverRider/srt-adapter-gptoss20b),
  [`RiverRider/srt-nla-av-gptoss20b`](https://huggingface.co/RiverRider/srt-nla-av-gptoss20b),
  [`RiverRider/srt-nla-gptoss20b-artifacts`](https://huggingface.co/datasets/RiverRider/srt-nla-gptoss20b-artifacts).
  Live trace Space:
  [`RiverRider/srt-nla-gptoss20b-trace`](https://huggingface.co/spaces/RiverRider/srt-nla-gptoss20b-trace).
- Gated sliding-window attention-mask support in `srt/adapter.py`
  (autodetected via `config.layer_types`), required for gpt-oss backbones;
  bit-exact parity vs the HF forward (`tests/test_gptoss_smoke.py`,
  `docs/PORTING_GPT_OSS_120B.md`).
- State-identity red-teaming instrument and studies on gpt-oss-20b
  (`scripts/redteam_states.py`, word-category / token-structure studies;
  `paper_nla.md` §11.5).
- Repository mirrored to
  [`space-bacon/SRT-Sunstone`](https://github.com/space-bacon/SRT-Sunstone)
  as a standalone public home for the Sunstone line.
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
