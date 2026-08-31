# Changelog

All notable changes to this project will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`srt_select`: choosing among K model replies, with no test suite and no
  training.** Run every candidate on synthesised inputs, group by the outputs
  they produce, return a member of the largest group. The entry point and the
  argument shapes are recovered from the candidates themselves, so it takes a
  chat turn rather than a benchmark row. On HumanEval it moves a random single
  draw from 0.1868 to **0.4426** against an oracle of 0.4954, and on MBPP from
  0.7185 to **0.8174** against 0.8887, which is 82.9% and 58.1% of the available
  headroom. It beats a verifier trained on 47,232 execution labels (0.2639 on
  HumanEval, and that verifier does not transfer to MBPP). Recovering the target
  from the chat turn alone costs coverage rather than accuracy: 55.0% of
  HumanEval problems resolve and 98.6% of MBPP ones, giving 0.3096 and 0.7991.
  Packaged with a CLI, a sandbox module documenting its own limits, and a test
  pinning it against the harness that produced the published numbers.
- **`hivemind_census.py`**, which recomputes the corpus census and the
  selection-decay slope so both can be gated instead of trusted.

### Fixed
- **The §3.1 head-swap control was degenerate, not weak, and the correction had
  never reached the prose.** The published version put one shared
  `all-MiniLM-L6-v2` in every vendor's caption slot, which deleted the swapped
  side's dependence on the text vendor, so both swapped arms measured the same
  quantity and returned 0.0833 against 0.0835. The ratio test reduced to
  `cross_native / within_native`. Rebuilt with one distinct caption tower per
  slot: within 1.2064 ± 0.0304, cross 1.2308 ± 0.0401, gap −0.0244 against a
  spread of 0.0476. Same conclusion, valid instrument. The same wrong numbers
  were live in the `srt-omni-crossvendor-states` card and are fixed there too.
- **`verify_cards.py` only ever checked cards held as module strings**, so any
  card living in a file was never gated. Now gated, which immediately found
  three numbers that existed only in prose. Also registered the artifact that
  actually sources `0.9390`, which had been passing by coincidentally matching
  an unrelated model's pass@k.
- **The `srt-omni-crossvendor-states` dataset shipped three of the twelve result
  files its paper's artifact index names.** Now ships twenty, including both
  files the §3.1 correction points at.
- **`RLIMIT_AS` at 4 GB raises on macOS**, so the sandbox guard killed every
  child and the selector reported "no candidate ran" on pools it should have
  resolved. Every limit is now clamped to the inherited hard limit and tolerates
  its own failure, because the guard must never be the thing that fails.

### Changed
- Three results that had been sitting in artifacts with no writeup are now in
  `paper_crossvendor.md`: the edge-identity control (monotone in 6/6 vendor
  permutations in both domains, with two-edge routes falling 0.078 and 0.063
  below the per-edge product), the floor as a distribution rather than a bound
  (0.5006 ± 0.0052 over 20 seeds, plus the 95% interval on the headline and the
  12-of-14 count null), and the `mteb#5330` replication.
- The selection-decay slope now names its read. −0.0704 is the chat-native read;
  agreement gives −0.0815 and execution-guided filtering −0.1192, the last
  fitting considerably tighter. The direction is read-independent, the magnitude
  is not, and the figure published is the most conservative of the three.

### Removed
- **The ensemble framing.** Pooling 48 candidates from six frontier models
  across five labs solves the same 154 of 164 problems as the best single
  member's own eight, a gain of **+0.0000**. The union ceiling of 0.9878 had
  been quoted as though it supported pooling, without a selector ever having
  been run over the pool it describes. It is a bound on what better selection
  could win and nothing more. Agreement returns the majority, so extra members
  mostly add votes for what the pool already believed.
- **Two follow-on directions, both measured and both dead.** Cluster reranking:
  the majority cluster is already correct on 160 of the 162 solvable problems
  and only 2 are minority-held, so the whole approach tops out at +2 problems.
  Probe resolution: sweeping `n_cases` over {2,4,6,12,24,48} crossed with K
  {2,4,8} moves accuracy by nothing at any K, and is faintly negative at the
  top, while cluster purity climbs 0.7076 to 0.8909. Sharper probes do split the
  impure clusters exactly as predicted; the candidates they separate are not the
  ones that pass. **K is the only lever that moves accuracy**, so latency is the
  currency and there is no cheap substitute for sampling.

- **The Lab-map reader re-measured without contamination, and it reaches
  human parity.** Refit the sunstone head from the same gemma L47 states with
  the 5,000 evaluation images held out of HEAD training as well as reader
  training, then retrained the reader on the resulting space. The first
  caption falls from median 8 to **57** and the two human references become
  indistinguishable, which is what two people describing one photograph should
  look like. Against a pool of 118,287: first human caption median 57, second
  human 48, **the reader 46**, both controls at chance (57,494 and 59,681).
  Per photograph the reader ranks the image above the first human caption on
  **49.6%** of images where a second human manages 48.0%, and that
  human-to-human arm is symmetric to the decimal, which is the validity check.
  The reader is inside the human band, not below a ceiling. These are not the
  deployed figures; the Lab still serves the shipped head.
  `artifacts/nla/verbalizer/caption_coverage/`.
- **Caption coverage in head training is worth about a third of retrieval
  rank.** The shipped head saw one caption per image, the first of COCO's five.
  Refitting with all five resampled per epoch, changing nothing else, moves
  unseen-caption retrieval from median 60/53 to **40/34** on a training loss
  that is higher (1.22 against 0.51): matching any of five descriptions is the
  harder objective and the one that generalises. **It does not carry through to
  the reader.** Retraining the reader on each space with the identical recipe
  moves the reader 46 to 40 (+12%) while the first human caption moves 57 to 39
  (+32%) and the second moves 48 to 28 (+41%). Everyone improves, the reader
  improves least, so its standing goes backwards: it beats the first human on
  49.6% of images under `cap0` and 44.8% under `all5`, against human-versus-human
  of 48.0% and 50.4%. The head's caption coverage is therefore **not** the
  reader's bottleneck; text-side organisation of the space and the reader's
  generative quality are largely separable. Scripts `encode_captions_l47.py`,
  `refit_sunstone_head.py` (`--mode cap0|all5`) and
  `run_caption_coverage_ab.sh`.
- **A map of the gallery you can stand anywhere on**, served as pane
  **04 · Stand somewhere**. All 118,287 COCO train2017 images projected through
  the sunstone head into one 1,024-d space and laid out in two dimensions. A
  ~36M-parameter prefix on a frozen Qwen3-0.6B reads any point and says what is
  there, including the open water between clusters where no photograph exists:
  the empty gap between the skiing and skateboarding regions reads as "A man is
  doing a trick on a snowboard." Every region name on the map was written by
  that reader rather than by us. Uploading a photograph encodes it through the
  Lab's live gemma-4-31B into the same space in about two seconds, and the
  eight nearest photographs are always shown beside the sentence so a caption
  prior cannot pass unnoticed. Map and reader assets on
  [`RiverRider/srt-sunstone-linear-head`](https://huggingface.co/RiverRider/srt-sunstone-linear-head);
  server in `scripts/sunstone_server.py`, figures via
  `scripts/make_map_figures.py`.
- **A 0.6B verbalizes a 31B's raw internal state** (`paper_nla.md` §11.8). A
  frozen Qwen3-0.6B with a 44.5M prefix MLP reads one raw gemma-4-31B L47 image
  state (d=5376) and writes a caption; re-encoding that caption through the
  shipped browser head retrieves the correct photograph from all 123,287 images
  at **median rank 25** (R@1 0.120), past the single human reference caption
  (median 39). Controls sit at chance: another image's state 62,970, the mean
  state 59,408. The gallery was built by an unrelated Qwen3.8-27B tower, so no
  shared representation carries the result. The win over the human caption is
  register and length, not caption quality: the model enumerates whole-scene
  inventory, which this head reads well, while the human references foreground
  arrangement, which it does not recover, and the metric rewards naming more
  true things about a scene than a single reference caption names. Raw states
  needed no re-encode.
  Scripts: `build_fullstate_pairs.py`, `train_shared_space_verbalizer.py`,
  `eval_shared_space_verbalizer.py`; artifacts `artifacts/nla/verbalizer/`.
  Checkpoints released as
  [`RiverRider/srt-verbalizer-v1`](https://huggingface.co/RiverRider/srt-verbalizer-v1)
  (matched 27B reader median 20, cross-model 31B reader 25, a gallery-vector
  reader for index-only deployments, and an EOS variant that writes better and
  scores worse). Served live as **02 · Read an image** at
  [lab.sunstonenorth.com](https://lab.sunstonenorth.com).
- **Median rank for the anchored Q4 browser arm at deployment scale**
  (`artifacts/nla/q4/cross_runtime_browser_rung_123k.json`). The measurement
  promised in public review: fp16 reference median 36, Q4 head as-is 44,578
  (chance ~61,644), Q4 plus the 4KB anchor **176**, the top 0.14% of the
  gallery. R@1 recovery of 32% understates the anchored arm, whose correct
  image is usually near the top and rarely first; that distinction decides
  whether a port is viable. `browser_rung` now reports median alongside recall.
- **Tunnel watchdog for the public lab** (`scripts/wg_keepalive.sh`,
  `deploy/com.sunstonenorth.wireguard.plist`). lab.sunstonenorth.com proxies
  every `/api/*` call through WireGuard to the machine holding the models, so
  when the tunnel drops the site still returns 200 and only the demo hangs.
  Nothing restarted it. Liveness is checked by route presence rather than
  `wg show`, which needs root and would report a healthy tunnel as down.
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
- **The Lab-map reader's "human ceiling" was a memorised training pair.**
  `sunstone_verb_eval.json` reported a gold-caption arm at median rank 8 and
  the prose around it called that the ceiling the reader falls short of. The
  sunstone head was fitted on COCO train2017 pairs built from each image's
  *first* caption (`gemma4_encode_pairs.py` writes `caption: caps[0]`) and the
  eval gallery is train2017, so that arm scored a pair the head had been
  trained to align. A second caption of the same photograph, which the head
  never saw, lands at median **45**, against the reader's 64. Counted per
  photograph, the reader ranks the image above the first human caption on
  **17.0%** of images, where a second human beats the first on **20.0%**. The
  arm is a wiring control and is now labelled as one in the script, the model
  card and the artifact's own `caveat` and `supersedes` fields, and the eval
  now carries an uncontaminated `second_human_caption` arm plus per-item ranks.
  Unlike every other correction in this list, this one had been making the
  result look **worse** than it measured. Confirmed experimentally the same
  day: holding the evaluation images out of head training moves the first
  caption from median 8 to 57 while the second caption barely moves, so the
  entire advantage was memorisation.
- **`srt-browser-head-118k` model card advertised the wrong recovery figure.**
  The published table reported 0.2300/0.0154/0.1952 and "85% recovery", which
  were measured on an earlier 4,000-image head against a 1,000-image pool, not
  on the deployment. Corrected to 0.1092/0.0000/0.0350, i.e. 32%, against all
  123,287 images on the head and gallery that ship, with the median-rank column
  added. Found in public review by
  [@dipankarsarkar](https://huggingface.co/dipankarsarkar).
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
