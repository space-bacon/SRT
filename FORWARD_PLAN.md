# Forward Plan — SRT / SRT-NLA

**Date:** 2026-07-08 (cross-modal Sunstone + gpt-oss-20b status addendum; Qwen3-235B scaling plan 2026-06-12; Stage-4 framing addendum 2026-05-20)
**Author of last edit:** Post-Sunstone documentation pass.

This is the single source of truth for "what is the next thing to work on"
across the workstreams in this repo. Supersedes the per-day handoff in
`SESSION_HANDOFF.md` (which remains a snapshot of 2026-07-08).

> **2026-07-08 status addendum.** Since the 235B plan below was written,
> three workstreams completed:
>
> 1. **Qwen3-235B Phase A: DONE and shipped** (see Status box in the
>    Stage-5 section below).
> 2. **gpt-oss-20b port: DONE, full Phase A+B plus NLA pipeline.**
>    Read-out validated (regime ECE 0.0009 / AUROC 0.974); the AV is an
>    honest negative on this backbone (best-of-64 centered 0.642 <
>    NN-retrieval 0.744), so the 4096-code VQ codebook is the shipped
>    decoder. Three HF repos + live trace Space
>    (`RiverRider/srt-nla-gptoss20b-trace`). State-identity red-teaming
>    instrument and findings in `paper_nla.md` §11.5. gpt-oss-120b
>    runbook prepared (`docs/PORTING_GPT_OSS_120B.md`), not yet run.
> 3. **Cross-modal (Stage 6, "SRT-Sunstone"): SHIPPED.** Text-trained
>    community read-out on frozen gemma-4-31B-it reads images zero-shot
>    (CIFAR-10 image→word retrieval@1 0.93; `paper_nla.md` §11.6).
>    Autostereogram boundary study in §11.6.2. Model
>    `RiverRider/Gemma-4-31B-it-SRT-Sunstone`, Space
>    `RiverRider/srt-sunstone`, repo mirrored to
>    `space-bacon/SRT-Sunstone`.
>
> **Open queue (2026-07-08 night, priority order):**
> - **Gemma-4 BASE checkpoint verbalizer run** (tests the
>   base-vs-instruction-tuned hypothesis from `paper_nla.md` §11.7;
>   identical pipeline, `scripts/train_gemma4_nla_all.sh` with
>   BACKBONE=google/gemma-4-31B, one box-day).
> - Press/outreach: WIRED pitch drafted (Sandra Upson); send decision
>   with the user.
> - MTEB(eng, v2) full 41-task run for `v22c_a050` leaderboard
>   submission (was launched on a since-retired box; check for partial
>   results, relaunch if needed).
> - ginigen Metacognition leaderboard: 4 backbones submitted, results
>   still unscored; pull `leaderboard_mcq.json` and run the
>   verbalizability-vs-gain correlation for §13.
> - Queued GPU experiments: multi-position spoof test (completeness flag
>   vs aggregation site); Qwen2.5-7B replication of the
>   L24-surface/L18-semantic split.
> - Dependabot: RESOLVED 2026-07-08 (gradio bumped to 6.19.0; transformers
>   CVE-2026-4372 dismissed as tolerable risk — 5.3.0 fix breaks KV-cached
>   generation; rationale at each pin, commit f896edab).
> - SRT-Sunstone repo curation: decide whether to trim it into a
>   Sunstone-focused public repo (README, scope) or keep as mirror.

> **2026-05-20 framing addendum.** NLA is **Stage 4** of the SRT
> program (Stages 1–2 = Lancaster 2025 / 2026a SSRN; Stage 3 = the
> SRT-Adapter manuscript under [`arxiv/`](arxiv/), repository-hosted,
> *not yet on arXiv*). See [`paper_nla.md`](paper_nla.md) §0, §1.5,
> §12 for the canonical framing. Cross-backbone work since the v1
> release: Llama-3.2-3B replication shipped (HF
> `RiverRider/srt-nla-av-llama32-3b`,
> `RiverRider/srt-nla-targets-llama32-3b-v1`); Gemma-2-2B replication
> in progress on Blackwell (placeholder in `paper_nla.md` §11).

---

## Scaling to Qwen3-235B-A22B (Stage 5)

**Goal.** Port the SRT read-out (divergence, regime `r̂`, community, and the
activation-verbalizer targets) onto a frozen **Qwen3-235B-A22B** backbone, and
ship a `srt-adapter-qwen3-235b` checkpoint plus the introspection signals the
showcase demo already renders.

> ### Status: Phase A DONE (2026-06-13)
>
> R0–R3 rungs all green; the read-only adapter is trained, evaluated on a
> held-out set, and shipped. Checkpoint:
> [`RiverRider/srt-adapter-qwen3-235b`](https://huggingface.co/RiverRider/srt-adapter-qwen3-235b).
>
> - **R0** `Qwen3-8B-Base` (dense): all green, fp32 parity bit-exact.
> - **R1** `Qwen3-14B-Base` (dense): smoke + 2000-step read-only train,
>   monotonic val.
> - **R2** `Qwen3-30B-A3B-Base` (MoE) + FP8 variant: smoke + train; MoE
>   read-only confirmed memory-bandwidth-bound.
> - **R3** `Qwen3-235B-A22B-FP8` (94-layer MoE, sharded across 8× RTX PRO 6000):
>   smoke parity **byte-identical** to HF forward after the SDPA `is_causal`
>   fix; Phase-A read-only training to step 2000 (best at step 1750).
>
> Validation `bif` improved 0.0999 → **0.0666** (~33%) warm-starting bs=16 →
> bs=128. Held-out probe (3000 rows, [`scripts/phaseA_probe.py`](scripts/phaseA_probe.py)):
>
> | Head | Metric | Value |
> |---|---|---|
> | Regime | ECE | **0.0005** |
> | Regime | AUROC | **0.9859** |
> | Regime | Brier | 0.0123 |
> | r̂ (bifurcation) | Pearson | 0.751 |
> | r̂ (bifurcation) | MAE | 0.571 (under-predicts scale; affine rescale halves it) |
> | Community | NMI | 0.6247 |
> | Community | ARI | 0.4040 |
>
> This **meets the R3 success criterion below** (regime ECE in the 1e-3 range,
> as on v8a; community beats random on held-out probes). It is the first SRT
> adapter on a frontier-scale (235B / 22B-active MoE) host, supporting the
> substrate-generality claim. **Phase B (inject-CE) remains deferred.**

### What is actually new vs the 7B port

The four SRT modules and the FiLM inject all operate on the **residual stream**
between backbone layers (see [`srt/adapter.py`](srt/adapter.py), which runs the
backbone layers in a manual Python loop). That logic is architecture-agnostic.
Only two things genuinely change going from Qwen2.5-7B to Qwen3-235B:

1. **The Qwen3 layer / rotary API.** Backbone is `Qwen3MoeForCausalLM`
   (`transformers>=4.51`; our `4.53.3` pin already covers it). Verify the
   manual loop against `Qwen3MoeDecoderLayer.forward` kwargs
   (`position_embeddings`, `attention_mask`, `past_key_value`,
   `cache_position`), relocate `rotary_emb` if needed, and handle
   `output_router_logits`.
2. **MoE layers.** 128 experts, 8 active per token. The tap and inject sit
   *between* layers, so the sparse MLP block is untouched. Confirm the
   residual-stream in/out shapes and that routing runs inside the manual loop.

### Backbone survey (live HF check, 2026-06-12)

Surveyed the full open-weight Qwen lineup via the HF API with authenticated
existence checks. Findings that constrain the plan:

- **There is no `Qwen3-235B-A22B-Base`** (authenticated check:
  `RepositoryNotFoundError`; same for `Qwen3-32B-Base`). Qwen released base
  checkpoints only up to 30B in the Qwen3 line (`Qwen3-0.6B/1.7B/4B/8B/14B-Base`
  and `Qwen3-30B-A3B-Base` all exist, ungated). The 235B target is therefore
  the post-trained **`Qwen/Qwen3-235B-A22B`** (hybrid thinking model, ungated).
  Plan for chat-template/`<think>` handling in target sampling, and re-verify
  side-channel transfer given v1.0 trained on a *base* host.
- **`Qwen/Qwen3-235B-A22B-FP8` exists, ungated** → the Phase-A cost-optimal
  pick is real.
- **The entire Qwen3.5 / Qwen3.6 generation is disqualified** for this port
  despite attractive active-param counts (e.g. `Qwen3.5-397B-A17B`,
  `Qwen3.5-122B-A10B`, `Qwen3.6-35B-A3B`): arch is
  `Qwen3_5MoeForConditionalGeneration` (multimodal wrapper, `text_config`
  nesting), **hybrid linear-attention** (3 linear : 1 full interval, gated attn
  output), different vocab/eos, and requires `transformers>=4.57`, violating
  the 4.53.3 pin and the manual-layer-loop assumptions. Same verdict for
  `Qwen3-Next-80B-A3B` (Gated DeltaNet hybrid, tf>=4.57).
- **`Qwen3-Coder-480B-A35B`**: instruct-only, code-specialised, ~960 GB bf16.
  Not worth 2× the memory of the 235B.
- A future Qwen3.5 port would be a **separate, larger effort** (hybrid-layer
  taps, new transformers pin, multimodal config surface). Out of scope here.

Consequence for the rungs: R0 uses `Qwen3-8B-Base`, R2 uses
`Qwen3-30B-A3B-Base`. There is no 32B base, so R1 uses `Qwen3-14B-Base`
(largest dense base) or is folded into R2 if R0 is clean.

Target backbone facts: `d=4096`, `L=94`, vocab 151936, GQA 64Q/4KV,
head_dim 128, 235B total / **22B active**, bf16 weights ≈ **470 GB**, FP8
≈ **235 GB**, ctx 40960 native (131072 with YaRN). Note `bos_token_id=151643`
and `eos_token_id=151645` are **distinct**, which sidesteps the Qwen2.5
`bos==eos` target-generation bug (commit `902b746`; re-verify anyway).

### The strategic insight that sets the budget

SRT's validated value is **read-only**: divergence, regime, `r̂`, community, and
the AV targets all *read* the residual stream. The FiLM **inject** path is the
only thing that needs CE-loss gradient to flow back through the backbone. Its
output effect is real but small (v4 checkpoint: mean logit delta ≈ 1.4;
seeded A/B in the showcase demo shows divergent but comparable-quality text),
and none of the validated read-out claims depend on it.

So the port splits into a cheap read-only phase and an optional expensive
inject phase:

- **Phase A, read-only (recommended).** Run the frozen 235B **forward only**
  (`inference_mode`, FP8-friendly), tap hidden states at the hook layers,
  **detach** them, and train only the ~16–20M head params. Cost is nearly
  independent of backbone size: it is a forward pass plus a tiny head update.
  Better still, cache the taps from one corpus pass and train the heads offline
  on the cached tensors (minutes to single hours on one GPU).
  - **Cache disk math (do not skip):** full-seq taps are huge. 100K rows ×
    512 tok × d=4096 × 4 tap layers × bf16 ≈ **1.7 TB**; the full 1M corpus
    ≈ 17 TB. Strategy: cache **pooled/last-token** vectors for the bulk corpus
    (1M × 4096 × 4 × 2 B ≈ 33 GB) for community/STS heads, and **full-seq**
    taps for only a 20–50K subset (~0.3–0.9 TB) for divergence/regime/chain.
- **Phase B, inject-CE (optional).** Only if we want the closed loop. CE
  backprop through the ~94 layers above the lowest inject point, with gradient
  checkpointing and the big multi-GPU box. Defer until Phase A proves
  valuable on 235B.

### Staged rollout (de-risk the Qwen3 API and MoE separately, cheaply)

Rungs are labelled R0–R3 to avoid colliding with the program-stage numbering
(Stages 1–5) and the training phases (A/B) above.

| Rung | Backbone | Why | GPUs | Wall-clock |
|---|---|---|---|---|
| R0. Smoke | `Qwen3-8B-Base` (dense) | Validate the Qwen3 layer/rotary API in the manual loop; fix bos/eos and kwargs | 1× (48–96 GB) | hours |
| R1. Dense scale | `Qwen3-14B-Base` (dense) | Confirm the Qwen3 dense port end-to-end (no 32B base exists); fold into R2 if R0 is clean | 1× H100/H200 or RTX PRO 6000 | ~½ day |
| R2. MoE de-risk | `Qwen3-30B-A3B-Base` (MoE, 3B active) | Validate the MoE port cheaply before paying 235B prices; verify FP8 here too | 1–2× H200 | ~½–1 day |
| R3. Target | **`Qwen3-235B-A22B`** (post-trained; no base exists) | Full target | 2–4× H200 (FP8) or 8× (bf16) | ~1–3 days |

R0–R2 cost little and remove most of the integration risk before any 235B
spend. Do not jump straight to 235B.

### Hardware (vast.ai) for the 235B stage

Memory math: bf16 weights 470 GB; FP8 ≈ 235 GB; adapter + AdamW states ≈ 0.2 GB
(negligible); read-only activations ≈ inference memory.

| Option | Mem | Fits | Notes | ~$/hr |
|---|---|---|---|---|
| 8× H200 SXM | 1128 GB | bf16, big headroom | NVLink → tensor-parallel. Reserve for Phase-B inject. | 18–28 |
| 4× H200 | 564 GB | bf16 (tight) | Fine for read-only / small batch | 9–14 |
| 8× RTX PRO 6000 Blackwell | 768 GB | bf16 | PCIe → sequential `device_map`, not TP; torch≥2.7+cu128 (known) | 8–16 |
| **2–4× H200 + FP8 backbone** | 282–564 GB | FP8 (~235 GB) | Frozen backbone → FP8 is acceptable for read-out, **but** (a) verify the FP8 checkpoint (`Qwen3-235B-A22B-FP8`, fine-grained fp8) runs through the *manual layer loop* under the 4.53.3 pin at R2, and (b) heads trained on FP8 taps should be served on FP8 (self-consistency). **Cost-optimal Phase-A pick.** | 5–12 |

Tensor parallelism wants NVLink (H100/H200 SXM). On PCIe boxes use pipeline /
sequential `device_map="auto"`, which is the natural fit for the manual layer
loop (move the hidden state across devices at layer boundaries).

### Time and cost estimate

Per-token forward scales with **active** params: 22B vs 7B ≈ 3.1×, made worse by
MoE routing + model-parallel comms, but offset by H200 speed and no backbone
backward in Phase A.

- Read-only step (8× H200, TP, bs=8, seq=512): ~1–2.5 s/step.
- Steps to a good checkpoint: side-channel heads converged at 10–17K steps on
  the 7B; budget **20–40K steps** with val-based selection.
- → **~10–25 GPU-hours of compute, ~1 day wall-clock** including data loading,
  val, checkpointing, and 1–2 restarts.
- Activation-cache variant: one corpus forward pass (the only expensive part)
  then head training in minutes to single hours on one GPU.
- Phase-B inject-CE: ~3–5× slower per step (full backprop + grad checkpointing)
  → budget **2–4 days** on 8× H200.

Cost ballpark: Phase-A read-only on FP8 2–4× H200 ≈ **$150–400**; bf16 8× H200
read-only ≈ **$400–700**; Phase-B inject ≈ **$1–2K**; R0–R2 rungs ≈
**$100–300** total.

### Work items (priority order)

1. **Qwen3 manual-loop port.** Verify `_cached_step` / `forward` in
   [`srt/adapter.py`](srt/adapter.py) against `Qwen3MoeDecoderLayer`; confirm
   under `transformers==4.53.3`. *(R0)*
2. **MoE pass-through.** Confirm residual in/out shapes and that the sparse MLP
   runs inside the manual loop; decide whether to log `output_router_logits` as
   a new observational signal. *(R2)*
3. **FP8 verification.** Load the FP8 checkpoint of Qwen3-30B-A3B (or 235B)
   through the manual loop; confirm quantized layers behave as drop-in modules
   and taps are non-degenerate. *(R2)*
4. **Device-aware loop.** Move `h`, masks, and rotary across each layer's
   device; keep SRT heads on one device; move taps to/from it. Choose TP
   (NVLink) vs sequential `device_map` (PCIe). *(R1+)*
5. **Read-only training mode.** Add a `detach taps + inference_mode backbone`
   path plus an activation-cache exporter (pooled for bulk, full-seq for a
   20–50K subset; see disk math above) so Phase A builds no backbone graph.
6. **Re-resolve hooks for L=94.** Auto-resolution gives MAH≈`[23,46,69]`,
   inject `[46,69]`, community≈`13`. A 94-layer model likely wants 4–6 taps;
   run a small sweep. Adds a few M params, still tiny.
7. **Gradient checkpointing** on backbone layers. Only needed for Phase B.
8. **Data.** rsync the local phase1 corpus from
   `/Users/burtron/development/SRT/data/phase1/` (1M train / 100K val) to the
   box; NLI-only for STS-targeted heads (see SRT-Adapter notes below).
9. **Defer the Activation Verbalizer port.** The AV needs a second 235B backbone
   copy, doubling memory. Keep the v1 AV on Qwen2.5-7B for the demo and port the
   AV to Qwen3 only after the read-out adapter proves out.

### Things to NOT do

- Do not fine-tune the backbone. The whole SRT premise is a frozen host.
- Do not start at 235B. R0–R2 cost a rounding error and catch the API/MoE
  bugs.
- Do not pay for bf16 if read-only, once FP8 is verified at R2.
- Do not use tensor parallelism on a PCIe box. Use sequential `device_map`.
- Do not port the AV in the same pass. It doubles backbone memory for no
  read-out benefit.
- Do not cache full-seq taps for the whole corpus (multi-TB). Pool for bulk,
  full-seq for a subset.
- Do not pick Qwen3.5/3.6 or Qwen3-Next as the host. Hybrid linear-attention
  layers + `transformers>=4.57` break the manual loop and the 4.53.3 pin.

### Success criteria

- R0: a clean read-only forward + tap on Qwen3-8B with non-degenerate
  divergence (`tap.std(dim=0).mean() > 0.1`).
- R3: a `srt-adapter-qwen3-235b` checkpoint whose regime classifier
  replicates calibration (ECE in the 1e-3 range as on v8a) and whose community
  / STS heads beat random on held-out probes, validated read-only.

---

## Two workstreams, one repo

| Workstream | Current SOTA | Branch | Status |
|---|---|---|---|
| **SRT-Adapter** (semiotic awareness for frozen LLMs) | `v22c_a050`, mean MTEB-STS 0.3744 (soup of v18+v21a) | `main` | Shipping. v1.0 on HF for downstream pinning. |
| **SRT-NLA** (activation verbalization, frozen backbone) | `srt-nla-av-v1`, best-of-64 ρ_norm = 0.92, greedy ρ_norm = 0.26 | `nla` | First public release 2026-05-18. Greedy gap is the open problem. |

---

## NLA — next push, in priority order

### 1. Ship the paper *(this week)*

Status: `paper_nla.md` is at v0.2 (post-cleanup). Action items:

- [ ] Add a §3.5 subsection summarising the K-curve and the
      Spearman(logp, oracle) ≈ 0.04 finding from `rerank_eval.py`. *(in
      this release)*
- [ ] Cross-link the HF model + dataset cards from §6 (Artifacts).
- [ ] One pass for unit consistency: every "0.28 / 0.99" claim should be
      `ρ_norm = 0.26 / 0.92` (post-rerank_eval triangulation).
- [ ] Decide venue: short methods note (≤8 pages) vs blog-post-with-arxiv.
      The contribution is the **reframe + the K-curve + the logp-death
      result**, not a new model.

**Why now**: the result is publishable as-is. Gating on closing the
greedy gap means gating on a separate research project.

### 2. Best-of-K oracle rerank as the deployable decode *(zero work)*

Already done — it ships as the recommended path in the model card and
release notes. Nothing more to do on this option; just make sure the
paper frames it as the headline decoding method.

### 3. BoK distillation at K=32 *(the policy-improvement experiment)*

From the released K-curve, win-of-32 ≈ 0.747 cen vs greedy 0.586 cen —
a real teacher gap, unlike the K=4 smoke (which regressed val).

- [ ] Profile memory on Blackwell: K=32 × seq_len=64 × batch=16 = 32k
      rollout tokens/step. May need `--batch=8 --samples-per-v=32`.
- [ ] Temperature anneal 1.5 → 0.7 over training.
- [ ] Keep hard-neg InfoNCE (J=8 NN from pool), α_bok=1.0, β_ctr=0.3,
      γ_act=0.
- [ ] Success criterion: greedy ρ_norm > 0.40 (currently 0.26) without
      regressing best-of-64 ρ_norm < 0.85 (currently 0.92).

**Risk**: even at K=32, teacher distribution may still concentrate on
greedy mode → no signal. Mitigation: monitor `win_top1_vs_greedy_rate`
during warmup; abort if < 0.2 after 500 steps.

### 4. Different-backbone sanity check *(one-script experiment)*

All current numbers are Qwen2.5-7B L20 (`‖μ‖ ≈ 55`). The greedy/best-of-K
gap may be largely an anisotropy story. Quick prior-art check:

- [ ] Run `oracle_ceiling.py` on LLaMA-3-8B L20 — measure `‖μ‖` and
      random_floor_cen.
- [ ] If `‖μ‖_LLaMA ≪ ‖μ‖_Qwen`, raw fve_nrm may already give a useful
      signal there → greedy gap may close *for free* on a less
      anisotropic backbone.
- [ ] One Blackwell day total; either confirms backbone-agnosticism of
      the centering claim or surfaces a new lever.

### 5. Things to NOT do *(based on negative results)*

- Don't build a logp-only reranker. Spearman 0.04.
- Don't rerun K=4 BoK. Teacher = noise.
- Don't add more prefix tokens past np=16. np=32 returned +0.003 raw.
- Don't add more multi-inject slots (M=4 returned 0).
- Don't retrain greedy-only with a higher CE weight — confirmed
  ρ_norm 0.26 ceiling on the warm-start.

---

## SRT-Adapter — next push, in priority order

### 1. Souping is the cheap Pareto move

Confirmed twice (v21b, v22c). Default behaviour for any new checkpoint
pair should be `soup_adapters.py` over `--alpha {0.3, 0.5, 0.7}` before
declaring a winner.

### 2. Acknowledged ceiling on Qwen2.5-7B-NLI

Mean STS 0.366 → 0.371 → 0.374 over 4 souping iterations suggests we
are within sampling noise of the InfoNCE-on-Qwen2.5-7B-NLI ceiling.
Further gains likely require one of:

- a different corpus (multilingual hard NLI, code-switched STS),
- a different backbone (LLaMA-3, Mistral, Phi-3.5),
- supervised distillation from a much larger embedder (e.g. e5-mistral-7b).

None of these are scheduled. SRT-Adapter is in maintenance mode pending
a new corpus or backbone delivery.

---

## Repo hygiene

Done in 2026-05-18 cleanup pass:

- ✓ Canonical metrics in [srt/nla/metrics.py](srt/nla/metrics.py).
- ✓ Dead REINFORCE arc archived to `scripts/_archive/` with explanation.
- ✓ README updated to mention NLA workstream + paper.
- ✓ `docs/nla_mission.md` banner pointing to `paper_nla.md`'s
  centered-unit conventions.
- ✓ HF model card, dataset card, release notes drafted.

Still to do (deferred):

- [ ] Backfill integration tests for `srt/data/dataset.py` (history of
      contrastive-collision bug).
- [ ] Promote `srt/nla/targets_check.py` into a pytest fixture that runs
      automatically on any new targets file.
- [ ] Consolidate `_last_token_h()` / `_last_h()` helpers (also
      duplicated across the surviving train_nla_*.py scripts).
- [ ] Replace `weights_only=False` calls with a centralized
      `srt.io.safe_load()` helper that documents the assumption.

---

## "If I am picking this up after a break" checklist

1. `git fetch origin && git checkout main && git pull --ff-only` (Qwen3 port
   work happens on `main`; NLA experiments live on `nla`).
2. Read this file top to bottom. The lead workstream is now
   [Scaling to Qwen3-235B-A22B](#scaling-to-qwen3-235b-a22b-stage-5). For the
   NLA framing read [`paper_nla.md`](paper_nla.md); for what shipped read
   [`RELEASE_NOTES_NLA_v1.md`](RELEASE_NOTES_NLA_v1.md).
3. If resuming the Qwen3 port: start at **R0** (Qwen3-8B smoke). The first
   real verification is that the manual decoder loop in
   [`srt/adapter.py`](srt/adapter.py) runs against `qwen3_moe` layers under the
   `transformers==4.53.3` pin. Do not provision a 235B box until R0–R2
   pass.
4. If resuming NLA instead: if the targets file is missing locally, pull from
   HF (`huggingface-cli download RiverRider/srt-nla-targets-v1 ...`); if the AV
   ckpt is missing, pull `RiverRider/srt-nla-av-v1 best_av.pt`. Smoke test:
   `pytest tests/test_nla_smoke.py -q`. Then pick item 1 / 3 / 4 from the NLA
   list based on available compute (paper = local; BoK = Blackwell day;
   backbone-check = Blackwell day).
