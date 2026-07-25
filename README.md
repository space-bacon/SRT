# SRT — Semiotic-Reflexive Transformer (Adapter Architecture)

**Meaning forks. SRT sees it.**

SRT-Adapter is a lightweight module that bolts semiotic awareness onto any
frozen causal language model.  The backbone runs natively — its own embeddings,
its own LM head, its own attention.  SRT modules are small taps that **read**
divergence from hidden states, **track** reflexive awareness, and optionally
**inject** semiotic corrections back into the stream.

## 30-second TL;DR

> - **What:** a ~12 M-parameter adapter that observes a frozen LLM at 3 layers and injects a FiLM correction at 2 of them, exposing per-token semiotic signals (divergence, reflexivity `r̂`, regime) plus a discourse/embedding vector.
> - **Why:** lightweight, portable instrumentation for a frozen backbone — no base-model weight updates, zero cross-entropy degradation, trains in hours at ≈0.17 % of backbone params. The released `v1.0` checkpoint targets semantic embeddings (MTEB-STS).
> - **New (July 2026) — the portability result:** the structure these taps read is **invariant across scale, precision, and hardware**. A 22 MB linear head gives a frozen multimodal LLM image↔text retrieval at fully-trained-2018-dual-encoder level (Karpathy 5k i2t R@1 = 0.416), and the *same head* survives a 10× host reduction (31B → 3B, no loss), 4-bit quantization (−0.01 R@1, unchanged weights), and a change of silicon (CUDA datacenter → Apple-Silicon Mac, 100 % head-space agreement). One artifact, every deployment tier from Raspberry-Pi-class to datacenter. See [SRT-Sunstone](#srt-sunstone--the-read-out-reads-images-cross-modal) and [docs/CROSSMODAL_LINEAR_HEAD.md](docs/CROSSMODAL_LINEAR_HEAD.md).
> - **How (one line):** read divergence → integrate in a GRU → emit `γ, β` → `h ← h·(1+γ) + β`.
> - **Reading order (5 min):** [Architecture](artifacts/explainers/00_architecture.png) → [Visual grammar](artifacts/explainers/00b_legend.png) → [One-token trace](artifacts/explainers/11_token_trace.png).


## Architecture

![SRT-Adapter architecture](artifacts/explainers/00_architecture.png)

The base LLM stays fully frozen. SRT taps the residual stream at three
layers (L7 / L14 / L21) into Metapragmatic Attention Heads, the GRU-based
Reflexive Reasoning Module integrates the divergence stream into a
meta-state, and a FiLM correction `h ← h·(1+γ) + β` is injected back into
L14 and L21. A community head taps L2 for discourse basin, and BEN reads
the RRM meta-state to emit per-token reflexivity `r̂` and a regime label.

> Full visual walkthrough with captions: [docs/EXPLAINERS.md](docs/EXPLAINERS.md).

## Key Ideas

1. **Zero CE degradation** — The backbone's native embeddings and LM head are
   untouched. Cross-entropy starts at pretrained quality (~3.5), not 200+.

2. **~14.6M trainable params** — Only the semiotic modules train. The 7B backbone
   is fully frozen. Trains in hours, not weeks.

3. **Unsupervised community discovery** — A small encoder discovers
   discourse-trajectory structure from hidden state patterns. No hardcoded
   labels. As of v8a the encoder output is the community vector directly
   (continuous trajectory mode); earlier checkpoints used a 32-prototype
   soft-argmax readout that turned out to be a discriminability bottleneck
   (see `arxiv/paper.md` §5.8–§5.9).

4. **Backbone-agnostic** — Works with any HuggingFace `AutoModelForCausalLM`:
   Qwen, LLaMA, Mistral, Phi, Gemma, etc.

5. **Portable** — Save/load just the 44MB adapter weights. Attach to any
   compatible backbone at inference time.

## Frozen-backbone probe — TruthfulQA-MC2 hidden-state detector

> **Scope note.** This result is a **separate diagnostic probe, not a capability
> of the SRT adapter.** It trains LightGBM on features extracted from a single
> forward pass of the **frozen backbone** (no adapter, no fine-tuning) and is
> reported here only to characterize the backbone's hidden-state geometry. The
> SRT adapter's own side-channels (`r̂`, regime, divergence) are observational
> signals; they are **not** a validated hallucination detector, and on free-form
> generation they do not, on their own, separate hallucinated from faithful
> answers above chance. Treat hallucination detection as out of scope for the
> adapter.

Using frozen-backbone features plus LightGBM, this repo reaches the top of the
published hidden-state-detector band on TruthfulQA-MC2, group-CV by question
(n=817, 5882 paired choices):

| backbone | params | LightGBM AUC |
|---|---:|---:|
| Gemma-2-2B | 2B | 0.8563 ± 0.016 |
| Llama-3.2-3B | 3B | 0.8475 ± 0.013 |
| **Qwen-2.5-7B** | **7B** | **0.8656 ± 0.011** |

Reference band: SAPLMA ≈ 0.72, SAR ≈ 0.75–0.83, INSIDE ≈ 0.78–0.85,
EigenScore ≈ 0.80–0.85.


Full protocol, ablations, and reproduction command in
[docs/TRUTHFULQA_RESULTS.md](docs/TRUTHFULQA_RESULTS.md). Evaluator:
[scripts/evals/truthfulqa_v3.py](scripts/evals/truthfulqa_v3.py).
Artifacts:
[Qwen](artifacts/truthfulqa/v3_qwen_n817.metrics.json),
[Llama](artifacts/truthfulqa/v3_llama32-3b_n817.metrics.json),
[Gemma](artifacts/truthfulqa/v3_gemma2-2b_n817.metrics.json).

## Explainer series

A 10-figure visual walkthrough of SRT — pipeline, modules, anisotropy,
losses, TQA ladder, cross-architecture results, and the demo map — with
captions kept separate from the visuals so they can be repositioned.

See [docs/EXPLAINERS.md](docs/EXPLAINERS.md). PNGs live in
[artifacts/explainers/](artifacts/explainers/) and re-render via
`python scripts/explainers/make_all.py`.

| Overview | MAH | TQA ladder | Cross-arch |
|---|---|---|---|
| ![01](artifacts/explainers/01_pipeline.png) | ![03](artifacts/explainers/03_mah.png) | ![08](artifacts/explainers/08_tqa_ladder.png) | ![09](artifacts/explainers/09_crossarch.png) |

## Modules

| Module | Purpose | Parameters |
|--------|---------|------------|
| **MAH** (Metapragmatic Attention Head) | Detects where meaning diverges across positions | ~2.7M × 3 layers |
| **RRM** (Reflexive Recurrent Module) | Tracks semiotic meta-state, injects corrections | ~2.2M |
| **BEN** (Bifurcation Estimation Network) | Estimates reflexivity coefficient r̂ and regime | ~0.2M |
| **Community Head** | Discovers discourse-trajectory structure unsupervised | ~0.2M |

## Quick Start

```bash
# install
git clone https://github.com/space-bacon/SRT.git
cd SRT
pip install -e .
```

### Run inference (frozen Qwen-7B + released adapter)

```python
from srt.adapter import SRTAdapter
from srt.config import SRTConfig
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import torch

repo = "RiverRider/srt-adapter-v1.0"          # or RiverRider/srt-adapter-v8a
cfg  = SRTConfig.from_json(hf_hub_download(repo, "config.json"))
adap = SRTAdapter(cfg).cuda().eval()
adap.load_state_dict(load_file(hf_hub_download(repo, "adapter.safetensors")), strict=False)
tok  = AutoTokenizer.from_pretrained(cfg.backbone_id)

enc = tok("meaning forks here", return_tensors="pt").to("cuda")
with torch.no_grad():
    out = adap(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
print(out.ben_output.r_hat.mean().item(), out.community_output.encoded.shape)
```

See [examples/](examples/) for end-to-end loading, scoring, and sentence-encoding scripts.

### Live demos

- **SRT-Sunstone** (gemma-4-31B-it cross-modal read-out): <https://huggingface.co/spaces/RiverRider/srt-sunstone>
- gpt-oss-20b full input→output trace: <https://huggingface.co/spaces/RiverRider/srt-nla-gptoss20b-trace>
- SRT showcase (Qwen2.5-7B live introspection, ZeroGPU): <https://huggingface.co/spaces/RiverRider/srt-showcase>
- v1.0 demo: <https://huggingface.co/spaces/RiverRider/srt-adapter-v1.0-demo>
- v8a demo: <https://huggingface.co/spaces/RiverRider/srt-adapter-v8a-demo>

### Released checkpoints

| Checkpoint | Backbone | Notes |
|---|---|---|
| [`RiverRider/srt-adapter-v1.0`](https://huggingface.co/RiverRider/srt-adapter-v1.0) | Qwen2.5-7B | Stable release, semantic embeddings (MTEB-STS). |
| [`RiverRider/srt-adapter-v8a`](https://huggingface.co/RiverRider/srt-adapter-v8a) | Qwen2.5-7B | Encoder-as-community headline run. |
| [`RiverRider/srt-adapter-qwen3-235b`](https://huggingface.co/RiverRider/srt-adapter-qwen3-235b) | Qwen3-235B-A22B-FP8 | Read-only port to a frozen frontier MoE. Held-out regime ECE 0.0005 / AUROC 0.986, community NMI 0.62. |
| [`RiverRider/srt-adapter-gptoss20b`](https://huggingface.co/RiverRider/srt-adapter-gptoss20b) | gpt-oss-20b (MXFP4 MoE) | Full Phase A+B port. Regime ECE 0.0009 / AUROC 0.974, r̂ Pearson 0.689, community NMI 0.42. |
| [`RiverRider/Gemma-4-31B-it-SRT-Sunstone`](https://huggingface.co/RiverRider/Gemma-4-31B-it-SRT-Sunstone) | gemma-4-31B-it (multimodal) | Text-trained community read-out that reads images zero-shot. See SRT-Sunstone below. |
| [`RiverRider/srt-sunstone-linear-head`](https://huggingface.co/RiverRider/srt-sunstone-linear-head) | gemma-4-31B-it (multimodal) | 22 MB (bf16) cross-modal retrieval head: i2t R@1 0.661 on our protocol, 0.416 Karpathy 5k. Quantization-robust; runs locally. |

The 235B checkpoint shows the SRT read-out transfers across backbone scale and
architecture (dense 7B → 94-layer, 22B-active MoE): only the ~15.9M side-channel
heads are trained, on a fully frozen, forward-only backbone.

## SRT-Sunstone — the read-out reads images (cross-modal)

A 12.3M community read-out head trained on **text only**, attached to a frozen
multimodal `google/gemma-4-31B-it`, interprets **images** with zero image
training. Point it at a picture and it retrieves the picture's own words
(CIFAR-10 image→word retrieval@1 = **0.93**, chance 0.10) and names the
nearest discourse community the head learned from prose. The same image state
also retrieves full **sentences** from an open pool of 10,000 COCO captions,
zero training (5/5 CIFAR natural images on-topic at rank 1; `paper_nla.md`
§11.6.3). Cross-modal alignment peaks at ~80% of backbone depth and collapses
at the final layer, the same late-is-surface signature seen on gpt-oss-20b
(`paper_nla.md` §11.6–§11.6.1).

This is the semiotic claim made concrete: the read-out taps the shared
**interpretant** in the residual stream, independent of whether the sign
arrived as a word or an image.

A boundary condition sharpens the claim (`paper_nla.md` §11.6.2): on a
random-dot **autostereogram**, whose figure exists only in binocular
disparity, the read-out honestly reports texture. A simulated
binocular-fusion front-end (`scripts/stereo_decode.py`) recovers the hidden
figure from the same pixels, and both the generative caption and the read-out
then name it. The capability gap is in the sensor, not the semiotics.

### The modality gap is linear — and one matrix unlocks it

A controlled ladder (`paper_nla.md` §11.6.4) established that after
per-modality centering, the image↔text gap inside the frozen backbone is
**anisotropic-linear**: an orthogonal rotation makes retrieval *worse* than
centering alone, a single trained linear map captures the gap, and an MLP
given 33× the data never beats the linear map (the gap *widens* with data).
Every rung carries a shuffled-pairs control.

| method (COCO val2017, 1000 imgs vs 5000 captions) | i2t R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| centered cosine (zero training) | 0.288 | 0.523 | 0.648 |
| orthogonal Procrustes (refuted) | 0.226 | 0.472 | 0.628 |
| **trained linear head** (117k pairs, InfoNCE, ~22 MB) | **0.661** | **0.911** | **0.967** |
| two-layer MLP (same data; never wins) | 0.567 | 0.887 | 0.943 |

On the literature-standard **Karpathy 5k test** (leakage-controlled) the head
scores i2t R@1/R@5/R@10 = **0.416 / 0.710 / 0.818**, matching fully-trained
2018 dual encoders (VSE++: 0.413/0.711/0.812) from a linear map over a frozen
chat model, with ~3,000× less pair data than CLIP-class systems. The claim is
never "beats CLIP"; it is **no new model**: retrieval as a free rider on the
LLM you already run.

### The capability is substrate-invariant: from Pi-class to datacenter

The deeper result is not any single deployment. It is that the structure the
head reads is a **stable property of the model class**, indifferent to the
three things deployments usually vary:

| axis | test | result |
|---|---|---|
| **host scale** | full ladder re-run on Qwen2.5-VL-**3B** (10× smaller) | identical fingerprint; linear head 0.577 R@1 at 39k pairs vs the 31B's 0.553–0.590 at the same budget — **no loss** |
| **weight precision** | bf16-trained head applied *unchanged* to 4-bit NF4 states | −0.01 R@1; a 42 KB mean recalibration recovers half of that |
| **hardware / runtime** | CUDA datacenter → Apple-Silicon MLX, different kernels, different quantization | local states retrieve their datacenter twins at **100 % R@1** through the head ([scripts/local_sunstone.py](scripts/local_sunstone.py)) |

Read together with the earlier read-out ports (Qwen-7B → gpt-oss-20b →
Qwen3-235B), the picture is: **meaning in these substrates is organized
linearly and stably enough that one small artifact reads it anywhere the
model runs.** Retrieval needs a single prefill pass, not generation, so the
same head serves a Raspberry-Pi-class device doing overnight photo tagging,
a Mac doing interactive local search, and a datacenter serving a fleet —
train once, read everywhere. The intelligence lives in the substrate; the
semiotic layer is a set of linear taps small enough to ship as a config
file and honest enough to audit by inspection.

Engineering guide (deployment tiers, calibration rules, reproduction recipe):
[docs/CROSSMODAL_LINEAR_HEAD.md](docs/CROSSMODAL_LINEAR_HEAD.md). Head:
[`RiverRider/srt-sunstone-linear-head`](https://huggingface.co/RiverRider/srt-sunstone-linear-head).

- **Live demo**: <https://huggingface.co/spaces/RiverRider/srt-sunstone>
- **Model**: [`RiverRider/Gemma-4-31B-it-SRT-Sunstone`](https://huggingface.co/RiverRider/Gemma-4-31B-it-SRT-Sunstone)
- **Demo source**: [demo/cross_modal_space/](demo/cross_modal_space/)

### Train from scratch

```bash
python scripts/train.py \
    --backbone Qwen/Qwen2.5-7B \
    --train-data data/all_train.jsonl \
    --val-data   data/all_val.jsonl \
    --output-dir checkpoints/adapter_v1 \
    --batch-size 16 --epochs 3 --lr 3e-4 --max-val-samples 5000
```

Resume from a saved `training_checkpoint.pt` with `--resume <path>` (restores optimizer, scheduler, step, epoch).

## Training Diagnostics

Every `--log-every` steps, the training script logs standard loss metrics
plus semiotic diagnostics:

| Diagnostic | What It Shows | Healthy Range |
|------------|--------------|---------------|
| `div_norms` | MAH divergence vector L2 norms per hook layer | > 0.1 (not collapsed) |
| `inj_norms` | RRM injection magnitudes at each injection point | ~1.0 (target norm) |
| `r_hat_mean±std` | BEN reflexivity predictions — distribution spread | std > 0.1 (not saturated) |
| `r_hat_min/max` | Range of r̂ across the batch | Should span [-1, 1] |

**Red flags to watch for:**
- `div_norms` → 0: divergence vectors collapsed, MAH not learning
- `r_hat_std` < 0.05: BEN stuck in trivial constant prediction
- `inj_norms` > 5: injection regularization not constraining norms (fixed in v3)
- CE climbing steadily: injections corrupting backbone representations
- Chain loss exactly 0.0: divergence collapsed to a constant

## Checkpointing

The training script saves:
- `training_checkpoint.pt` — full state (adapter weights + optimizer + scheduler + step + epoch) at every validation step, for seamless resumption
- `best_adapter.pt` — adapter weights only, at best validation loss
- `adapter_epoch{N}.pt` — adapter weights at end of each epoch
- `final_adapter.pt` — adapter weights at end of training
- `train_log.jsonl` — all metrics + diagnostics in structured format

## SRT-NLA — Stage 4 of the SRT program

**Activation verbalization — read any hidden state of a frozen backbone as a sentence.**

SRT-NLA is the **fourth stage** of the SRT program:

1. *Stages 1–2* — semiotic theory and pretraining-time architecture
   (Lancaster, 2025 [SSRN 5987495]; Lancaster, 2026a [SSRN 6349978]).
2. *Stage 3* — frozen-backbone bolt-on adapter (the SRT-Adapter
   manuscript under [`arxiv/`](arxiv/); repository-hosted, not yet on
   arXiv).
3. *Stage 4* — **NLA**: a small (~12.7M-param) Activation Verbalizer
   (AV) is trained so that given a target mid-layer hidden vector `v`
   from a fully frozen backbone, it generates text whose own
   re-encoded activation `h` matches `v` under an anisotropy-corrected
   metric `fve_nrm_cen = ½(1 + cos(h−μ, v−μ))`, normalised against a
   random-text floor and a same-source paraphrase ceiling to give a
   backbone-agnostic `ρ_norm ∈ [0, 1]`.

The full Stage-4 framing — Peircean interpretant completion, Kockelman
sieving, Silverstein metapragmatic awareness as decoding capacity, and
the substrate-asymmetry hypothesis (text port strong / hidden-state
port weak) — lives in [`paper_nla.md`](paper_nla.md) §1.5 and §12.

- **Paper (Stage 4)**: [`paper_nla.md`](paper_nla.md)
- **Release notes (Qwen v1)**: [`RELEASE_NOTES_NLA_v1.md`](RELEASE_NOTES_NLA_v1.md)
- **Forward plan**: [`FORWARD_PLAN.md`](FORWARD_PLAN.md)
- **Mission & stakes (historical, superseded units)**: [`docs/nla_mission.md`](docs/nla_mission.md)
- **Architecture & phased plan**: [`docs/SRT_NLA_PLAN.md`](docs/SRT_NLA_PLAN.md)

### Headline numbers

`ρ_norm` is anchored at 0 = random unrelated text, 1 = a same-source
paraphrase. Best-of-K is oracle rerank against the target `v`, which is
available at deploy time.

| Backbone | Layer | Greedy `ρ_norm` | Best-of-64 `ρ_norm` |
|---|---|---|---|
| Qwen2.5-7B | L20 | 0.26 | **0.92** |
| Llama-3.2-3B | L19 (73% depth) | reported in [`paper_nla.md`](paper_nla.md) §10 | reported in §10 |
| Gemma-2-2B | L19 (73% depth) | 0.30 | **1.33** (overshoots paraphrase ceiling) |

The greedy-vs-rerank gap is the central artifact of the program: the
verbalizer *can* express paraphrase-quality outputs, but argmax does
not surface them on the first try. The bag-of-K self-distillation
attempt to close that gap (Lever B) returned a clean negative result
(`paper_nla.md` §6); deploy-time best-of-K oracle rerank (Lever A)
remains the only mechanism that closes the gap on this backbone.

### HF artifacts

| Backbone | Model | Targets dataset |
|---|---|---|
| Qwen2.5-7B | [`RiverRider/srt-nla-av-v1`](https://huggingface.co/RiverRider/srt-nla-av-v1) | [`RiverRider/srt-nla-targets-v1`](https://huggingface.co/datasets/RiverRider/srt-nla-targets-v1) |
| Llama-3.2-3B | [`RiverRider/srt-nla-av-llama32-3b`](https://huggingface.co/RiverRider/srt-nla-av-llama32-3b) | [`RiverRider/srt-nla-targets-llama32-3b-v1`](https://huggingface.co/datasets/RiverRider/srt-nla-targets-llama32-3b-v1) |
| Gemma-2-2B | [`RiverRider/srt-nla-av-gemma2-2b-v1`](https://huggingface.co/RiverRider/srt-nla-av-gemma2-2b-v1) | [`RiverRider/srt-nla-targets-gemma2-2b-v1`](https://huggingface.co/datasets/RiverRider/srt-nla-targets-gemma2-2b-v1) |
| gpt-oss-20b | [`RiverRider/srt-nla-av-gptoss20b`](https://huggingface.co/RiverRider/srt-nla-av-gptoss20b) *(retrieval/codebook decode recommended)* | [`RiverRider/srt-nla-gptoss20b-artifacts`](https://huggingface.co/datasets/RiverRider/srt-nla-gptoss20b-artifacts) |
| gemma-4-31B-it | [`RiverRider/srt-nla-av-gemma4`](https://huggingface.co/RiverRider/srt-nla-av-gemma4) *(vision + text state read-out; retrieval decode recommended)* | [`RiverRider/srt-nla-gemma4-artifacts`](https://huggingface.co/datasets/RiverRider/srt-nla-gemma4-artifacts) |

### Reproducibility caveat

A bug in `scripts/sample_targets.py` (Qwen2.5 sets
`bos_token_id == eos_token_id == 151643`, which caused the BOS prompt
to register as the first EOS and collapsed every target activation into
one constant vector) was fixed on `2026-05-16` (commit `902b746`). All
NLA-branch results before that date are invalidated. The Llama-3.2-3B
and Gemma-2-2B sample paths do not have this trap. The released
SRT-Adapter checkpoints (`v1.0` / `v8a` / `v18` / `v21a` / `v22c_a050`)
are on a separate codepath and are unaffected.

## srt_introspect — adaptive-density reasoning trace (Stage 3 + 4 sidecar)

`srt_introspect` is a read-only product wrapper that turns the trained
adapter (Stage 3) and the activation verbalizer (Stage 4) into a single
generate-with-trace call. It produces text alongside a *non-uniform*
trace: dense narration where the model's internal state is moving fast
(high MAH divergence) and sparse where it's coasting.

```python
from srt_introspect import Trace

t = Trace.load()  # defaults: RiverRider/srt-adapter-v1.0 + RiverRider/srt-nla-av-v1
result = t.generate(
    "Q: What killed the dinosaurs?\nA:",
    max_new_tokens=200,
    budget=12,   # adaptive verbalization slots
    k=8,         # AV samples per slot (consensus)
)
print(result.text)
for s in result.selected():
    print(s.token_idx, repr(s.token), s.divergence, s.regime, "→", s.verbalization)
```

Each `Step` carries `token_idx, token, divergence, regime, r_hat,
verbalization` (the last populated only for scheduler-selected sites).
The scheduler (`srt_introspect.scheduler.quantile_by_density`) places
verbalizations at equal-mass quantiles of the per-token divergence series
so the trace density tracks where the model's metapragmatic state is
changing.

Throughput on an RTX Pro 6000 Blackwell:

| op | latency |
|---|---|
| Trace.load (cold) | ~10s |
| adapter forward (10-tok prompt) | 21 ms |
| AV verbalize, K=8 (32 new tok) | 770 ms |
| 120-token generation + 8 verbalizations | ~7 s |

CLI demo, JSON dump, and self-contained HTML viewer:

```bash
PYTHONPATH=. python scripts/demos/trace_demo.py \
    --max-new-tokens 200 --budget 12 --k 8 \
    --out-json artifacts/trace.json
PYTHONPATH=. python scripts/demos/render_trace_html.py \
    artifacts/trace.json --out artifacts/trace.html
```

Mini benchmark over a small prompt set:

```bash
PYTHONPATH=. python scripts/evals/trace_bench.py \
    --out-dir artifacts/trace_bench
```

## Theoretical Foundation

SRT is grounded in C.S. Peirce's semiotics. Language models process signs
(representamens) but are blind to when meaning forks — when the same word
means different things to different communities. SRT makes the model
*reflexively aware* of its own semiotic processing:

- **MAH** implements metapragmatic awareness: detecting that "freedom" carries
  different interpretive weight in libertarian vs. socialist discourse.
- **RRM** implements reflexive recursion: the model's awareness of its own
  awareness, tracking how divergence propagates through the interpretant chain.
- **BEN** estimates the bifurcation point: where a sign tips from stable
  (subcritical) to contested (supercritical) interpretation.

See [Lancaster (2025)](arxiv/paper.md) — the full SRT-Adapter manuscript
and its LaTeX source live under [`arxiv/`](arxiv/) (`paper.md`, `paper.tex`,
`paper.pdf`). The folder name is forward-looking: this manuscript is
**not yet on arXiv**. The only currently posted Lancaster preprints are
the two SSRN entries cited from [`paper_nla.md`](paper_nla.md) §14
(SSRN 5987495, SSRN 6349978); the SRT-Adapter and SRT-NLA manuscripts
are repository-hosted at the time of writing.

## Versioning policy

Two tiers exist on Hugging Face:

- **Stable product release.** [`RiverRider/srt-adapter-v1.0`](https://huggingface.co/RiverRider/srt-adapter-v1.0) is the only checkpoint we recommend pinning from external code, papers, or downstream products. Semver applies to this lineage going forward (`v1.0`, `v1.1`, `v2.0`, ...).
- **Research checkpoints.** Every other repo of the form `RiverRider/srt-adapter-vNNx*` (e.g. `v8a`, `v18`, `v21b_a070`, `v22c_a050`, `v23*`) is an internal research-iteration release. Weights are open under Apache-2.0 for reproducibility of paper results, but the labels are research generations, not versions in the semver sense — mentally, these are `v0.8a`, `v0.18`, `v0.22c_a050`, etc. They may be moved, retired, or renamed without notice.

If you are integrating SRT into a product (including [`RiverRider/zooL4nD3r-v0.1`](https://huggingface.co/RiverRider/zooL4nD3r-v0.1)), pin `srt-adapter-v1.0`.

## Released checkpoints

| Repo | Tier | Notes |
|---|---|---|
| [`RiverRider/srt-adapter-v1.0`](https://huggingface.co/RiverRider/srt-adapter-v1.0) | **Stable release** | First semver release. Use this for downstream pinning. (Internal lineage: v15a.) |
| [`RiverRider/srt-adapter-v8a`](https://huggingface.co/RiverRider/srt-adapter-v8a) | Research checkpoint | Encoder-as-community headline result (Reddit recall@1 0.484). Paper §5.9. |
| [`RiverRider/srt-adapter-v18`](https://huggingface.co/RiverRider/srt-adapter-v18) | Research checkpoint | CoSENT supervised STS, English-purist tier. Paper §5.14. |
| [`RiverRider/srt-adapter-v21a`](https://huggingface.co/RiverRider/srt-adapter-v21a) | Research checkpoint | mxbai-distilled CoSENT, multilingual-leaning. Paper §5.14. |
| [`RiverRider/srt-adapter-v22c_a050`](https://huggingface.co/RiverRider/srt-adapter-v22c_a050) | Research checkpoint | Souping `v18 + v21a` at α=0.5; MTEB-STS SOTA (mean 0.3744). Paper §5.14. |

## Citation

```bibtex
@misc{lancaster2025srtadapter,
  title  = {The Semiotic-Reflexive Transformer Adapter: Lightweight Semiotic Awareness for Frozen Causal Language Models},
  author = {Lancaster, Burton},
  year   = {2025},
  url    = {https://github.com/space-bacon/SRT},
}
```

See `CITATION.cff` for machine-readable metadata.

## License

Apache-2.0 — see [LICENSE](LICENSE). The released adapter weights on Hugging
Face are also Apache-2.0; the underlying `Qwen/Qwen2.5-7B` backbone is released
under its own Qwen license, which applies whenever the backbone is loaded.
