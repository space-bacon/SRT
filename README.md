# SRT — Semiotic-Reflexive Transformer (Adapter Architecture)

**Train once, read everywhere.**

The structure inside frozen language models — meaning, divergence,
reflexivity, even the bridge between images and words — is linearly
readable, and what the taps read is **invariant across host scale (3B →
235B), weight precision (bf16 → 4-bit), and hardware (datacenter GPU →
the chip in your laptop)**. One small trained artifact reads the same
structure anywhere the model runs, from Raspberry-Pi-class devices to
server fleets. Deployment tiers differ in latency and cost, never in
capability.

SRT-Adapter is the instrument: a lightweight module that bolts semiotic
awareness onto any frozen causal language model. The backbone runs
natively — its own embeddings, its own LM head, its own attention. SRT
modules are small taps that **read** divergence from hidden states,
**track** reflexive awareness, and optionally **inject** semiotic
corrections back into the stream. *Meaning forks. SRT sees it.*

## 30-second TL;DR

> - **What:** a ~12 M-parameter adapter that observes a frozen LLM at 3 layers and injects a FiLM correction at 2 of them, exposing per-token semiotic signals (divergence, reflexivity `r̂`, regime) plus a discourse/embedding vector.
> - **Why:** lightweight, portable instrumentation for a frozen backbone — no base-model weight updates, zero cross-entropy degradation, trains in hours at ≈0.17 % of backbone params. The released `v1.0` checkpoint is pinned for downstream compatibility; the sentence-embedding series it came from is a closed line and not where the current results are.
> - **New (July 2026) — the portability result:** the structure these taps read is **invariant across scale, precision, and hardware**. A 22 MB linear head gives a frozen multimodal LLM image↔text retrieval at fully-trained-2018-dual-encoder level (Karpathy 5k i2t R@1 = 0.416), and the *same head* survives a 10× host reduction (31B → 3B, no loss), 4-bit quantization (−0.01 R@1, unchanged weights), and a change of silicon (CUDA datacenter → Apple-Silicon Mac: 97.0% head-space text agreement against a 99.96% same-runtime ceiling, and on-device image→text retrieval within 3 R@1 points of the datacenter reference, 0.640 vs 0.670). One artifact, every deployment tier from Raspberry-Pi-class to datacenter. See [SRT-Sunstone](#srt-sunstone--the-read-out-reads-images-cross-modal) and [docs/CROSSMODAL_LINEAR_HEAD.md](docs/CROSSMODAL_LINEAR_HEAD.md).
> - **New (August 2026) — the whole system runs in a browser tab:** a **382 MB** Qwen3-0.6B and a **2.1 MB** head, in WebAssembly on the CPU, search **123,287** photographs a 27B model encoded offline. No server, no GPU, no API key, and it works with the network off. The 27B never runs at inference; what ships is its reading. Crossing runtimes needs a **4 KB** anchor, without which the read-out is at chance rather than merely degraded. [Try it](https://huggingface.co/spaces/RiverRider/0.6b-reads-27b), [details below](#the-browser-tier-the-whole-system-in-a-tab).
> - **New (August 2026) — a small model reads a large one:** a frozen **Qwen3-0.6B** (382 MB at Q4), handed one **raw gemma-4-31B** hidden state and nothing else, writes a sentence that retrieves the right photograph out of **123,287** at **median rank 25**, against 39 for a human reference caption. The gallery was built by an unrelated 27B tower, so no shared representation carries it, and both controls (another image's state, the mean state) sit at chance. The human-caption comparison is a register effect, not a captioning claim; see [§11.8](paper_nla.md) for why. [Details below](#a-06b-puts-words-to-a-31bs-internal-state).
> - **New (August 2026) — index with one vendor, search with another:** a ridge map between two vendors' image states moves a picture from one company's encoder into another's and lands on the right picture at **r@1 0.8024** (shuffled floor 0.0007), across Qwen3-Omni 30B, Gemma-4 31B, Mistral Small 3.1 and Aria. Embedding lock-in is weaker than assumed. This also **retired our own `retention` metric**: within-vendor text-to-image is 0.1050 against 0.8024 vendor-to-vendor, so the ratio was throttled by the caption head rather than the vendor boundary. [Details below](#index-with-one-vendor-search-with-another).
> - **New (August 2026) — frozen states beat a fine-tuned baseline on chest radiographs:** a linear probe on frozen `gemma-4-31B-it` states scores **0.7590** mean AUROC on all 112,120 images of ChestX-ray14 using the **official split**, against **0.7451** for the dataset authors' fine-tuned ResNet-50 on that same split, ahead on 12 of 14 findings. No radiology training, no fine-tuning, patient-level cluster bootstrap. [Details below](#medical-imaging--frozen-states-linear-probes-split-matched).
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
| **`srt_select`** | Picks one of K model replies by what they compute, no training | 0 |

## `srt_select` — choose among K replies without a test suite

```python
from srt_select import select
best = select(user_message, replies)
```

Run every candidate on the same synthesised inputs, group them by the outputs
they produce, return a member of the largest group. No reference solution, no
test suite, no scoring model, no training, no weights. The entry point and the
argument shapes are recovered from the candidates themselves, so it works on a
chat turn rather than a benchmark row.

| | problems | arms | floor | selected | oracle |
|---|---:|---:|---:|---:|---:|
| HumanEval | 164 | 36 | 0.1868 | **0.4426** | 0.4954 |
| MBPP | 425 | 10 | 0.7185 | **0.8174** | 0.8887 |

That is 82.9% and 58.1% of the headroom an oracle would capture. It beats a
verifier trained on 47,232 execution labels, which reached 0.2639 on HumanEval
and did not transfer to MBPP. Recovering the target from the chat turn alone
costs coverage rather than accuracy: HumanEval resolves on 55.0% of problems and
MBPP on 98.6%, giving 0.3096 and 0.7991 with unresolved problems scored as an
arbitrary pick.

**Three things it does not do.** It is not a correctness check, it returns the
pool's majority, and a pool that is confidently wrong together defeats it.
Its value decays with scale at −0.0704 per decade, from +0.1538 at 3B to +0.0097
at 32B, so it is a weak-model amplifier. And **pooling several models buys
nothing**: over 48 candidates from six frontier models across five labs it solves
the same 154 of 164 problems as the best single member's own eight, a gain of
+0.0000. Point it at one model's samples, not at an ensemble.

### K samples do not cost K times one answer

This is the economics the method rests on, and it is easy to get wrong.

Autoregressive decode is **memory-bandwidth bound, not compute bound**. Each
decode step streams the model weights out of HBM once, and that cost is the same
whether the step advances one sequence or thirty-two. The prompt prefill is paid
once and shared across all K samples. So K samples generated in a *single batch*
cost far less than K separate answers, and the marginal sample is close to free
until K exceeds what one batch holds.

The cost curve therefore has a knee rather than a slope: flat while the batch
absorbs K, stepping each time K spills into another batch. Where the knee sits is
a property of the card and the model, so measure it on the hardware you serve
from with `scripts/k_latency.py`.

This is what makes the ladder result a product claim rather than a curiosity. A
14B with selection scores 0.8706 on MBPP against a 32B's 0.8609 alone. If K=8
cost eight answers you would simply run the 32B; because it does not, you get
32B-class coding from a model that fits far more comfortably.

Two caveats. Under concurrency, batching K samples for one user consumes
capacity that would otherwise serve other users, so it is nearly free for a
single user and a real throughput cost under load. And selection needs the K
samples before it can choose, so it cannot improve a published `pass@1` number,
which is single-sample by definition.

Selecting means executing every candidate, so it is a remote code execution
primitive if pointed at the wrong host. Read `srt_select/sandbox.py` before
deploying. Full documentation in `srt_select/README.md`, method and controls in
`paper_hivemind.md` §5.4, artifacts in
[`RiverRider/srt-hivemind`](https://huggingface.co/datasets/RiverRider/srt-hivemind).

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

- **0.6B reads 27B, entirely in your browser** (no server, no GPU, works offline
  once loaded): <https://huggingface.co/spaces/RiverRider/0.6b-reads-27b>
- **Sunstone Lab** (gemma-4-31B-it chat, captioning, and retrieval on one
  frozen backbone): <https://lab.sunstonenorth.com>
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
| [`RiverRider/srt-browser-head-118k`](https://huggingface.co/RiverRider/srt-browser-head-118k) | Qwen3-0.6B text × Qwen3.8-27B image | 2.1 MB head for the browser tier. Search 123,287 photographs from a tab, offline. Ships with the 4 KB runtime anchor, without which the cross-runtime read-out is at chance. |
| [`RiverRider/srt-verbalizer-v1`](https://huggingface.co/RiverRider/srt-verbalizer-v1) | frozen Qwen3-0.6B reading gemma-4-31B / Qwen3.8-27B states | ~44M prefix that turns one raw hidden state into a sentence. Median rank 20 of 123,287 against 39 for a human caption; both controls at chance. Four checkpoints incl. a documented negative. |

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
| **weight precision** | bf16-trained head applied *unchanged* to 4-bit NF4 states | −0.011 R@1 on the 1,000-image pool; a 42 KB mean recalibration returns 27% of that at R@1, 50% at R@10 |
| **hardware / runtime** | CUDA datacenter → Apple-Silicon MLX, different kernels, different quantization | local states retrieve their datacenter twins at **100 % R@1** through the head ([scripts/local_sunstone.py](scripts/local_sunstone.py)) |

Read together with the earlier read-out ports (Qwen-7B → gpt-oss-20b →
Qwen3-235B), the picture is: **meaning in these substrates is organized
linearly and stably enough that one small artifact reads it anywhere the
model runs.** Retrieval needs a single prefill pass, not generation, so the
same head serves a Raspberry-Pi-class device doing overnight photo tagging,
a Mac doing interactive local search, and a datacenter serving a fleet —
train once, read everywhere. The intelligence lives in the substrate; the
semiotic layer is a set of linear taps small enough to ship as a config
file and small enough to audit by inspection.

Engineering guide (deployment tiers, calibration rules, reproduction recipe):
[docs/CROSSMODAL_LINEAR_HEAD.md](docs/CROSSMODAL_LINEAR_HEAD.md). Head:
[`RiverRider/srt-sunstone-linear-head`](https://huggingface.co/RiverRider/srt-sunstone-linear-head).

- **Live demo**: <https://huggingface.co/spaces/RiverRider/srt-sunstone>
- **Model**: [`RiverRider/Gemma-4-31B-it-SRT-Sunstone`](https://huggingface.co/RiverRider/Gemma-4-31B-it-SRT-Sunstone)
- **Demo source**: [demo/cross_modal_space/](demo/cross_modal_space/)

### The browser tier: the whole system in a tab

The far end of that invariance axis is a web page. A **382 MB** Qwen3-0.6B at
Q4_0 and a **2.1 MB** head run in WebAssembly on the CPU, and search
**123,287** photographs that a 27B model encoded offline. No server, no GPU,
no API key; once the tab has loaded it works with the network off.

The 27B appears nowhere at runtime. It encoded the gallery once, months
earlier, and what ships is the index. That is the deployment shape the whole
program argues for: the large model's reading is a durable artifact, and the
small model is enough to *use* it.

Crossing runtimes is where this gets interesting, and it is the part most
easily got wrong. A head fitted on PyTorch/fp16 states does not transfer to
candle/Q4_0, and the failure is silent, returning confidently ranked wrong
answers. Measured on the head and gallery that ship, 5,001 captions against
all 123,287 images:

| runtime | t2i R@1 | median rank |
|---|---|---|
| PyTorch fp16 (reference) | 0.1092 | 36 |
| candle Q4_0, head as-is | 0.0000 | 44,578 |
| candle Q4_0, + 4 KB anchor | 0.0350 | **176** |

Chance median is ~61,644, so the unanchored read-out is at chance: not
degraded, gone. A 4,096-byte mean vector measured on 200 held-out sentences
restores it to **32%** of the fp16 reference at R@1, and to the **top 0.14%**
of the gallery by median rank. Read both columns. R@1 alone reads like a
broken port; the median says the right photograph is usually near the top and
just rarely first, which is weak for "I feel lucky" and perfectly serviceable
behind a grid of results. An earlier version of this table advertised 85%
recovery, measured on a different head against a 1,000-image pool; that number
was wrong for the deployment and the correction came out of public review.

- **Live demo**: <https://huggingface.co/spaces/RiverRider/0.6b-reads-27b>
- **Head + full measurement**: [`RiverRider/srt-browser-head-118k`](https://huggingface.co/RiverRider/srt-browser-head-118k)
- **Raw 27B states**: [`RiverRider/srt-qwen38-coco-states`](https://huggingface.co/datasets/RiverRider/srt-qwen38-coco-states)
- **Artifact**: [`artifacts/nla/q4/cross_runtime_browser_rung_123k.json`](artifacts/nla/q4/cross_runtime_browser_rung_123k.json)

### Index with one vendor, search with another

Four multimodal backbones from four companies (Qwen3-Omni 30B, Gemma-4 31B,
Mistral Small 3.1, Aria) encode the same gallery. A ridge map fitted between
two vendors' image states, train rows only, moves a picture from one
vendor's space into another's and lands on the right picture:

| | r@1 | pool |
|---|---:|---:|
| **cross-vendor image agreement, direct map** | **0.8024** | 1000 |
| routed through a third vendor | 0.7864 | 1000 |
| shuffled floor | 0.0007 | 1000 |

Embedding lock-in is weaker than usually assumed: a gallery encoded once
remains searchable by a different vendor's encoder.

**The retention metric we started with was the wrong instrument, and this is
the correction.** `retention = cross r@1 / within r@1` reported ~0.99 on
photographs and held across satellite and radiology. It also refused to move
under isotropic noise, spectral truncation, and spectral complement. The
reason is that both terms are limited by the same component: within-vendor
text-to-image retrieval is **0.1050** while vendor-to-vendor image agreement
is **0.8024**. A ratio of two numbers throttled by the same caption head is
close to insensitive to the vendor boundary it is named after. Report the
legs separately.

Anisotropy is load-bearing throughout. Raw mean pairwise cosine on the image
states runs 0.873 to **0.998**; centering on the train mean takes all four
vendors to 0.005 or below. Every number here is centered.

- **Live demo**: <https://huggingface.co/spaces/RiverRider/srt-omni-demo>
- **States**: [`RiverRider/srt-omni-crossvendor-states`](https://huggingface.co/datasets/RiverRider/srt-omni-crossvendor-states)
- **Artifacts**: [`artifacts/nla/omni/triadic_composition_roco.json`](artifacts/nla/omni/triadic_composition_roco.json),
  [`artifacts/nla/omni/geometry_compare_roco.json`](artifacts/nla/omni/geometry_compare_roco.json)

### Single-pass tests could not resolve structure that iteration exposes

Transporting a state around a closed cycle of vendors returns it almost
exactly after one lap: holonomy gap **+0.0043 ± 0.0037** across 18 cycles,
which is the same null every single-pass perturbation gave. Iterating the
same maps separates them cleanly. Hop counts are matched at 12, 24 and 36,
where every route's period divides evenly:

| route | distinct edges | encloses area | 12 | 24 | 36 |
|---|---:|---|---:|---:|---:|
| self-loop | 0 | no | 1.0000 | 1.0000 | 1.0000 |
| there-and-back | 1 | no | 0.9470 | 0.8190 | 0.7220 |
| palindrome | 3 | **no** | 0.8130 | 0.5640 | 0.3880 |
| four-cycle | 4 | yes | 0.7850 | 0.5230 | 0.3200 |

**The first reading of this was wrong and is withdrawn.** We originally
reported a three-way ordering and read it as evidence about enclosed area.
Dipankar Sarkar pointed out that there-and-back also differs by composing a
map with its own approximate inverse, so errors cancel pairwise, and proposed
the palindrome `A → B → C → D → C → B → A`: four vendors, every edge
retraced, zero enclosed area. The palindrome tracks the four-cycle, so area
is not the variable. It also retraces everything, so retracing is not the
protection.

Holding area at zero and retracing fixed, and varying only how many distinct
vendor boundaries a route crosses:

| distinct edges | 12 | 24 | 36 |
|---:|---:|---:|---:|
| 1 | 0.9470 | 0.8190 | 0.7220 |
| 2 | 0.9090 | 0.7170 | 0.5580 |
| 3 | 0.8130 | 0.5640 | 0.3880 |

Monotone. That also orders the original result, where the self-loop crosses
none, there-and-back one, and the four-cycle four.

**Replicated on satellite**, same protocol, only the pixels change. The single
domain was the weakness of the first version of this claim, and it is the same
weakness that cost us the routing recipe:

| distinct edges | 12 | 24 | 36 |
|---:|---:|---:|---:|
| 1 | 0.9650 | 0.7760 | 0.5840 |
| 2 | 0.9170 | 0.5800 | 0.3770 |
| 3 | 0.7820 | 0.4230 | 0.2510 |

Monotone at every hop count in both domains, six of six.

**Caveat carried on the face of the claim**: iterating any non-normal linear
map collapses toward its dominant eigenspace and every route pays that cost.
Hop counts are matched, so read the ordering rather than the size of the
split. The practical consequence is that two models tying on a single-pass
retrieval benchmark is not evidence they carry the same structure.

- **Artifacts**: [`artifacts/nla/omni/semiosis_holonomy_roco.json`](artifacts/nla/omni/semiosis_holonomy_roco.json),
  [`artifacts/nla/omni/holonomy_palindrome_roco.json`](artifacts/nla/omni/holonomy_palindrome_roco.json),
  [`artifacts/nla/omni/holonomy_palindrome_rsicd.json`](artifacts/nla/omni/holonomy_palindrome_rsicd.json)

### A probe trained on one backbone reads another at native accuracy

Retrieval says two vendors agree about *which picture*. It does not say they
agree about *what is in it*. This asks the labelled version, on satellite
imagery: fit a linear scene probe on one vendor's frozen image states, then read
it on a different vendor's states through a ridge map fitted on train rows only.
17 land-use classes, four backbones, 12 cross directions.

| | mean AUROC |
|---|---:|
| native, each vendor probing itself | 0.9507 (spread 0.0277) |
| self-map control | 0.9517 |
| **transported, probe from one backbone read on another** | **0.9484** |
| shuffled floor | 0.5014 |

Transport costs **0.0024**, and 4 of the 12 transported pairs beat the native
target outright. Train once, read anywhere, as a measurement rather than a
slogan.

**The labels are weak and the number must not be quoted without this sentence.**
Scene classes are keyword-matched out of the RSICD captions, which is the same
shape of supervision ChestX-ray14 uses for its fourteen findings, and they are
coarser than chest pathology, which is part of why 0.95 is reachable. The probe
reads the image tower while the label comes from the caption, so the text side
cannot leak the answer.

- **Artifact**: [`artifacts/nla/omni/rsicd_scene_probe.json`](artifacts/nla/omni/rsicd_scene_probe.json)

### One frame for four backbones, and reading them together

Everything above is bilateral: one ridge map per ordered pair, twelve for four
vendors. Fitting a *single* shared frame from all four at once (MAXVAR
generalised CCA) gives four encoders and four decoders, eight maps instead of
twelve, and costs little:

| | satellite | radiology |
|---|---:|---:|
| direct, 12 pairwise maps | 0.8425 | 0.8817 |
| **via one shared frame, 8 maps** | **0.8164** | **0.8400** |
| cost | 0.0261 | 0.0417 |

**Three vendors read together beat the best single vendor reading alone, on all
four targets in both domains, eight for eight.** Satellite gains +0.0345 on
average (mistral 0.8310 → 0.8810), radiology +0.0165 (qwen3omni 0.9670 →
0.9940). The backbones are not redundant: each carries something the others do
not, and pooling recovers it.

**A reading we caught before publishing.** Routing every hop through the shared
frame flattens the edge-count ladder above to 1.0000 at 36 hops. That is not the
frame buying back the degradation: the composed map collapses to rank 128 of
256, exactly the joint width, so later hops cannot discard dimensions already
gone. Truncating every pairwise map to the same rank is the control, and
rank-matched pairwise degrades identically to full-rank (0.3720 → 0.3720 at one
edge, 36 hops). So the flat ladder is not a bottleneck artifact, but it may only
be restating that the joint route reuses one subspace per hop while pairwise
routes rotate between mismatched ones. The artifact carries a `ladder_caveat`
field forbidding comparison against full-rank pairwise.

- **Artifacts**: [`artifacts/nla/omni/joint_frame_roco.json`](artifacts/nla/omni/joint_frame_roco.json),
  [`artifacts/nla/omni/joint_frame_rsicd.json`](artifacts/nla/omni/joint_frame_rsicd.json)

### The caption head is the bottleneck, tested by replacing it

The claim that `retention` is insensitive to the vendor boundary rests on both
of its terms being limited by the caption head. The test that settles it is to
hold the image side fixed and swap the head. An unrelated off-the-shelf
encoder, `all-MiniLM-L6-v2`, as the caption tower on ROCO:

| | native head | swapped head | ratio |
|---|---:|---:|---:|
| within r@1 | 0.0887 | 0.0833 | 0.938 |
| cross r@1 | 0.0853 | 0.0830 | 0.973 |

Both terms scale by nearly the same factor, gap 0.0347. That is what
"limited by the shared head" predicts, and it is the test designed to break
the reframe. Test proposed by Dipankar Sarkar.

- **Artifact**: [`artifacts/nla/omni/head_swap_roco.json`](artifacts/nla/omni/head_swap_roco.json)

## Medical imaging — frozen states, linear probes, split-matched

A linear probe on frozen `gemma-4-31B-it` states, no fine-tuning and no
radiology training, on all 112,120 images of ChestX-ray14 using the
**official `test_list.txt`** (86,524 train / 25,596 test, 30,805 patients,
patient overlap 0). Confidence intervals use a patient-level cluster
bootstrap, since the unit that repeats is the patient rather than the film.

| ChestX-ray14, official split | mean AUROC | method |
|---|---:|---|
| Wang et al. 2017 (dataset authors) | 0.7451 | ResNet-50, fine-tuned end to end |
| **ours** | **0.7590** | frozen backbone, linear probe |
| shuffled floor | 0.5002 | |
| view-position only | 0.5883 | |

Ahead on 12 of 14 findings. Behind on Hernia (−0.0888, 227 positives in the
whole dataset) and Fibrosis (−0.0321). Reference numbers from CheXNet (0.8414)
and Yao (0.8027) are **not** split-matched: those are a random 70/10/20
partition. Which split is harder is not established, since Wang scored 0.7381
random against 0.7451 official, so we assert only that the two are not
comparable. `scripts/cxr_probe.py` labels them as context rather than as a
head-to-head.

**The headline is backbone-specific, and that is a limit on the claim.** The
identical probe, split and protocol run on four backbones gives a spread of
0.057:

| backbone, official split | mean AUROC | vs Wang 0.7451 |
|---|---:|---:|
| **Qwen3-Omni-30B-A3B** | **0.7650** | **+0.0199** |
| Gemma-4-31B-it | 0.7590 | +0.0139 |
| Aria | 0.7080 | −0.0371 |

All three clear the view-position baseline on all 14 findings and sit far above
the shuffled floor, so "frozen general-purpose states carry chest pathology
linearly" holds for every backbone tested. "Ahead of the split-matched
baseline" does not: Aria is behind it. Mistral is absent because a shard was
lost, see the note below.

**A probe fitted on one backbone reads another, and often reads it better than
its own probe does.** Fitting the ridge map on train rows only, four of six
cross directions beat the target backbone's *native* probe:

| | mean AUROC |
|---|---:|
| native, each backbone probing itself | 0.7440 |
| self-map control | 0.7450 |
| **transported across backbones** | **0.7511** |
| round-trip cycle | 0.7426 |
| shuffled floor | 0.5020 |

The best single reading in the whole study is a transported one: Gemma-4's probe
read on Qwen3-Omni's states scores **0.7708**, and Qwen3-Omni's probe read on
Gemma-4's states scores 0.7707, both above either backbone's own probe. Transport
cost is **−0.0071**, meaning it is negative: moving a probe between backbones
costs nothing and on average gains. The satellite scene probe predicted this,
where transport cost 0.0024; pathology turns out to share direction even more
readily than land use.

**Averaging probe scores across backbones extends the lead, with zero added
parameters.** Aria is individually 0.0371 behind Wang and still helps:

| | mean AUROC | vs best single | vs Wang 0.7451 |
|---|---:|---:|---:|
| Qwen3-Omni alone | 0.7650 | | +0.0199 |
| Gemma-4 alone | 0.7590 | −0.0060 | +0.0139 |
| Aria alone | 0.7080 | −0.0570 | −0.0371 |
| **mean of three probes' logits** | **0.7774** | **+0.0124** | **+0.0323** |
| concatenated features | 0.7627 | −0.0023 | +0.0176 |
| control: best single concatenated with itself | 0.7626 | −0.0024 | +0.0175 |

Significant under a paired patient-clustered bootstrap: **+0.0124, CI [+0.0082,
+0.0168]**. Averaging logits adds no parameters at all, so the gain cannot be
capacity and has to be information one backbone holds and the others do not.
This is the same effect the joint-frame run found on retrieval, where three
vendors read together beat the best single vendor eight times out of eight.

The concatenation row went the other way, and the control is why that is
readable. Concatenating the best backbone **with itself** landed at 0.7626
against real concatenation's 0.7627, a gap of 0.0001. So the whole concat effect
is width, not content. Concatenation was never retuned, so read it as untuned
rather than refuted.

**Mistral is missing, and the reason is ours.** Its four encode shards all
finished, but one wrote a truncated `.npz`. The chain gated on the file
*existing* rather than on it being readable, so it proceeded to merge, hit
`BadZipFile`, and then ran an unconditional `rm` that deleted all four shards
including the three good ones. Both faults are fixed in
`scripts/cxr_shard_chain.sh`: the gate now opens the archive before counting a
shard done, and shards are deleted only after a merge that succeeded.

**A metric bug found and fixed, 2026-08-29.** Our `auroc` promised tie-averaged
ranks and did not implement them. Continuous probe scores have no ties, so
0.7590 and every per-finding number are unaffected. The view-position baseline
scores a single binary feature, which is nothing but ties, so its value depended
on platform sort order: 0.5896 on macOS, 0.5827 on Linux, both wrong. The
correct order-invariant value is **0.5883**, verified against scikit-learn over
200 tie-heavy trials, and it has been corrected on every public surface.

Longitudinal CT on real NLST volumes (620 slices, 40 participants, 116
studies): probe AUROC **0.9380**, CI [0.906, 0.964], against a position-only
baseline of 0.5353 and a shuffled floor of 0.4651. 37 of 38 studies rank the
lesion-bearing slice above the others. There is no leaderboard comparison for
this because LUNA16 and LIDC score localisation with FROC, which is a
different task, and manufacturing a comparison would be dishonest.

**Scope**: these labels describe what is visible in the image, so this is
detection and not early detection. Nothing here speaks to catching disease
before it is apparent.

- **Artifacts**: [`artifacts/nla/cxr14_probe_full112k.json`](artifacts/nla/cxr14_probe_full112k.json),
  [`artifacts/nla/cxr14_pool_sweep.json`](artifacts/nla/cxr14_pool_sweep.json),
  [`artifacts/nla/cxr14_ensemble3.json`](artifacts/nla/cxr14_ensemble3.json),
  [`artifacts/nla/cxr14_transport.json`](artifacts/nla/cxr14_transport.json),
  [`artifacts/nla/cxr14_vendor_compare.json`](artifacts/nla/cxr14_vendor_compare.json),
  [`artifacts/nla/nlst_probe.json`](artifacts/nla/nlst_probe.json)

### Where the states live, and one filename that will bite you

The `.npz` state dumps are gitignored, so the copies that matter are elsewhere.
All three backbones cover the identical 112,120 images in manifest row order and
can be indexed against each other directly.

| | rows | published | cold storage |
|---|---:|---|---|
| `cxr14_gemma4_full112k.npz` | 112,120 | `RiverRider/srt-cxr14-frozen-probe` | Supabase `cxr14/`, LaCie |
| `cxr14_aria.npz` | 112,120 | same dataset | same |
| `cxr14_qwen3omni.npz` | 112,120 | same dataset | same |
| `cxr14_gemma4_PILOT35k.npz` | **34,999** | not published | Supabase, LaCie |

**The pilot is the trap.** It sat at `cxr14_gemma4.npz` for a while, which is the
name every script reaches for, and it holds under a third of the images. It has
been renamed on disk and in cold storage, and the mislabelled archive copy was
deleted. Any script that slices states should check `len(z["ok"])` against the
manifest row count before doing anything with the result;
`scripts/build_pooled_demo_subset.py` refuses outright on a mismatch, which is
what caught it.

Chest radiographs themselves are not stored anywhere here. `scripts/get_cxr14.py`
re-fetches them from the NIH release.

### Banked negatives

Results that cut against us, kept because they bound the claims above.

| hypothesis | result |
|---|---|
| Attention-style pooling beats mean for focal findings | **Falsified.** max −0.0537, top16 −0.0225 on focal findings; mean-pool wins at every depth tested |
| Depth of readout matters for the probe | **No.** 0.7600 to 0.7605 across 0.4/0.6/0.8 of backbone depth |
| A within-vendor r@1 of 0.015 bounds where retention is usable | **Withdrawn.** It was one perturbation geometry and did not survive a change of geometry |
| The shared subspace is a thin remnant in low-variance directions | **Refuted.** Cross tracks within at every spectral level, both keeping and dropping the head |
| Interpretants fail to compose (triadic irreducibility) | **Negative.** Routing through a third vendor costs 0.0160 beyond one extra fitted hop; the dyadic reduction holds at one pass |
| Vendor-first routing beats a directly fitted cross-vendor map | **Photographs only.** 12/12 on COCO at p=0.0002, but 8/12 on radiology (p=0.19) and 5/12 on satellite (p=0.81). Published as general and withdrawn to one domain |
| The four-cycle penalty is about enclosed area | **Withdrawn.** A palindrome route with zero area tracks the four-cycle. Degradation is monotone in distinct vendor boundaries crossed, which was confounded with area |
| The chest-radiograph result holds for any frozen backbone | **Scoped.** Across three backbones the spread is 0.057: Qwen3-Omni 0.7650, Gemma-4 0.7590, Aria 0.7080. Linear presence of pathology replicates on all three; beating the split-matched baseline does not, since Aria is 0.0371 behind it |
| Concatenating backbones' features beats either alone | **Falsified as run.** 0.7627 against the best single 0.7650, and the duplicate-vendor control scored 0.7626, within 0.0001. The whole effect is width, not content. Averaging logits, which adds no parameters, gains +0.0124 instead. Untuned, not refuted |
| A shared frame buys back the loop degradation | **Not established.** The via-joint ladder is flat, but the composed map collapses to exactly the joint width. Rank-matched pairwise rules out the trivial bottleneck reading; what remains may only restate that one subspace is reused every hop |
| Concatenating two backbones' features beats either alone | **Falsified as run.** 0.7543 against Gemma-4's 0.7590. The duplicate-vendor control also lost 0.0022, so a wider probe is worse-conditioned at these hyperparameters whatever fills the columns. Untuned, not refuted. Averaging logits, which adds no parameters, wins instead |
| Pathology lives in a backbone-specific direction | **Refuted.** A probe fitted on one backbone, read on another through a train-only ridge map, beats the target's own probe in 4 of 6 directions. Transport cost is −0.0071 |

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

### A 0.6B puts words to a 31B's internal state

Every verbalizer above reads a state produced by its own backbone. This one
reads across models: a **frozen Qwen3-0.6B** (382 MB at Q4) with a 44.5M
prefix MLP is handed one **raw gemma-4-31B layer-47 image state** (d = 5376)
and writes a sentence. Nothing about the photograph reaches the small model
except that vector, which the 31B produced and which the 0.6B has never had
an image with which to make.

Scoring stays outside both models. The generated caption is re-encoded with
the shipped browser head and retrieved against all **123,287** gallery images,
a gallery built by an unrelated **Qwen3.8-27B** tower. No shared representation
can carry the result. On val2017, held out of both head and verbalizer
training:

| arm | R@1 | median rank |
|---|---|---|
| the image's own state | **0.120** | **25** |
| a human reference caption | 0.101 | 39 |
| another image's state | 0.000 | 62,970 |
| the mean state | 0.000 | 59,408 |

Chance median is ~61,644, so both controls sit at chance. The controls are the
result: the *foreign* arm is the real arm's vectors rolled by one position and
it returns the real arm's captions rolled by one position, while the *mean* arm
emits a single sentence for every input.

Running it again on the **Qwen3.8-27B** states that actually built the gallery
reaches **median 20**, against **25** for the cross-model gemma run. Five ranks
in 123,287. That near-parity matters: if a representation shared between the
two towers were carrying the result rather than the sentence being descriptive,
the matched pair should dominate. It does not.

**Read the human-caption row carefully.** This is not better captioning. Two
measurable things produce it, neither of them superior description. The first
is register: the model enumerates whole-scene inventory, which this head
recovers well (detection AUC 0.883 over 80 COCO categories), while the human
references foreground arrangement and oddity ("a woman *stands*", "mounted
*upside-down*"), which the same head is documented *not* to recover. The second
is length: the metric rewards naming more true things about a scene, and a
human reference names one scene once. The EOS checkpoint that stops on its own
in eleven tokens drops to median 46, below the human caption it used to beat,
which is that mechanism showing up directly. Gold is
one reference caption, not best-of-five. What the number does establish is that
a 382 MB model can describe a 31B's reading precisely enough to identify the
photograph among 123,287 candidates. Details and caveats:
[`paper_nla.md`](paper_nla.md) §11.8. Checkpoints:
[`RiverRider/srt-verbalizer-v1`](https://huggingface.co/RiverRider/srt-verbalizer-v1).
Live under **02 · Read an image** at
[lab.sunstonenorth.com](https://lab.sunstonenorth.com), where you can upload
your own photograph and see the same reader given no record beside it.

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
| [`RiverRider/srt-adapter-v22c_a050`](https://huggingface.co/RiverRider/srt-adapter-v22c_a050) | Research checkpoint | Souping `v18 + v21a` at α=0.5; best of the STS series (mean 0.3744 over 40 splits, our harness). Superseded line, kept for provenance. Paper §5.14. |
| [`RiverRider/srt-cxr14-linear-probe`](https://huggingface.co/RiverRider/srt-cxr14-linear-probe) | Probe | `Linear(5376, 14)` on frozen gemma-4-31B-it. 0.7590 on the official ChestX-ray14 split. |
| [`RiverRider/srt-cxr14-pooled-probe`](https://huggingface.co/RiverRider/srt-cxr14-pooled-probe) | Probe | Three backbones' probes plus their normalisation. Logit average scores 0.7774, +0.0323 over the split-matched baseline, with no added parameters. |
| [`RiverRider/srt-cxr14-frozen-probe`](https://huggingface.co/datasets/RiverRider/srt-cxr14-frozen-probe) | Dataset | 4.51 GB. All three backbones' states for the same 112,120 images, the manifest, and every result json. |
| [`RiverRider/srt-hivemind`](https://huggingface.co/datasets/RiverRider/srt-hivemind) | Dataset | 529 MB. 110,704 generations across 97 arms, 12 models' states, the selection results, and the `srt_select` package. |

Live surfaces: [`srt-cxr14-probe`](https://huggingface.co/spaces/RiverRider/srt-cxr14-probe)
reads a held-out film with one backbone or with all three side by side, and
[`0.6b-decodes-31b`](https://huggingface.co/spaces/RiverRider/0.6b-decodes-31b)
lets a 0.6B model describe a 31B's hidden state, then scramble or interpolate it.
Both run on CPU against precomputed states, so neither waits for a GPU.

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
