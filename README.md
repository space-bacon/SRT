# SRT — Semiotic-Reflexive Transformer (Adapter Architecture)

**Meaning forks. SRT sees it.**

SRT-Adapter is a lightweight module that bolts semiotic awareness onto any
frozen causal language model.  The backbone runs natively — its own embeddings,
its own LM head, its own attention.  SRT modules are small taps that **read**
divergence from hidden states, **track** reflexive awareness, and optionally
**inject** semiotic corrections back into the stream.

## Architecture

```
tokens ──► Backbone Embeddings (native, frozen)
               │
         ┌─────┴─────┐
         │  Layer 0-6 │  (frozen)
         └─────┬─────┘
               │
         ┌─────┴─────┐
  ┌─────►│  Layer 7   │──────► MAH₁ reads divergence ──► RRM step
  │      └─────┬─────┘
  │            │
  │      ┌─────┴─────┐
  │      │ Layer 8-13 │  (frozen)
  │      └─────┬─────┘
  │            │
  │      ┌─────┴─────┐
  ├─────►│  Layer 14  │──────► MAH₂ reads ──► RRM step ──► inject
  │      └─────┬─────┘                                       │
  │            │◄────────────────────────────────────────────┘
  │      ┌─────┴─────┐
  │      │ Layer 15-20│  (frozen, with semiotic correction)
  │      └─────┬─────┘
  │            │
  │      ┌─────┴─────┐
  └─────►│  Layer 21  │──────► MAH₃ reads ──► RRM step ──► inject
         └─────┬─────┘                                       │
               │◄────────────────────────────────────────────┘
         ┌─────┴─────┐
         │ Layer 22-27│  (frozen, with semiotic correction)
         └─────┬─────┘
               │
         Backbone LM Head (native, frozen) ──► logits + CE loss
               │
         BEN (from RRM meta-state) ──► r̂, regime, modulation
```

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
from srt.config import build_config_from_json
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import torch

repo = "RiverRider/srt-adapter-v1.0"          # or RiverRider/srt-adapter-v8a
cfg  = build_config_from_json(hf_hub_download(repo, "config.json"))
adap = SRTAdapter(cfg).cuda().eval()
adap.load_state_dict(load_file(hf_hub_download(repo, "adapter.safetensors")), strict=False)
tok  = AutoTokenizer.from_pretrained(cfg.backbone_id)

enc = tok("meaning forks here", return_tensors="pt").to("cuda")
with torch.no_grad():
    out = adap(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
print(out.r_hat.mean().item(), out.community_output.encoded.shape)
```

See [examples/](examples/) for end-to-end loading, scoring, and sentence-encoding scripts.

### Live demos

- v1.0 demo: <https://huggingface.co/spaces/RiverRider/srt-adapter-v1.0-demo>
- v8a demo: <https://huggingface.co/spaces/RiverRider/srt-adapter-v8a-demo>

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

See [Lancaster (2025)](arxiv/paper.md) — the full paper and arXiv source live
under [`arxiv/`](arxiv/) (`paper.md`, `paper.tex`, `paper.pdf`).

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
