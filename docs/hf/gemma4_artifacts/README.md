---
license: apache-2.0
tags:
  - interpretability
  - srt
  - gemma-4
  - retrieval
  - activation-states
pretty_name: SRT-NLA gemma-4-31B-it L47 artifacts
---

# SRT-NLA gemma-4-31B-it artifacts (L47)

**The retrieval indexes that turn a frozen gemma-4-31B-it's L47 states into
full sentences, with zero training.**

Layer 47 is the backbone's cross-modal alignment peak. Encode any L47
state — a text's last token, or the mean over an image's soft-tokens —
and these indexes decode it by nearest-neighbour retrieval
([github.com/space-bacon/SRT](https://github.com/space-bacon/SRT),
`paper_nla.md` §11.6–§11.7).

> **On "peak".** That ranking is measured with the raw-cosine read-out
> these indexes use. A fitted linear probe orders the layers differently
> and puts L20 ahead of L47 (R@1 0.293 against 0.230 on 4,000 held-out
> COCO pairs). The depth profile is a joint property of the backbone and
> the probe, so the two orderings are both real and answer different
> questions. Full nine-layer tables, scaling curves and scripts:
> [`RiverRider/srt-depth-probe-artifacts`](https://huggingface.co/datasets/RiverRider/srt-depth-probe-artifacts).
> L47 remains the tap these artifacts are built and validated at.

## What they power

- **Open-vocabulary image captioning**: 5/5 CIFAR natural images
  retrieve on-topic captions at rank 1 from the 10k pool (paper
  §11.6.3), live in the
  [SRT-Sunstone demo](https://huggingface.co/spaces/RiverRider/srt-sunstone).
- **Text-state read-out** at centered fve 0.695 (NN) inside a calibrated
  anchor frame (replay 0.994 / floor 0.494).
- Retrieval is the primary decoder on this backbone; the generative
  companion for sampled candidates is
  [`RiverRider/srt-nla-av-gemma4`](https://huggingface.co/RiverRider/srt-nla-av-gemma4)
  (best-of-K with oracle rerank).

## Files

| file | contents |
|---|---|
| `caption_index_L47.pt` | `{vectors: (10000, 5376) fp16, texts}` — last-token L47 states of 10k deduped COCO captions. The open-vocabulary image→caption retrieval index. |
| `corpus_index_L47.pt` | `{vectors: (10000, 5376) fp16, texts}` — last-token L47 states of 10k forum passages. The text-side NN baseline pool. |
| `anchors_L47.json` | Centered reference frame: replay 0.994 / NN 0.695 / floor 0.494. |
| `kcurve_ce.jsonl` | Per-target best-of-K rollouts + scores for the K-curve. |
| `decode_probe_ce.jsonl` | Decoded verbalizer samples (paper §11.7 measurement set). |
| `vision_caps_retrieval.json`, `vision_caps_cifar.json` | Image→caption retrieval results (stereo images; CIFAR naturals 5/5 rank-1). |
| `caption_pool.jsonl`, `corpus_targets.jsonl` | The source texts (regenerate either index in ~8 min with `scripts/sample_targets.py --corpus`). |

## Usage (retrieval decode)

```python
import torch, torch.nn.functional as F
from huggingface_hub import hf_hub_download

idx = torch.load(hf_hub_download("RiverRider/srt-nla-gemma4-artifacts",
                                 "caption_index_L47.pt", repo_type="dataset"),
                 map_location="cpu", weights_only=False)
pool, texts = idx["vectors"].float(), idx["texts"]
mu = pool.mean(0)
pc = F.normalize(pool - mu, dim=-1)
# v: your L47 query state (text last-token, or mean over image soft-tokens,
# centered by a mean of the SAME modality — see paper §11.6.3)
sims = pc @ F.normalize(v - mu_query_modality, dim=-1)
print(texts[int(sims.argmax())])
```

Protocol notes: gemma-4 must be loaded as `Gemma4ForConditionalGeneration`
(not CausalLM), and every text re-encode must prepend BOS.
