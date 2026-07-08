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

Retrieval indexes, anchors, and evaluation results for the SRT-NLA work
on a frozen `google/gemma-4-31B-it`
([github.com/space-bacon/SRT](https://github.com/space-bacon/SRT),
`paper_nla.md` §11.6–§11.7).

## The recommended decoder for this backbone is retrieval

The trained verbalizer mode-collapses under greedy decoding on this
chat-tuned host (see
[`RiverRider/srt-nla-av-gemma4`](https://huggingface.co/RiverRider/srt-nla-av-gemma4)),
while zero-training nearest-neighbour retrieval against these indexes
reaches centered fve 0.695 on text and retrieves on-topic captions for
5/5 CIFAR natural images at rank 1 (paper §11.6.3). These indexes power
the open-vocabulary caption panel in the live
[SRT-Sunstone demo](https://huggingface.co/spaces/RiverRider/srt-sunstone).

## Files

| file | contents |
|---|---|
| `caption_index_L47.pt` | `{vectors: (10000, 5376) fp16, texts}` — last-token L47 states of 10k deduped COCO captions. The open-vocabulary image→caption retrieval index. |
| `corpus_index_L47.pt` | `{vectors: (10000, 5376) fp16, texts}` — last-token L47 states of 10k forum passages. The text-side NN baseline pool. |
| `anchors_L47.json` | Centered reference frame: replay 0.994 / NN 0.695 / floor 0.494. |
| `kcurve_ce.jsonl` | Per-target best-of-K rollouts + scores for the K-curve (greedy 0.50 → best-of-32 0.59). |
| `decode_probe_ce.jsonl` | Decoded samples showing the greedy mode collapse. |
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
