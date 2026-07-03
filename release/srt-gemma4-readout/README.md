---
license: apache-2.0
base_model: google/gemma-4-31B-it
base_model_relation: adapter
library_name: pytorch
tags:
  - srt
  - semiotic
  - read-out
  - interpretability
  - cross-modal
  - discourse-community
  - frozen-backbone
pipeline_tag: feature-extraction
---

# SRT Cross-Modal Read-Out — gemma-4-31B

A **12.3 M-parameter side-channel read-out** for a **frozen `google/gemma-4-31B-it`**.
It reads the model's residual stream and produces a discourse-community
*interpretant*: a compact code that says which community of language use a passage
belongs to. Trained on **text only**, it transfers to **images** with zero image
training, which is the point — the SRT read-out is **semiotic, not linguistic**.

- **Base model:** `google/gemma-4-31B-it` (frozen; never modified)
- **Adapter type:** read-out / side-channel (not a fine-tune, not LoRA). The
  backbone runs unchanged; the head only *reads* hidden states.
- **Trainable parameters:** 12.3 M (community head + 3 MAH divergence heads +
  chain predictor). Backbone frozen at 32 B.
- **Read layers:** community head @ text-layer 8; MAH divergence @ layers
  15/30/45 (of 60).

## What it does

1. **Discourse-community read-out (text).** Given a passage, the community head
   pools the residual stream at layer 8 into a 64-dim code and assigns it to one
   of 35 discourse communities. On held-out passages: **centered top-1 0.535**
   against a 0.029 chance floor (18.7×).
2. **Cross-modal transfer (images).** Applied to gemma-4 image soft-tokens with
   **no image training**: images cluster by semantic class (kNN **0.64**, chance
   0.10) and map to sensible discourse communities (cars → `cars`,
   frogs/birds → `biology`, deer/horse → `gardening`, cats/dogs → `knitting`).
   Image↔word retrieval **0.27** (chance 0.10).
3. **Metacognition read (secondary, honest).** On the ginigen Metacognition-Bench,
   a mid-layer probe over the frozen stream predicts gemma-4's *own* trap
   failures at **AUROC 0.70**, beating the model's verbalized confidence (0.66)
   and the last layer (0.67); on the pivot-detection subset the gap widens
   (0.69 vs 0.53). Caveat: only 14 failures in 300, so this is suggestive, not
   conclusive.

## How it was trained

- Frozen gemma-4-31B (`Gemma4ForConditionalGeneration`), bf16.
- Corpus: 150 K discourse passages across 35 communities (Reddit-derived).
- Objective: supervised-contrastive on `community_id` (grouped sampling,
  8 communities × 4 passages per batch) + self-supervised MAH divergence/chain +
  a divergence target-norm penalty. Backbone frozen; only the head updates.
- **Checkpoint selected by held-out community accuracy**, not training loss
  (train loss is dominated by batch composition and mis-selects; see the
  selection curve in the repo). Selected step 2250 of 3000.

## Usage

```python
import torch
from transformers import Gemma4ForConditionalGeneration, AutoTokenizer
from srt.config import SRTConfig
from srt.modules.community import CommunityDiscoveryHead

MID = "google/gemma-4-31B-it"
ck = torch.load("readout_selected.pt", map_location="cpu", weights_only=False)
cc = ck["config"]

tok = AutoTokenizer.from_pretrained(MID)
model = Gemma4ForConditionalGeneration.from_pretrained(
    MID, dtype=torch.bfloat16, device_map="cuda").eval()

cfg = SRTConfig(backbone_id=MID)
head = CommunityDiscoveryHead(cfg.community, cc["d_backbone"]).cuda().eval()
head.load_state_dict({k[2:]: v for k, v in ck["heads"].items() if k.startswith("0.")})

enc = tok("Another stoic maxim says...", return_tensors="pt").to("cuda")
hs = model(**enc, output_hidden_states=True).hidden_states
code = head(hs[cc["community_layer"]].float(), attention_mask=enc.attention_mask).encoded
# `code` is the 64-dim discourse interpretant; compare to community centroids.
```

Code, training/eval/selection scripts, and the cross-modal demo:
https://github.com/space-bacon/SRT

## Limitations

- **Not a classifier.** The output is a discourse-community code, not an object
  or topic label. Image mappings reflect *discourse* associations, not taxonomy.
- **Image↔word is lossy** (0.27): the 64-dim code discards fine visual detail.
- **Metacognition result is underpowered** (14 failures / 300); treat as a
  direction, not a headline.
- Requires the full 62.5 GB gemma-4-31B backbone to compute hidden states; the
  head itself is tiny but the backbone is not.

## License

Apache-2.0. Base model governed by Google's gemma-4 license.
