---
license: apache-2.0
tags:
  - srt
  - cross-modal-retrieval
  - sentence-transformers
  - coco
  - webgpu
  - browser
base_model:
  - BAAI/bge-small-en-v1.5
  - intfloat/e5-base-v2
  - thenlper/gte-base
  - sentence-transformers/all-MiniLM-L6-v2
base_model_relation: adapter
datasets:
  - detection-datasets/coco
---

# SRT reader heads

Four text heads, 0.8 to 1.6 MB each, that let an ordinary sentence encoder search the SRT browser gallery of 123,287 COCO photographs in place of a 600M-parameter LLM text tower. Every number below can be recomputed from the files in this repository with `check_head.py`.

**Where they run.** The SRT in-browser chat engine loads `bge-small-en-v1.5` plus `text_head_bge-small_v3gallery.safetensors` as its reader: the encoder embeds every caption, note and passage the tab reads; the head projects text into the gallery's space for photograph retrieval. The same encoder runs the tab's text memory on WebGPU at several hundred passages a second; on a phone it runs in wasm.

## What a head is

The shipped gallery (`gallery_123k_v3.srtidx`, on [RiverRider/srt-browser-head-118k](https://huggingface.co/RiverRider/srt-browser-head-118k)) holds one 1,024-d vector per photograph, projected from Qwen3.8-27B image states (layer 52) by a head fitted against Qwen3-0.6B text. Replacing the text side without touching those vectors is the point: a device that already holds the 130 MB gallery should not download it again because the reader changed.

So the image side is frozen and only a text head is fitted:

```
embedding = encoder(text)              # pooling and prefix as the encoder specifies, L2-normalised
projection = (embedding - mu_txt) @ txt.weight.T + txt.bias     # 1,024-d
score = cosine(projection, gallery_row)
```

Each `.safetensors` file carries three tensors in fp16: `mu_txt` (the encoder's training-set mean, `d_txt`), `txt.weight` (`1024 x d_txt`) and `txt.bias` (`1024`).

The heads were fitted against the fixed gallery vectors on COCO train2017 captions only, three head-initialisation seeds. Evaluation: 5,001 val2017 captions (`replay_123k.json`, each with its gold gallery row) against all 123,287 rows (chance R@1 = 0.000008), plus the 1,000-image sub-pool those captions belong to. Per-seed results: `reader_swap_123k.json`.

## Results (`reader_swap_123k.json`)

Text-to-image retrieval over the full gallery, mean of three seeds with [min, max]:

| reader | params | encoder fp16 | head | t2i R@1 | R@5 | R@10 | median rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| `text_head_e5-base_v3gallery` | 110M | 219 MB | 1.6 MB | **0.142** [0.139, 0.146] | 0.292 | 0.382 | **24** |
| `text_head_gte-base_v3gallery` | 110M | 219 MB | 1.6 MB | 0.132 [0.131, 0.133] | 0.283 | 0.368 | 26 |
| `text_head_bge-small_v3gallery` | 33M | 67 MB | 0.8 MB | 0.115 [0.112, 0.116] | 0.254 | 0.335 | 32 |
| `text_head_minilm_v3gallery` | 23M | 45 MB | 0.8 MB | 0.106 [0.103, 0.107] | 0.251 | 0.333 | 34 |
| Qwen3-0.6B text tower, the previous reader | 600M | 382 MB (Q4) | | 0.109 | 0.244 | 0.331 | 36 |

On the 1,000-image sub-pool: t2i R@1 0.63 to 0.66, i2t R@1 0.74 to 0.80, median rank 1 for every reader. Shuffled-pairing controls score 0.000 with median rank 57,000 to 63,000.

What the table says: with the gallery untouched, a 110M sentence encoder beats the 600M LLM tower by 0.033 R@1 and moves the median correct photograph from rank 36 to 24 of 123,287; bge-small matches the tower at a sixth of the bytes; MiniLM is level at an eighth. The reader and the chat model are separate purchases.

Which to use: **bge-small** is the default in the browser engine (best accuracy per byte, and the same model already runs the tab's text memory). **e5-base** when 220 MB is affordable and retrieval quality matters most; remember its `query: ` prefix. **MiniLM** for the smallest footprint.

## Parity fixtures (`parity_bge-small.json`, `parity_e5-base.json`)

Eight probe captions with the PyTorch encoder embedding, the projection through the head, and the top-5 gallery rows and keys. The engine's candle port reproduces them to a maximum embedding difference of 2e-7, projection cosine 0.999999, top-1 and top-5 identical on every probe; `check_head.py` here recomputes the same three stages from the encoder and the head (`max |embedding gap| 1.79e-07, top-1 8/8, top-5 8/8` on bge-small). That is why no per-runtime recalibration vector ships with these heads: the 4 KB mean that earlier SRT heads carried corrected a Q4 LLM tap, and an fp32 sentence encoder has nothing to correct.

## Checking the numbers

```
pip install sentence-transformers safetensors huggingface_hub numpy torch
python check_head.py --reader bge-small
```

Fetches the head, the fixture, the evaluation set and the gallery from the Hub, runs the parity check, then scores the 5,001 captions against all 123,287 rows (about 2 s on an M2 for the scoring). Expected: `t2i R@1 0.1152  R@5 0.2541  R@10 0.3365  median rank 32`, against the tower's 0.1092 / 0.2442 / 0.3307 / 36.

## Using a head

Python, any of the four:

```python
import torch
from safetensors.torch import load_file
from sentence_transformers import SentenceTransformer

head = load_file("text_head_bge-small_v3gallery.safetensors")           # mu_txt, txt.weight, txt.bias
enc = SentenceTransformer("BAAI/bge-small-en-v1.5")                    # cls pooling, normalised; e5-base wants "query: " + text
e = torch.tensor(enc.encode(["a red double-decker bus"], normalize_embeddings=True))
p = (e - head["mu_txt"].float()) @ head["txt.weight"].float().T + head["txt.bias"].float()
p = torch.nn.functional.normalize(p, dim=-1)                           # compare by cosine against the gallery rows
```

Encoder settings per head: bge-small `cls` pooling, no prefix; e5-base `mean` pooling, prefix `query: `; gte-base and MiniLM `mean` pooling, no prefix; all L2-normalised, max length 64 tokens at fitting time.

Gallery rows: `gallery_123k_v3.srtidx` on [RiverRider/srt-browser-head-118k](https://huggingface.co/RiverRider/srt-browser-head-118k); `read_srtidx` in `check_head.py` reads the format.

## Files

| file | what |
|---|---|
| `text_head_{bge-small,e5-base,gte-base,minilm}_v3gallery.safetensors` | the heads, fp16, seed 0 of the three fitted |
| `reader_swap_123k.json` | per-seed and mean scores, controls, the 0.6B reference row |
| `replay_123k.json` | the 5,001 evaluation captions with their gold gallery rows |
| `parity_bge-small.json`, `parity_e5-base.json` | eight-probe fixtures |
| `check_head.py` | recomputes parity and the replay scores from the files above |

## Scope

These heads read the v3 gallery only; a gallery projected by a different image head needs its own fit. The encoders saw COCO caption text in their own pretraining, so the comparison against the 0.6B tower is a drop-in measurement, not a claim about text the encoders never met; the shuffled controls are the clean rung. Fitted 2026-09-02.
