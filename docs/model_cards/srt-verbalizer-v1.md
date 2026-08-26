---
license: apache-2.0
language:
- en
tags:
- interpretability
- activation-verbalization
- frozen-backbone
- readout-head
- cross-modal
- image-captioning
base_model: Qwen/Qwen3-0.6B
pipeline_tag: text-generation
library_name: pytorch
---

# SRT Verbalizer v1 — a 0.6B reads a large model's record aloud

When a large vision-language model looks at a photograph, it forms an internal
reading of it: a few thousand numbers, partway through the network. That record
is not hidden and not mysterious. The SRT program's whole finding is that it is
orderly enough to read with small, auditable instruments. What it is not, is
**in words**.

This is the translator. A frozen **Qwen3-0.6B**, plus a ~44M-parameter prefix
network, takes one such record and writes a sentence describing the photograph.
The small model has no vision path. It never sees the image. It only ever
receives the vector.

Measured on 5,000 held-out photographs, the sentence it writes retrieves the
correct image out of **123,287** at **median rank 20**, against **39** for a
human-written reference caption.

> Not mind reading. Reading aloud. The record was always legible; this puts it
> in English.

## What is in this repository

| file | reads | d | params | median rank |
|---|---|---|---|---|
| `qwen38_27b_L52.pt` | Qwen3.8-27B layer 52 | 5120 | 44.1M | **20** |
| `gemma4_31b_L47.pt` | gemma-4-31B layer 47 | 5376 | 44.6M | **25** |
| `browser_gallery_1024.pt` | the shipped gallery vector | 1024 | 35.7M | 18 † |
| `gemma4_31b_L47_eos.pt` | gemma-4-31B layer 47 | 5376 | 44.6M | 46 ‡ |

† Not comparable to the others. See *Which checkpoint to use*.
‡ A documented negative. See *Negative results*.

## The result, and the controls that make it one

A plausible-sounding caption proves nothing: a model that learned COCO's
caption prior would produce fluent sentences about nothing. So the arms below
matter more than the headline. All are val2017 photographs, held out of both
head training and verbalizer training, scored against the full 123,287-image
gallery. Chance median is ~61,644.

**Qwen3.8-27B layer 52** (`qwen38_27b_L52.pt`)

| arm | R@1 | median rank |
|---|---|---|
| the image's own record | **0.123** | **20** |
| a human reference caption | 0.101 | 39 |
| **another image's record** | 0.000 | 63,541 |
| **the mean record** | 0.000 | 59,911 |

**gemma-4-31B layer 47** (`gemma4_31b_L47.pt`)

| arm | R@1 | median rank |
|---|---|---|
| the image's own record | 0.120 | 25 |
| a human reference caption | 0.101 | 39 |
| another image's record | 0.000 | 62,970 |
| the mean record | 0.000 | 59,408 |

Both controls sit at chance, and they are legible in the text itself:

- **another image's record**: the vectors are rolled by one position, and the
  captions come out rolled by one position. The words follow the vector.
- **the mean record**: one sentence for every input, every time. *"A man is
  standing on a bench next to a basket."* Fluent, plausible, about nothing.
  That is the floor a real reading has to clear.

### The cross-model control

The gemma-4-31B checkpoint is scored by a gallery an **unrelated Qwen3.8-27B**
tower built. No shared representation is available to carry that result, and it
lands within five ranks of the matched pair. If a shared representation were
doing the work rather than the sentence being descriptive, the matched arm
should dominate. It does not.

## Caveats that travel with the number

**It is not better captioning than a human.** It scores above a human reference
for two measurable reasons, neither of which is superior description.

1. **Register.** The model enumerates whole-scene inventory, which is what this
   retrieval head recovers well (detection AUC 0.883 over 80 COCO categories).
   Human references foreground arrangement and oddity: *"a woman **stands** in
   the dining area"*, *"a stop sign mounted **upside-down**"*. That is exactly
   the information the head is documented not to carry. The humans encode more;
   the instrument reads less of it.
2. **Length.** The metric rewards naming more true things about a scene, and a
   human reference names one scene once. Given more tokens this model continues
   the reading and names more, and enough of that is correct to help. Forcing
   length artificially by repetition makes it *worse* (see below), so the
   verbosity that helps is genuine continued reading, not filler.

**Gold is one reference caption**, not the best of an image's five.

**Scored in one instrument.** These are retrieval numbers through one head and
one gallery. A caption-similarity metric such as CIDEr would very likely favour
the human references, and that would not contradict anything here.

## Which checkpoint to use

- **`qwen38_27b_L52.pt`** if your records come from the tower that built the
  shipped gallery. Best number, single coherent system.
- **`gemma4_31b_L47.pt`** if you want the scientifically cleanest claim. Its
  scoring gallery shares no weights with the model being read, so it is the arm
  that rules out a shared-representation shortcut. This is the checkpoint the
  Sunstone Lab serves.
- **`browser_gallery_1024.pt`** for deployment where only the projected gallery
  vector is available, such as a browser holding an index. Every image in the
  gallery becomes describable with **no extra download**. Its median of 18 is
  **not comparable** to the others: the gallery row it reads is also the
  retrieval target, so the input carries the answer more directly than a hidden
  state does. The controls still discriminate and training was cross-entropy on
  captions rather than on retrieval, so the reading is real, but do not put that
  18 in a table beside the 20 and 25.

## Negative results

Kept because they were expensive to learn and they constrain what to try next.

**Teaching it to stop costs retrieval.** No training target originally contained
an end-of-sequence token, because Qwen's tokenizer does not append one, so the
model never learned that a caption ends and greedy decoding loops the last
clause until the budget runs out. Appending EOS fixes the text completely: it
stops on its own in 11–15 tokens, no repetition, no trimming needed. It also
drops to **median 46**, below the human caption it used to beat. Saying its
piece in eleven tokens names fewer objects. That checkpoint is included as
`gemma4_31b_L47_eos.pt` because its prose is strictly better and some uses will
prefer it.

**Length by repetition is worse than brevity.** Joining three of an image's
five reference captions into one target gives length, and the model fills it by
emitting the most likely caption three times: given only the vector it cannot
know which three references are wanted, so the mode repeated is the lowest-loss
output. **Median 68.** Abandoned at a third of training.

## How it works

```
image ──► large model ──► hidden state (5376-d or 5120-d)
                              │
                              ▼
                    prefix MLP  d_in → 2048 → 16 × 1024      (the only trained part)
                              │
                              ▼
                16 soft tokens ──► frozen Qwen3-0.6B ──► a sentence
```

Training: cross-entropy on COCO captions for 118,287 `train2017` images and
591,753 captions, 3 epochs, batch 32, lr 3e-4, bf16, ~30 min on one RTX PRO
6000. The backbone is frozen throughout; only the prefix learns.

Raw states are strongly anisotropic, so inputs are centred by the train-split
mean and scaled by its mean radius. **Those statistics ride along in the
checkpoint** (`mu`, `sd`) and must be reapplied at inference: feeding an
uncentred vector silently degrades the output rather than failing.

## Usage

```python
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ck  = torch.load("gemma4_31b_L47.pt", map_location="cpu", weights_only=False)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
lm  = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B",
                                           dtype=torch.float32).eval()

class Prefix(torch.nn.Module):
    def __init__(self, d_in, d_model, n_tok, hidden=2048):
        super().__init__()
        self.n_tok, self.d_model = n_tok, d_model
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, n_tok * d_model))
    def forward(self, v):
        return self.net(v).view(v.size(0), self.n_tok, self.d_model)

pre = Prefix(ck["d_in"], ck["d_model"], ck["n_tok"], ck.get("hidden", 2048))
pre.load_state_dict(ck["prefix"]); pre.eval()

v = np.load("one_image_state.npy")              # (5376,) gemma-4-31B layer 47
x = torch.tensor((v - ck["mu"]) / ck["sd"],     # the frame it was fit in
                 dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    soft = pre(x)
    out = lm.generate(inputs_embeds=soft,
                      attention_mask=torch.ones(soft.shape[:2], dtype=torch.long),
                      max_new_tokens=32, do_sample=False,
                      pad_token_id=tok.eos_token_id)
print(tok.decode(out[0], skip_special_tokens=True).strip())
```

The non-EOS checkpoints will not stop on their own; trim to the last complete
sentence for display, or use `gemma4_31b_L47_eos.pt`.

## Try it

The Sunstone Lab serves this live at
[lab.sunstonenorth.com](https://lab.sunstonenorth.com) under **02 · Read an
image**: upload a photograph, gemma-4-31B encodes it, and this model says what
is in the record beside the same reader given no record at all. Warm latency is
about 2 s to encode and 2 s to read.

Note that the prefix was fit on CUDA/bf16 states and reads Apple-MLX/Q4 states
from the Lab correctly with no adaptation, which is the program's
substrate-invariance result appearing in a serving path.

## Provenance

- Paper: [`paper_nla.md` §11.8](https://github.com/space-bacon/SRT/blob/main/paper_nla.md)
- Code: [`scripts/train_shared_space_verbalizer.py`](https://github.com/space-bacon/SRT/blob/main/scripts/train_shared_space_verbalizer.py),
  [`eval_shared_space_verbalizer.py`](https://github.com/space-bacon/SRT/blob/main/scripts/eval_shared_space_verbalizer.py),
  [`build_fullstate_pairs.py`](https://github.com/space-bacon/SRT/blob/main/scripts/build_fullstate_pairs.py)
- Every arm above is a banked JSON under
  [`artifacts/nla/verbalizer/`](https://github.com/space-bacon/SRT/tree/main/artifacts/nla/verbalizer)
- Input states, already published, no re-encode required:
  [`srt-nla-gemma4-artifacts`](https://huggingface.co/RiverRider/srt-nla-gemma4-artifacts)
  (`procrustes/train_pairs/`) and
  [`srt-qwen38-coco-states`](https://huggingface.co/datasets/RiverRider/srt-qwen38-coco-states)
  (`raw118k/`)
- Scoring head and gallery:
  [`srt-browser-head-118k`](https://huggingface.co/RiverRider/srt-browser-head-118k)

## Citation

```bibtex
@software{lancaster_srt_verbalizer_2026,
  author = {Lancaster, Burton},
  title  = {SRT Verbalizer v1: reading a large model's record aloud with a 0.6B},
  year   = {2026},
  url    = {https://huggingface.co/RiverRider/srt-verbalizer-v1}
}
```
