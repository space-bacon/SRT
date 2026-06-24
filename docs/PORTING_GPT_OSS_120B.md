# Porting SRT to gpt-oss-120b

A runbook for standing up an SRT read-out adapter on OpenAI's `gpt-oss-120b`.
It follows the same staged process that put SRT on Qwen3-235B-A22B (a 94-layer
MoE), which is already validated end to end. gpt-oss is architecturally close to
that target, so most of this is "repeat the Qwen3 process with two new wrinkles."

## Why this is genuinely easy

SRT runs the backbone's decoder layers in a manual Python loop and does all of
its work on the **residual stream** between layers (`srt/adapter.py`). Nothing
in the read-out path touches attention internals or the MoE router. So:

- **MoE is a pass-through.** Taps and injections sit between layers; the sparse
  expert block runs untouched inside each layer. Validated on Qwen3-30B-A3B and
  Qwen3-235B-A22B (128 experts).
- **The read-out heads are tiny** (~12-20M params) and train in **read-only**
  mode: the frozen backbone runs under `no_grad`, gradients flow only into the
  heads. Cost is essentially the cost of inference, independent of backbone
  size. You can even cache taps once and train heads offline.
- **One device.** A DGX Spark (128 GB unified memory, GB10 Grace-Blackwell)
  holds the MXFP4 120b (~63 GB) with room to spare. No multi-GPU sharding, which
  is the part that made the 235B fiddly. This is simpler than the 235B port.

## Target facts to confirm (read these from the model's `config.json`)

`gpt-oss-120b` is `GptOssForCausalLM`, `model_type: gpt_oss`. Do not trust the
numbers below blindly; print the config and verify, because the heads' layer
indices and dtype handling depend on them.

| field | gpt-oss-120b (expected) | why it matters |
|---|---|---|
| `num_hidden_layers` | 36 | sets MAH / inject / community indices |
| `hidden_size` | 2880 | head dims; divergence-std sanity floor |
| `num_local_experts` / `num_experts_per_tok` | 128 / 4 | MoE pass-through; ~5.1B active |
| `num_attention_heads` / `num_key_value_heads` | 64 / 8 | GQA |
| `head_dim` | 64 | rotary |
| `sliding_window` | 128 | **new wrinkle 1** (see below) |
| `layer_types` | alternating sliding/full | **new wrinkle 1** |
| attention sinks | yes (learned per-head) | **new wrinkle 2** |
| `rope_scaling` | YaRN | pass `position_embeddings` per layer |
| `vocab_size` | ~201,088 (o200k_harmony) | CE / lm_head |
| bos / eos ids | **VERIFY** | target-sampling bug guard (see below) |

Resolved head indices for L=36 (auto, `config.resolve_layer_indices`):
`MAH@[9,18,27]`, `inject@[18,27]`, `community@5`. For the NLA verbalizer the
extraction layer is ~73% depth, i.e. **L26**.

`gpt-oss-20b` (24 layers, same width and attention design) is the cheap smoke
rung. Do it first, exactly like Qwen3-8B was the smoke rung for the 235B.

## Environment

- **transformers >= 4.55** (when `gpt_oss` landed). The SRT manual loop is
  verified bit-exact under transformers 5.7.0 too, so any `>=4.55,<6` is fine.
  The old `==4.53.3` pin in the repo predates gpt-oss; bump it in a dedicated
  venv for this work.
- **torch >= 2.7 + cu128** for Blackwell (sm_120/sm_121). The DGX Spark is
  Blackwell, so MXFP4 Triton kernels need a recent triton (>= 3.4) plus the
  `kernels` package: `pip install -U transformers kernels triton`.
- **MXFP4 is the point.** Load the model as shipped (MXFP4 MoE weights, ~63 GB).
  Do **not** dequantize to bf16 (that is ~240 GB and will not fit). If MXFP4
  kernels are unavailable, transformers falls back to bf16 dequant and you OOM,
  so confirm the kernels load.
- One venv: `python -m venv .venv-gptoss && pip install -U "transformers>=4.55" kernels triton torch --index-url https://download.pytorch.org/whl/cu128`, then `pip install -e .` for SRT.

## The two new wrinkles vs Qwen3

Everything else is identical to the Qwen3 port. These two are the only real
engineering items.

### Wrinkle 1: sliding-window layers need the right mask (DONE)

gpt-oss alternates **sliding-window attention** (window 128) and **full
attention** layers per `config.layer_types`. The Qwen3 lesson was: on deep MoE,
any deviation from the reference attention kernel gets amplified by discrete
expert routing into real logit divergence. For Qwen3 the fix was to pass
`attention_mask=None` so SDPA takes the `is_causal=True` fast path, byte-identical
to HF.

That trick does **not** transfer to gpt-oss, for two reasons: (a) passing `None`
would give full causal attention on the sliding-window layers, and (b) HF's own
gpt-oss forward builds **explicit** per-layer masks instead of using the
`is_causal` fast path, so to stay bit-exact you must match it.

**This is already implemented in `srt/adapter.py`.** When the backbone config
declares `layer_types` with a `sliding_window` (gpt-oss does), the adapter
autodetects it (`self._has_sliding`) and routes a per-layer mask: full layers
get a plain causal mask, sliding layers get a banded window mask. Both the
training/probing path (`forward`) and the KV-cache generate path
(`_cached_step`) are handled, the latter masking correctly over the full cached
length so it stays correct past the 128-token window. Every other backbone
(Qwen, Llama, Mistral, Gemma) keeps `_has_sliding = False` and the original
`None` / `is_causal` path, so their numerics are unchanged.

Validate it locally with no download (tiny random `gpt_oss`, fp32 CPU,
bit-exact vs HF forward and cached prefill):

```bash
python -m pytest tests/test_gptoss_smoke.py -q
```

The helpers are `_make_sliding_window_mask` (forward) and
`_make_sliding_window_mask_cached` (KV cache) in `srt/adapter.py`.

### Wrinkle 2: attention sinks (no action, but know they exist)

Each gpt-oss attention layer has learned **sink** logits concatenated into the
softmax. They live inside `GptOssAttention` (`self.sinks`). As long as the manual
loop calls the decoder layer normally (it does), the sinks are applied. Nothing
to add; just do not try to reimplement attention by hand.

## Step-by-step

### 0. Smoke on gpt-oss-20b (cheap, fits in ~16 GB)

First validate the layer API and mask routing locally with **no download**
(tiny random `gpt_oss`, fp32 CPU):

```bash
python -m pytest tests/test_gptoss_smoke.py -q   # bit-exact vs HF forward + cache
```

Then smoke the real 20b on the box. The existing smoke script is
backbone-agnostic and already takes `--backbone`:

```bash
python scripts/qwen3_smoke.py --backbone openai/gpt-oss-20b --dtype bfloat16
```

The five checks are: load, **parity vs HF forward**, tap non-degeneracy
(div-std > 0.1), KV-cache generate, read-only grad isolation. Parity is the one
to watch: in bf16 accept top-1 agreement >= 0.99 on decided positions (margin
> 0.05); in fp32 demand bit-exact `max|diff| = 0`. If parity fails it is the
attention masking, which the tiny CPU test above already exercises, so fix it
there first.

### 1. Verify the target-sampling guard

Qwen2.5 and Qwen3-Base have `bos == eos`, which silently broke target sampling
before commit `902b746`. **Check gpt-oss:**

```python
tok = AutoTokenizer.from_pretrained("openai/gpt-oss-120b")
print(tok.bos_token_id, tok.eos_token_id)
```

If they collide, the 902b746 guard already handles it, but confirm the smoke
script prints the warning. Harmony uses distinct control tokens
(`<|start|>`, `<|end|>`, `<|return|>`), so they likely differ, but verify.

### 2. Config + load on the DGX Spark

```python
from srt.config import SRTConfig
from srt.adapter import SRTAdapter

cfg = SRTConfig(backbone_id="openai/gpt-oss-120b", backbone_dtype="bfloat16")
# Single device, MXFP4 fits in 128 GB unified memory: no device_map needed.
adapter = SRTAdapter(cfg).to("cuda")
```

Let `resolve_layer_indices(36)` pick `MAH@[9,18,27]`, `inject@[18,27]`,
`community@5`. Consider a small sweep to 4-6 taps later; the auto default is the
right starting point.

### 3. Smoke on 120b

```bash
python scripts/qwen3_smoke.py --backbone openai/gpt-oss-120b --dtype bfloat16
```

Single device, so no `--device-map`. Expect all five green, div-std ~1-3,
coherent generation.

### 4. Read-only Phase-A training

```bash
python scripts/train.py \
  --backbone openai/gpt-oss-120b --read-only \
  --train-data data/all_train.jsonl --val-data data/all_val.jsonl \
  --max-val-samples 5000 --batch-size 16 --val-every 2000 \
  --output-dir checkpoints/gptoss_120b_phaseA
```

Always pass `--max-val-samples 5000`. In read-only mode CE is constant (frozen
backbone), which is correct; judge on `bif`, `chain`, and the divergence losses.
On the Spark (one device, MXFP4, memory-bandwidth-bound MoE) expect a modest
step rate; the heads converged at 10-17K steps on every prior backbone. If
memory is tight, cache last-token taps to disk in one pass and train the heads
offline on the cached tensors.

### 5. Held-out probe

```bash
python scripts/phaseA_probe.py --backbone openai/gpt-oss-120b \
  --ckpt checkpoints/gptoss_120b_phaseA/best_adapter.pt \
  --val-data data/all_val.jsonl --n 3000 \
  --out artifacts/regime_calibration/gptoss_120b_phaseA.json
```

Targets from the 235B for a "good" read-out: regime ECE ~0.0005, AUROC ~0.99,
community NMI ~0.62, r_hat Pearson ~0.75 (under-predicts magnitude; an affine
rescale fixes the scale). The probe persists raw community vectors before
clustering, so a missing sklearn will not cost you the forward pass.

## Optional: the NLA verbalizer (read a hidden state as text)

The activation verbalizer (`srt.nla`) is a separate, second port and needs its
own frozen copy of the backbone, which doubles memory. On a single 128 GB Spark
two MXFP4 120b copies (~126 GB) will not both fit, so do the read-out adapter
first and treat the AV as a later effort (possibly on 20b, or with the read-out
adapter unloaded). The AV uses only `backbone(output_hidden_states=True)`, the
most stable API, so it has none of the manual-loop mask concerns; the only cost
is the extra backbone.

## Register it in BlackWindow

Once the read-out adapter (and optionally the AV) exists, expose it through the
BlackWindow runtime by adding one `AdapterSpec`:

```python
from blackwindow.registry import register_adapter, AdapterSpec

register_adapter(AdapterSpec(
    key="gpt-oss-120b",
    label="gpt-oss-120b (L26)",
    backbone_id="openai/gpt-oss-120b",
    extraction_layer=26,                       # ~73% depth
    av_repo="<your-hf>/srt-nla-av-gptoss-120b", # if/when you train the AV
    srt_adapter_repo="<your-hf>/srt-adapter-gptoss-120b",
    srt_adapter_filename="best_adapter.pt",
    notes="MXFP4 on a single DGX Spark. Read-out Phase-A.",
))
```

## Gotchas carried from the Qwen3 port

- **Mask path is everything on deep MoE.** Discrete routing amplifies any
  attention-kernel difference. Get parity bit-exact in fp32 before trusting bf16.
- **MoE read-only is memory-bandwidth-bound, not compute-bound.** Expect <100%
  utilization; that is normal, the experts are being streamed.
- **Do not `.to()` the whole adapter** if you ever shard the backbone. (Not
  needed on a single Spark, but true if you scale out.)
- **Back up checkpoints off the box** after every good val. They are small
  (~32 MB); rsync to local and/or push to HF.
- **Verify `bos != eos`** before sampling AV targets.

## One honest caveat to set expectations

SRT's validated value is **read-out / monitoring**, not capability lift. On the
7B, closed-loop injection did not improve task accuracy (it hurt GSM8K), and
selecting answers by SRT signals lost to plain self-consistency. The genuinely
useful, pre-registered finding is that low divergence predicts wrong answers
(AUROC ~0.71-0.75 held out). So frame a gpt-oss-120b port as "a calibrated
introspection and monitoring channel on a frontier-class open model," which is
exactly what the regime/community/r_hat heads deliver, not as a way to make
gpt-oss answer better.
