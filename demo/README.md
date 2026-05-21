# SRT-NLA Demo

A single Gradio app exposing three views of the trained Activation
Verbalizer (AV) sitting on top of three frozen backbones (Qwen-2.5-7B,
Llama-3.2-3B, Gemma-2-2B):

1. **Playground** - prompt -> hidden state -> ranked candidate
   verbalisations with raw and centred round-trip fidelity.
2. **Live thought trace** - watch the backbone generate token by token
   while the AV periodically verbalises its running hidden state.
3. **Steer by editing** - verbalise the prompt's hidden into plain
   English, edit the text, re-encode it, and patch the difference into
   layer L while the backbone generates a new continuation.

All three tabs share a backbone selector at the top and a lazy in-memory
cache of loaded backbones. The first call on a given backbone downloads
the AV checkpoint and the centring pool from HuggingFace.

## Run locally on a GPU box

```bash
cd /workspace/srt-adapter
pip install -r demo/requirements.txt
python demo/app.py
```

By default the app launches on `http://0.0.0.0:7860` and serves all
three backbones lazily. Set `NLA_DEFAULT_BACKBONE` to one of
`qwen2.5-7b`, `llama-3.2-3b`, `gemma-2-2b` to change the initial
selection.

Hardware (bf16):

| Backbone        | VRAM  | First-load time |
|-----------------|------:|----------------:|
| Qwen-2.5-7B     | ~16GB | ~60s            |
| Llama-3.2-3B    | ~8GB  | ~30s            |
| Gemma-2-2B      | ~6GB  | ~20s            |

## Deploy as an HF Space

The app supports both standard GPU Spaces and **ZeroGPU**. When the
`spaces` package is importable and `SPACES_ZERO_GPU` is set in the
environment (HF Spaces does this automatically on a ZeroGPU tier), the
three callbacks are wrapped with `@spaces.GPU` and request CUDA on
demand. Otherwise they run on whatever device `torch.cuda.is_available()`
reports.

1. Create a new Space (Gradio template). For ZeroGPU pick the
   *zero-a10g* tier (free); otherwise pick *A10G* or larger.
2. Mirror this `demo/` directory and the `srt/` package into the Space
   repo (the app imports `srt.nla.verbalizer`).
3. Add `HF_TOKEN` as a secret if any of the backbones is gated for your
   account (Llama-3.2-3B is gated).
4. Set `app_file: demo/app.py` in the Space `README.md` frontmatter.

Notes on ZeroGPU:

- First call on a backbone loads weights from HuggingFace; the duration
  budget on each call is 120s (180s for the steer/trace tabs). Qwen-7B
  cold-start can saturate this — prefer Gemma-2-2B as the default and
  only switch to Qwen on Pro hardware.
- Models stay resident in the persistent control process between calls,
  so subsequent invocations only pay the GPU-attach overhead.

## How it works

Each `NLAPipeline` holds:

- the frozen backbone (`AutoModelForCausalLM`, `requires_grad=False`),
- the AV (`srt.nla.verbalizer.ActivationVerbalizer`) loaded from the
  matching `RiverRider/srt-nla-av-*` HF repo,
- a centring pool (2K real layer-L last-token hiddens from the matching
  `RiverRider/srt-nla-targets-*` HF dataset) used to compute centred
  cosine fidelity that corrects for the backbone's anisotropic mean.

The verbaliser is invoked through `av.generate(v, ...)`, which prepends
`v` (projected) plus a learned static prefix to the backbone's
input-embeddings stream and runs greedy or sampled generation. The
backbone's weights are never touched.

Steering is a single `register_forward_hook` on
`backbone.model.layers[L]` that adds the layer-L residual difference
between the re-encoded edited text and the original prompt; once the
generation is finished the hook is removed.

## Caveats

- The AV is trained on layer-L last-token hiddens of natural-language
  prompts. Out-of-distribution hiddens (e.g. mid-token, code, foreign
  scripts) will produce lower fidelity verbalisations.
- Best-of-K with the centred metric is the default ranking. Raw cosine
  is also reported for comparison; raw is biased upward by the
  backbone's anisotropic mean (see paper §3).
- Steering with `|alpha| > 1` often pushes the model off-distribution
  fast; small values (0.1 - 0.5) are usually more interesting.
