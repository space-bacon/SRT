# Hugging Face Space deployment

The Space and the GitHub repo can share code — push this repo to a Hugging Face
Space, but the **first commit's `README.md` must begin with a YAML front-matter
block** describing the Space. Replace the project README at the Space root with:

```markdown
---
title: SRT · introspect
emoji: 🧭
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: "4.40.0"
app_file: app.py
python_version: "3.10"
pinned: false
hardware: zero-a100
short_description: Adaptive-density reasoning traces from a frozen Qwen-2.5-7B
---

# SRT · introspect

Live demo of the SRT-Adapter (Stage 3) + Activation Verbalizer (Stage 4).
Source: https://github.com/space-bacon/SRT
```

(The body below the front-matter is rendered as the Space's card description;
adjust freely.)

## One-time setup

1. **Create the Space** on https://huggingface.co/spaces with:
   - SDK: `Gradio`
   - Hardware: `ZeroGPU` (free; ~A100 burst, ~5 min wall-time per request)
   - Python: `3.10`
2. **Add the `HF_TOKEN` secret** in *Settings → Variables and secrets*.
   The token needs `read` access and must have accepted the
   [Qwen-2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) gated terms,
   otherwise the backbone will fail to download at boot.
3. **Add the repo as a remote** locally and push:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-user>/srt-introspect
   git push space main
   ```

The Space's build will:
- `pip install -r requirements.txt` (gradio + transformers + torch + spaces)
- import `app.py`, which imports `build_app()` from `demo/srt_introspect_app.py`
- detect `_ON_ZEROGPU=True` (set by the `spaces` shim) and route the
  `cb_generate` callback through `@spaces.GPU(duration=300)`

## First-request latency on ZeroGPU

- Cold: ~60-90 s (GPU acquisition + 17 GB weight download from the HF cache)
- Warm: ~7-10 s per generation (same as the local A6000/Blackwell numbers)

The UI shows a placeholder warning about this in the trace pane.

## Public limits applied in `cb_generate`

- `max_new_tokens` capped at 200 (slider also enforces)
- `prompt` truncated to 1500 chars server-side
- `queue(default_concurrency_limit=1, max_size=20)` — one user at a time,
  reject after 20 queued
- Stop button cancels the in-flight `.click()` event

## Local smoke

```bash
PYTHONPATH=. python app.py
# open http://localhost:7860/
```
