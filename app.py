"""HF Space entrypoint shim.

Hugging Face Spaces expects an `app.py` at the repo root. The actual
implementation lives in `demo/srt_introspect_app.py`; this file just
calls `build_app()` and launches.
"""

from __future__ import annotations

import os

from demo.srt_introspect_app import build_app, _get_trace, _ON_ZEROGPU, DEVICE  # noqa: E402

demo = build_app()
demo.queue(default_concurrency_limit=1, max_size=20)

if DEVICE == "cuda" and not _ON_ZEROGPU:
    try:
        _get_trace()
    except Exception as e:  # pragma: no cover
        print(f"warmup skipped: {e}")

if __name__ == "__main__":
    # On HF Spaces we let the platform supply server_name/server_port via env;
    # locally we default to 0.0.0.0:7860.
    if _ON_ZEROGPU or os.environ.get("SPACE_ID"):
        demo.launch()
    else:
        demo.launch(
            server_name="0.0.0.0",
            server_port=int(os.environ.get("PORT", 7860)),
        )
