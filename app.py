"""HF Space entrypoint shim.

Hugging Face Spaces expects an `app.py` at the repo root. The actual
implementation lives in `demo/srt_introspect_app.py`; this file just
calls `build_app()` and launches.
"""

from __future__ import annotations

import os

from demo.srt_introspect_app import build_app, _get_trace, _ON_ZEROGPU, DEVICE  # noqa: E402

app = build_app()

if DEVICE == "cuda" and not _ON_ZEROGPU:
    try:
        _get_trace()
    except Exception as e:  # pragma: no cover
        print(f"warmup skipped: {e}")

if __name__ == "__main__":
    app.queue(default_concurrency_limit=1, max_size=20).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
