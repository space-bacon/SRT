"""Pre-bake a trace HTML snippet that the demo shows on first load.

Usage (on a GPU box):

    PYTHONPATH=. python scripts/cache_demo_traces.py \\
        --prompt "def quicksort(arr):\\n    if len(arr) <= 1:\\n        return arr\\n    pivot = arr[len(arr) // 2]\\n" \\
        --max-new 160 --budget 10 --k 6 \\
        --out demo/cached_traces/default.html

The output HTML is committed to the repo / Space so visitors land on a
populated demo instead of waiting 60-90s on a cold ZeroGPU slice. The
cached HTML is self-contained: it references CSS classes that the demo's
`_TRACE_CSS` block injects, so the file is meaningful only inside the app.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

# Make the demo module importable so we can reuse its renderer verbatim.
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from demo.srt_introspect_app import _render_trace_html  # noqa: E402
from srt_introspect import Trace  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--rep-pen", type=float, default=1.15)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    args = ap.parse_args()

    print("[cache] loading adapter ...", flush=True)
    t = Trace.load()
    layers = list(getattr(t.adapter.config, "mah_layer_indices", []) or [])

    print(f"[cache] generating {args.max_new} tokens ...", flush=True)
    t0 = time.perf_counter()
    result = t.generate(
        args.prompt,
        max_new_tokens=args.max_new,
        budget=args.budget,
        k=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.rep_pen,
    )
    elapsed = time.perf_counter() - t0
    html = _render_trace_html(result, args.prompt, elapsed, layer_indices=layers)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"[cache] wrote {args.out}  ({len(html):,} bytes, {elapsed:.1f}s gen)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
