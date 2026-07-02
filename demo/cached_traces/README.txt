Cached pre-rendered trace HTML lives here. Generate with:

    PYTHONPATH=. python scripts/cache_demo_traces.py \
        --prompt "..." --out demo/cached_traces/default.html

The app loads `default.html` on first render so the panel isn't empty
during the ~60s cold-start weight load. CSS classes used by the cache
HTML are injected by `_TRACE_CSS` in `srt_introspect_app.py`, so the
file is only meaningful inside the demo.
