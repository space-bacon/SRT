# Public Release Checklist

This file tracks the steps required to take a new SRT-Adapter generation from
trained checkpoint → public release on Hugging Face → tagged GitHub release.

> Patent / publication review status: training and research source code is
> currently held back. Public releases are **inference-only** packages built
> under `release/<version>/` (gitignored locally; published only to Hugging
> Face). Do not flip this GitHub repository to public until that hold lifts.

## Per-release artifact (under `release/srt-adapter-<vN>/`)

- [ ] `adapter.safetensors` (and `adapter.pt`) — trained weights
- [ ] `config.json` — adapter config + backbone reference
- [ ] `requirements.txt` — pinned inference deps
- [ ] `src/` — inference-only `SRTAdapter` loader
- [ ] `examples/` — minimal usage snippet
- [ ] `benchmarks/` — `metrics.json` + `traces.json` from `scripts/benchmark.py`
- [ ] `data/val_200.jsonl` — calibration baseline used by the demo
- [ ] `paper.pdf` — frozen at release tag
- [ ] `SUA.md` — Semantic Uncertainty Appendix for any new load-bearing terms
- [ ] `README.md` — model card (license, base model, intended use, limitations)
- [ ] `LICENSE` — Apache-2.0
- [ ] `VALIDATION_HISTORY.md` — what changed vs the previous release
- [ ] `SHA256SUMS` — checksums of every file in the package

## Pre-publish gates

- [ ] No personal email addresses, drafts, or substack progress notes in the
      package (cross-check `git ls-files | grep -iE 'email|substack|newsletter'`
      returns nothing under `release/`)
- [ ] No training scripts, loss code, or dataset construction code in `src/`
- [ ] `python -m examples.minimal` runs end-to-end on a clean venv
- [ ] Demo Space (`RiverRider/srt-adapter-<vN>-demo`) loads the new weights
      and the live tunnel is registered in
      `RiverRider/srt-adapter-<vN>/live_url.json`

## Publish

- [ ] `huggingface_hub` upload to `RiverRider/srt-adapter-<vN>` (model repo)
- [ ] Update Space `index.html` fallback URL + `live_url.json`
- [ ] Tag local repo: `git tag srt-adapter-<vN> && git push --tags` (only
      once the GitHub repo goes public; until then, tag locally only)

## Current state

| Generation | Status                                  | Released |
|------------|-----------------------------------------|----------|
| v8a        | Public on HF, paper-aligned             | yes      |
| v9         | Benchmarked; SUA written; package staged| no       |
| v10        | Trained; superseded by v11              | no       |
| v11        | Training in progress on A6000 (vast.ai) | no       |

Best-to-date public artifact remains **v8a** until v11 finishes and clears the
gates above.
