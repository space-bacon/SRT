# arXiv package — the deployment paper

**Read Everywhere, Verify There: What It Takes to Put a Frozen-Model Read-Out
on the Visitor's Hardware**

Companion to *Train Once, Read Everywhere* (SSRN 7264778, in `../arxiv_program/`).
That paper establishes substrate invariance. This one reports the three
failures that only appear when you build the browser tier and measure it on an
end task.

- `paper.md` — **canonical source** (unlike `arxiv_program/`, there is no
  separate top-level markdown; edit this file). pandoc metadata block with the
  abstract in the `abstract:` field, and `DeclareUnicodeCharacter` mappings so
  the source compiles under **pdflatex** (arXiv does not support XeLaTeX).
- `./build.sh` — pandoc + pdflatex → `paper.pdf`; injects `\pdfoutput=1` into
  the preamble per arXiv requirements.
- `abstract.txt` — for the submission form.

## Every number is backed by an artifact

| claim | artifact |
|---|---|
| cross-runtime collapse and 4 KB recovery (§3) | `artifacts/nla/q4/cross_runtime_browser_rung.json` |
| gallery scale cost, 1K vs 123,287 (§4) | `artifacts/nla/q4/gallery_scale_cost.json`, `gallery_scale_cost_head118k.json` |
| head-space steering, retention + 32 random controls (§5) | `artifacts/nla/q4/headspace_axis_validation.json` |
| int8 index is free (§6.1) | `artifacts/nla/q4/gallery_precision_cost.json` |
| six-arm ablation, registered predictions (§7) | `artifacts/nla/q4/browser_head_v2_arms.json` |
| final head at deployment scale (§7) | `artifacts/nla/q4/browser_head_v3_report.json` |
| one-forward-pass probe, why retrieval cannot share generation's prefill (§2) | `artifacts/nla/q4/one_forward_pass_probe.json` |
| word-order swap probe, at chance (§8) | `artifacts/nla/q4/swap_probe.json`, `swap_probe_pairs.json` |
| abliteration prior (§8) | `artifacts/nla/q4/` abliteration battery + geometry reports |

Reproduction scripts live in `scripts/` (`browser_rung_reference.py`,
`retrieval_reference.py`, `gallery_scale_cost.py`, `gallery_precision_cost.py`,
`headspace_axis_validation.py`, `browser_head_v2.py`, `export_head_safetensors.py`,
`export_index_srtidx.py`, `pack_thumbnails.py`).

Read-out geometry is the public Rust crate `rust/srt-geometry/` (13 tests, no
model dependency, builds native + `wasm32-unknown-unknown`). The deployment
runtime and browser engine are private.

## Pre-submission checklist

- [ ] Author read-through for voice and claims
- [ ] Confirm the SSRN companion is cited with its final title/ID
- [ ] Decide public/private boundary for §2.4 implementation detail
- [ ] Verify no undefined refs / missing glyphs in the pdflatex build log
- [ ] Build `srt-deploy-arxiv.tar.gz` (paper.tex + any figs) for the upload form
- [ ] Optional: add figures (K-curve of scale cost, retention-vs-purity plot,
      ablation bar chart) — the paper currently carries tables only
