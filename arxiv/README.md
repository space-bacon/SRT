# arXiv Submission Package

Source for the arXiv preprint of *Semiotic Taps: Lightweight Adapter Modules
for Bifurcation Detection in Frozen Language Models*.

## Contents

- `paper.tex` — single-file LaTeX source (pandoc-generated from `../paper.md`).
- `paper.pdf` — locally built reference PDF (xelatex, 41 pages).
- `00README.XXX` — arXiv processing hints (declares xelatex toplevel).
- `build.sh` — rebuild script.

No `.bib` or external figure files. References are inline author-year prose.

## Build locally

```bash
./build.sh
```

Or manually:
```bash
xelatex -interaction=nonstopmode paper.tex
xelatex -interaction=nonstopmode paper.tex   # second pass for longtable + refs
```

## arXiv submission

1. Upload `paper.tex` plus `00README.XXX` as a tarball:
   ```bash
   tar czf srt-adapter-arxiv.tar.gz paper.tex 00README.XXX
   ```
2. arXiv's autotex driver detects `fontspec` / `unicode-math` and runs
   xelatex automatically. No special declarations needed beyond that.
3. Suggested categories:
   - Primary: `cs.CL` (Computation and Language)
   - Cross-list: `cs.LG` (Machine Learning), `cs.AI` (Artificial Intelligence)
4. Title and abstract: full abstract is in `abstract.txt` (~3600 chars).
   arXiv's submission form limits the abstract field to ~1920 characters, so
   it will need trimming before paste. The full abstract remains intact in
   the PDF.
5. License: arXiv's default (non-exclusive license to distribute) or CC-BY 4.0.

## Notes

- Source uses `unicode-math` for symbols (Δ, ×, →, ≈, ±). pdflatex will
  reject these directly; the `iftex` switch in the preamble handles
  engine detection.
- 41 pages, ~200 KB PDF, no external assets.
- Companion artifacts (model weights, benchmarks, code) live at
  - https://huggingface.co/RiverRider/srt-adapter-v8a
  - https://github.com/space-bacon/SRT
