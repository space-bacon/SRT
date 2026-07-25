# arXiv package — the SRT program paper

- Canonical source: ../paper_srt_program.md (edit THAT, then regenerate here)
- `paper.md` here is the arXiv-adapted copy: pandoc metadata block with the
  abstract moved into the `abstract:` field (proper title page + abstract
  environment), repo-relative links flattened to code spans, figure paths
  rewritten to figs/, and `DeclareUnicodeCharacter` mappings so the source
  compiles under **pdflatex** (arXiv does not support XeLaTeX). Regenerate
  from the canonical source before building if the canonical changed.
- `./build.sh` — pandoc + pdflatex → paper.pdf (12 pages); injects
  `\pdfoutput=1` into the preamble per arXiv requirements
- `srt-program-arxiv.tar.gz` — paper.tex + figs/*.png, for the arXiv upload form
- `abstract.txt` — for the submission form

Pre-submission checklist (see repo memory / leverage.md):
- [x] SSRN reference titles filled (Treachery of Signs = 5987495; SRT architecture = 6349978)
- [x] Literature numbers verified (VSE++ 41.3/71.1/81.2 COCO 5k ft-ResNet; CLIP zero-shot 58.4 i2t R@1; detector band per docs/TRUTHFULQA_RESULTS.md with attributions)
- [x] Citations completed (Kockelman Semiotica 157; Silverstein in Basso & Selby eds.; Peirce CP Hartshorne-Weiss-Burks; Azaria & Mitchell, Chen et al., Duan et al. added)
- [x] arXiv compliance: pdflatex-compilable source, \pdfoutput=1, title page with \maketitle + abstract environment, figures in tarball, no missing glyphs / undefined refs in the build log
- [ ] Author read-through for voice and claims
