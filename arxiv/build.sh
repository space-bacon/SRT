#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Regenerate LaTeX from the canonical Markdown source, then build PDF.
pandoc paper.md -o paper.tex --pdf-engine=xelatex --standalone
xelatex -interaction=nonstopmode paper.tex >/dev/null
xelatex -interaction=nonstopmode paper.tex >/dev/null
rm -f paper.aux paper.log paper.out paper.toc
echo "built: $(pwd)/paper.pdf ($(wc -c < paper.pdf) bytes)"
