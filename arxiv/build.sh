#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
xelatex -interaction=nonstopmode paper.tex >/dev/null
xelatex -interaction=nonstopmode paper.tex >/dev/null
rm -f paper.aux paper.log paper.out paper.toc
echo "built: $(pwd)/paper.pdf ($(wc -c < paper.pdf) bytes)"
