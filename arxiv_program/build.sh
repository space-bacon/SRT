#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Regenerate LaTeX from the canonical Markdown source, then build PDF.
# arXiv compiles with pdflatex (TeX Live); XeLaTeX is not supported there,
# so this build targets pdflatex. Unicode math chars are mapped via
# newunicodechar in the metadata header-includes.
pandoc paper.md -o paper.tex --pdf-engine=pdflatex --standalone
# arXiv requires \pdfoutput=1 within the first 5 lines of the preamble.
perl -i -pe 'print "\\pdfoutput=1\n" if $. == 2' paper.tex
pdflatex -interaction=nonstopmode paper.tex >/dev/null || true
pdflatex -interaction=nonstopmode paper.tex >/dev/null
rm -f paper.aux paper.log paper.out paper.toc
echo "built: $(pwd)/paper.pdf ($(wc -c < paper.pdf) bytes)"
