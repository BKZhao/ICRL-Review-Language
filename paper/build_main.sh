#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if command -v tectonic >/dev/null 2>&1; then
  echo "[build] Using tectonic"
  tectonic --keep-logs --keep-intermediates main.tex
elif command -v pdflatex >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then
  echo "[build] Using pdflatex + bibtex"
  pdflatex -interaction=nonstopmode main.tex
  bibtex main
  pdflatex -interaction=nonstopmode main.tex
  pdflatex -interaction=nonstopmode main.tex
else
  echo "[error] No TeX engine found."
  echo "[hint] Install one of:"
  echo "  1) tectonic (recommended for quick setup)"
  echo "  2) texlive binaries providing pdflatex+bibtex"
  exit 1
fi

echo "[build] Done: main.pdf"
