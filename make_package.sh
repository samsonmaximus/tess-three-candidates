#!/bin/bash
# Build the paper and the release package:
#   Fraser2026_three_candidates.pdf  the paper
#   arxiv_source.tar.gz              LaTeX source for arXiv, with a slim anc/ directory (code, small inputs,
#                                    results, posterior samples, checksums) that arXiv serves as ancillary files
#   anc_code_and_data.tar.gz         the full package for Zenodo (adds the PRF bundle) ...
#   lightcurve_bundles.tar.gz        ... and its light-curve bundles; extract both in one folder
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$ROOT/dist}"
mkdir -p "$OUT"
cd "$ROOT/scripts"
if ls "$ROOT"/results/fit_*.pkl > /dev/null 2>&1; then python3 s04b_export_posteriors.py > /dev/null; fi   # chains are not in the public packages
python3 s08_make_numbers.py --final
cd "$ROOT/paper"
rm -f three_candidates.aux three_candidates.bbl three_candidates.blg
pdflatex -interaction=nonstopmode three_candidates.tex > build1.log
bibtex three_candidates > bib.log
pdflatex -interaction=nonstopmode three_candidates.tex > build2.log
pdflatex -interaction=nonstopmode three_candidates.tex > build3.log
if grep -E "^!" build3.log; then echo "LaTeX errors"; exit 1; fi
if grep -E "undefined" build3.log; then echo "undefined references"; exit 1; fi
cp three_candidates.pdf "$OUT/Fraser2026_three_candidates.pdf"

# full code+data package (Zenodo)
ANC="$OUT/anc_code_and_data"; rm -rf "$ANC"; mkdir -p "$ANC"
cp -r "$ROOT/README.md" "$ROOT/make_package.sh" "$ROOT/scripts" "$ROOT/tcemine" "$ROOT/tools" "$ROOT/figures" "$ANC/"
mkdir -p "$ANC/data" "$ANC/results/fpp" "$ANC/results/posteriors" "$ANC/paper"
cp -r "$ROOT/data/"* "$ANC/data/"
cp "$ROOT/results/"*.json "$ANC/results/"; cp "$ROOT/results/fpp/"*.json "$ANC/results/fpp/"
cp "$ROOT/results/posteriors/"*.csv.gz "$ANC/results/posteriors/"
cp "$ROOT/paper/"three_candidates.tex "$ROOT/paper/"numbers.tex "$ROOT/paper/"tab_ephem.tex "$ROOT/paper/"refs.bib "$ROOT/paper/"openjournal.cls "$ANC/paper/"
find "$ANC" -name "__pycache__" -type d -prune -exec rm -rf {} +
(cd "$ANC" && find . -type f ! -name SHA256SUMS -exec sha256sum {} + | sort -k2 > SHA256SUMS)
# two archives for Zenodo, each under 30 MB; extracted in the same folder they rebuild the full tree
(cd "$OUT" && tar czf anc_code_and_data.tar.gz --exclude="anc_code_and_data/data/lc_*.bin.gz" anc_code_and_data)
(cd "$OUT" && tar czf lightcurve_bundles.tar.gz anc_code_and_data/data/lc_toi678.bin.gz anc_code_and_data/data/lc_hip31126.bin.gz anc_code_and_data/data/lc_toi5997.bin.gz)

# arXiv source: flat directory with the figures next to the .tex, plus anc/ (everything except the large
# light-curve and PRF bundles, which are re-extractable from MAST with tools/ and are in the Zenodo package)
SRC="$OUT/arxiv_source"; rm -rf "$SRC"; mkdir -p "$SRC"
cp three_candidates.tex numbers.tex tab_ephem.tex refs.bib three_candidates.bbl openjournal.cls "$SRC/"
cp "$ROOT/figures/"fig_transits.pdf "$ROOT/figures/"fig_pixels.pdf "$ROOT/figures/"fig_context.pdf "$SRC/"
cp -r "$ANC" "$SRC/anc"
rm -f "$SRC/anc/data/"lc_*.bin.gz "$SRC/anc/data/tess_prf_s0004.json.gz"
(cd "$SRC/anc" && find . -type f ! -name SHA256SUMS -exec sha256sum {} + | sort -k2 > SHA256SUMS)
(cd "$SRC" && tar czf "$OUT/arxiv_source.tar.gz" .)
# test that the arXiv source builds on its own
TST="$OUT/_arxiv_test"; rm -rf "$TST"; mkdir -p "$TST"; tar xzf "$OUT/arxiv_source.tar.gz" -C "$TST"
(cd "$TST" && pdflatex -interaction=nonstopmode three_candidates.tex > b1.log && pdflatex -interaction=nonstopmode three_candidates.tex > b2.log)
if grep -E "^!" "$TST/b2.log"; then echo "arXiv source does not build"; exit 1; fi
if grep -E "undefined" "$TST/b2.log"; then echo "arXiv source has undefined references"; exit 1; fi
rm -rf "$TST"
if [ -f "$ROOT/SUBMISSION_NOTES.md" ]; then cp "$ROOT/SUBMISSION_NOTES.md" "$OUT/"; fi
ls -la "$OUT"
