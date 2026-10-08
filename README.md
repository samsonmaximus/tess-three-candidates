# Three small transiting planet candidates from TESS around TOI-678, GJ 237 and TOI-5997

**Samson Fraser** · Independent researcher, Sicamous, British Columbia, Canada · preprint, October 2026

**Paper:** [`Fraser2026_three_candidates.pdf`](Fraser2026_three_candidates.pdf)
· Zenodo DOI: *to be added*
· arXiv: *to be added*

> These are **planet candidates, not validated planets.** The paper gives the observations that would validate them.

## Summary

The TESS pipeline (SPOC) detected each of these three signals as a threshold-crossing event in its multi-sector searches, but none has become a TESS Object of Interest.

| Candidate | Host | Period (d) | Radius (R⊕) | Insolation (S⊕) | Transits | FPP (TRICERATOPS) |
|---|---|---|---|---|---|---|
| TOI-678 (130 d) | G dwarf, 207 pc; also hosts TOI-678.01 | 130.145 | 3.60 ± 0.28 | 3.9 | 3 | 0.021 ± 0.002 |
| HIP 31126 (29.3 d) | GJ 237, pair of early M dwarfs 3.3″ apart, 23.8 pc | 29.286 | 1.88 ± 0.11 (star A) or 2.39 ± 0.14 (star B) | 2.9 (A) or 1.16 (B) | 7 | 0.40 ± 0.02 |
| TOI-5997 (14.2 d) | K dwarf, 46.6 pc; also hosts the validated TOI-5997 b | 14.216 | 1.27 ± 0.09 | 21 | 9 | (3.7 ± 0.2) × 10⁻⁴ |

- With a conservative allowance for correlated noise the signals have S/N 8.4–9.8.
- Depths agree from transit to transit and season to season, in simple-aperture as well as corrected photometry. None shows an odd–even difference, a secondary eclipse or a period alias.
- A fit of the TESS pixel response function to difference images places each source within about 1σ of its target.
- For HIP 31126 the FPP is high almost entirely because the host star is unknown; eclipsing-binary scenarios carry 0.0174 ± 0.0018 of the probability.
- TOI-5997's FPP is below the usual validation threshold, but its S/N is the lowest of the three, so it is presented as a candidate.

## Upcoming transits

Barycentric UTC, rounded to the minute; the light-travel time to an observer on Earth differs by up to ±8 min. Full list in Table B1 of the paper.

| Star | Mid-transit (UTC) | σ (min) | Duration (h) |
|---|---|---|---|
| HIP 31126 | 2026-10-17 04:51 | 8 | 2.1 |
| HIP 31126 | 2026-11-15 11:42 | 8 | 2.1 |
| TOI-678 | 2026-12-29 08:31 | 16 | 6.1 |
| TOI-5997 | 2027-04-11 08:32 | 10 | 3.2 |

Seeing-limited photometry during one transit of HIP 31126 would show which of the two stars hosts the signal. If you can observe one of these transits, please get in touch (email in the paper).

## What is in this repository

This is the code and data package of the paper. Details: [`docs/PACKAGE_README.md`](docs/PACKAGE_README.md).

- `Fraser2026_three_candidates.pdf` the paper
- `paper/` LaTeX source
- `tcemine/` analysis library (detrending, vetting statistics, transit model and fits, PRF localisation)
- `scripts/` the pipeline, `s02` to `s08` in order
- `data/` public inputs (Gaia DR3, TIC 8.2, ExoFOP contrast curves, TESS difference images and PRF models)
- `results/` fit outputs, pixel localisation, TRICERATOPS outputs and posterior samples
- `figures/` the three figures
- `tools/` the scripts that extracted the light curves and difference images from MAST

**Not in this repository:** the TESS light-curve bundles (`data/lc_*.bin.gz`, 23 MB). They will be in the Zenodo record and can be rebuilt from MAST with `tools/`. `SHA256SUMS` covers the full Zenodo package, so the three bundle lines fail here, and `README.md` here is this landing page. The package's own README is `docs/PACKAGE_README.md`.

### Reproducing

Python 3.11+ with numpy, scipy, pandas, astropy, matplotlib, batman-package, emcee, wotan and triceratops 1.1.0 (tess-point for the sector check). Run the scripts in `scripts/` in the order given in `docs/PACKAGE_README.md`. `make_package.sh` rebuilds the numbers, the paper and the release archives.

## Use of AI

The analysis code, the literature checks and the draft text were prepared with the assistance of Claude (Anthropic) under the author's direction, and an AI-assisted review of the draft was carried out. Most numbers in the text and all numbers in the tables are written by `scripts/s08_make_numbers.py` from the analysis outputs. The author checked the results and takes responsibility for the content.

## Citation

Fraser, S. (2026), *Three small transiting planet candidates from TESS around TOI-678, GJ 237 and TOI-5997*, preprint. See [`CITATION.cff`](CITATION.cff). Please cite the Zenodo DOI or arXiv ID once available.

## License

- Paper, figures, data products and text: [CC BY 4.0](LICENSE)
- Code (`tcemine/`, `scripts/`, `tools/`, `make_package.sh`): [MIT](LICENSE-CODE)
- Third-party inputs in `data/` (TESS, Gaia, TIC, ExoFOP, NASA Exoplanet Archive) remain under their providers' terms; please also cite them as the paper does.

## Acknowledgements

This work uses data from the TESS mission (MAST), ExoFOP, the NASA Exoplanet Archive, Gaia DR3 and 2MASS, and archival imaging from Zorro, 'Alopeke, PHARO and SOAR. See the paper's acknowledgements.
