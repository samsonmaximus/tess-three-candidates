# Three small transiting planet candidates from TESS (TOI-678, GJ 237 / HIP 31126, TOI-5997)

Code and data for the paper by S. Fraser (2026). Everything needed to reproduce the numbers, tables and
figures is here; the TESS data were taken from MAST and the imaging data products from ExoFOP. The copy
attached to arXiv as ancillary files leaves out the light-curve bundles (`data/lc_*.bin.gz`) and the PRF models
(`data/tess_prf_s0004.json.gz`); they are in the full package on Zenodo and can be rebuilt from MAST with `tools/`.
On Zenodo the package comes as two archives (`anc_code_and_data.tar.gz` and `lightcurve_bundles.tar.gz`);
extract both in the same folder. `SHA256SUMS` covers the full tree.

## Layout

- `data/` inputs, all public
  - `lc_*.bin.gz` SPOC 2-min light curves (PDCSAP, SAP, centroids, quality) for each star, in the compact
    bundle format read by `tcemine.lightcurves.load_bundle` (built in the browser by `tools/bundle_b64.js`)
  - `tpf_diffimages.json.gz` per-transit and null difference images from the SPOC target pixel files
    (built by `tools/tpf_diffimg.js`), with apertures and WCS
  - `tess_prf_s0004.json.gz` TESS PRF models (MAST, start_s0004) for the cameras/CCDs used
  - `gaia_dr3.json` Gaia DR3 rows for the hosts, the cones used, and the background samples for TRICERATOPS
  - `tic82_cones.json` TIC 8.2 cones (0.07 deg) around each target
  - `imaging/` raw contrast-curve files from ExoFOP (Zorro, 'Alopeke, PHARO; the SOAR sensitivity plot)
  - `cc_*.dat` contrast curves passed to TRICERATOPS (made by `scripts/s05c_contrast.py`)
  - `ffi/toi678_ffi_windows.txt` TESS-SPOC and QLP full-frame-image light-curve windows of TOI-678
  - `nasa_pscomppars_2026-10-08.csv` NASA Exoplanet Archive table used for Fig. 3
- `tcemine/` the analysis library (detrending, vetting statistics, transit model and fits, PRF localisation)
- `scripts/` the pipeline, run in this order:
  1. `s02_stellar.py` stellar parameters
  2. `s03_vet.py` light curves and the tests of every signal (candidates and controls)
  3. `s05a_pixel.py` PRF localisation; `s04a_dilution.py` flux fractions and the HIP 31126 dilution factors
  4. `s04_fit.py SYSTEM [rho|free] [A|B]` transit fits; `s04b_export_posteriors.py` posterior samples as CSV
  5. `s05c_contrast.py` contrast curves; `s05b_fpp.py SYSTEM SIGNAL CC EXCL SEED` TRICERATOPS runs
  6. `s06_ffi_check.py` full-frame-image check of TOI-678
  7. `s07_figures.py` figures; `s08_make_numbers.py --final` LaTeX macros and table rows for the paper
- `results/` outputs: `numbers.json`, `fit_*.json`, `pixel_*.json`, `fpp/*.json`, `ffi_toi678.json`, and
  `posteriors/posterior_*.csv.gz` (every 4th stored MCMC sample, named columns; P in days, T0 in BJD_TDB - 2457000)
- `paper/` LaTeX source (`three_candidates.tex`, `numbers.tex`, `tab_ephem.tex`, `refs.bib`, `openjournal.cls`)
- `figures/` the three figures

## Requirements

Python 3.11+ with numpy, scipy, pandas, astropy, matplotlib, batman-package, emcee, wotan and
triceratops 1.1.0; tess-point for the sector check. The fits take about an hour for TOI-678 on one core;
each TRICERATOPS run (10^6 draws) takes 3-5 minutes.

## Notes

- Box depths, S/N and consistency tests are given with two estimates of correlated noise (see the paper).
- The pixel-level exclusions require a star to be excluded at 3 sigma by both the bootstrap (Mahalanobis)
  test and the calibrated chi-square test; for TOI-678 and TOI-5997 no neighbour meets this.
- TRICERATOPS uses the Gaia DR3 field population (0.1 deg^2, G < 21) instead of TRILEGAL.
