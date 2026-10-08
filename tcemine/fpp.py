"""Offline harness for TRICERATOPS (Giacalone et al. 2021).

TRICERATOPS normally queries MAST (TIC), Gaia and TESScut itself. Where those services are unreachable,
this module builds the `target` object from local inputs instead:
  * TIC v8.2 neighbours (VizieR IV/39/tic82) within 10 TESS pixels,
  * the Gaia DR3 field population (VizieR I/355) used for the blended scenarios,
  * the SPOC optimal-aperture masks and WCS from each sector's light-curve file.
It requires the `triceratops` package (tested with v1.1.0, Python 3.11).
"""
import numpy as np, pandas as pd

def _tsv(s):
    L = [l for l in s.split("\n") if l and not l.startswith("#")]
    hdr = [h.strip() for h in L[0].split("\t")]
    df = pd.DataFrame([[x.strip() for x in l.split("\t")] for l in L[3:]], columns=hdr)
    return df

def build_target(tic, tic_tsv, gaia_tsv, apertures, sector_epochs=None, sep_epoch=2024.0):
    """sector_epochs: {sector: decimal year} to propagate TIC (epoch 2000) positions with proper motion."""
    import astroquery.gaia
    from astropy.table import Table
    from astropy.io.fits import Header
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    import triceratops.triceratops as trm, triceratops.funcs as tf
    t = _tsv(tic_tsv)
    for c in t.columns:
        if c not in ("Disp", "m_TIC"):
            t[c] = pd.to_numeric(t[c], errors="coerce")
    t = t[~t.Disp.isin(["ARTIFACT", "DUPLICATE", "SPLIT"])].sort_values("_r").reset_index(drop=True)
    assert int(t.TIC[0]) == tic
    stars = pd.DataFrame({"ID": t.TIC.astype(int), "Tmag": t.Tmag, "Vmag": t.Vmag, "GAIAmag": t.Gmag, "gaiabp": t.BPmag,
                          "gaiarp": t.RPmag, "Jmag": t.Jmag, "Hmag": t.Hmag, "Kmag": t.Kmag, "ebv": t["E(B-V)"],
                          "ra": t.RAJ2000, "dec": t.DEJ2000, "mass": t.Mass, "rad": t.Rad, "Teff": t.Teff, "logg": t.logg,
                          "plx": t.Plx, "disposition": t.Disp, "duplicate_id": t.m_TIC})
    stars = stars[stars.Tmag.notna()].reset_index(drop=True)
    pmra = pd.to_numeric(t.loc[stars.index, "pmRA"], errors="coerce").fillna(0).values if "pmRA" in t else np.zeros(len(stars))
    pmde = pd.to_numeric(t.loc[stars.index, "pmDE"], errors="coerce").fillna(0).values if "pmDE" in t else np.zeros(len(stars))
    ra0, de0 = stars.ra.values.copy(), stars.dec.values.copy()
    def at_epoch(yr):
        dt = yr - 2000.0
        return ra0 + pmra * dt / 3.6e6 / np.cos(np.radians(de0)), de0 + pmde * dt / 3.6e6
    g = _tsv(gaia_tsv)
    for c in g.columns:
        g[c] = pd.to_numeric(g[c], errors="coerce")
    tab = Table({"phot_g_mean_mag": g.Gmag.values, "phot_bp_mean_mag": g.BPmag.values, "phot_rp_mean_mag": g.RPmag.values,
                 "parallax": g.Plx.values, "parallax_over_error": g.RPlx.values})
    class _Job:
        def get_results(self):
            return tab
    astroquery.gaia.Gaia.launch_job = lambda *a, **k: _Job()
    astroquery.gaia.Gaia.launch_job_async = lambda *a, **k: _Job()
    bg = tf.query_gaia_background(stars.ra[0], stars.dec[0], tic, verbose=0)
    T = object.__new__(trm.target)
    T.ID = tic; T.mission = "TESS"; T.search_radius = 10; T.N_pix = 22
    T.trilegal_fname = bg; T.trilegal_url = None; T.background_population_source = "gaia"
    pix, aps, secs = [], [], []
    strkeys = {"CTYPE1", "CTYPE2", "RADESYS", "EXTNAME", "CUNIT1", "CUNIT2", "XTENSION", "WCSNAMEP", "CTYPE1P", "CTYPE2P", "WCSNAME", "INHERIT"}
    for s in apertures["sectors"]:
        H = Header()
        for k, v in s["h2"].items():
            try:
                H[k] = v if k in strkeys else float(v)
            except Exception:
                pass
        w = WCS(H, naxis=2); c0, r0 = float(s["h2"]["CRVAL1P"]), float(s["h2"]["CRVAL2P"])
        ra_e, de_e = at_epoch(sector_epochs.get(s["sector"], 2000.0) if sector_epochs else 2000.0)
        xy = np.array(w.all_world2pix(ra_e, de_e, 0)).T
        pix.append(np.column_stack([c0 + xy[:, 0], r0 + xy[:, 1]]))
        img = np.array(s["img"]); jj, ii = np.where((img & 2) > 0)
        aps.append(np.column_stack([c0 + ii, r0 + jj])); secs.append(s["sector"])
    T.sectors = np.array(secs); T.pix_coords = pix; T.TESS_images = [np.zeros((22, 22))] * len(secs)
    T.col0s = [0] * len(secs); T.row0s = [0] * len(secs)
    ra_s, de_s = at_epoch(sep_epoch)
    stars["ra"], stars["dec"] = ra_s, de_s
    c = SkyCoord(ra_s, de_s, unit="deg")
    stars["sep (arcsec)"] = c[0].separation(c).arcsec.round(3); stars["PA (E of N)"] = c[0].position_angle(c).deg.round(3)
    T.stars = stars; T.estimate_stellar_params(verbose=0)
    return T, aps

def fold_binned(t, f, P, T0, half_window_d, bin_min=5.0):
    n = np.round((t - T0) / P); x = t - T0 - n * P; w = np.abs(x) < half_window_d
    x, y = x[w], f[w]; bw = bin_min / 1440
    edges = np.arange(x.min(), x.max() + bw, bw); idx = np.digitize(x, edges)
    keep = [i for i in range(1, len(edges)) if (idx == i).sum() > 3]
    xb = np.array([x[idx == i].mean() for i in keep]); yb = np.array([y[idx == i].mean() for i in keep])
    return xb, yb
