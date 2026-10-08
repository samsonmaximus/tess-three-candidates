"""Pixel-level localisation of a transit signal with the TESS PRF model.

Inputs per sector: the mean out-of-transit image, per-transit difference images (out - in) and "null"
difference images built the same way at random times (tools/tpf_diffimg.js), the stamp WCS, and the
PRF grid of the camera/CCD (MAST prf_fitsfiles, start_s0004).

Method:
  1. PRF at the target's CCD position: bilinear interpolation of the four surrounding grid PRFs.
  2. Registration: the mean image is fitted with all catalogue stars (positions propagated to the sector
     epoch, fluxes in fixed ratio from TESS magnitudes) times a free scale, plus a background, and a free
     common shift (dx, dy) of the catalogue positions. This absorbs WCS and pointing errors.
  3. Localisation: a single point source at sky offset (dRA*, dDec) from the target, amplitude and constant
     per sector free, is fitted to all sectors' mean difference images at once; per-pixel errors are the
     scatter of the null images (divided by sqrt of the number of transits averaged).
  4. Uncertainty: the best-fit model is added to averages of randomly chosen null images and refitted;
     the scatter of the refitted positions gives the covariance (correlated pixel noise included).
"""
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize

GRID_R = np.array([1, 513, 1025, 1536, 2048]); GRID_C = np.array([45, 557, 1069, 1580, 2092])

def grid_prf(prfs, cam, ccd, col, row):
    """Bilinear interpolation of the 4 grid PRFs around (col, row). prfs: dict from data/tess_prf_s0004.json.gz."""
    def br(g, v):
        i = int(np.clip(np.searchsorted(g, v) - 1, 0, len(g) - 2)); return g[i], g[i + 1]
    r0, r1 = br(GRID_R, row); c0, c1 = br(GRID_C, col)
    def get(r, c):
        d = prfs[f"{cam}_{ccd}_row{r:04d}-col{c:04d}"]; return np.array(d["data"], float).reshape(d["n2"], d["n1"])
    wr = np.clip((row - r0) / (r1 - r0), 0, 1); wc = np.clip((col - c0) / (c1 - c0), 0, 1)
    P = (1 - wr) * (1 - wc) * get(r0, c0) + (1 - wr) * wc * get(r0, c1) + wr * (1 - wc) * get(r1, c0) + wr * wc * get(r1, c1)
    return P

class StampPRF:
    """Unit-flux image of a point source on a stamp, from a 9x oversampled TESS PRF."""
    def __init__(self, P, nsamp=9):
        self.P = P; self.n = nsamp; self.c = (P.shape[0] - 1) / 2
    def image(self, xs, ys, nrow, ncol):
        jj, ii = np.mgrid[0:nrow, 0:ncol]
        ky = self.c + self.n * (jj - ys); kx = self.c + self.n * (ii - xs)
        return map_coordinates(self.P, [ky.ravel(), kx.ravel()], order=1, mode="constant", cval=0.0).reshape(nrow, ncol)

def btjd_to_year(btjd):
    return 2000.0 + (btjd + 2457000.0 - 2451545.0) / 365.25

def propagate(ra, dec, pmra, pmdec, epoch0, epoch1):
    """Positions (deg) moved by proper motion (mas/yr; pmra includes cos dec) from epoch0 to epoch1 (years)."""
    dt = epoch1 - epoch0; pmra = np.nan_to_num(pmra); pmdec = np.nan_to_num(pmdec)
    return ra + pmra * dt / 3.6e6 / np.cos(np.radians(dec)), dec + pmdec * dt / 3.6e6

def linfit(D, W, comps):
    """Weighted linear least squares D ~ sum_k a_k comps_k. Returns (coef, chi2, model)."""
    A = np.array([c.ravel() for c in comps]).T; w = W.ravel(); d = D.ravel()
    ok = np.isfinite(d) & np.isfinite(w) & (w > 0)
    Aw = A[ok] * np.sqrt(w[ok])[:, None]; dw = d[ok] * np.sqrt(w[ok])
    coef, *_ = np.linalg.lstsq(Aw, dw, rcond=None)
    model = (A @ coef).reshape(D.shape)
    return coef, float(np.sum(w[ok] * (d[ok] - model.ravel()[ok]) ** 2)), model

def register(img, prf, xy, rel):
    """Fit img with sum_k rel_k PRF(x - xk - dx, y - yk - dy) * S + b; returns (dx, dy, S, b, chi2r)."""
    nrow, ncol = img.shape; W = 1 / np.maximum(np.abs(img), 1.0)          # ~Poisson weights
    def chi(p):
        m = sum(r * prf.image(x + p[0], y + p[1], nrow, ncol) for (x, y), r in zip(xy, rel))
        return linfit(img, W, [m, np.ones_like(img)])[1]
    best = min(((chi((a, b)), a, b) for a in np.linspace(-1, 1, 11) for b in np.linspace(-1, 1, 11)))
    x0 = np.array([best[1], best[2]])
    r = minimize(chi, x0, method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-6,
                 initial_simplex=np.array([x0, x0 + [0.1, 0.0], x0 + [0.0, 0.1]])))
    m = sum(rr * prf.image(x + r.x[0], y + r.x[1], nrow, ncol) for (x, y), rr in zip(xy, rel))
    coef, c2, _ = linfit(img, W, [m, np.ones_like(img)])
    return float(r.x[0]), float(r.x[1]), float(coef[0]), float(coef[1]), c2 / img.size

class Localiser:
    """Joint fit of one point source to the mean difference images of several sectors.

    The source's flux deficit is the same fraction A of the target's flux in every sector (one eclipse depth),
    so the model for sector s is A * S_s * PRF_s(x - x_src, y - y_src) + c_s, where S_s is the target flux from
    the registration fit, c_s a constant; A >= 0 (a brightening is not a transit).
    sectors: list of dicts with keys D (mean diff image), sig (per-pixel error image), prf (StampPRF), scale (S_s),
    to_pix (function (ra, dec) -> registered stamp x, y), ra0, dec0 (target at that sector's epoch).
    """
    def __init__(self, sectors):
        self.S = sectors
    def pos(self, s, dra, ddec):
        ra = s["ra0"] + dra / 3600 / np.cos(np.radians(s["dec0"])); dec = s["dec0"] + ddec / 3600
        return s["to_pix"](ra, dec)
    def _solve(self, dra, ddec, data=None):
        cols, ys, ws, nps = [], [], [], []
        n = len(self.S)
        for i, s in enumerate(self.S):
            D = s["D"] if data is None else data[i]; nrow, ncol = D.shape
            x, y = self.pos(s, dra, ddec)
            m = s["prf"].image(x, y, nrow, ncol).ravel() * s["scale"]
            ind = np.zeros((D.size, n)); ind[:, i] = 1.0
            cols.append(np.column_stack([m, ind])); ys.append(D.ravel()); ws.append((1 / s["sig"] ** 2).ravel()); nps.append(D.size)
        A = np.vstack(cols); y = np.concatenate(ys); w = np.concatenate(ws)
        sw = np.sqrt(w); coef, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
        if coef[0] < 0:                      # no negative eclipses: refit background only
            coef = np.r_[0.0, np.linalg.lstsq(A[:, 1:] * sw[:, None], y * sw, rcond=None)[0]]
        model = A @ coef; chi2 = float(np.sum(w * (y - model) ** 2))
        return coef, chi2, model, nps
    def chi2(self, dra, ddec, data=None):
        return self._solve(dra, ddec, data)[1]
    def amplitudes(self, dra, ddec):
        coef, chi2, model, nps = self._solve(dra, ddec)
        out, k = [], 0
        for i, s in enumerate(self.S):
            m = model[k:k + nps[i]].reshape(s["D"].shape); k += nps[i]
            out.append(dict(amp=float(coef[0] * s["scale"]), A=float(coef[0]), bkg=float(coef[1 + i]),
                            chi2=float(np.sum(((s["D"] - m) / s["sig"]) ** 2)), npix=int(s["D"].size), model=m))
        return out
    def fit(self, data=None, half=60.0, step=2.0, start=None):
        if start is None:
            g = np.arange(-half, half + 1e-9, step)
            c = np.array([[self.chi2(a, b, data) for a in g] for b in g])
            j, i = np.unravel_index(np.argmin(c), c.shape); start = (g[i], g[j])
        x0 = np.array(start, float)
        r = minimize(lambda p: self.chi2(p[0], p[1], data), x0, method="Nelder-Mead",
                     options=dict(xatol=0.01, fatol=1e-4, initial_simplex=np.array([x0, x0 + [1.5, 0.0], x0 + [0.0, 1.5]])))
        return float(r.x[0]), float(r.x[1]), float(r.fun)
