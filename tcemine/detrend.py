"""Biweight detrending with transits masked out of the trend estimate."""
import numpy as np
from wotan import flatten

def in_transit(t, P, T0, dur_d, fac=1.0):
    """True where |phase| < fac*dur/2."""
    ph = ((t - T0) / P + 0.5) % 1 - 0.5
    return np.abs(ph * P) < fac * dur_d / 2

def masked_biweight(t, f, sector, window, masks=()):
    """Fit the trend to out-of-transit points only, per sector, and interpolate across transits.

    masks: iterable of (P, T0, dur_d); each is masked at 1.5x its duration.
    """
    oot = np.ones_like(t, bool)
    for P, T0, dur in masks:
        oot &= ~in_transit(t, P, T0, dur, 1.5)
    trend = np.full_like(f, np.nan)
    for s in np.unique(sector):
        m = sector == s
        mo = m & oot
        _, tr = flatten(t[mo], f[mo], method="biweight", window_length=window,
                        edge_cutoff=0.0, break_tolerance=0.4, return_trend=True)
        good = np.isfinite(tr)
        if good.sum() < 10:
            continue
        trend[m] = np.interp(t[m], t[mo][good], tr[good], left=np.nan, right=np.nan)
    return f / trend

def masked_biweight_additive(t, x, sector, window, masks=()):
    """Same as masked_biweight but subtracts the trend (for centroid series, which can be any sign)."""
    out = np.full_like(x, np.nan, dtype=float)
    for s in np.unique(sector):
        m = sector == s
        med = np.nanmedian(x[m])
        y = masked_biweight(t[m], x[m] - med + 100.0, sector[m], window, masks)
        out[m] = (y - 1.0) * 100.0  # approx (x - trend) for small residuals relative to 100
    return out
