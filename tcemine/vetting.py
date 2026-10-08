"""Model-independent vetting statistics for a periodic transit signal."""
import numpy as np

def phase_offset(t, P, T0, phase0=0.0):
    """Time from the nearest event at phase `phase0` (days)."""
    return (((t - T0) / P - phase0 + 0.5) % 1 - 0.5) * P

def box_depth(t, f, P, T0, dur, phase0=0.0, inner=0.8):
    """Depth = median(out-of-transit ring) - mean(in-transit), with a white-noise error.

    In-transit: |dt| < inner*dur/2. Ring: dur < |dt| < 3 dur.
    Returns (depth, err, n_in).
    """
    dt = np.abs(phase_offset(t, P, T0, phase0))
    inn = dt < inner * dur / 2
    out = (dt > dur) & (dt < 3 * dur)
    if inn.sum() < 3 or out.sum() < 10:
        return np.nan, np.nan, int(inn.sum())
    return np.median(f[out]) - np.mean(f[inn]), np.std(f[out]) / np.sqrt(inn.sum()), int(inn.sum())

def per_transit(t, f, P, T0, dur, min_cov=0.6):
    """Depth of every individual transit with enough coverage. Rows: epoch, tc, depth, err."""
    n = np.round((t - T0) / P); rows = []
    cad = np.median(np.diff(t))
    for k in np.unique(n):
        tc = T0 + k * P; m = n == k
        inn = m & (np.abs(t - tc) < 0.4 * dur); out = m & (np.abs(t - tc) > dur) & (np.abs(t - tc) < 3 * dur)
        if inn.sum() < min_cov * 0.8 * dur / cad or out.sum() < 10:
            continue
        # require ring on both sides (no gap-edge events)
        if not ((t[out] < tc).any() and (t[out] > tc).any()):
            continue
        rows.append((int(k), tc, np.median(f[out]) - np.mean(f[inn]), np.std(f[out]) / np.sqrt(inn.sum())))
    return np.array(rows)

def per_sector_centroid(t, c1, c2, sector, P, T0, dur, n=300, seed=2):
    """Per-sector in-transit minus out-of-transit centroid shift (column, row), with the scatter of the
    same statistic at random phases in that sector. Uses the same in/ring windows as box_depth."""
    rng = np.random.default_rng(seed); rows = []
    for s in np.unique(sector):
        m = sector == s; tt, x, y = t[m], c1[m], c2[m]
        def stat(T):
            dt = np.abs(phase_offset(tt, P, T, 0)); inn = dt < 0.4 * dur; out = (dt > dur) & (dt < 3 * dur)
            if inn.sum() < 5 or out.sum() < 20:
                return None
            return (np.mean(x[inn]) - np.mean(x[out]), np.mean(y[inn]) - np.mean(y[out]))
        real = stat(T0)
        if real is None:
            continue
        null = [stat(T0 + rng.uniform(2 * dur, P - 2 * dur)) for _ in range(n)]
        null = np.array([v for v in null if v is not None])
        rows.append(dict(sector=int(s), dcol=real[0], drow=real[1], sd_col=float(null[:, 0].std()), sd_row=float(null[:, 1].std()),
                         mu_col=float(null[:, 0].mean()), mu_row=float(null[:, 1].mean())))
    return rows

def chi2_constant(depths, errs):
    w = 1 / errs**2; mu = np.sum(depths * w) / np.sum(w)
    return float(np.sum(((depths - mu) / errs) ** 2)), len(depths) - 1, float(mu)

def odd_even(t, f, P, T0, dur):
    n = np.round((t - T0) / P)
    o = box_depth(t[n % 2 == 1], f[n % 2 == 1], P, T0, dur)
    e = box_depth(t[n % 2 == 0], f[n % 2 == 0], P, T0, dur)
    return o, e, abs(o[0] - e[0]) / np.hypot(o[1], e[1])

def secondary_scan(t, f, P, T0, dur, step=0.002):
    """Box depth at every phase away from the transit; returns phases, depth, err."""
    ph = np.arange(0.05, 0.95 + 1e-9, step); d = []; s = []
    for p in ph:
        a, b, _ = box_depth(t, f, P, T0, dur, phase0=p); d.append(a); s.append(b)
    return ph, np.array(d), np.array(s)

def random_phase_null(t, f, P, T0, dur, extra=(), n=600, seed=1):
    """Compare a statistic at the true ephemeris with the same statistic at random phases.

    `extra` is a list of additional arrays (e.g. centroids) to test alongside flux.
    Returns list of (real, null_mean, null_sd, z) for flux and each extra series.
    """
    series = [f] + list(extra)
    def stat(T):
        dt = np.abs(phase_offset(t, P, T, 0))
        inn = dt < 0.4 * dur; out = (dt > dur) & (dt < 3 * dur)
        if inn.sum() < 20:
            return None
        return [np.median(x[out]) - np.mean(x[inn]) if j == 0 else np.mean(x[inn]) - np.mean(x[out]) for j, x in enumerate(series)]
    real = stat(T0); rng = np.random.default_rng(seed); null = []
    for _ in range(n):
        x = stat(T0 + rng.uniform(2 * dur, P - 2 * dur))
        if x is not None:
            null.append(x)
    null = np.array(null)
    return [(real[j], null[:, j].mean(), null[:, j].std(), (real[j] - null[:, j].mean()) / null[:, j].std())
            for j in range(len(series))]

def seasons(sector, gap=3):
    """Group sectors into observing seasons (consecutive sectors within `gap`)."""
    us = np.unique(sector); g = np.cumsum(np.r_[0, np.diff(us) > gap])
    return g[np.searchsorted(us, sector)]

def bls_top(t, f, e, dur, pmin=0.5, pmax=40.0, nper=None, ntop=5):
    """Blind BLS over a log period grid; returns the top peaks separated by >1%."""
    from astropy.timeseries import BoxLeastSquares
    T = t.max() - t.min(); pmax = min(pmax, T / 2.5)
    nper = nper or int(20000 + 60000 * min(1, T / 100))
    periods = np.exp(np.linspace(np.log(pmin), np.log(pmax), nper))
    r = BoxLeastSquares(t, f, e).power(periods, np.array([0.5, 0.75, 1.0, 1.4]) * dur, objective="snr")
    top = []
    for j in np.argsort(r.power)[::-1]:
        if all(abs(r.period[j] / q - 1) > 0.01 for q, _ in top):
            top.append((float(r.period[j]), float(r.power[j])))
        if len(top) >= ntop:
            break
    return top
