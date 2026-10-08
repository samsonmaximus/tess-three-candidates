"""Transit model and fits (batman + emcee)."""
import numpy as np, batman, emcee

G = 6.674e-8

def ars_from_rho(rho, P):
    return (G * rho * (P * 86400) ** 2 / (3 * np.pi)) ** (1 / 3)

def q_to_u(q1, q2):
    s = np.sqrt(q1); return 2 * s * q2, s * (1 - 2 * q2)

class TransitModel:
    """Sum of circular-orbit planets sharing one star."""
    def __init__(self, t, nplanet):
        self.t = t; self.n = nplanet
        self.pm = batman.TransitParams(); pm = self.pm
        pm.t0 = 0; pm.per = 1; pm.rp = 0.01; pm.a = 20; pm.inc = 90; pm.ecc = 0; pm.w = 90
        pm.u = [0.3, 0.2]; pm.limb_dark = "quadratic"
        self.m = batman.TransitModel(pm, t)
    def flux(self, planets, u):
        out = np.ones_like(self.t)
        for (P, T0, rp, a, b) in planets:
            pm = self.pm; pm.per = P; pm.t0 = T0; pm.rp = rp; pm.a = a; pm.u = list(u)
            pm.inc = np.degrees(np.arccos(min(b / a, 1.0)))
            out += self.m.light_curve(pm) - 1
        return out

def joint_fit(t, f, guesses, rho_prior=None, ld_prior=None, groups=None, nwalk=48, nstep=15000, burn=5000, seed=0, free_rho=False, dilution=1.0):
    """Fit N circular-orbit planets sharing one star.

    guesses: list of dicts with P, T0, rp, b (and a if free_rho).
    rho_prior: (mean, sd) Gaussian prior on the stellar density in g/cc (sampled directly).
    ld_prior: ((q1, sd), (q2, sd)) Gaussian priors on the Kipping (2013) q1, q2.
    groups: integer array (same length as t) assigning each cadence to a noise group (e.g. observing season);
            each group gets its own white-noise scale factor.
    dilution: factor g with model = 1 - g * (1 - transit model); g = (true host flux fraction in the aperture) / CROWDSAP
              when the PDCSAP crowding correction assumed a different flux fraction for the host (HIP 31126).
    Parameter vector: per planet [P, T0, rp, b] (+ ln a/R* if free_rho), then [rho] (unless free_rho),
    q1, q2, then one ln(noise factor) per group.
    """
    n = len(guesses); M = TransitModel(t, n)
    groups = np.zeros(len(t), int) if groups is None else np.asarray(groups)
    ug = np.unique(groups); gi = np.searchsorted(ug, groups); ng = len(ug)
    sig0 = np.array([np.std(f[gi == k]) for k in range(ng)])
    k_pl = 5 if free_rho else 4
    def unpack(x):
        pl = []
        rho = None if free_rho else x[k_pl * n]
        off = k_pl * n + (0 if free_rho else 1)
        q1, q2 = x[off:off + 2]; lnj = x[off + 2:off + 2 + ng]
        for i in range(n):
            P, T0, rp, b = x[k_pl * i:k_pl * i + 4]
            a = np.exp(x[k_pl * i + 4]) if free_rho else ars_from_rho(rho, P)
            pl.append((P, T0, rp, a, b))
        return pl, rho, q1, q2, lnj
    def lnp(x):
        pl, rho, q1, q2, lnj = unpack(x)
        if not (0 < q1 < 1 and 0 < q2 < 1 and np.all(np.abs(lnj) < 2)):
            return -np.inf
        if rho is not None and not (0.05 < rho < 50):
            return -np.inf
        lp = 0.0
        for (P, T0, rp, a, b), g in zip(pl, guesses):
            if not (0.001 < rp < 0.2 and 0 <= b < 1 + rp and 1.5 < a < 400 and abs(P / g["P"] - 1) < 2e-3 and abs(T0 - g["T0"]) < 0.1):
                return -np.inf
        if rho_prior and rho is not None:
            lp += -0.5 * ((rho - rho_prior[0]) / rho_prior[1]) ** 2
        if ld_prior:
            lp += -0.5 * (((q1 - ld_prior[0][0]) / ld_prior[0][1]) ** 2 + ((q2 - ld_prior[1][0]) / ld_prior[1][1]) ** 2)
        mod = 1.0 - dilution * (1.0 - M.flux(pl, q_to_u(q1, q2)))
        s2 = (sig0[gi] * np.exp(lnj[gi])) ** 2
        return lp - 0.5 * np.sum((f - mod) ** 2 / s2 + np.log(s2))
    x0, scale = [], []
    for g in guesses:
        x0 += [g["P"], g["T0"], g["rp"], g.get("b", 0.3)]; scale += [1e-5, 1e-3, 1e-3, 0.05]
        if free_rho:
            x0 += [np.log(g.get("a", 20))]; scale += [0.05]
    if not free_rho:
        x0 += [rho_prior[0] if rho_prior else 3.0]; scale += [0.1 * (rho_prior[1] if rho_prior else 1.0)]
    x0 += [ld_prior[0][0] if ld_prior else 0.36, ld_prior[1][0] if ld_prior else 0.3] + [0.0] * ng
    scale += [0.02, 0.02] + [0.01] * ng
    x0 = np.array(x0); ndim = len(x0); rng = np.random.default_rng(seed)
    p0 = x0 + np.array(scale) * rng.standard_normal((nwalk, ndim))
    S = emcee.EnsembleSampler(nwalk, ndim, lnp)
    S.run_mcmc(p0, nstep, progress=False)
    chain = S.get_chain(discard=burn, thin=10, flat=True)
    try:
        tau = S.get_autocorr_time(discard=burn, quiet=True)
    except Exception:
        tau = np.full(ndim, np.nan)
    return dict(chain=chain, tau=tau, unpack=unpack, acc=float(np.mean(S.acceptance_fraction)), nstep=nstep, burn=burn, groups=ug.tolist())

def fit_times(t, f, planet, u, dur, window=3.0, half_range=1.5, nboot=200, seed=0):
    """Fit each transit's mid-time with the shape fixed.

    The uncertainty is empirical: the best-fit model is re-injected at the fitted time into
    cyclically shifted residuals of the same window (preserving correlated noise) and refitted
    `nboot` times; sigma is the scatter of the recovered times.
    Returns rows (epoch, tc, sigma, delta_chi2_of_transit).
    """
    P, T0, rp, a, b = planet; n = np.round((t - T0) / P); rows = []
    rng = np.random.default_rng(seed)
    for k in np.unique(n):
        tc0 = T0 + k * P; m = (np.abs(t - tc0) < window * dur)
        if m.sum() < 30 or not ((t[m] < tc0 - dur).any() and (t[m] > tc0 + dur).any()):
            continue
        tt, ff = t[m], f[m]; M = TransitModel(tt, 1); sig = np.std(ff[np.abs(tt - tc0) > dur])
        grid = np.linspace(tc0 - half_range * dur, tc0 + half_range * dur, 241)
        mods = np.array([M.flux([(P, g, rp, a, b)], u) for g in grid])
        def best(y):
            c = np.sum((y[None, :] - mods) ** 2, axis=1) / sig ** 2
            i = np.argmin(c)
            if 0 < i < len(grid) - 1:   # parabolic refinement
                y0, y1, y2 = c[i - 1], c[i], c[i + 1]; den = y0 - 2 * y1 + y2
                off = 0.5 * (y0 - y2) / den if den > 0 else 0.0
                return grid[i] + off * (grid[1] - grid[0]), c
            return grid[i], c
        tc, c = best(ff)
        dchi = float(np.sum((ff - 1) ** 2) / sig ** 2 - c.min())
        res = ff - M.flux([(P, tc, rp, a, b)], u); base = M.flux([(P, tc, rp, a, b)], u)
        boots = [best(base + np.roll(res, rng.integers(10, len(res) - 10)))[0] for _ in range(nboot)]
        rows.append((int(k), float(tc), float(np.std(boots)), dchi))
    return np.array(rows)
