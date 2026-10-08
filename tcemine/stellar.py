"""Empirical M-dwarf relations used for TOI-6284."""
import numpy as np

def abs_mag(m, plx_mas):
    """Absolute magnitude from apparent magnitude and parallax (mas)."""
    return m + 5 * np.log10(plx_mas / 1000.0) + 5

def mann15_radius(MKs):
    """Mann et al. (2015) R*-M_Ks relation, Table 1 (2.89% scatter). Valid 4.6 < M_Ks < 9.8."""
    return 1.9515 - 0.3520 * MKs + 0.01680 * MKs**2

def mann19_mass(MKs):
    """Mann et al. (2019) M*-M_Ks relation, n=5 fit (2.2% scatter). Valid 4 < M_Ks < 11."""
    a = [-0.642, -0.208, -8.43e-4, 7.87e-3, 1.42e-4, -2.13e-4]
    x = MKs - 7.5
    return 10 ** sum(ai * x**i for i, ai in enumerate(a))

def density_cgs(M, R):
    """Mean stellar density in g/cm^3 from M (Msun) and R (Rsun)."""
    return 1.40894 * M / R**3

def mdwarf_params(Ks, eKs, plx, eplx, n=200000, seed=0):
    """Monte Carlo R, M, rho with relation scatter included."""
    rng = np.random.default_rng(seed)
    k = rng.normal(Ks, eKs, n); p = rng.normal(plx, eplx, n)
    MK = abs_mag(k, p)
    R = mann15_radius(MK) * (1 + rng.normal(0, 0.0289, n))
    M = mann19_mass(MK) * (1 + rng.normal(0, 0.022, n))
    rho = density_cgs(M, R)
    q = lambda x: np.percentile(x, [16, 50, 84])
    return dict(MKs=q(MK), R=q(R), M=q(M), rho=q(rho))
