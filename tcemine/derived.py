"""Derived planet properties (predictions, not measurements)."""
import numpy as np

def chen_kipping_mass(R):
    """Mass (Earth masses) from radius (Earth radii), deterministic Chen & Kipping (2017) relation.

    Terran: R = 1.008 M^0.279 for M < 2.04 (R < 1.23); Neptunian: R = 0.808 M^0.589 up to ~0.41 M_Jup.
    """
    R = np.asarray(R, dtype=float)
    terran = (R / 1.008) ** (1 / 0.279)
    nept = (R / 0.808) ** (1 / 0.589)
    return np.where(R < 1.008 * 2.04 ** 0.279, terran, nept)

def rv_semi_amplitude(Mp_earth, Mstar_sun, P_days, inc_deg=90.0, e=0.0):
    """K in m/s for a planet of mass Mp (Earth masses) on a P-day orbit around Mstar (solar masses)."""
    return (28.4329 * (Mp_earth / 317.828) * np.sin(np.radians(inc_deg)) * Mstar_sun ** (-2 / 3)
            * (P_days / 365.25) ** (-1 / 3) / np.sqrt(1 - e ** 2))

def mutual_hill_separation(a1, a2, m1, m2, Mstar):
    """Separation in mutual Hill radii; a in au, masses in solar masses."""
    a1, a2 = min(a1, a2), max(a1, a2)
    RH = ((m1 + m2) / (3 * Mstar)) ** (1 / 3) * (a1 + a2) / 2
    return (a2 - a1) / RH
