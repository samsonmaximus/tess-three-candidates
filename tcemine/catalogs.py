"""Cross-match SPOC multi-sector TCEs against the TOI and CTOI lists and apply vetting cuts."""
import numpy as np, pandas as pd

TESS_SYSTEMATIC_PERIODS = (13.7, 27.4, 6.85, 1.0, 0.5, 2.0)

def harmonic_match(P, Pk, tol=0.004, nmax=6):
    """True if P and Pk are equal or integer multiples (1..nmax) within fractional tol."""
    if not (Pk > 0):
        return False
    for r in (P / Pk, Pk / P):
        n = round(r)
        if 1 <= n <= nmax and abs(r - n) / n < tol:
            return True
    return False

def epoch_overlap(P, T0, dur_d, Pk, T0k, durk_d):
    """True if a known object's transit falls on the TCE ephemeris, or the TCE epoch falls on the known one."""
    win = max(dur_d, durk_d if np.isfinite(durk_d) else 0)
    on = abs((((T0k - T0) / P + 0.5) % 1 - 0.5) * P) < win
    on2 = Pk > 0 and abs((((T0 - T0k) / Pk + 0.5) % 1 - 0.5) * Pk) < win
    return bool(on or on2)

def known_objects(toi, ctoi):
    rows = []
    for _, r in toi.iterrows():
        disp = r["TFOPWG Disposition"] if isinstance(r["TFOPWG Disposition"], str) else r["TESS Disposition"]
        rows.append((int(r["TIC ID"]), f"TOI-{r['TOI']}", disp, r["Period (days)"], r["Epoch (BJD)"], r["Duration (hours)"]))
    for _, r in ctoi.iterrows():
        rows.append((int(r["TIC ID"]), f"CTOI-{r['CTOI']}", r["User Disposition"], r["Period (days)"], r["Transit Epoch (BJD)"], r["Duration (hrs)"]))
    return pd.DataFrame(rows, columns=["tic", "name", "disp", "P", "T0", "dur_h"])

def expected_duration_h(P, rho):
    a = (6.674e-8 * rho * (P * 86400) ** 2 / (3 * np.pi)) ** (1 / 3)
    return P * 24 / np.pi * np.arcsin(np.clip(1 / a, 0, 1))

def shortlist(tce, known, snr=8.5, mes=8.5, ntr=3, pmin=0.5, rp=(0.5, 6.0), rstar_max=1.8, teff=(2800, 7000)):
    """Apply the paper's cuts and return unclaimed TCEs (one row per star+period)."""
    t = tce.copy()
    t["dur_ratio"] = t.tce_duration / expected_duration_h(t.tce_period, t.tce_sdensity)
    byt = {k: g for k, g in known.groupby("tic")}
    def is_known(r):
        g = byt.get(r.ticid)
        return g is not None and any(harmonic_match(r.tce_period, p) for p in g.P)
    t["matched"] = [is_known(r) for r in t.itertuples()]
    c = t[~t.matched]
    c = c[(c.tce_model_snr >= snr) & (c.tce_max_mult_ev >= mes) & (c.tce_num_transits >= ntr) & (c.tce_period >= pmin)
          & (c.tce_prad > rp[0]) & (c.tce_prad <= rp[1]) & (c.tce_sradius <= rstar_max) & c.tce_steff.between(*teff)
          & (c.tce_bin_oedp_stat < 3) & (c.tce_cap_stat > c.tce_hap_stat) & (c.tce_cap_stat > 0)
          & ((c.tce_dicco_msky_err <= 0) | (c.tce_dicco_msky / c.tce_dicco_msky_err < 3))
          & (c.boot_fap >= 0) & (c.boot_fap < 1e-8) & c.dur_ratio.between(0.25, 1.5)]
    bad = np.zeros(len(c), bool)
    for Pb in TESS_SYSTEMATIC_PERIODS:
        bad |= (abs(c.tce_period / Pb - 1) < 0.02).values
    c = c[~bad].sort_values("tce_model_snr", ascending=False)
    keep, seen = [], []
    for i, r in c.iterrows():
        if any(s[0] == r.ticid and abs(r.tce_period / s[1] - 1) < 0.003 for s in seen):
            continue
        seen.append((r.ticid, r.tce_period)); keep.append(i)
    c = c.loc[keep].copy()
    c["n_known"] = [len(byt[x]) if x in byt else 0 for x in c.ticid]
    flags = []
    for r in c.itertuples():
        g = byt.get(r.ticid); f = []
        if g is not None:
            for k in g.itertuples():
                if np.isfinite(k.T0) and epoch_overlap(r.tce_period, r.tce_time0bt + 2457000, r.tce_duration / 24, k.P, k.T0, (k.dur_h or 0) / 24):
                    f.append(k.name)
        flags.append(";".join(f))
    c["epoch_overlap"] = flags
    return c
