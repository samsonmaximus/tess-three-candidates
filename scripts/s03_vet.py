"""Step 3: light-curve preparation and vetting statistics, the same code for every signal.

For each system:
  * detrending (a): biweight fitted with every known signal of the system masked (3 x T14 wide), used for
    depths, fits and tests; detrending (b): the same filter with nothing masked, used for blind searches;
  * correlated-noise factor beta = scatter of depth/sigma over a full-phase box scan (as in the companion
    TOI-6284 analysis); every error and significance is inflated by beta;
  * the pre-registered C1 statistic: box S/N with beta from the RMS of binned out-of-transit residuals
    (time-averaging method, bins of T14), as worded in ACCEPTANCE_overnight.md;
  * empirical significance: box depth at the ephemeris against 3000 random epochs;
  * per-transit, per-season, odd/even, secondary-eclipse scan, period aliases P/2..P/10 (and 2P),
    SAP vs PDCSAP depth, flux-weighted centroid shifts (per sector, calibrated at random phases);
  * blind BLS of all data (other signals masked) on 30-min bins.
Controls (TOI-5997 b, TOI-678.01, the 14.81 d field signal of HIP 31126, the 6.56 d TCE of TOI-678)
go through the same code.
"""
from common import *
from tcemine.lightcurves import load_bundle, clean
from tcemine.detrend import masked_biweight, masked_biweight_additive, in_transit
from tcemine import vetting as V
from wotan import flatten
from scipy.stats import chi2 as chi2d

def plain_biweight(t, f, s, window):
    out = np.full_like(f, np.nan)
    for q in np.unique(s):
        m = s == q
        out[m] = flatten(t[m], f[m], method="biweight", window_length=window, edge_cutoff=0.0, break_tolerance=0.4)
    return out

def prepare(S, ephems, window):
    secs = load_bundle(R("data", S["bundle"]))[S["tic"]]
    t, f, e, s, c1, c2 = clean(secs)
    _, fsap, _, _, _, _ = clean(secs, flux="SAP_FLUX")
    masks = [(P, T0, 2 * T14) for P, T0, T14 in ephems]          # masked at 1.5 x (2 x T14) = 3 x T14 wide
    fl = masked_biweight(t, f, s, window, masks)
    fs = masked_biweight(t, fsap, s, window, masks)
    fp = plain_biweight(t, f, s, window)
    c1d = masked_biweight_additive(t, c1, s, window, masks); c2d = masked_biweight_additive(t, c2, s, window, masks)
    ok = np.isfinite(fl) & np.isfinite(fs)
    r = fl - 1; mad = 1.4826 * np.nanmedian(np.abs(r[ok] - np.nanmedian(r[ok]))); ok &= r < 5 * mad   # remove flares
    okp = np.isfinite(fp) & ((fp - 1) < 5 * mad)
    meta = []
    for d in secs:
        mm = d["meta"]
        meta.append(dict(sector=d["sector"], cadence_s=round(float(np.nanmedian(np.diff(d["TIME"]))) * 86400),
                         crowdsap=float(mm.get("CROWDSAP") or np.nan), flfrcsap=float(mm.get("FLFRCSAP") or np.nan),
                         camera=mm.get("CAMERA"), ccd=mm.get("CCD"), tstart=float(np.nanmin(d["TIME"])), tstop=float(np.nanmax(d["TIME"]))))
    return dict(t=t[ok], f=fl[ok], fsap=fs[ok], e=e[ok], s=s[ok], c1=c1d[ok], c2=c2d[ok],
                tp=t[okp], fp=fp[okp], ep=e[okp], sp=s[okp], mad=mad, meta=meta)

def beta_timeavg(t, r, T14, cad=2 / 1440):
    """Time-averaging beta (Pont et al. 2006; Winn et al. 2008) at bin length T14: rms of binned residuals over
    the white-noise expectation. Bins are formed within each continuous stretch."""
    n = max(2, int(round(T14 / cad))); s1 = np.std(r); binned = []
    gaps = np.r_[0, np.where(np.diff(t) > 0.5)[0] + 1, len(t)]
    for a, b in zip(gaps[:-1], gaps[1:]):
        k = (b - a) // n
        if k >= 1:
            binned.append(r[a:a + k * n].reshape(k, n).mean(1))
    rb = np.concatenate(binned); M = len(rb)
    return float(max(1.0, np.std(rb) / (s1 / np.sqrt(n) * np.sqrt(M / (M - 1)))))

def alias_tests(t, f, P, T0, dur, beta):
    """Box depth at the extra epochs implied by P/n (n=2..10): events of P/n that are not events of P.
    Also 2P: depth of odd and even epochs (reported by odd_even)."""
    out = []
    for nn in range(2, 11):
        Pn = P / nn; dt = V.phase_offset(t, Pn, T0, 0)
        k = np.round((t - T0 - dt) / Pn).astype(int)            # event index at P/n
        extra = (k % nn) != 0                                   # not an event of P
        d = V.box_depth(t[extra], f[extra], Pn, T0, dur)
        out.append(dict(n=nn, P=Pn, depth_ppm=d[0] * 1e6, err_ppm=d[1] * 1e6 * beta, n_in=d[2]))
    return out

def vet_signal(L, P, T0, dur, other, exclude=(), blind_pmax=None, nnull=3000):
    t, f, s = L["t"], L["f"], L["s"]
    m = np.ones_like(t, bool)
    for Po, T0o, Do in other:
        m &= ~in_transit(t, Po, T0o, Do, 2.0)
    n = np.round((t - T0) / P)
    for k in exclude:
        m &= n != k
    tt, ff, ss, fsap = t[m], f[m], s[m], L["fsap"][m]
    ph, sd, se = V.secondary_scan(tt, ff, P, T0, dur)
    z = sd / se; beta = float(max(1.0, np.nanstd(z)))
    d = V.box_depth(tt, ff, P, T0, dur)
    dsap = V.box_depth(tt, fsap, P, T0, dur)
    oot = ~in_transit(tt, P, T0, dur, 3.0)
    beta_c1 = beta_timeavg(tt[oot], ff[oot] - 1, dur)
    o, ev, oe = V.odd_even(tt, ff, P, T0, dur)
    pt = V.per_transit(tt, ff, P, T0, dur)
    if len(pt) >= 2:
        c2w, dof, _ = V.chi2_constant(pt[:, 2], pt[:, 3]); c2b = c2w / beta ** 2
    else:
        c2w, dof, c2b = np.nan, 0, np.nan
    g = V.seasons(ss); seas = []
    for k in np.unique(g):
        q = g == k; dd = V.box_depth(tt[q], ff[q], P, T0, dur)
        if np.isfinite(dd[0]):
            seas.append(dict(sectors=sorted(set(int(x) for x in ss[q])), depth=dd[0] * 1e6, err=dd[1] * 1e6 * beta, n_in=dd[2]))
    if len(seas) >= 2:
        sc2, sdof, _ = V.chi2_constant(np.array([x["depth"] for x in seas]), np.array([x["err"] for x in seas]))
    else:
        sc2, sdof = np.nan, 0
    null = V.random_phase_null(tt, ff, P, T0, dur, extra=[L["c1"][m], L["c2"][m]], n=nnull)
    # empirical tail fraction of the flux statistic
    rng = np.random.default_rng(7); ndeeper = 0; ntot = 0
    for _ in range(nnull):
        Tr = T0 + rng.uniform(2 * dur, P - 2 * dur); dr = V.box_depth(tt, ff, P, Tr, dur)[0]
        if np.isfinite(dr):
            ntot += 1; ndeeper += dr >= d[0]
    i05 = np.argmin(abs(ph - 0.5))
    rec = dict(P=P, T0=T0, T14_h=dur * 24, beta=beta, beta_c1=beta_c1,
               depth_ppm=d[0] * 1e6, depth_err_white_ppm=d[1] * 1e6, depth_err_ppm=d[1] * 1e6 * beta, snr_box=d[0] / (d[1] * beta),
               snr_c1=d[0] / (d[1] * beta_c1), n_in=d[2],
               sap_depth_ppm=[dsap[0] * 1e6, dsap[1] * 1e6 * beta],
               odd_ppm=[o[0] * 1e6, o[1] * 1e6 * beta], even_ppm=[ev[0] * 1e6, ev[1] * 1e6 * beta], oddeven_sigma=oe / beta,
               sec05_ppm=[sd[i05] * 1e6, se[i05] * 1e6 * beta], sec_max_dip_z=float(np.nanmax(z)) / beta,
               sec_max_dip_phase=float(ph[np.nanargmax(z)]), sec_max_bright_z=float(-np.nanmin(z)) / beta,
               sec_n_dip3=int(np.sum(z / beta > 3)), sec_n_bright3=int(np.sum(z / beta < -3)),
               n_transits=len(pt), per_transit=pt.tolist(), pertransit_chi2=c2b, pertransit_dof=dof,
               pertransit_p=float(chi2d.sf(c2b, dof)) if dof else np.nan,
               seasons=seas, season_chi2=sc2, season_dof=sdof, season_p=float(chi2d.sf(sc2, sdof)) if sdof else np.nan,
               null_flux_z=null[0][3], null_col_z=null[1][3], null_row_z=null[2][3], null_frac_deeper=ndeeper / max(ntot, 1), null_n=ntot,
               col_shift_pix=null[1][0], row_shift_pix=null[2][0], col_null_sd=null[1][2], row_null_sd=null[2][2],
               per_sector_centroid=V.per_sector_centroid(tt, L["c1"][m], L["c2"][m], ss, P, T0, dur),
               aliases=alias_tests(tt, ff, P, T0, dur, beta),
               excluded_epochs=list(exclude))
    if blind_pmax:
        tp, fp, ep, sp = L["tp"], L["fp"], L["ep"], L["sp"]
        mp = np.ones_like(tp, bool)
        for Po, T0o, Do in other:
            mp &= ~in_transit(tp, Po, T0o, Do, 2.0)
        tb, fb, eb = bin30(tp[mp], fp[mp], ep[mp])
        top = bls_all(tb, fb, eb, dur, pmin=1.0, pmax=blind_pmax)
        rank = next((i + 1 for i, (pp, _) in enumerate(top) if min(abs(pp / P - 1), abs(pp * 2 / P - 1), abs(pp / 2 / P - 1)) < 0.002), None)
        rec["blind_bls"] = dict(pmax=blind_pmax, top=top[:5], rank_of_true=rank,
                                rank_exact=next((i + 1 for i, (pp, _) in enumerate(top) if abs(pp / P - 1) < 0.002), None))
    return rec

def bin30(t, f, e, w=30 / 1440):
    k = np.floor((t - t.min()) / w).astype(int); u, inv, cnt = np.unique(k, return_inverse=True, return_counts=True)
    tb = np.bincount(inv, t) / cnt; fb = np.bincount(inv, f) / cnt; eb = np.sqrt(np.bincount(inv, e ** 2)) / cnt
    ok = cnt >= 5
    return tb[ok], fb[ok], eb[ok]

def bls_all(t, f, e, dur, pmin, pmax, ntop=8):
    from astropy.timeseries import BoxLeastSquares
    T = t.max() - t.min(); durs = np.array([0.6, 0.8, 1.0, 1.3]) * dur
    nper = int(min(600000, 3 * T / (0.25 * durs.min()) * np.log(pmax / pmin)))
    periods = np.exp(np.linspace(np.log(pmin), np.log(pmax), nper))
    r = BoxLeastSquares(t, f, e).power(periods, durs, objective="snr")
    top = []
    for j in np.argsort(r.power)[::-1]:
        if all(abs(r.period[j] / q - 1) > 0.01 for q, _ in top):
            top.append((float(r.period[j]), float(r.power[j])))
        if len(top) >= ntop:
            break
    return top

if __name__ == "__main__":
    only = sys.argv[1:] or list(SYSTEMS)
    allout = load_numbers("vetting")
    for sysname in only:
        S = SYSTEMS[sysname]
        eph = {"new": get_ephem(sysname, "new")}
        for k in S["others"]:
            eph[k] = get_ephem(sysname, k)
        window = max(0.75, 3 * eph["new"][2])
        L = prepare(S, list(eph.values()), window)
        np.savez(R("results", f"lc_{S['key']}.npz"), t=L["t"], f=L["f"], fsap=L["fsap"], e=L["e"], s=L["s"], c1=L["c1"], c2=L["c2"])
        np.savez(R("results", f"lcplain_{S['key']}.npz"), t=L["tp"], f=L["fp"], e=L["ep"], s=L["sp"])
        allout[f"{sysname}:sectors"] = L["meta"]; allout[f"{sysname}:rms_2min_ppm"] = L["mad"] * 1e6
        allout[f"{sysname}:window_d"] = window
        pmax = {"HIP 31126": 60.0, "TOI-678": 150.0, "TOI-5997": 40.0}[sysname]
        for which in ["new"] + [k for k, v in S["others"].items() if v["control"]]:
            other = [eph[k] for k in eph if k != which]
            rec = vet_signal(L, *eph[which], other, exclude=S["exclude_epochs"].get(which, []),
                             blind_pmax=pmax if which == "new" else None)
            allout[f"{sysname}:{which}"] = rec
            print(sysname, which, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()
                                   if k not in ("seasons", "per_sector_centroid", "aliases", "per_transit", "blind_bls")})
            print("   seasons", [(x["sectors"], round(x["depth"]), round(x["err"])) for x in rec["seasons"]])
            print("   aliases", [(a["n"], round(a["depth_ppm"]), round(a["err_ppm"])) for a in rec["aliases"][:5]])
            if "blind_bls" in rec:
                print("   blind", rec["blind_bls"])
        save_numbers("vetting", allout)
