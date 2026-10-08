"""Step 5a: pixel-level localisation of every signal (candidates and controls) with the TESS PRF.

See tcemine/pixel.py for the method. Output: results/pixel_<key>.json (best-fit sky offset of the source from
the target, bootstrap covariance, per-sector registration and amplitudes, and a test of every TIC star that
could produce the signal) and results/pixel_<key>_figdata.npz for the figure.
"""
from common import *
from tcemine.pixel import grid_prf, StampPRF, btjd_to_year, propagate, register, Localiser, linfit
from astropy.wcs import WCS
from astropy.io.fits import Header
from scipy.stats import chi2 as chi2d, norm

DIF = json.load(gzip.open(R("data", "tpf_diffimages.json.gz")))
PRF = json.load(gzip.open(R("data", "tess_prf_s0004.json.gz")))
TIC = json.load(open(R("data", "tic82_cones.json")))
ST = load_numbers("stellar")
CONF = {"HIP 31126": dict(cone="H31126", sig={"new": "c29", "f14": "f14"}),
        "TOI-678": dict(cone="T678", sig={"new": "c130", "b11": "b11"}),
        "TOI-5997": dict(cone="T5997", sig={"new": "c14", "b5": "b5"})}
STRKEYS = ("CTYPE1", "CTYPE2", "RADESYS", "CUNIT1", "CUNIT2")
NBOOT = int(os.environ.get("NBOOT", 500))

def stars_for(sysname, S):
    df = tic_field(TIC[CONF[sysname]["cone"]])
    df = df[df.Tmag.notna()].copy()
    for c in ("ra", "dec", "pmRA", "pmDEC", "Tmag"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if sysname == "HIP 31126":   # TIC magnitudes of the pair are blended; use Gaia-based T for both components
        df.loc[df.ID == "293689266", "Tmag"] = ST["HIP 31126"]["Tmag_gaia"]
        df.loc[df.ID == "293689267", "Tmag"] = ST["HIP 31126 B"]["Tmag_gaia"]
    t0 = df[df.ID == str(S["tic"])].iloc[0]
    df["dstArcSec"] = np.hypot((df.ra - t0.ra) * np.cos(np.radians(t0.dec)), df.dec - t0.dec) * 3600
    df = df.sort_values("dstArcSec").reset_index(drop=True)
    assert df.ID[0] == str(S["tic"])
    return df

def wcs_of(d):
    H = Header()
    for k, v in d["wcs"].items():
        if k.endswith("P") and (k.startswith("CRVAL") or k.startswith("CRPIX") or k.startswith("CDELT") or k.startswith("CTYPE") or k.startswith("CUNIT")):
            continue
        try:
            H[k] = v if k in STRKEYS else float(v)
        except Exception:
            pass
    return WCS(H, naxis=2)

def sector_setup(d, st, Ttarget):
    nrow, ncol = d["nrow"], d["ncol"]; w = wcs_of(d)
    ep = btjd_to_year(0.5 * (d["tmin"] + d["tmax"]))
    ra, dec = propagate(st.ra.values, st.dec.values, st.pmRA.values, st.pmDEC.values, 2000.0, ep)
    x, y = w.all_world2pix(ra, dec, 0)
    prf = StampPRF(grid_prf(PRF, d["camera"], d["ccd"], d["col0"] + x[0], d["row0"] + y[0]))
    img = np.array(d["meanimg"], float).reshape(nrow, ncol)
    sel = (st.Tmag.values < Ttarget + 6) & (x > -3) & (x < ncol + 2) & (y > -3) & (y < nrow + 2)
    rel = 10 ** (-0.4 * (st.Tmag.values[sel] - Ttarget))
    dx, dy, scale, bkg, c2r = register(img, prf, list(zip(x[sel], y[sel])), rel)
    def to_pix(r, dd, w=w, dx=dx, dy=dy):
        xx, yy = w.all_world2pix(np.atleast_1d(r), np.atleast_1d(dd), 0)
        return float(xx[0]) + dx, float(yy[0]) + dy
    ap = (np.array(d["aperture"]).reshape(nrow, ncol) & 2) > 0
    return dict(nrow=nrow, ncol=ncol, wcs=w, ep=ep, ra=ra, dec=dec, x=x + dx, y=y + dy, prf=prf, img=img, ap=ap,
                reg=dict(dx=dx, dy=dy, scale=scale, bkg=bkg, chi2r=c2r, n_stars=int(sel.sum())), to_pix=to_pix)

def run(sysname):
    S = SYSTEMS[sysname]; st = stars_for(sysname, S); Tt = float(st.Tmag[0])
    prods = [d for d in DIF.values() if d["tic"] == S["tic"]]
    setups = {d["sector"]: sector_setup(d, st, Tt) for d in prods}
    out = dict(target_Tmag=Tt, sectors={int(k): v["reg"] for k, v in setups.items()}, signals={})
    rng = np.random.default_rng(11)
    for which, name in CONF[sysname]["sig"].items():
        secs = []
        for d in prods:
            sg = d["signals"][name]
            if not sg["transits"]:
                continue
            u = setups[d["sector"]]; n = len(sg["transits"])
            Dm = np.mean([np.array(t["diff"], float) for t in sg["transits"]], axis=0).reshape(u["nrow"], u["ncol"])
            nulls = np.array([np.array(z["diff"], float) for z in sg["nulls"]]).reshape(-1, u["nrow"], u["ncol"])
            mad = 1.4826 * np.median(np.abs(nulls - np.median(nulls, 0)), axis=0)
            sig = np.maximum(mad, 1e-3 * np.nanmax(np.abs(Dm))) / np.sqrt(n)
            secs.append(dict(sector=d["sector"], D=Dm, sig=sig, nulls=nulls, ntr=n, prf=u["prf"], to_pix=u["to_pix"], scale=u["reg"]["scale"],
                             ra0=u["ra"][0], dec0=u["dec"][0], u=u))
        L = Localiser(secs)
        dra, ddec, c2 = L.fit(half=120.0, step=3.0)
        amps = L.amplitudes(dra, ddec)
        # bootstrap: best model + averaged nulls, refitted from a random start within +-6" of the injected position.
        # For each bootstrap we also keep dchi2_true = chi2(injected position) - chi2(refit minimum), which calibrates
        # the per-pixel chi2 surface for correlated noise (its median over the bootstraps vs the chi2_2 median).
        boots, dtrue = [], []
        for b in range(NBOOT):
            data = []
            for s, a in zip(secs, amps):
                idx = rng.choice(len(s["nulls"]), size=min(s["ntr"], len(s["nulls"])), replace=False)
                data.append(a["model"] + s["nulls"][idx].mean(0))
            st0 = (dra + rng.uniform(-6, 6), ddec + rng.uniform(-6, 6))
            bx, by, bc = L.fit(data=data, start=st0)
            boots.append((bx, by)); dtrue.append(max(L.chi2(dra, ddec, data) - bc, 0.0))
        boots = np.array(boots); C = np.cov(boots.T); bias = boots.mean(0) - np.array([dra, ddec])
        dtrue = np.array(dtrue); kscale = max(1.0, float(np.median(dtrue) / chi2d.ppf(0.5, 2)))
        Ci0 = np.linalg.inv(C); m2boot = np.einsum("ij,jk,ik->i", boots - [dra, ddec], Ci0, boots - [dra, ddec])
        # chi2 of the target hypothesis and of every star able to produce the signal
        c2t = L.chi2(0.0, 0.0)
        mean_ep = np.mean([s["u"]["ep"] for s in secs])
        ra_m, dec_m = propagate(st.ra.values, st.dec.values, st.pmRA.values, st.pmDEC.values, 2000.0, mean_ep)
        off_e = (ra_m - ra_m[0]) * np.cos(np.radians(dec_m[0])) * 3600; off_n = (dec_m - dec_m[0]) * 3600
        # flux of each star inside the SPOC aperture, relative to the total in the aperture (first sector with transits)
        u0 = secs[0]["u"]; fl = 10 ** (-0.4 * (st.Tmag.values - Tt))
        inap = np.array([fl[k] * (u0["prf"].image(u0["x"][k], u0["y"][k], u0["nrow"], u0["ncol"]) * u0["ap"]).sum() for k in range(len(st))])
        frac = inap / inap.sum()
        vet = load_numbers("vetting").get(f"{sysname}:{which}", {})
        depth = vet.get("depth_ppm", np.nan) * 1e-6 * float(np.nanmean([m["crowdsap"] for m in load_numbers("vetting")[f"{sysname}:sectors"]]))
        Ci = np.linalg.inv(C); rows = []
        for k in range(len(st)):
            need = depth / frac[k] if frac[k] > 0 else np.inf
            sep = float(np.hypot(off_e[k], off_n[k]))
            if (need > 1.0 and k > 0) or sep > 150:
                continue
            dv = np.array([off_e[k] - dra, off_n[k] - ddec]); m2 = float(dv @ Ci @ dv)
            p = float(chi2d.sf(m2, 2)); zeq = float(norm.isf(p / 2)) if p > 0 else 40.0
            dc = float(L.chi2(off_e[k], off_n[k]) - c2); pc = float(chi2d.sf(dc / kscale, 2))
            zc = float(norm.isf(pc / 2)) if pc > 0 else 40.0
            rows.append(dict(TIC=st.ID[k], Tmag=float(st.Tmag[k]), sep=sep, off_e=float(off_e[k]), off_n=float(off_n[k]),
                             frac_ap=float(frac[k]), need_depth=float(need), mahal2=m2, p=p, sigma=zeq,
                             frac_boot_beyond=float(np.mean(m2boot >= m2)), dchi2=dc, sigma_dchi2=zc,
                             excl3=bool(zeq >= 3 and zc >= 3)))
        out["signals"][which] = dict(name=name, sectors=[s["sector"] for s in secs], ntr=[s["ntr"] for s in secs],
                                     best_offset_arcsec=[dra, ddec], chi2_min=c2, chi2_target=c2t,
                                     npix=int(sum(a["npix"] for a in amps)), cov=C.tolist(),
                                     sd_arcsec=[float(np.sqrt(C[0, 0])), float(np.sqrt(C[1, 1]))], boot_bias=bias.tolist(),
                                     amps=[dict(sector=s["sector"], amp=a["amp"], A=a["A"], bkg=a["bkg"], chi2=a["chi2"], npix=a["npix"]) for s, a in zip(secs, amps)],
                                     A_target=float(L.amplitudes(0.0, 0.0)[0]["A"]), A_best=float(amps[0]["A"]),
                                     depth_in_aperture=depth, stars=rows, nboot=NBOOT, chi2_scale=kscale,
                                     dchi2_true_median=float(np.median(dtrue)), max_boot_m2=float(m2boot.max()),
                                     dchi2_nosource=float(L.chi2(5000.0, 5000.0) - c2))
        d0 = np.hypot(dra, ddec); print("   chi2 scale %.2f, median dchi2_true %.2f, max boot m2 %.1f" % (kscale, np.median(dtrue), m2boot.max()))
        print(sysname, which, "offset E,N = %.2f, %.2f arcsec (|%.2f|), sd %.2f %.2f; target: %.2f sigma" %
              (dra, ddec, d0, np.sqrt(C[0, 0]), np.sqrt(C[1, 1]), [r["sigma"] for r in rows if r["TIC"] == str(S["tic"])][0]))
        for r in sorted(rows, key=lambda r: r["sep"])[:12]:
            print("    TIC %s T=%.2f sep=%.1f\" need=%.3f  -> %.2f sigma (p=%.2g) dchi2=%.1f (%.2f sigma scaled) beyond=%.3f excl3=%s" % (r["TIC"], r["Tmag"], r["sep"], r["need_depth"], r["sigma"], r["p"], r["dchi2"], r["sigma_dchi2"], r["frac_boot_beyond"], r["excl3"]))
        # figure data: the sector whose difference image has the highest matched-filter S/N at the target
        def mf_snr(s):
            x0, y0 = s["u"]["x"][0], s["u"]["y"][0]; Mp = s["prf"].image(x0, y0, *s["D"].shape)
            return float(np.nansum(s["D"] * Mp / s["sig"] ** 2) / np.sqrt(np.nansum(Mp ** 2 / s["sig"] ** 2)))
        sbest = max(secs, key=mf_snr)
        np.savez(R("results", f"pixel_{S['key']}_{which}_fig.npz"), D=sbest["D"], sig=sbest["sig"], img=sbest["u"]["img"], ap=sbest["u"]["ap"],
                 x=sbest["u"]["x"], y=sbest["u"]["y"], tmag=st.Tmag.values, ids=st.ID.values.astype(str),
                 fit=np.array(sbest["to_pix"](sbest["ra0"] + dra / 3600 / np.cos(np.radians(sbest["dec0"])), sbest["dec0"] + ddec / 3600)),
                 sector=sbest["sector"], off_e=off_e, off_n=off_n, best=np.array([dra, ddec]), cov=C, boots=boots)
    json.dump(out, open(R("results", f"pixel_{S['key']}.json"), "w"), indent=1, default=float)

if __name__ == "__main__":
    for sysname in (sys.argv[1:] or list(SYSTEMS)):
        run(sysname)
