"""Step 8: write paper/numbers.tex (LaTeX macros) and the table-row files paper/tab_*.tex from the analysis outputs.

Derived numbers in the paper come from here. With --final, any missing input is an error, so the paper cannot be
built with a placeholder; without it, missing values print as a bold ?? (drafting only).
Macro suffixes: Ts = TOI-678, Hp = HIP 31126 (host A; HpB = host B), Tf = TOI-5997;
controls: Tsb = TOI-678.01, Tfb = TOI-5997 b, Hpf = the 14.8-d signal in the HIP 31126 field, Tst = the 6.56-d TCE.
"""
import glob
from common import *
from astropy.time import Time
import astropy.units as u
from scipy.stats import chi2 as chi2d, norm
FINAL = "--final" in sys.argv
N = load_numbers(); V = N["vetting"]; ST = N["stellar"]
TAG = {"TOI-678": "Ts", "HIP 31126": "Hp", "TOI-5997": "Tf"}
CTRL = {"TOI-678:b11": "Tsb", "TOI-5997:b5": "Tfb", "HIP 31126:f14": "Hpf", "TOI-678:t6": "Tst"}
M = {}; missing = []

def miss(name):
    missing.append(name); return r"\textbf{??}"
def ndec(err, sig=1):
    """decimals so that err has `sig` significant digits (at least 0)."""
    if not np.isfinite(err) or err <= 0:
        return 2
    return int(max(0, sig - 1 - np.floor(np.log10(err))))
def pm(q, nd=None):
    lo, me, hi = q; a, b = me - lo, hi - me
    if nd is None:
        nd = ndec(min(a, b), 2 if min(a, b) / 10 ** np.floor(np.log10(min(a, b))) < 2 else 1)
    if abs(a - b) / max(a, b, 1e-30) < 0.25:
        return f"{me:.{nd}f} \\pm {0.5 * (a + b):.{nd}f}"
    return f"{me:.{nd}f}^{{+{b:.{nd}f}}}_{{-{a:.{nd}f}}}"
def pmv(v, e, nd):
    return f"{v:.{nd}f} \\pm {e:.{nd}f}"
def sci(x, nd=1):
    """x >= 1e-3: plain decimal with nd significant digits; smaller: m x 10^e."""
    if x <= 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    if e >= -3:
        return f"{x:.{max(0, nd - 1 - e)}f}"
    return f"{x / 10 ** e:.{nd - 1}f}\\times10^{{{e}}}"
def meansd(vals):
    """mean +- sample sd, the sd with 1 significant digit (2 if it starts with 1) and the mean to match."""
    v = np.array(vals, float)
    if len(v) == 1:
        return sci(v[0], 2)
    m, s = v.mean(), v.std(ddof=1)
    if m < 1e-3:
        e = int(np.floor(np.log10(max(m, 1e-12))))
        return f"({m / 10 ** e:.1f} \\pm {s / 10 ** e:.1f})\\times10^{{{e}}}"
    nd = ndec(s, 2 if s / 10 ** np.floor(np.log10(s)) < 2 else 1) if s > 0 else 3
    return f"{m:.{nd}f} \\pm {s:.{nd}f}"
def zsig(p):
    return float(norm.isf(p / 2)) if p > 0 else np.inf
def loadj(path):
    return json.load(open(path)) if os.path.exists(path) else None

# ---------------------------------------------------------------- stars
for sysname, T in TAG.items():
    st = ST[sysname]; g = st["gaia"]; tm = st["2mass"]
    M[f"Gaia{T}"] = g["source_id"]; M[f"twomass{T}"] = tm["original_ext_source_id"]
    M[f"pm{T}"] = f"{g['pmra']:+.2f}, {g['pmdec']:+.2f}"; M[f"plx{T}"] = f"{g['parallax']:.3f} \\pm {g['parallax_error']:.3f}"
    M[f"dist{T}"] = pm(st["dist_pc"], 2); M[f"distr{T}"] = f"{st['dist_pc'][1]:.1f}"
    M[f"ruwe{T}"] = f"{g['ruwe']:.2f}"; M[f"G{T}"] = f"{st['G']:.2f}"; M[f"bprp{T}"] = f"{st['bp_rp']:.2f}"
    M[f"J{T}"] = f"{tm['j_m']:.2f}"; M[f"Ks{T}"] = f"{tm['ks_m']:.2f}"
    M[f"rv{T}"] = (f"{g['radial_velocity']:+.1f} \\pm {g['radial_velocity_error']:.1f}" if g.get("radial_velocity") is not None else "--")
    M[f"rvp{T}"] = f"{g['rv_chisq_pvalue']:.2f}" if g.get("rv_chisq_pvalue") is not None else "--"
    M[f"rvamp{T}"] = f"{g['rv_amplitude_robust']:.1f}" if g.get("rv_amplitude_robust") is not None else "--"
    M[f"Teff{T}"] = f"{st['Teff'][1]:.0f} \\pm {0.5 * (st['Teff'][2] - st['Teff'][0]):.0f}"
    M[f"Rs{T}"] = pm(st["R"], 3); M[f"Ms{T}"] = pm(st["M"], 3); M[f"rhos{T}"] = pm(st["rho"], 2)
    M[f"Ls{T}"] = pm(st["L"], 3 if st["L"][1] < 0.5 else 2); M[f"logg{T}"] = pm(st["logg"], 2)
    M[f"feh{T}"] = pm(st["feh"], 2) if "feh" in st else "--"
    if "logg_spec" in st:
        M[f"loggspec{T}"] = f"{st['logg_spec'][1]:.2f}"
st = ST["HIP 31126 B"]; g = st["gaia"]
M.update(GaiaHpB=g["source_id"], GHpB=f"{st['G']:.2f}", bprpHpB=f"{st['bp_rp']:.2f}", plxHpB=f"{g['parallax']:.3f} \\pm {g['parallax_error']:.3f}",
         pmHpB=f"{g['pmra']:+.2f}, {g['pmdec']:+.2f}", ruweHpB=f"{g['ruwe']:.2f}", rvHpB=f"{g['radial_velocity']:+.1f} \\pm {g['radial_velocity_error']:.1f}",
         rvpHpB=f"{g['rv_chisq_pvalue']:.2f}",
         TeffHpB=f"{st['Teff'][1]:.0f} \\pm {0.5 * (st['Teff'][2] - st['Teff'][0]):.0f}", RsHpB=pm(st["R"], 3), MsHpB=pm(st["M"], 3),
         rhosHpB=pm(st["rho"], 2), LsHpB=pm(st["L"], 4), loggHpB=pm(st["logg"], 2), distHpB=pm(st["dist_pc"], 2),
         TmagHpA=f"{ST['HIP 31126']['Tmag_gaia']:.2f}", TmagHpB=f"{st['Tmag_gaia']:.2f}")
xc = ST["HIP 31126"]["xcheck_2MASS"]
M["MKsoffHp"] = f"{ST['HIP 31126']['MKs'][1] - xc['MKs']:.2f}"
M["RxcheckpctHp"] = f"{100 * (xc['R'] / ST['HIP 31126']['R'][1] - 1):.0f}"
M["ebvTs"] = f"{ST['TOI-678']['ebv']:.3f}"
# binary: separation (Gaia epoch 2016.0), projected separation, proper-motion and parallax differences
gA, gB = ST["HIP 31126"]["gaia"], ST["HIP 31126 B"]["gaia"]
sep = np.hypot((gB["ra"] - gA["ra"]) * np.cos(np.radians(gA["dec"])), gB["dec"] - gA["dec"]) * 3600
dpm = np.hypot(gB["pmra"] - gA["pmra"], gB["pmdec"] - gA["pmdec"])
dA = 1000 / gA["parallax"]; vt = 4.74047 * dpm / 1000 * dA; aproj = sep * dA
vorb = 29.78 * np.sqrt((ST["HIP 31126"]["M"][1] + ST["HIP 31126 B"]["M"][1]) / aproj)
dplx = gB["parallax"] - gA["parallax"]; edplx = np.hypot(gA["parallax_error"], gB["parallax_error"])
M.update(sepAB=f"{sep:.2f}", aprojAB=f"{aproj:.0f}", dpmAB=f"{dpm:.1f}", dvtAB=f"{vt:.1f}", vorbAB=f"{vorb:.1f}",
         dplxAB=f"{abs(dplx):.3f} \\pm {edplx:.3f}", dplxsigAB=f"{abs(dplx) / edplx:.1f}")

# ---------------------------------------------------------------- data and vetting
for sysname, T in TAG.items():
    secs = V[f"{sysname}:sectors"]; v = V[f"{sysname}:new"]
    M[f"nsec{T}"] = str(len(secs)); M[f"rms{T}"] = f"{V[f'{sysname}:rms_2min_ppm']:.0f}"
    M[f"crowd{T}"] = f"{np.mean([m['crowdsap'] for m in secs]):.3f}"
    M[f"ntr{T}"] = str(v["n_transits"]); M[f"boxdep{T}"] = pmv(v["depth_ppm"], v["depth_err_ppm"], 0)
    snr_full = v["depth_ppm"] / v["depth_err_ppm"]
    M[f"snr{T}"] = f"{v['snr_c1']:.1f}"; M[f"snrfull{T}"] = f"{snr_full:.1f}"
    M[f"beta{T}"] = f"{v['beta']:.2f}"; M[f"betac{T}"] = f"{v['beta_c1']:.2f}"
    sapc = np.array(v["sap_depth_ppm"]) / np.mean([m["crowdsap"] for m in secs])
    M[f"sap{T}"] = pmv(*v["sap_depth_ppm"], 0); M[f"sapc{T}"] = pmv(*sapc, 0)
    M[f"odd{T}"] = pmv(*v["odd_ppm"], 0); M[f"even{T}"] = pmv(*v["even_ppm"], 0)
    M[f"oe{T}"] = f"{v['oddeven_sigma']:.1f}"; M[f"sec{T}"] = pmv(*v["sec05_ppm"], 0)
    M[f"secmax{T}"] = f"{v['sec_max_dip_z']:.1f}"; M[f"secmaxph{T}"] = f"{v['sec_max_dip_phase']:.2f}"
    # per-transit and per-season chi2 with both noise factors (stored chi2 is divided by the full-phase beta^2)
    for kind in ("pertransit", "season"):
        c2b, dof = v[f"{kind}_chi2"], v[f"{kind}_dof"]; c2c = c2b * v["beta"] ** 2 / v["beta_c1"] ** 2
        tag = "pt" if kind == "pertransit" else "sea"
        M[f"{tag}chi{T}"] = f"{c2b:.1f}"; M[f"{tag}dof{T}"] = str(dof); M[f"{tag}p{T}"] = f"{chi2d.sf(c2b, dof):.2f}"
        M[f"{tag}chic{T}"] = f"{c2c:.1f}"; M[f"{tag}pc{T}"] = f"{chi2d.sf(c2c, dof):.2f}"
    M[f"nseas{T}"] = str(len(v["seasons"])); M[f"nullz{T}"] = f"{v['null_flux_z']:.1f}"; M[f"nulln{T}"] = f"{v['null_n']:,}".replace(",", "\\,")
    p = V[f"{sysname}:new"]["P"]; M[f"nindep{T}"] = f"{v['P'] / (v['T14_h'] / 24):.0f}"
    al = v["aliases"]; worst = max(al, key=lambda a: a["depth_ppm"] / a["err_ppm"])
    M[f"aliasworst{T}"] = f"{worst['depth_ppm'] / worst['err_ppm']:.1f}"
    M[f"aliasmin{T}"] = f"{min((v['depth_ppm'] - a['depth_ppm']) / np.hypot(a['err_ppm'], v['depth_err_ppm']) for a in al):.0f}"
    bl = v.get("blind_bls") or {}
    M[f"blindpmax{T}"] = f"{bl.get('pmax', 0):.0f}"
    if bl.get("top"):
        M[f"blindsde{T}"] = f"{bl['top'][0][1]:.1f}"; M[f"blindnext{T}"] = f"{bl['top'][1][1]:.1f}"
        tops = sorted(x[0] for x in bl["top"]); M[f"blindlo{T}"] = f"{tops[0]:.0f}"; M[f"blindhi{T}"] = f"{tops[-1]:.0f}"
    M[f"nshiftcol{T}"] = f"{v['null_col_z']:.1f}"; M[f"nshiftrow{T}"] = f"{v['null_row_z']:.1f}"
    M[f"Tdurbox{T}"] = f"{v['T14_h']:.1f}"
for k, T in CTRL.items():
    v = V[k]; M[f"snr{T}"] = f"{v['snr_c1']:.1f}"; M[f"snrfull{T}"] = f"{v['depth_ppm'] / v['depth_err_ppm']:.1f}"
    M[f"boxdep{T}"] = pmv(v["depth_ppm"], v["depth_err_ppm"], 0); M[f"oe{T}"] = f"{v['oddeven_sigma']:.1f}"
    M[f"ptp{T}"] = f"{v['pertransit_p']:.3f}" if v["pertransit_p"] < 0.1 else f"{v['pertransit_p']:.2f}"
    M[f"seap{T}"] = f"{v['season_p']:.3f}" if v["season_p"] < 0.1 else f"{v['season_p']:.2f}"
    M[f"ntr{T}"] = str(v["n_transits"]); M[f"nullz{T}"] = f"{v['null_flux_z']:.1f}"; M[f"P{T}x"] = f"{v['P']:.3f}"
v = V["TOI-678:b11"]; s8 = [s for s in v["seasons"] if 90 in s["sectors"]][0]
M["deplateTsb"] = pmv(s8["depth"], s8["err"], 0); M["deplatepctTsb"] = f"{100 * (1 - s8['depth'] / v['depth_ppm']):.0f}"
v = V["TOI-5997:b5"]; s5 = [s for s in v["seasons"] if 52 in s["sectors"]][0]; s5c = [s for s in V["TOI-5997:new"]["seasons"] if 52 in s["sectors"]][0]
M["depfiveTfb"] = pmv(s5["depth"], s5["err"], 0); M["depfiveTf"] = pmv(s5c["depth"], s5c["err"], 0)
M["depfivepctTfb"] = f"{100 * (1 - s5['depth'] / v['depth_ppm']):.0f}"

# per-transit depths of the TOI-678 candidate (with the full-phase beta); longest planet transit for the 14.8-d signal
v = V["TOI-678:new"]
for i, (ep, tc, dd, ee) in enumerate(v["per_transit"]):
    M["ptdep" + "abc"[i] + "Ts"] = f"{1e6 * dd:.0f} \\pm {1e6 * ee * v['beta']:.0f}"
for nm, key in (("A", "HIP 31126"), ("B", "HIP 31126 B")):
    rho = ST[key]["rho"][1] * 1000.0; Pd = 14.8104 * 86400
    ars = (6.674e-11 * rho * Pd ** 2 / (3 * np.pi)) ** (1 / 3)
    M[f"fmaxdur{nm}"] = f"{Pd / np.pi * np.arcsin(1.05 / ars) / 3600:.1f}"

# ---------------------------------------------------------------- dilution (HIP 31126)
dl = N["dilution"]; g = dl["HIP 31126:g"]
fa = np.mean([r["f_target"] for r in dl["HIP 31126"] if r["sector"] in g["sectors"]])
fb = np.mean([r["f_second"] for r in dl["HIP 31126"] if r["sector"] in g["sectors"]])
M.update(fAap=f"{fa:.2f}", fBap=f"{fb:.2f}", gA=f"{g['A']:.3f}", gB=f"{g['B']:.3f}")
dev = max(abs(r["f_target"] / r["crowdsap"] - 1) for s in ("TOI-678", "TOI-5997") for r in dl[s])
M["crowdagree"] = f"{np.ceil(100 * dev):.0f}"
vH = V["HIP 31126:new"]
M["depAtrue"] = f"{vH['depth_ppm'] / g['A'] / 1000:.2f}"; M["depBtrue"] = f"{vH['depth_ppm'] / g['B'] / 1000:.1f}"

# ---------------------------------------------------------------- pixel localisation
def star(rows, tic):
    return [r for r in rows if r["TIC"] == tic][0]
for sysname, T in TAG.items():
    P = json.load(open(R("results", f"pixel_{SYSTEMS[sysname]['key']}.json")))["signals"]
    for which, TT in (("new", T), ("f14", "Hpf"), ("b11", "Tsb"), ("b5", "Tfb")):
        if which not in P:
            continue
        s = P[which]; rows = s["stars"]; tgt = star(rows, str(SYSTEMS[sysname]["tic"]))
        M[f"off{TT}"] = f"{s['best_offset_arcsec'][0]:+.1f}, {s['best_offset_arcsec'][1]:+.1f}"
        M[f"offsd{TT}"] = f"{s['sd_arcsec'][0]:.1f}, {s['sd_arcsec'][1]:.1f}"
        M[f"pixsig{TT}"] = f"{tgt['sigma']:.1f}"; M[f"pixsigc{TT}"] = f"{tgt.get('sigma_dchi2', np.nan):.1f}"
        others = [r for r in rows if r["TIC"] != tgt["TIC"]]
        M[f"nnb{TT}"] = str(len(others)); M[f"nnbx{TT}"] = str(sum(bool(r.get("excl3", r["sigma"] >= 3)) for r in others))
        M[f"pixdet{TT}"] = f"{np.sqrt(max(s.get('dchi2_nosource', 0.0), 0.0)):.0f}"
        dd = max(s.get("dchi2_nosource", 0.0), 0.0) / s.get("chi2_scale", 1.0)
        M[f"pixdetc{TT}"] = f"{zsig(chi2d.sf(dd, 2)):.1f}"
        M[f"chiscale{TT}"] = f"{s.get('chi2_scale', np.nan):.1f}"
        ex = sorted([r for r in others if r.get("excl3", r["sigma"] >= 3)], key=lambda r: r["sep"])
        if ex:
            M[f"nbxTIC{TT}"] = ex[0]["TIC"]; M[f"nbxsep{TT}"] = f"{ex[0]['sep']:.0f}"; M[f"nbxT{TT}"] = f"{ex[0]['Tmag']:.1f}"
            M[f"nbxsig{TT}"] = f"{ex[0]['sigma']:.1f}"; M[f"nbxsigc{TT}"] = f"{ex[0].get('sigma_dchi2', np.nan):.1f}"
            M[f"nbxneed{TT}"] = f"{100 * ex[0]['need_depth']:.0f}"
M["nbootpix"] = str(json.load(open(R("results", "pixel_T678.json")))["signals"]["new"].get("nboot", 200))
pA = json.load(open(R("results", "pixel_H31126.json")))["signals"]
rB = star(pA["new"]["stars"], "293689267"); rA = star(pA["new"]["stars"], "293689266")
M.update(pixsigHpBstar=f"{rB['sigma']:.1f}", pixsigcHpBstar=f"{rB.get('sigma_dchi2', np.nan):.1f}",
         dchiHpA=f"{rA['dchi2']:.1f}", dchiHpB=f"{rB['dchi2']:.1f}")
r14 = sorted(pA["f14"]["stars"], key=lambda r: r["sigma"])
M.update(fsrcTIC=r14[0]["TIC"], fsrcsep=f"{r14[0]['sep']:.0f}", fsrcT=f"{r14[0]['Tmag']:.1f}", fsrcsig=f"{r14[0]['sigma']:.1f}",
         fsrcneed=f"{100 * r14[0]['need_depth']:.1f}",
         fABsig=f"{min(r['sigma'] for r in r14 if r['TIC'] in ('293689266', '293689267')):.0f}")
p678 = json.load(open(R("results", "pixel_T678.json")))["signals"]["new"]["stars"]
for r in p678:
    if r["TIC"] == "294395926":
        continue
for tic, nm in (("766625715", "Tsn"), ("766625695", "Tsm"), ("766625709", "Tso")):
    r = star(p678, tic); M[f"sig{nm}"] = f"{r['sigma']:.1f}"; M[f"sigc{nm}"] = f"{r.get('sigma_dchi2', np.nan):.1f}"
    M[f"sep{nm}"] = f"{r['sep']:.0f}"; M[f"T{nm}"] = f"{r['Tmag']:.1f}"; M[f"need{nm}"] = f"{100 * r['need_depth']:.0f}"
    M[f"x{nm}"] = "yes" if r.get("excl3") else "no"
r = star(p678, "294395924"); M.update(sigTsp=f"{r['sigma']:.1f}", sigcTsp=f"{r['sigma_dchi2']:.1f}", TTsp=f"{r['Tmag']:.1f}",
                                     sepTsp=f"{r['sep']:.0f}", needTsp=f"{100 * r['need_depth']:.0f}")
r = star(p678, "294395934"); M.update(sigTsq=f"{r['sigma']:.1f}", sigcTsq=f"{r['sigma_dchi2']:.1f}", TTsq=f"{r['Tmag']:.1f}",
                                     sepTsq=f"{r['sep']:.0f}", needTsq=f"{100 * r['need_depth']:.1f}")
nb = [r for r in p678 if r["TIC"] != "294395926"]
M["needminTs"] = f"{100 * min(r['need_depth'] for r in nb):.0f}"; M["needmaxTs"] = f"{100 * max(r['need_depth'] for r in nb):.0f}"
p5 = json.load(open(R("results", "pixel_T5997.json")))["signals"]["new"]["stars"]
r = sorted([x for x in p5 if x["TIC"] != "39516274"], key=lambda x: x["sep"])[0]
M.update(nbTICTf=r["TIC"], nbTTf=f"{r['Tmag']:.1f}", nbsepTf=f"{r['sep']:.0f}", nbsigTf=f"{r['sigma']:.1f}", nbsigcTf=f"{r['sigma_dchi2']:.1f}")
notex678 = [r for r in p678 if r["TIC"] != "294395926" and not r.get("excl3", r["sigma"] >= 3)]
M["nnotxTs"] = str(len(notex678))
M["notxlistTs"] = ", ".join(f"TIC~{r['TIC']}" for r in sorted(notex678, key=lambda r: r["sep"]))

# ---------------------------------------------------------------- transit fits and derived quantities
def mass_ck17(r):
    return r ** 3.58 if r < 1.23 else (r / 0.808) ** 1.698
fits = {}
for sysname, T in TAG.items():
    F = loadj(R("results", f"fit_{SYSTEMS[sysname]['key']}.json"))
    if F is None:
        miss(f"fit {sysname}"); continue
    fits[sysname] = F
    parts = [(F, T, ST[sysname])]
    if sysname == "HIP 31126":
        parts.append((json.load(open(R("results", "fit_H31126_B.json"))), "HpB", ST["HIP 31126 B"]))
    for FF, TT, star_ in parts:
        p = FF["new"]
        M[f"P{TT}"] = pm(p["P"]); M[f"Tz{TT}"] = pm(p["T0"], 4)
        M[f"Pshort{TT}"] = f"{p['P'][1]:.3f}"
        M[f"rprs{TT}"] = pm(p["rp_rs"], 4); M[f"ars{TT}"] = pm(p["a_rs"], 1); M[f"b{TT}"] = pm(p["b"], 2); M[f"inc{TT}"] = pm(p["inc"], 2)
        M[f"Tdur{TT}"] = pm(p["T14_h"], 2); M[f"Tdurr{TT}"] = f"{p['T14_h'][1]:.1f}"; M[f"dep{TT}"] = pm(p["depth_ppm"], 0)
        M[f"Rp{TT}"] = pm(p["Rp_Re"], 2); M[f"Rpr{TT}"] = f"{p['Rp_Re'][1]:.1f}"
        M[f"a{TT}"] = pm(p["a_au"], 3); M[f"ntau{TT}"] = f"{FF['n_tau']:.0f}"; M[f"rhofit{TT}"] = pm(FF["rho"], 2)
        # insolation and equilibrium temperature from the same a (Kepler's law with M*), zero albedo, full redistribution
        L = star_["L"][1]; eL = 0.5 * (star_["L"][2] - star_["L"][0]); a = p["a_au"][1]; ea = 0.5 * (p["a_au"][2] - p["a_au"][0])
        S = L / a ** 2; eS = S * np.hypot(eL / L, 2 * ea / a)
        Teq = 278.6 * S ** 0.25; eTeq = 0.25 * Teq * eS / S
        Rp = p["Rp_Re"][1]; Mp = mass_ck17(Rp)
        K = 28.4329 * Mp * 0.0031457 * (p["P"][1] / 365.25) ** (-1 / 3) * star_["M"][1] ** (-2 / 3)
        M[f"S{TT}"] = (f"{S:.2f} \\pm {eS:.2f}" if S < 2 else f"{S:.1f} \\pm {eS:.1f}") if S < 10 else f"{S:.0f} \\pm {eS:.0f}"
        M[f"Sr{TT}"] = f"{S:.1f}"
        M[f"Teq{TT}"] = f"{Teq:.0f} \\pm {eTeq:.0f}"; M[f"Teqr{TT}"] = f"{Teq:.0f}"; M[f"Mp{TT}"] = f"{Mp:.1f}"
        M[f"K{TT}"] = f"{K:.1f}"
    M[f"qone{T}"] = pm(F["q1"], 2); M[f"qtwo{T}"] = pm(F["q2"], 2)
    M[f"rhoprior{T}"] = f"{F['rho_prior'][0]:.2f} \\pm {F['rho_prior'][1]:.2f}"
ntaus = [fits[s]["n_tau"] for s in fits] + [json.load(open(R("results", "fit_H31126_B.json")))["n_tau"]]
M["ntaumin"] = f"{min(ntaus):.0f}"; M["ntaumax"] = f"{max(ntaus):.0f}"
if "TOI-678" in fits:
    F = fits["TOI-678"]; tau = max(F["tau"]); M["neffTs"] = f"{48 * (F['nstep'] - F['burn']) / tau:.0f}"
for sysname, T, k in (("TOI-678", "Tsb", "b11"), ("TOI-5997", "Tfb", "b5")):
    if sysname in fits:
        p = fits[sysname][k]; M[f"Rp{T}"] = pm(p["Rp_Re"], 2); M[f"b{T}"] = pm(p["b"], 2); M[f"P{T}"] = pm(p["P"])
        M[f"Tdur{T}"] = pm(p["T14_h"], 2)
    else:
        for x in ("Rp", "b", "P", "Tdur"):
            M[f"{x}{T}"] = miss(f"fit {sysname}")
if "HIP 31126" in fits:
    pa = fits["HIP 31126"]["new"]; pb = json.load(open(R("results", "fit_H31126_B.json")))["new"]
    M["radratio"] = f"{pb['Rp_Re'][1] / pa['Rp_Re'][1]:.2f}"
    SA = ST["HIP 31126"]["L"][1] / pa["a_au"][1] ** 2; SB = ST["HIP 31126 B"]["L"][1] / pb["a_au"][1] ** 2
    M["insratio"] = f"{SA / SB:.1f}"
if "TOI-5997" in fits:
    r = fits["TOI-5997"]["new"]["P"][1] / fits["TOI-5997"]["b5"]["P"][1]
    M["perratioTf"] = f"{r:.3f}"; M["perresoffTf"] = f"{100 * abs(r / 2.5 - 1):.1f}"
    # tenth transit: the epoch masked in the vetting because it overlaps a transit of b
    pc, pbb = fits["TOI-5997"]["new"], fits["TOI-5997"]["b5"]
    secs = V["TOI-5997:sectors"]; vt = [x[1] for x in V["TOI-5997:new"]["per_transit"]]
    for n in range(-80, 80):
        tc = pc["T0"][1] + n * pc["P"][1]
        if any(s_["tstart"] < tc < s_["tstop"] for s_ in secs) and min(abs(tc - x) for x in vt) > 0.5:
            nb = np.round((tc - pbb["T0"][1]) / pbb["P"][1]); dtb = (pbb["T0"][1] + nb * pbb["P"][1] - tc) * 24
            if abs(dtb) < 3:
                M["tenthTf"] = f"{tc:.2f}"; M["tenthdtTf"] = f"{abs(dtb):.1f}"
# free-density fits
for sysname, T in TAG.items():
    tag = SYSTEMS[sysname]["key"] + "_free" + ("_A" if sysname == "HIP 31126" else "")
    FR = loadj(R("results", f"fit_{tag}.json"))
    if FR:
        lo_, me_, hi_ = FR["new"]["rho_implied"]; M[f"rhofree{T}"] = f"{me_:.1f}^{{+{hi_ - me_:.1f}}}_{{-{me_ - lo_:.1f}}}"
        M[f"ntaufree{T}"] = f"{FR['n_tau']:.0f}"
    else:
        M[f"rhofree{T}"] = miss(f"free {sysname}")
    if FR:
        ri = FR["new"]["rho_implied"]
        for nm, key in (("A", "HIP 31126"), ("B", "HIP 31126 B")) if sysname == "HIP 31126" else (("", sysname),):
            rs = ST[key]["rho"][1]; srs = 0.5 * (ST[key]["rho"][2] - ST[key]["rho"][0])
            e = (ri[2] - ri[1]) if rs > ri[1] else (ri[1] - ri[0])
            M[f"rhofreesig{T}{nm}"] = f"{abs(ri[1] - rs) / np.hypot(e, srs):.1f}"

# ---------------------------------------------------------------- FFI check of TOI-678 (Sectors 6 and 13)
FFI = loadj(R("results", "ffi_toi678.json"))
if FFI:
    for key, nm in (("13spoc:pipeline", "ffiS"), ("13spoc:SAP", "ffiSsap"), ("13qlp:pipeline", "ffiQ"), ("13qlp:SAP", "ffiQsap"),
                    ("6spoc:pipeline", "ffisix"), ("6qlp:pipeline", "ffisixQ")):
        r = FFI[key]; M[nm] = f"{r['scale']:.2f}^{{+{r['hi']:.2f}}}_{{-{r['lo']:.2f}}}"
        M[nm + "z"] = f"{np.sqrt(max(r['dchi2_zero'], 0)):.1f}"; M[nm + "one"] = f"{np.sqrt(max(r['dchi2_one'], 0)):.1f}"
else:
    miss("ffi")

# ---------------------------------------------------------------- TRICERATOPS
EBS = {"EB", "EBx2P", "PEB", "PEBx2P", "SEB", "SEBx2P", "DEB", "DEBx2P", "BEB", "BEBx2P", "NEB", "NEBx2P"}
def fpp_runs(key, which, cc, excl):
    """runs with 10^6 draws; the TESS-only variant may use fewer (file name ends in _N<draws>)."""
    fs = sorted(glob.glob(R("results", "fpp", f"{key}_{which}_{cc}_{excl}_s*.json")))
    return [json.load(open(f)) for f in fs if "_N" not in os.path.basename(f) or (cc == "none" and excl == "none")]
FPPSUM = {}
for sysname, T in TAG.items():
    key = SYSTEMS[sysname]["key"]
    for cc, excl, suf in (("cc", "none", ""), ("cc", "pix", "pix"), ("none", "none", "tess")):
        runs = fpp_runs(key, "new", cc, excl)
        if not runs and suf == "pix" and M.get(f"nnbx{T}") == "0":
            M[f"FPP{suf}{T}"] = r"\text{as adopted}"; M[f"NFPP{suf}{T}"] = r"\text{as adopted}"; M[f"nrun{suf}{T}"] = "0"
            continue
        if not runs:
            M[f"FPP{suf}{T}"] = miss(f"fpp {sysname} {cc} {excl}"); M[f"NFPP{suf}{T}"] = miss(f"fpp {sysname} {cc} {excl}")
            M[f"nrun{suf}{T}"] = "0"
            if suf == "":
                for k in ("pEB", "pEBmax", "pplanet", "pSTP", "pNTP", "psys", "pBTP"):
                    M[f"{k}{T}"] = miss(f"fpp {sysname} {cc} {excl}")
            continue
        M[f"FPP{suf}{T}"] = meansd([r["FPP"] for r in runs]); M[f"NFPP{suf}{T}"] = meansd([r["NFPP"] for r in runs])
        M[f"nrun{suf}{T}"] = str(len(runs))
        sc = {}
        for r in runs:
            for k, v in r["scenarios"].items():
                sc.setdefault(k, []).append(v)
        scm = {k: np.sum(v) / len(runs) for k, v in sc.items()}
        FPPSUM[f"{sysname}:{cc}:{excl}"] = dict(runs=len(runs), FPP=[r["FPP"] for r in runs], NFPP=[r["NFPP"] for r in runs], scen=scm)
        per = lambda ks: [sum(r["scenarios"].get(k, 0.0) for k in ks) for r in runs]
        if suf == "pix":
            M[f"pEBpix{T}"] = meansd(per(EBS)); M[f"pEBpixmax{T}"] = sci(max(per(EBS)), 1)
        if suf == "":
            M[f"pEB{T}"] = meansd(per(EBS)); M[f"pEBmax{T}"] = sci(max(per(EBS)), 1)
            M[f"pplanet{T}"] = meansd(per(["TP", "PTP", "DTP"])); M[f"pSTP{T}"] = meansd(per(["STP"]))
            M[f"pNTP{T}"] = meansd(per(["NTP"])); M[f"pBTP{T}"] = meansd(per(["BTP"]))
            sysp = per(["TP", "PTP", "DTP", "STP"] + (["NTP"] if sysname == "HIP 31126" else []))
            M[f"psys{T}"] = meansd(sysp)
            if sysname == "TOI-678":
                q = [sum(x[2] for x in r.get("table", []) if x[0] == "294395934") for r in runs]
                M["pnbqTs"] = meansd(q)
            if sysname == "HIP 31126":
                # NTP on B specifically, and neighbour-EB share
                ntpB = [sum(x[2] for x in r.get("table", []) if x[0] == "293689267" and x[1] == "NTP") for r in runs]
                M["pNTPBHp"] = meansd(ntpB) if any(r.get("table") for r in runs) else M["pNTPHp"]
for which, T in (("b5", "Tfb"), ("b11", "Tsb")):
    key = "T5997" if which == "b5" else "T678"; runs = fpp_runs(key, which, "cc", "none")
    M[f"FPP{T}"] = meansd([r["FPP"] for r in runs]) if runs else miss(f"fpp {which}")
    M[f"NFPP{T}"] = meansd([r["NFPP"] for r in runs]) if runs else miss(f"fpp {which}")
    M[f"nrun{T}"] = str(len(runs))
for k in ("pNTPBHp", "pEBpixHp", "pEBpixmaxHp"):
    if k not in M:
        M[k] = miss("fpp " + k)
save_numbers("fpp_summary", FPPSUM)

# ---------------------------------------------------------------- upcoming transits (2026 Oct 15 to 2027 Dec 31)
rows = []
t_start, t_end = Time("2026-10-15").jd - 2457000, Time("2027-12-31").jd - 2457000
for sysname, T in TAG.items():
    if sysname not in fits:
        continue
    F = fits[sysname]; p = F["new"]; P, T0 = p["P"][1], p["T0"][1]; beta = V[f"{sysname}:new"]["beta"]
    sP = 0.5 * (p["P"][2] - p["P"][0]); sT = 0.5 * (p["T0"][2] - p["T0"][0]); cov = F["new"].get("cov_P_T0", 0.0)
    ts = (Time("2027-04-01").jd - 2457000) if sysname == "TOI-5997" else t_start
    n0 = int(np.ceil((ts - T0) / P)); n1 = int(np.floor((t_end - T0) / P))
    nshow = {"TOI-678": 3, "HIP 31126": 5}.get(sysname, 4)
    for n in range(n0, n1 + 1)[:nshow]:
        tc = T0 + n * P; sig = beta * np.sqrt(sT ** 2 + (n * sP) ** 2 + 2 * n * cov) * 1440
        utc = (Time(tc + 2457000, format="jd", scale="tdb").utc + 30 * u.s).iso[:16]  # nearest minute
        rows.append((sysname, utc, tc + 2457000, sig, p["T14_h"][1]))
    M[f"sigtc{T}"] = f"{beta * np.sqrt(sT ** 2 + (n0 * sP) ** 2 + 2 * n0 * cov) * 1440:.0f}"
os.makedirs(R("paper"), exist_ok=True)
with open(R("paper", "tab_ephem.tex"), "w") as f:
    last = None
    for sysname, iso, jd, sig, dur in rows:
        name = sysname if sysname != last else ""; last = sysname
        f.write(f"{name} & {iso} & {jd:.4f} & {sig:.0f} & {dur:.1f} \\\\\n")
if "TOI-678" in fits:
    nxt = [r for r in rows if r[0] == "TOI-678"]
    M["nextTsa"] = Time(nxt[0][2], format="jd", scale="tdb").utc.strftime("%Y %B %-d")
    M["nextTsb"] = Time(nxt[1][2], format="jd", scale="tdb").utc.strftime("%Y %B %-d")

# ---------------------------------------------------------------- context: comparable known transiting planets
df = pd.read_csv(R("data", "nasa_pscomppars_2026-10-08.csv"))
m = (df.tran_flag == 1) & (df.pl_orbper > 100) & (df.pl_rade > 2) & (df.pl_rade < 4) & (df.st_teff > 5200) & (df.st_teff < 6200) & (df.sy_vmag < 12)
M["ncompTs"] = str(int(m.sum())); M["ntransarchive"] = f"{int((df.tran_flag == 1).sum()):,}".replace(",", "\\,")

# ---------------------------------------------------------------- write
with open(R("paper", "numbers.tex"), "w") as f:
    f.write("% generated by scripts/s08_make_numbers.py -- do not edit\n")
    for k, v in sorted(M.items()):
        assert k.isalpha(), k
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(len(M), "macros;", len(set(missing)), "missing groups:", sorted(set(missing)))
if FINAL and missing:
    raise SystemExit("missing inputs: " + ", ".join(sorted(set(missing))))
