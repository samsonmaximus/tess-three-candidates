"""Step 2: stellar parameters.

HIP 31126 A and B (M dwarfs 3.3" apart): 2MASS blends the pair (B has Ks quality flag F), so M_Ks is taken
from the resolved Gaia DR3 M_G through the Mamajek dwarf sequence (version 2022.04.16; Pecaut & Mamajek 2013),
with 0.10 mag added in quadrature, and R and M follow from the M_Ks relations of Mann et al. (2015, 2019)
including their scatter. Teff from the same sequence. The 2MASS-based values for A are kept as a cross-check.
TOI-678 and TOI-5997 (G and K dwarfs): L from Gaia DR3 G with the BC_G(Teff) of Andrae et al. (2018), Teff,
log g and [Fe/H] from the spectroscopic analyses on ExoFOP (CHIRON for TOI-678, TRES/SPC for TOI-5997) with a
2% Teff floor (Tayar et al. 2022), R from Stefan-Boltzmann, M from Torres et al. (2010) with 6.4% scatter.
Extinction: A_G = 2.74 E(B-V) with E(B-V) from TIC 8.2 (zero for the two nearby systems).
"""
from common import *
rng = np.random.default_rng(42); N = 200000
q = lambda z: [float(np.percentile(z, 16)), float(np.percentile(z, 50)), float(np.percentile(z, 84))]
G = json.load(open(R("data", "gaia_dr3.json")))
hosts = pd.read_csv(io.StringIO(G["hosts"]), dtype={"source_id": str}).set_index("source_id")
tm = pd.read_csv(io.StringIO(G["tmass"]), dtype={"source_id": str}).set_index("source_id")
# Mamajek sequence rows K9V..M3V: Teff, R, Bp-Rp, M_G, M_Ks, M
EEM = np.array([[3930, 0.608, 1.79, 8.03, 5.01, 0.59], [3850, 0.588, 1.84, 8.16, 5.15, 0.57], [3770, 0.544, 1.97, 8.44, 5.36, 0.54],
                [3660, 0.501, 2.09, 8.82, 5.64, 0.50], [3620, 0.482, 2.13, 8.98, 5.75, 0.47], [3560, 0.446, 2.23, 9.29, 5.98, 0.44],
                [3470, 0.421, 2.39, 9.67, 6.18, 0.40], [3430, 0.361, 2.50, 10.05, 6.55, 0.37]])
mann15_R = lambda MK: 1.9515 - 0.3520 * MK + 0.01680 * MK ** 2
def mann19_M(MK):
    a = [-0.642, -0.208, -8.43e-4, 7.87e-3, 1.42e-4, -2.13e-4]; x = MK - 7.5
    return 10 ** sum(a[i] * x ** i for i in range(6))
def mdwarf_from_G(sid):
    h = hosts.loc[sid]
    plx = rng.normal(h.parallax, h.parallax_error, N); d = 1000 / plx
    MG = rng.normal(h.phot_g_mean_mag, 0.003, N) - 5 * np.log10(d / 10)
    MK = np.interp(MG, EEM[:, 3], EEM[:, 4]) + rng.normal(0, 0.10, N)
    Teff = np.interp(MG, EEM[:, 3], EEM[:, 0]) + rng.normal(0, 100, N)
    Rs = mann15_R(MK) * rng.normal(1, 0.0289, N); Ms = mann19_M(MK) * rng.normal(1, 0.020, N)
    rho = 1.40894 * Ms / Rs ** 3; L = Rs ** 2 * (Teff / 5772) ** 4
    return dict(G=float(h.phot_g_mean_mag), bp_rp=float(h.bp_rp), plx=float(h.parallax), dist_pc=q(d), MG=q(MG), MKs=q(MK),
                Teff=q(Teff), R=q(Rs), M=q(Ms), rho=q(rho), L=q(L), logg=q(np.log10(Ms / Rs ** 2) + 4.438))
def fgk(sid, teff, eteff, logg, elogg, feh, efeh, ebv=0.0):
    h = hosts.loc[sid]
    plx = rng.normal(h.parallax, h.parallax_error, N); d = 1000 / plx
    T = rng.normal(teff, np.hypot(eteff, 0.02 * teff), N)
    dT = T - 5772; BC = 6.000e-02 + 6.731e-05 * dT - 6.647e-08 * dT ** 2 + 2.859e-11 * dT ** 3 - 7.197e-15 * dT ** 4
    MG = rng.normal(h.phot_g_mean_mag, 0.003, N) - 5 * np.log10(d / 10) - 2.74 * ebv
    L = 10 ** (-0.4 * (MG + BC - 4.74)); Rs = np.sqrt(L) / (T / 5772) ** 2
    lg = rng.normal(logg, elogg, N); fe = rng.normal(feh, efeh, N); X = np.log10(T) - 4.1
    Ms = 10 ** (1.5689 + 1.3787 * X + 0.4243 * X ** 2 + 1.139 * X ** 3 - 0.1425 * lg ** 2 + 0.01969 * lg ** 3 + 0.1010 * fe) * rng.normal(1, 0.064, N)
    rho = 1.40894 * Ms / Rs ** 3
    return dict(G=float(h.phot_g_mean_mag), bp_rp=float(h.bp_rp), plx=float(h.parallax), dist_pc=q(d), MG=q(MG), Teff=q(T), R=q(Rs), M=q(Ms),
                rho=q(rho), L=q(L), logg_spec=[logg - elogg, logg, logg + elogg], feh=[feh - efeh, feh, feh + efeh],
                logg=q(np.log10(Ms / Rs ** 2) + 4.438), ebv=ebv)
out = {}
A, B = "5568872394338114560", "5568872398635858816"
out["HIP 31126"] = mdwarf_from_G(A); out["HIP 31126 B"] = mdwarf_from_G(B)
# cross-check for A: 2MASS Ks (quality AAA, but blended with B at 3.3")
h = hosts.loc[A]; Ks = tm.loc[A].ks_m; MKa = Ks - 5 * np.log10(1000 / h.parallax / 10)
out["HIP 31126"]["xcheck_2MASS"] = dict(Ks=float(Ks), MKs=float(MKa), R=float(mann15_R(MKa)), M=float(mann19_M(MKa)))
out["HIP 31126"]["method"] = "M_Ks from Gaia M_G via Mamajek sequence; R Mann+2015; M Mann+2019; Teff Mamajek sequence"
out["TOI-678"] = fgk("5490097505810081024", 5536, 100, 4.343, 0.10, 0.105, 0.08, ebv=0.0455)
out["TOI-678"]["method"] = "L Gaia G + BC_G (Andrae+2018); Teff/logg/[M/H] CHIRON (ExoFOP); R Stefan-Boltzmann; M Torres+2010"
out["TOI-5997"] = fgk("4602034956332491264", 4717, 50, 4.621, 0.10, -0.339, 0.08, ebv=0.0)
out["TOI-5997"]["method"] = "L Gaia G + BC_G (Andrae+2018); Teff/logg/[Fe/H] TRES/SPC (ExoFOP); R Stefan-Boltzmann; M Torres+2010"
# T magnitudes from Gaia (Stassun et al. 2019, eq. 1) for the two components of HIP 31126
for k, sid in (("HIP 31126", A), ("HIP 31126 B", B)):
    h = hosts.loc[sid]; c = h.bp_rp
    out[k]["Tmag_gaia"] = float(h.phot_g_mean_mag - 0.00522555 * c ** 3 + 0.0891337 * c ** 2 - 0.633923 * c + 0.0324473)
# Gaia quality and RV indicators for the table
for k, sid in (("HIP 31126", A), ("HIP 31126 B", B), ("TOI-678", "5490097505810081024"), ("TOI-5997", "4602034956332491264")):
    h = hosts.loc[sid]
    out[k]["gaia"] = {c: (None if pd.isna(h[c]) else (h[c] if isinstance(h[c], str) else float(h[c]))) for c in
                      ("ra", "dec", "pmra", "pmdec", "parallax", "parallax_error", "ruwe", "non_single_star", "ipd_frac_multi_peak",
                       "radial_velocity", "radial_velocity_error", "rv_nb_transits", "rv_amplitude_robust", "rv_chisq_pvalue",
                       "phot_variable_flag", "phot_g_mean_mag", "bp_rp")}
    out[k]["gaia"]["source_id"] = sid
    if sid in tm.index:
        out[k]["2mass"] = {c: (None if pd.isna(tm.loc[sid][c]) else (tm.loc[sid][c] if isinstance(tm.loc[sid][c], str) else float(tm.loc[sid][c])))
                           for c in ("original_ext_source_id", "j_m", "j_msigcom", "h_m", "h_msigcom", "ks_m", "ks_msigcom", "ph_qual")}
for k, v in out.items():
    print(k, {a: (np.round(b, 4).tolist() if isinstance(b, list) else b) for a, b in v.items() if a in ("R", "M", "rho", "Teff", "L", "dist_pc", "MKs", "Tmag_gaia", "xcheck_2MASS")})
save_numbers("stellar", out)
