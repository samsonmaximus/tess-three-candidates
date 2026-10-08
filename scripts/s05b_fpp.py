"""Step 5b: TRICERATOPS 1.1.0 false-positive probabilities, one recipe for all three systems.

Usage: python s05b_fpp.py SYSTEM SIGNAL CC EXCL SEED      e.g.  python s05b_fpp.py TOI-678 new cc pix 3
  SIGNAL: new | b5 (TOI-5997 b control) | b11 (TOI-678.01 control)
  CC:     cc (all of the system's high-resolution contrast curves, see CC below) | none
  EXCL:   pix (stars excluded at >= 3 sigma by both pixel-level metrics of s05a get zero depth) | none
Recipe (as Collier et al. 2026 for TOI-5997 b, with offline inputs):
  * folded light curve from the transit-masked detrending, other fitted planets divided out with their best-fit
    model and other signals masked, trimmed to +-2 T14 and averaged into 200 bins;
  * TIC 8.2 field within 0.07 deg, positions moved by proper motion to the mean epoch of the transit sectors,
    our stellar parameters for the target (and for HIP 31126 B, with its Gaia-based T magnitude);
  * SPOC apertures and WCS of the sectors with transits (from the target pixel files);
  * Gaia DR3 field population (0.1 deg^2, G < 21) for the blended-background scenarios;
  * HIP 31126: PDC used a blended T magnitude for B, so the light curve is converted back to the aperture
    (depth x CROWDSAP) and passed with dilution_corrected=False; TOI-678 and TOI-5997 use PDCSAP as is.
"""
from common import *
import batman
from astropy.wcs import WCS
from astropy.io.fits import Header
from astropy.table import Table
from astropy.coordinates import SkyCoord
import triceratops.triceratops as trm, triceratops.funcs as tf
from tcemine.detrend import in_transit
from tcemine.pixel import btjd_to_year, propagate

sysname, which, ccmode, excl, seed = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
S = SYSTEMS[sysname]; key = S["key"]; ST = load_numbers("stellar")
FIT = {"HIP 31126": ["new"], "TOI-678": ["new", "b11"], "TOI-5997": ["new", "b5"]}[sysname]
MASK = {"HIP 31126": ["f14"], "TOI-678": ["t6"], "TOI-5997": []}[sysname]
# high-resolution imaging: (file, TRICERATOPS filter). SOAR I band and the 832 nm speckle channels are treated as
# TESS band; the 562 nm channel as "Vis"; PHARO Br-gamma as K. TRICERATOPS adopts the tightest constraint.
CC = {"HIP 31126": [(R("data", "cc_H31126_soar.dat"), "TESS")],
      "TOI-678": [(R("data", "cc_T678_zorro832.dat"), "TESS"), (R("data", "cc_T678_zorro562.dat"), "Vis")],
      "TOI-5997": [(R("data", "cc_T5997_alopeke832.dat"), "TESS"), (R("data", "cc_T5997_alopeke562.dat"), "Vis"),
                   (R("data", "cc_T5997_pharoBrg.dat"), "K")]}[sysname]
NSIM = int(os.environ.get("NSIM", 1000000))
PIXNAME = {"new": "new", "b5": "b5", "b11": "b11"}[which]
os.makedirs(R("results", "fpp"), exist_ok=True)

# ---------- light curve ----------
F = json.load(open(R("results", f"fit_{key}.json")))
d = np.load(R("results", f"lc_{key}.npz")); t, f = d["t"], d["f"]
def model(name, tt):
    p = F[name]; P, T0, rp, a, b = p["P"][1], p["T0"][1], p["rp_rs"][1], p["a_rs"][1], p["b"][1]
    q1, q2 = F["q1"][1], F["q2"][1]; u1, u2 = 2 * np.sqrt(q1) * q2, np.sqrt(q1) * (1 - 2 * q2)
    bp = batman.TransitParams(); bp.t0 = T0; bp.per = P; bp.rp = rp; bp.a = a; bp.inc = np.degrees(np.arccos(b / a))
    bp.ecc = 0; bp.w = 90; bp.u = [u1, u2]; bp.limb_dark = "quadratic"
    m = batman.TransitModel(bp, tt, supersample_factor=3, exp_time=120 / 86400).light_curve(bp)
    return 1.0 - F["dilution"] * (1.0 - m)
for k in FIT:
    if k != which:
        f = f / model(k, t)
for k in MASK:
    P_, T0_, D_ = get_ephem(sysname, k); m_ = ~in_transit(t, P_, T0_, D_, 2.0); t, f = t[m_], f[m_]
P, Tc, T14 = F[which]["P"][1], F[which]["T0"][1], F[which]["T14_h"][1] / 24
vet = load_numbers("vetting"); crowd = np.mean([m["crowdsap"] for m in vet[f"{sysname}:sectors"]])
dil_corr = True
if sysname == "HIP 31126":
    f = 1.0 - crowd * (1.0 - f); dil_corr = False
x = (((t - Tc) / P + 0.5) % 1 - 0.5) * P; w = np.abs(x) < 2 * T14; x, y = x[w], f[w]
r = y - np.median(y); s = 1.4826 * np.median(np.abs(r)); keep = np.abs(r) < 5 * s + 3 * (1 - np.min(y)); x, y = x[keep], y[keep]
edges = np.linspace(-2 * T14, 2 * T14, 201); idx = np.digitize(x, edges)
xb = np.array([x[idx == i].mean() for i in range(1, 201) if (idx == i).sum() > 3])
yb = np.array([y[idx == i].mean() for i in range(1, 201) if (idx == i).sum() > 3])
err = float(np.std(yb[np.abs(xb) > 0.6 * T14])); depth = float(1 - np.median(yb[np.abs(xb) < 0.25 * T14]))

# ---------- stars ----------
TIC = json.load(open(R("data", "tic82_cones.json")))
td = tic_field(TIC[{"HIP 31126": "H31126", "TOI-678": "T678", "TOI-5997": "T5997"}[sysname]])
td = td[pd.to_numeric(td.Tmag, errors="coerce").notna()].copy()
num = ["ra", "dec", "pmRA", "pmDEC", "Tmag", "Vmag", "GAIAmag", "gaiabp", "gaiarp", "Jmag", "Hmag", "Kmag", "ebv", "mass", "rad", "Teff", "logg", "plx"]
for c in num:
    td[c] = pd.to_numeric(td[c], errors="coerce")
DIF = json.load(gzip.open(R("data", "tpf_diffimages.json.gz")))
PX = json.load(open(R("results", f"pixel_{key}.json")))["signals"][PIXNAME]
prods = [p for p in DIF.values() if p["tic"] == S["tic"] and p["sector"] in PX["sectors"]]
ep = np.mean([btjd_to_year(0.5 * (p["tmin"] + p["tmax"])) for p in prods])
td["ra"], td["dec"] = propagate(td.ra.values, td.dec.values, td.pmRA.values, td.pmDEC.values, 2000.0, ep)
t0 = td[td.ID == str(S["tic"])].iloc[0]
td["sep"] = np.hypot((td.ra - t0.ra) * np.cos(np.radians(t0.dec)), td.dec - t0.dec) * 3600
td = td.sort_values("sep").reset_index(drop=True); assert td.ID[0] == str(S["tic"])
stars = pd.DataFrame({"ID": td.ID.astype(int), "Tmag": td.Tmag, "Vmag": td.Vmag, "GAIAmag": td.GAIAmag, "gaiabp": td.gaiabp,
                      "gaiarp": td.gaiarp, "Jmag": td.Jmag, "Hmag": td.Hmag, "Kmag": td.Kmag, "ebv": td.ebv, "ra": td.ra, "dec": td.dec,
                      "mass": td.mass, "rad": td.rad, "Teff": td.Teff, "logg": td.logg, "plx": td.plx,
                      "disposition": td.disposition, "duplicate_id": td.duplicate_id})
star = ST[sysname]
stars.loc[0, "rad"] = star["R"][1]; stars.loc[0, "mass"] = star["M"][1]; stars.loc[0, "Teff"] = star["Teff"][1]
if sysname == "HIP 31126":
    stars.loc[0, "Tmag"] = star["Tmag_gaia"]
    j = int(np.where(stars.ID == 293689267)[0][0]); B = ST["HIP 31126 B"]
    stars.loc[j, "Tmag"] = B["Tmag_gaia"]; stars.loc[j, "rad"] = B["R"][1]; stars.loc[j, "mass"] = B["M"][1]; stars.loc[j, "Teff"] = B["Teff"][1]
    stars.loc[j, "logg"] = B["logg"][1]
# ---------- Gaia background population ----------
G = json.load(open(R("data", "gaia_dr3.json")))
g = pd.read_csv(io.StringIO(G["bg_" + key]))
gt = Table({"phot_g_mean_mag": g.phot_g_mean_mag.values, "phot_bp_mean_mag": g.phot_bp_mean_mag.values,
            "phot_rp_mean_mag": g.phot_rp_mean_mag.values, "parallax": g.parallax.values, "parallax_over_error": g.parallax_over_error.values})
class FakeJob:
    def get_results(self): return gt
import astroquery.gaia
astroquery.gaia.Gaia.launch_job = lambda *a, **k: FakeJob(); astroquery.gaia.Gaia.launch_job_async = lambda *a, **k: FakeJob()
os.chdir(R("results", "fpp"))
bg = tf.query_gaia_background(float(stars.ra[0]), float(stars.dec[0]), S["tic"], verbose=0)
# ---------- target object ----------
T = object.__new__(trm.target)
T.ID = S["tic"]; T.mission = "TESS"; T.search_radius = 10; T.N_pix = 22
T.trilegal_fname = bg; T.trilegal_url = None; T.background_population_source = "gaia"
pix, aps, secs = [], [], []
for p in prods:
    H = Header()
    for k, v in p["wcs"].items():
        if k.endswith("P") and k[:5] in ("CRVAL", "CRPIX", "CDELT", "CTYPE", "CUNIT"):
            continue
        try:
            H[k] = v if k in ("CTYPE1", "CTYPE2", "RADESYS", "CUNIT1", "CUNIT2") else float(v)
        except Exception:
            pass
    wcs = WCS(H, naxis=2); c0, r0 = float(p["col0"]), float(p["row0"])
    xy = np.array(wcs.all_world2pix(stars.ra.values, stars.dec.values, 0)).T
    pix.append(np.column_stack([c0 + xy[:, 0], r0 + xy[:, 1]]))
    img = np.array(p["aperture"]).reshape(p["nrow"], p["ncol"]); jj, ii = np.where((img & 2) > 0)
    aps.append(np.column_stack([c0 + ii, r0 + jj])); secs.append(p["sector"])
T.sectors = np.array(secs); T.pix_coords = pix; T.TESS_images = [np.zeros((22, 22))] * len(secs); T.col0s = [0] * len(secs); T.row0s = [0] * len(secs)
c = SkyCoord(stars.ra.values, stars.dec.values, unit="deg")
stars["sep (arcsec)"] = c[0].separation(c).arcsec.round(3); stars["PA (E of N)"] = c[0].position_angle(c).deg.round(3)
T.stars = stars; T.estimate_stellar_params(verbose=0)
T.calc_depths(tdepth=depth, all_ap_pixels=aps, dilution_corrected=dil_corr)
nb = f"neighbours_{key}_{which}.csv"
if not os.path.exists(nb):
    T.stars[["ID", "Tmag", "sep (arcsec)", "PA (E of N)", "fluxratio", "tdepth"]].to_csv(nb, index=False)
excluded = []
if excl == "pix":
    for row in PX["stars"]:
        if row.get("excl3", row["sigma"] >= 3.0) and int(row["TIC"]) != S["tic"]:
            excluded.append(int(row["TIC"]))
    # stars not tested by the localisation (needed depth > 100%, or beyond 150") cannot produce the signal anyway
    T.stars.loc[T.stars.ID.isin(excluded), "tdepth"] = 0.0
np.random.seed(seed)
T.calc_probs(time=xb, flux_0=yb, flux_err_0=err, P_orb=float(P), contrast_curve_file=([c[0] for c in CC] if ccmode == "cc" else None),
             filt=([c[1] for c in CC] if ccmode == "cc" else "TESS"), N=NSIM, verbose=0, dilution_corrected=dil_corr, exptime=0.00139)
scen = T.probs.groupby("scenario").prob.sum().sort_values(ascending=False)
on_star = T.probs.groupby("ID").prob.sum().sort_values(ascending=False)
out = dict(system=sysname, signal=which, cc=ccmode, excl=excl, seed=seed, N=NSIM, FPP=float(T.FPP), NFPP=float(T.NFPP),
           depth_used=depth, flux_err=err, n_bins=int(len(xb)), P=float(P), Tc=float(Tc), T14_h=T14 * 24, dilution_corrected=dil_corr,
           excluded_ids=excluded, n_stars_nonzero_depth=int((T.stars.tdepth > 0).sum()),
           scenarios={k: float(v) for k, v in scen.items() if v > 1e-6}, by_star={str(k): float(v) for k, v in on_star.head(8).items()},
           table=[[str(i), str(sc), float(pr)] for i, sc, pr in zip(T.probs.ID, T.probs.scenario, T.probs.prob) if pr > 1e-7])
json.dump(out, open(f"{key}_{which}_{ccmode}_{excl}_s{seed}" + ("" if NSIM == 1000000 else f"_N{NSIM}") + ".json", "w"))
print(json.dumps(out)[:1500])
