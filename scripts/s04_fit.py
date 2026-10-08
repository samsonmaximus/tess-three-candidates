"""Step 4: transit fits (batman + emcee). Usage: python s04_fit.py SYSTEM [MODE]
MODE: rho (stellar-density prior; default) | free (candidate only, a/R* free, other signals masked: density check)
For HIP 31126 an extra argument HOST (A or B) selects the host: its density prior and the dilution factor
g = f_host / CROWDSAP (s04a). Other signals in the light curve are masked at 2 x T14.
Fitted planets: HIP 31126 -> the candidate; TOI-678 -> candidate + TOI-678.01; TOI-5997 -> candidate + TOI-5997 b.
"""
import pickle
from common import *
from tcemine.transit import joint_fit
from tcemine.detrend import in_transit
from tcemine.vetting import seasons
sysname = sys.argv[1]; mode = sys.argv[2] if len(sys.argv) > 2 else "rho"; host = sys.argv[3] if len(sys.argv) > 3 else "A"
S = SYSTEMS[sysname]; ST = load_numbers("stellar")
FIT = {"HIP 31126": ["new"], "TOI-678": ["new", "b11"], "TOI-5997": ["new", "b5"]}[sysname]
MASK = {"HIP 31126": ["f14"], "TOI-678": ["t6"], "TOI-5997": []}[sysname]
if mode == "free":       # density check: the candidate alone (a/R* free), every other signal masked
    MASK = MASK + [k for k in FIT if k != "new"]; FIT = ["new"]
def u2q(u1, u2):
    return (u1 + u2) ** 2, u1 / (2 * (u1 + u2))
LD = {"HIP 31126:A": (0.30, 0.32), "HIP 31126:B": (0.25, 0.35), "TOI-678:A": (0.40, 0.22), "TOI-5997:A": (0.45, 0.19)}[f"{sysname}:{host}"]
q1c, q2c = u2q(*LD); ld = ((q1c, 0.15), (q2c, 0.15))
star = ST["HIP 31126 B"] if (sysname == "HIP 31126" and host == "B") else ST[sysname]
rho = star["rho"]; rho_prior = (rho[1], (rho[2] - rho[0]) / 2)
g = 1.0
if sysname == "HIP 31126":
    g = load_numbers("dilution")["HIP 31126:g"][host]
d = np.load(R("results", f"lc_{S['key']}.npz")); t, f, s = d["t"], d["f"], d["s"]
eph = {k: get_ephem(sysname, k) for k in FIT + MASK}
keep = np.zeros_like(t, bool)
for k in FIT:
    keep |= in_transit(t, eph[k][0], eph[k][1], eph[k][2], 6.0)
for k in MASK:
    keep &= ~in_transit(t, eph[k][0], eph[k][1], eph[k][2], 2.0)
for k in FIT:
    for e in S["exclude_epochs"].get(k, []):
        n = np.round((t - eph[k][1]) / eph[k][0]); keep &= ~((n == e) & in_transit(t, eph[k][0], eph[k][1], eph[k][2], 6.0))
grp = seasons(s)[keep]; t, f = t[keep], f[keep]
vet = load_numbers("vetting")
guess = []
F0 = json.load(open(R("results", f"fit_{S['key']}.json"))) if (mode == "free" and os.path.exists(R("results", f"fit_{S['key']}.json"))) else {}
for k in FIT:
    P_, T0_, D_ = eph[k]; dep = vet[f"{sysname}:{k}"]["depth_ppm"] * 1e-6 / g
    a0, b0 = (F0[k]["a_rs"][1], F0[k]["b"][1]) if k in F0 else (20.0, 0.4)     # free mode starts at the density-prior solution
    guess.append(dict(P=P_, T0=T0_, rp=np.sqrt(max(dep, 1e-5)), b=b0, a=a0))
res = joint_fit(t, f, guess, rho_prior=rho_prior if mode == "rho" else None, ld_prior=ld, groups=grp, free_rho=(mode == "free"),
                nwalk=48, nstep=int(os.environ.get("NSTEP", 15000)), burn=int(os.environ.get("NBURN", 5000)), seed=1, dilution=g)
ch = res["chain"]; unpack = res["unpack"]
q = lambda x: np.percentile(x, [16, 50, 84]).tolist()
out = dict(mode=mode, host=host, dilution=g, npts=int(t.size), acc=res["acc"], tau_max=float(np.nanmax(res["tau"])),
           nstep=res["nstep"], burn=res["burn"], rho_prior=rho_prior, ld_prior=ld, ld_u=LD)
samples = [unpack(x) for x in ch[:: max(1, len(ch) // 6000)]]
rng = np.random.default_rng(2)
Rs = rng.normal(star["R"][1], (star["R"][2] - star["R"][0]) / 2, len(samples))
Ms = rng.normal(star["M"][1], (star["M"][2] - star["M"][0]) / 2, len(samples))
for i, nm in enumerate(FIT):
    P = np.array([z[0][i][0] for z in samples]); T0 = np.array([z[0][i][1] for z in samples])
    rp = np.array([z[0][i][2] for z in samples]); a = np.array([z[0][i][3] for z in samples]); b = np.array([z[0][i][4] for z in samples])
    inc = np.degrees(np.arccos(np.clip(b / a, 0, 1)))
    T14 = P / np.pi * np.arcsin(np.sqrt(np.clip((1 + rp) ** 2 - b ** 2, 0, None)) / a / np.sin(np.radians(inc))) * 24
    rhop = 3 * np.pi * a ** 3 / (6.674e-8 * (P * 86400) ** 2)
    au = (Ms * (P / 365.25) ** 2) ** (1 / 3)
    out[nm] = dict(P=q(P), T0=q(T0), rp_rs=q(rp), a_rs=q(a), b=q(b), inc=q(inc), T14_h=q(T14), depth_ppm=q(rp ** 2 * 1e6),
                   rho_implied=q(rhop), Rp_Re=q(rp * Rs * 109.076), a_au=q(au), cov_P_T0=float(np.cov(P, T0)[0, 1]))
out["q1"] = q(np.array([z[2] for z in samples])); out["q2"] = q(np.array([z[3] for z in samples]))
out["noise_factor_per_season"] = [q(np.exp([z[4][j] for z in samples])) for j in range(len(res["groups"]))]
out["seasons"] = res["groups"]; out["tau"] = [float(x) for x in res["tau"]]
out["n_tau"] = float((res["nstep"] - res["burn"]) / np.nanmax(res["tau"]))
if mode == "rho":
    out["rho"] = q(np.array([z[1] for z in samples]))
tag = S["key"] + ("" if mode == "rho" else "_free") + (f"_{host}" if sysname == "HIP 31126" else "")
pickle.dump(dict(chain=ch, out=out), open(R("results", f"fit_{tag}.pkl"), "wb"))
json.dump(out, open(R("results", f"fit_{tag}.json"), "w"), indent=1, default=float)
if mode == "rho" and host == "A":     # the adopted fit defines the ephemerides used by the other steps
    json.dump(out, open(R("results", f"fit_{S['key']}.json"), "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k in FIT + ["n_tau", "acc", "dilution"]}, default=float)[:3000])
