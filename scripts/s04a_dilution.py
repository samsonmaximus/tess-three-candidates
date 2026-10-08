"""Step 4a: flux fraction of each star inside the SPOC aperture, per sector, from the PRF model and the
registered star positions (TIC 8.2; Gaia-based T magnitudes for the two components of HIP 31126), compared
with the SPOC CROWDSAP value used by PDC. For HIP 31126 this gives the dilution factor g = f_host / CROWDSAP
for host A or B (PDC assumed the TIC magnitude of B, which is blended)."""
from common import *
import s05a_pixel as PX

def run(sysname):
    S = SYSTEMS[sysname]; st = PX.stars_for(sysname, S); Tt = float(st.Tmag[0])
    vet = load_numbers("vetting"); crowd = {m["sector"]: m["crowdsap"] for m in vet[f"{sysname}:sectors"]}
    fl = 10 ** (-0.4 * (st.Tmag.values - Tt)); rows = []
    for d in [d for d in PX.DIF.values() if d["tic"] == S["tic"]]:
        u = PX.sector_setup(d, st, Tt)
        inap = np.array([fl[k] * (u["prf"].image(u["x"][k], u["y"][k], u["nrow"], u["ncol"]) * u["ap"]).sum() for k in range(len(st))])
        tot = inap.sum()
        rows.append(dict(sector=d["sector"], crowdsap=crowd.get(d["sector"]), f_target=float(inap[0] / tot),
                         f_second=float(inap[1] / tot), second_id=st.ID[1], second_sep=float(st.dstArcSec[1])))
    return rows

if __name__ == "__main__":
    out = {}
    for sysname in SYSTEMS:
        rows = run(sysname); out[sysname] = rows
        print(sysname, [(r["sector"], round(r["crowdsap"], 3), round(r["f_target"], 3), round(r["f_second"], 3)) for r in rows])
    # HIP 31126: dilution factors for host A and B, averaged over the sectors with transits of the candidate
    fit_secs = set(json.load(open(R("results", "pixel_H31126.json")))["signals"]["new"]["sectors"]) if os.path.exists(R("results", "pixel_H31126.json")) else None
    r = [x for x in out["HIP 31126"] if (fit_secs is None or x["sector"] in fit_secs)]
    gA = np.mean([x["f_target"] / x["crowdsap"] for x in r]); gB = np.mean([x["f_second"] / x["crowdsap"] for x in r])
    out["HIP 31126:g"] = dict(A=float(gA), B=float(gB), sectors=[x["sector"] for x in r])
    print("HIP 31126 dilution factors g_A=%.3f g_B=%.3f" % (gA, gB))
    save_numbers("dilution", out)
