"""Step 6: check of the predicted transits of the TOI-678 candidate that fall in sectors observed only in full-frame
images (Sector 13: epoch -10; Sector 6: an epoch of the P/2 alias). Input: data/ffi/toi678_ffi_windows.txt, the
TESS-SPOC (PDCSAP, SAP) and QLP (KSPSAP, SAP) light curves within 2.5 d of each epoch, extracted in the browser from
the MAST HLSP files. The transit shape is fixed to the 2-min fit; free depth scale and a quadratic baseline."""
import json, numpy as np, batman
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
raw = open("/home/claude/paper3/repo/data/ffi/toi678_ffi_windows.txt").read()
F = json.load(open("/home/claude/paper3/repo/results/fit_T678.json")); pn = F["new"]
P, T0 = pn["P"][1], pn["T0"][1]; q1, q2 = F["q1"][1], F["q2"][1]
def mdl(t, tc):
    bp = batman.TransitParams(); bp.t0 = tc; bp.per = P; bp.rp = pn["rp_rs"][1]; bp.a = pn["a_rs"][1]
    bp.inc = np.degrees(np.arccos(pn["b"][1] / pn["a_rs"][1])); bp.ecc = 0; bp.w = 90
    bp.u = [2 * np.sqrt(q1) * q2, np.sqrt(q1) * (1 - 2 * q2)]; bp.limb_dark = "quadratic"
    return batman.TransitModel(bp, t, supersample_factor=5, exp_time=30 / 1440).light_curve(bp)
res = {}
fig, axs = plt.subplots(2, 2, figsize=(9, 6))
for ib, blk in enumerate(raw.split("## ")[1:]):
    head, *lines = blk.strip().split("\n"); key = head.split()[0]
    if key.startswith("3spoc"):
        continue
    a = np.array([[float(x) for x in l.split()] for l in lines]); t, f, sap, fe, q = a.T
    ok = (q == 0) & np.isfinite(f); t, f, sap = t[ok], f[ok], sap[ok]; tc = float(head.split("tc=")[1])
    for j, (nm, y) in enumerate((("pipeline", f), ("SAP", sap))):
        y = y / np.median(y)
        def chi(sc, dt=0.0):
            m = 1 - sc * (1 - mdl(t, tc + dt)); A = np.vstack([np.ones_like(t), t - tc, (t - tc) ** 2]).T
            c, *_ = np.linalg.lstsq(A * m[:, None], y, rcond=None); return np.sum((y - m * (A @ c)) ** 2), m * (A @ c), A @ c
        grid = np.linspace(-1.0, 2.5, 141); cg = np.array([chi(g)[0] for g in grid]); best = grid[np.argmin(cg)]
        n = len(t); s2 = cg.min() / (n - 4); d = (cg - cg.min()) / s2; one = grid[d < 1]
        r = dict(n=n, scale=float(best), lo=float(best - one.min()), hi=float(one.max() - best), dchi2_zero=float((chi(0.0)[0] - cg.min()) / s2),
                 dchi2_one=float((chi(1.0)[0] - cg.min()) / s2), rms_ppm=float(np.sqrt(s2) * 1e6))
        # free mid-time scan (depth fixed at 1x) to see where the dip is
        dts = np.linspace(-0.5, 0.5, 41); cd = np.array([chi(1.0, x)[0] for x in dts]); r["dt_best_h"] = float(dts[np.argmin(cd)] * 24)
        res[f"{key}:{nm}"] = r; print(key, nm, r)
        if key in ("13spoc", "13qlp"):
            ax = axs[0 if key == "13spoc" else 1, j]; _, mod, base = chi(best)
            ax.plot((t - tc) * 24, y / base, ".", color="0.5"); ax.plot((t - tc) * 24, mod / base, "r-")
            ax.set_title(f"{key} {nm}: scale {best:.2f}"); ax.axvline(0, color="k", lw=0.5)
fig.tight_layout(); fig.savefig("/home/claude/paper3/repo/results/ffi13_check.png", dpi=80)
json.dump(res, open("/home/claude/paper3/repo/results/ffi_toi678.json", "w"), indent=1)
