"""Step 7: figures of the paper (written to figures/).
  fig_transits.pdf   light curves: individual transits and the phase-folded transit with the adopted model
  fig_pixels.pdf     pixel-level localisation of the three candidates and of the 14.8-d field signal (control)
  fig_context.pdf    period-radius diagram of known planets with the three candidates
"""
from common import *
import batman
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from tcemine.detrend import in_transit

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
                     "font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.direction": "in", "ytick.direction": "in",
                     "xtick.top": True, "ytick.right": True, "savefig.dpi": 300})
os.makedirs(R("figures"), exist_ok=True)
ORDER = ["TOI-678", "HIP 31126", "TOI-5997"]
FIT = {"HIP 31126": ["new"], "TOI-678": ["new", "b11"], "TOI-5997": ["new", "b5"]}
MASK = {"HIP 31126": ["f14"], "TOI-678": ["t6"], "TOI-5997": []}
LABEL = {"HIP 31126": "HIP 31126 (GJ 237), $P = 29.29$ d", "TOI-678": "TOI-678, $P = 130.1$ d", "TOI-5997": "TOI-5997, $P = 14.22$ d"}

def model(F, name, tt, exp=120 / 86400):
    p = F[name]; P, T0, rp, a, b = p["P"][1], p["T0"][1], p["rp_rs"][1], p["a_rs"][1], p["b"][1]
    q1, q2 = F["q1"][1], F["q2"][1]
    bp = batman.TransitParams(); bp.t0 = T0; bp.per = P; bp.rp = rp; bp.a = a; bp.inc = np.degrees(np.arccos(b / a))
    bp.ecc = 0; bp.w = 90; bp.u = [2 * np.sqrt(q1) * q2, np.sqrt(q1) * (1 - 2 * q2)]; bp.limb_dark = "quadratic"
    m = batman.TransitModel(bp, np.asarray(tt, float), supersample_factor=5, exp_time=exp).light_curve(bp)
    return 1.0 - F["dilution"] * (1.0 - m)

def binned(x, y, w):
    k = np.floor(x / w).astype(int); u, inv, cnt = np.unique(k, return_inverse=True, return_counts=True)
    xb = np.bincount(inv, x) / cnt; yb = np.bincount(inv, y) / cnt
    sb = np.sqrt(np.maximum(np.bincount(inv, y ** 2) / cnt - yb ** 2, 0)) / np.sqrt(np.maximum(cnt - 1, 1))
    ok = cnt >= 4
    return xb[ok], yb[ok], sb[ok]

def light_curve_data(sysname):
    S = SYSTEMS[sysname]; F = json.load(open(R("results", f"fit_{S['key']}.json")))
    d = np.load(R("results", f"lc_{S['key']}.npz")); t, f, s = d["t"], d["f"], d["s"]
    for k in FIT[sysname]:
        if k != "new":
            f = f / model(F, k, t)
    for k in MASK[sysname]:
        P_, T0_, D_ = get_ephem(sysname, k); m = ~in_transit(t, P_, T0_, D_, 2.0); t, f, s = t[m], f[m], s[m]
    return F, t, f, s

def have_fit(sysname):
    return os.path.exists(R("results", f"fit_{SYSTEMS[sysname]['key']}.json"))

def fig_transits():
    fig = plt.figure(figsize=(7.1, 8.6))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.25], hspace=0.32, wspace=0.22, left=0.08, right=0.98, top=0.975, bottom=0.055)
    for row, sysname in enumerate(ORDER):
        if not have_fit(sysname):
            continue
        F, t, f, s = light_curve_data(sysname); p = F["new"]
        P, T0, T14 = p["P"][1], p["T0"][1], p["T14_h"][1] / 24
        n = np.round((t - T0) / P); x = t - (T0 + n * P)
        win = np.abs(x) < 3 * T14
        # the transits with full coverage, as counted by the vetting (s03)
        epochs = [int(round((pt[1] - T0) / P)) for pt in load_numbers("vetting")[f"{sysname}:new"]["per_transit"]]
        # left: individual transits
        ax = fig.add_subplot(gs[row, 0]); depth = 1e-6 * p["depth_ppm"][1] * F["dilution"]
        s30 = np.std(f[win & (np.abs(x) > T14)]) / np.sqrt(15)
        off = max(4.0 * depth, 9.0 * s30)
        xm = np.linspace(-3 * T14, 3 * T14, 600)
        for i, k in enumerate(epochs):
            m = win & (n == k); sec = int(np.median(s[m]))
            xb, yb, sb = binned(x[m] * 24, f[m], 30 / 60)
            ax.errorbar(xb, yb - i * off, sb, fmt="o", ms=2.2, color="0.25", elinewidth=0.5, capsize=0)
            ax.plot(xm * 24, model(F, "new", T0 + k * P + xm) - i * off, color="C3", lw=0.9)
            ax.text(-3 * T14 * 24 * 0.95, 1 - i * off + 0.3 * off, f"S{sec}", fontsize=6.3, va="center", color="0.3")
        ax.set_xlim(-3 * T14 * 24, 3 * T14 * 24); ax.set_xlabel("Hours from mid-transit")
        ax.set_ylabel("Relative flux + offset"); ax.set_title(LABEL[sysname], fontsize=8.5, loc="left")
        # right: folded transit and residuals
        sub = gs[row, 1].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
        a1 = fig.add_subplot(sub[0]); a2 = fig.add_subplot(sub[1], sharex=a1)
        xx = x[win] * 24; yy = f[win]; res = yy - model(F, "new", t[win])
        a1.plot(xx, (yy - 1) * 1e3, ",", color="0.75", rasterized=True)
        xb, yb, sb = binned(xx, yy, 15 / 60)
        a1.errorbar(xb, (yb - 1) * 1e3, sb * 1e3, fmt="o", ms=2.5, color="k", elinewidth=0.6, capsize=0)
        a1.plot(xm * 24, (model(F, "new", T0 + xm) - 1) * 1e3, color="C3", lw=1.2)
        rb = binned(xx, res, 15 / 60)
        a2.errorbar(rb[0], rb[1] * 1e3, rb[2] * 1e3, fmt="o", ms=2.5, color="k", elinewidth=0.6, capsize=0)
        a2.axhline(0, color="C3", lw=0.8)
        lim = 2.2 * depth * 1e3
        a1.set_ylim(-lim, 0.9 * lim); a2.set_ylim(-0.55 * lim, 0.55 * lim)
        a1.set_ylabel("Flux $-$ 1 (ppt)"); a2.set_ylabel("Res."); a2.set_xlabel("Hours from mid-transit")
        plt.setp(a1.get_xticklabels(), visible=False); a1.set_xlim(-3 * T14 * 24, 3 * T14 * 24)
        a1.text(0.02, 0.06, f"{len(epochs)} transits, 15-min bins", transform=a1.transAxes, fontsize=6.5, color="0.3")
    fig.savefig(R("figures", "fig_transits.pdf")); fig.savefig(R("figures", "fig_transits.png"), dpi=150); plt.close(fig)

def ellipse(ax, mu, C, nsig, **kw):
    w, v = np.linalg.eigh(C); ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
    # chi2 (2 dof) quantile matching the two-sided Gaussian nsig probability
    from scipy.stats import chi2, norm
    k = np.sqrt(chi2.ppf(1 - 2 * norm.sf(nsig), 2))
    ax.add_patch(Ellipse(mu, 2 * k * np.sqrt(w[1]), 2 * k * np.sqrt(w[0]), angle=ang, fill=False, **kw))

def fig_pixels():
    cols = [("HIP 31126", "new", "H31126", "HIP 31126: 29.3-d candidate"), ("HIP 31126", "f14", "H31126", "HIP 31126 field: 14.8-d signal"),
            ("TOI-678", "new", "T678", "TOI-678: 130.1-d candidate"), ("TOI-5997", "new", "T5997", "TOI-5997: 14.2-d candidate")]
    fig, axs = plt.subplots(2, 4, figsize=(7.1, 3.9), gridspec_kw=dict(hspace=0.36, wspace=0.34, left=0.06, right=0.975, top=0.93, bottom=0.11))
    for j, (sysname, which, key, title) in enumerate(cols):
        d = np.load(R("results", f"pixel_{key}_{which}_fig.npz"), allow_pickle=True)
        PX = json.load(open(R("results", f"pixel_{key}.json")))["signals"][which]
        D, ap, x, y, tm = d["D"], d["ap"], d["x"], d["y"], d["tmag"]
        ax = axs[0, j]; vmax = np.nanmax(np.abs(D[ap])) if ap.any() else np.nanmax(np.abs(D))
        ax.imshow(np.clip(D / vmax, -1, 1), origin="lower", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        ny, nx = D.shape
        for (yy, xx) in zip(*np.where(ap)):     # aperture outline
            for dy, dx, seg in ((0, -0.5, "v"), (0, 0.5, "v"), (-0.5, 0, "h"), (0.5, 0, "h")):
                y2, x2 = int(yy + 2 * dy), int(xx + 2 * dx)
                if not (0 <= y2 < ny and 0 <= x2 < nx and ap[y2, x2]):
                    if seg == "v":
                        ax.plot([xx + dx] * 2, [yy - 0.5, yy + 0.5], color="k", lw=0.7)
                    else:
                        ax.plot([xx - 0.5, xx + 0.5], [yy + dy] * 2, color="k", lw=0.7)
        Tt = tm[0]; sel = (tm < Tt + 7) & (x > -0.5) & (x < nx - 0.5) & (y > -0.5) & (y < ny - 0.5)
        ax.scatter(x[sel][1:], y[sel][1:], s=np.clip(30 * 10 ** (-0.2 * (tm[sel][1:] - Tt)), 2, 30), facecolors="none", edgecolors="0.1", lw=0.6)
        ax.plot(x[0], y[0], "*", ms=8, mfc="gold", mec="k", mew=0.5)
        ax.plot(*d["fit"], "x", ms=7, color="lime", mew=1.6)
        ax.set_xlim(-0.5, nx - 0.5); ax.set_ylim(-0.5, ny - 0.5)
        ax.set_title(title, fontsize=7.2); ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.03, 0.04, f"S{int(d['sector'])}", transform=ax.transAxes, fontsize=6.5, color="k")
        # bottom: sky-plane offsets
        b = axs[1, j]; best = d["best"]; C = d["cov"]; boots = d["boots"]
        b.plot(boots[:, 0], boots[:, 1], ".", ms=1.2, color="0.6", rasterized=True)
        ellipse(b, best, C, 1, color="C0", lw=0.9); ellipse(b, best, C, 3, color="C0", lw=0.9, ls="--")
        lim = 45 if which == "f14" else 40
        rows = {r["TIC"]: r for r in PX["stars"]}; placed = []
        for k in range(len(d["ids"])):
            oe, on = d["off_e"][k], d["off_n"][k]
            if abs(oe) > lim or abs(on) > lim or tm[k] > Tt + 9:
                continue
            if k == 0:
                b.plot(0, 0, "*", ms=9, mfc="gold", mec="k", mew=0.5, zorder=5)
                continue
            r = rows.get(str(d["ids"][k]))
            col = "k" if r is not None else "0.6"
            b.plot(oe, on, "o", ms=max(2, 6 - 0.4 * (tm[k] - Tt)), mfc="none", mec=col, mew=0.7)
            if r is not None and np.hypot(oe, on) < 5:
                b.text(oe - 2.0, on - 5.5, "B", fontsize=6.5, color="k")
            elif r is not None:
                lx, ly = oe + 1.5, on + 1.5
                while any(abs(lx - px) < 9 and abs(ly - py) < 4.5 for px, py in placed):
                    ly -= 5.0
                placed.append((lx, ly))
                b.text(lx, ly, f"{r['sigma']:.1f}$\\sigma$" if r["sigma"] < 10 else ">10$\\sigma$", fontsize=5.5, color="C3")
        b.plot(*best, "x", ms=6, color="C0", mew=1.4, zorder=6)
        b.set_xlim(lim, -lim); b.set_ylim(-lim, lim); b.set_aspect("equal")
        b.set_xlabel(r"$\Delta\alpha$ (arcsec)", fontsize=7.5)
        if j == 0:
            b.set_ylabel(r"$\Delta\delta$ (arcsec)", fontsize=7.5)
    fig.savefig(R("figures", "fig_pixels.pdf")); fig.savefig(R("figures", "fig_pixels.png"), dpi=150); plt.close(fig)

def fig_context():
    df = pd.read_csv(R("data", "nasa_pscomppars_2026-10-08.csv"))
    fig, ax = plt.subplots(figsize=(3.45, 2.9)); fig.subplots_adjust(left=0.15, right=0.97, top=0.97, bottom=0.14)
    df = df[df.tran_flag == 1]
    tess = df.disc_facility.fillna("").str.contains("TESS")
    ax.plot(df.pl_orbper[~tess], df.pl_rade[~tess], ".", ms=1.6, color="0.78", rasterized=True, label="Transiting planets")
    ax.plot(df.pl_orbper[tess], df.pl_rade[tess], ".", ms=2.2, color="C0", alpha=0.6, rasterized=True, label="TESS discoveries")
    mk = {"TOI-678": ("s", "C3"), "HIP 31126": ("o", "C2"), "TOI-5997": ("D", "C1")}
    for sysname in [x for x in ORDER if have_fit(x)]:
        S = SYSTEMS[sysname]; F = json.load(open(R("results", f"fit_{S['key']}.json"))); p = F["new"]
        P = p["P"][1]; r = p["Rp_Re"]
        ax.errorbar(P, r[1], [[r[1] - r[0]], [r[2] - r[1]]], fmt=mk[sysname][0], ms=5, mfc=mk[sysname][1], mec="k", mew=0.6,
                    ecolor="k", elinewidth=0.7, capsize=0, zorder=5, label=sysname + (" (host A)" if sysname == "HIP 31126" else ""))
        if sysname == "HIP 31126":
            FB = json.load(open(R("results", "fit_H31126_B.json")))["new"]["Rp_Re"]
            ax.errorbar(P, FB[1], [[FB[1] - FB[0]], [FB[2] - FB[1]]], fmt="o", ms=5, mfc="white", mec=mk[sysname][1], mew=1.0,
                        ecolor=mk[sysname][1], elinewidth=0.7, capsize=0, zorder=5, label="HIP 31126 (host B)")
            ax.plot([P, P], [r[1], FB[1]], ":", color=mk[sysname][1], lw=0.8, zorder=4)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.5, 500); ax.set_ylim(0.6, 25)
    ax.set_xlabel("Orbital period (d)"); ax.set_ylabel(r"Planet radius ($R_\oplus$)")
    ax.set_yticks([1, 2, 4, 10, 20]); ax.set_yticklabels(["1", "2", "4", "10", "20"])
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="0.8", handletextpad=0.3, borderaxespad=0.3, markerscale=1.0, fontsize=6.3)
    fig.savefig(R("figures", "fig_context.pdf")); fig.savefig(R("figures", "fig_context.png"), dpi=150); plt.close(fig)

if __name__ == "__main__":
    what = sys.argv[1:] or ["transits", "pixels", "context"]
    for w in what:
        globals()["fig_" + w](); print("done", w)
