"""Step 5c: contrast curves for TRICERATOPS from the high-resolution imaging data products on ExoFOP.

Inputs (data/imaging/, as posted on ExoFOP):
  HIP 31126  SOAR/HRCam I band, 2019-05-18 (Ziegler et al. 2020): sensitivity plot TIC293689267I-cz20190518.pdf;
             the plotted 5-sigma limits were digitised from the vector graphics (see DIGITISED below).
  TOI-678    Gemini-S/Zorro 562 and 832 nm, 2020-12-28: TOI678I-ef20201228-{562,832}_sensitivity.dat
  TOI-5997   Gemini-N/'Alopeke 562 and 832 nm, 2024-05-23: TOI5997I-cc20240523-{562,832}_sensitivity.dat
             Palomar/PHARO Br-gamma, 2023-06-06: TOI5997I-dc20230606-Brgamma_plot.tbl
For the speckle data we use the fitted curve that the Gemini team provides in each file. TRICERATOPS needs the
contrast to increase with separation, so every curve is replaced by its conservative monotone envelope
C'(s) = min over s' >= s of C(s') (a dip at large separation lowers the curve inside it, never raises it).
Output: data/cc_<system>_<instrument>.dat (separation in arcsec, delta-mag), comma separated.
"""
from common import *

# SOAR: centres of the plotted filled circles, converted with the axis ticks (0..8 arcsec, 0..8 mag) of the PDF
DIGITISED_SOAR = [(0.063, 1.00), (0.100, 2.04), (0.150, 2.52), (0.200, 5.01), (1.000, 5.36), (8.064, 7.95)]

def gemini_fit(path):
    rows, on = [], False
    for line in open(path):
        if line.startswith("# fit"):
            on = True; continue
        if on and line.strip() and not line.startswith("#"):
            s, m = map(float, line.split()[:2]); rows.append((s, m))
    return [(s, m) for s, m in rows if m > 0]

def pharo(path):
    rows = []
    for line in open(path):
        p = [x.strip() for x in line.split(",")]
        try:
            s, m = float(p[0]), float(p[1])
        except (ValueError, IndexError):
            continue
        if s > 0:
            rows.append((s, m))
    return rows

def envelope(rows):
    """Sorted by separation, de-duplicated, conservative monotone (non-decreasing) envelope, strictly increasing."""
    rows = sorted(dict(rows).items())
    s = np.array([r[0] for r in rows]); m = np.array([r[1] for r in rows])
    env = np.minimum.accumulate(m[::-1])[::-1]
    out = []
    for si, mi in zip(s, env):
        if out and mi <= out[-1][1]:
            mi = out[-1][1] + 1e-6      # plateau: keep the point, contrast nudged so np.interp sees increasing values
        out.append((si, mi))
    return out

CURVES = {
    "cc_H31126_soar.dat": DIGITISED_SOAR,
    "cc_T678_zorro832.dat": gemini_fit(R("data", "imaging", "TOI678I-ef20201228-832_sensitivity.dat")),
    "cc_T678_zorro562.dat": gemini_fit(R("data", "imaging", "TOI678I-ef20201228-562_sensitivity.dat")),
    "cc_T5997_alopeke832.dat": gemini_fit(R("data", "imaging", "TOI5997I-cc20240523-832_sensitivity.dat")),
    "cc_T5997_alopeke562.dat": gemini_fit(R("data", "imaging", "TOI5997I-cc20240523-562_sensitivity.dat")),
    "cc_T5997_pharoBrg.dat": pharo(R("data", "imaging", "TOI5997I-dc20230606-Brgamma_plot.tbl")),
}

if __name__ == "__main__":
    summ = {}
    for name, rows in CURVES.items():
        env = envelope(rows)
        with open(R("data", name), "w") as f:
            for s, m in env:
                f.write("%.4f,%.6f\n" % (s, m))
        s = np.array([e[0] for e in env]); m = np.array([e[1] for e in env])
        at = {str(x): float(np.interp(x, s, m)) for x in (0.1, 0.15, 0.5, 1.0)}
        summ[name] = dict(n=len(env), smin=float(s[0]), smax=float(s[-1]), dmag_at=at, max_lowering=float(np.max(np.array([r[1] for r in sorted(dict(rows).items())]) - m)))
        print(name, len(env), "pts; dmag at 0.1/0.15/0.5/1.0 arcsec:", ["%.2f" % v for v in at.values()], "max lowering by envelope %.2f" % summ[name]["max_lowering"])
    save_numbers("contrast", summ)
