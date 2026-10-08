"""Shared settings for the three-candidate paper (HIP 31126, TOI-678, TOI-5997)."""
import json, gzip, io, os, sys
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
R = lambda *p: os.path.join(ROOT, *p)

# Initial ephemerides (BTJD = BJD_TDB - 2457000); they are replaced by the fitted values once s04 has run.
# "new": the candidate. "others": other signals in the same light curve, masked when testing the candidate.
# control=True: run through the same vetting as a comparison object.
SYSTEMS = {
    "HIP 31126": dict(tic=293689266, key="H31126", bundle="lc_hip31126.bin.gz",
        new=dict(label="HIP 31126 (29.3 d)", P=29.285536, T0=3334.99159, dur_h=2.05),
        others={"f14": dict(label="14.81 d field signal", P=14.8104, T0=1470.2213, dur_h=3.87, control=True)},
        exclude_epochs={}),
    "TOI-678": dict(tic=294395926, key="T678", bundle="lc_toi678.bin.gz",
        new=dict(label="TOI-678 (130.1 d)", P=130.1449, T0=2972.2631, dur_h=6.4),
        others={"b11": dict(label="TOI-678.01", P=11.3298459, T0=1493.024505, dur_h=4.475, control=True),
                "t6": dict(label="6.56 d TCE", P=6.56223, T0=1389.0843, dur_h=2.81, control=True)},
        exclude_epochs={}),
    "TOI-5997": dict(tic=39516274, key="T5997", bundle="lc_toi5997.bin.gz",
        new=dict(label="TOI-5997 (14.2 d)", P=14.216053, T0=2758.28181, dur_h=3.06),
        others={"b5": dict(label="TOI-5997 b", P=5.654999, T0=2759.89543, dur_h=1.67, control=True)},
        exclude_epochs={}),
}

def save_numbers(section, values):
    p = R("results", "numbers.json")
    allv = json.load(open(p)) if os.path.exists(p) else {}
    allv[section] = values
    json.dump(allv, open(p, "w"), indent=1, default=float)

def load_numbers(section=None):
    p = R("results", "numbers.json")
    allv = json.load(open(p)) if os.path.exists(p) else {}
    return allv.get(section, {}) if section else allv

def get_ephem(sysname, which="new"):
    """(P, T0, T14 in days): fitted values from s04 if present, else the initial values above."""
    S = SYSTEMS[sysname]; p = R("results", f"fit_{S['key']}.json")
    if os.path.exists(p):
        F = json.load(open(p))
        if which in F:
            return F[which]["P"][1], F[which]["T0"][1], F[which]["T14_h"][1] / 24
    d = S["new"] if which == "new" else S["others"][which]
    return d["P"], d["T0"], d["dur_h"] / 24

def tic_field(psv_text):
    """TIC 8.2 cone (pipe-separated text) -> DataFrame of real stars. Entries flagged ARTIFACT are dropped. Entries
    flagged DUPLICATE or SPLIT are dropped unless no other entry represents the same Gaia source, in which case one
    of them is kept (TIC 8.2 can leave a real Gaia source represented only by flagged entries)."""
    df = pd.read_csv(io.StringIO(psv_text), sep="|", dtype={"ID": str, "GAIA": str, "duplicate_id": str})
    disp = df.disposition.fillna("")
    keep = ~disp.isin(["ARTIFACT", "DUPLICATE", "SPLIT"])
    have = set(df.GAIA[keep].dropna())
    for g, grp in df[disp.isin(["DUPLICATE", "SPLIT"])].groupby("GAIA"):
        if g not in have:
            j = grp.index[np.argsort((grp.disposition != "SPLIT").values)[0]]   # prefer the entry a duplicate points to
            keep[j] = True; have.add(g)
    return df[keep].copy()
