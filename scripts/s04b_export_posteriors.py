"""Step 4b: export the MCMC posterior samples of every transit fit (results/fit_*.pkl) as gzipped CSV files with
named columns in results/posteriors/. Every 4th stored sample is kept (12 000 for the adopted fits, 6 000 for the
free-density checks). Parameter layout as documented in tcemine.transit.joint_fit: per planet [P, T0, rp, b]
(+ ln a/R* in the free-density fits), then the stellar density in g/cm^3 (adopted fits only), q1, q2 and one
ln(noise factor) per observing season. P in days, T0 in BJD_TDB - 2457000."""
import gzip, pickle
from common import *

FITS = {  # file tag -> (planet names in fit order, free-density fit?)
    "T678": (["candidate", "TOI-678.01"], False), "T678_free": (["candidate"], True),
    "H31126_A": (["candidate"], False), "H31126_B": (["candidate"], False), "H31126_free_A": (["candidate"], True),
    "T5997": (["candidate", "TOI-5997b"], False), "T5997_free": (["candidate"], True),
}
os.makedirs(R("results", "posteriors"), exist_ok=True)
for tag, (names, free) in FITS.items():
    d = pickle.load(open(R("results", f"fit_{tag}.pkl"), "rb")); ch = d["chain"][::4]; out = d["out"]
    cols = []
    for nm in names:
        cols += [f"{nm}_P", f"{nm}_T0", f"{nm}_rp_rs", f"{nm}_b"] + ([f"{nm}_ln_a_rs"] if free else [])
    if not free:
        cols += ["rho_star"]
    cols += ["q1", "q2"] + [f"ln_noise_season{s}" for s in out["seasons"]]
    assert len(cols) == ch.shape[1], (tag, len(cols), ch.shape)
    path = R("results", "posteriors", f"posterior_{tag}.csv.gz")
    with gzip.open(path, "wt") as f:
        f.write(",".join(cols) + "\n")
        for row in ch:
            f.write(",".join(f"{v:.10g}" for v in row) + "\n")
    print(tag, ch.shape, path)
