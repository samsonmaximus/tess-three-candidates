"""Light-curve input: the browser-built bundle format, or FITS files fetched normally."""
import gzip, json, struct
import numpy as np

MAGIC = 0x424F5845  # 'EXOB'

def load_bundle(path):
    """Read a bundle written by tools/fetch_bundle.js: {tic: [sector dicts]}."""
    raw = gzip.open(path, "rb").read()
    magic, hlen = struct.unpack("<II", raw[:8])
    if magic != MAGIC:
        raise ValueError("not a tcemine bundle")
    hdr = json.loads(raw[8:8 + hlen]); pad = (8 - (8 + hlen) % 8) % 8; base = 8 + hlen + pad
    out = {}
    for it in hdr["items"]:
        d = {k: np.frombuffer(raw, dtype="<" + dt, count=n, offset=base + off).copy()
             for k, (off, n, dt) in it["arrs"].items()}
        d["meta"] = it["meta"]; d["sector"] = int(it["sector"])
        out.setdefault(int(it["tic"]), []).append(d)
    for k in out:
        out[k].sort(key=lambda d: d["sector"])
    return out

def load_fits(paths):
    """Read SPOC *_lc.fits files with astropy (for users with direct MAST access)."""
    from astropy.io import fits
    secs = []
    for p in paths:
        with fits.open(p) as h:
            d = {c: np.array(h[1].data[c]) for c in ("TIME", "PDCSAP_FLUX", "PDCSAP_FLUX_ERR", "SAP_FLUX",
                                                     "QUALITY", "MOM_CENTR1", "MOM_CENTR2")}
            d["meta"] = dict(h[0].header); d["sector"] = int(h[0].header["SECTOR"])
            secs.append(d)
    return sorted(secs, key=lambda d: d["sector"])

def clean(secs, flux="PDCSAP_FLUX"):
    """Quality==0, finite, per-sector median-normalised arrays."""
    T, F, E, S, C1, C2 = [], [], [], [], [], []
    for d in secs:
        t = d["TIME"]; f = d[flux].astype(float); e = d["PDCSAP_FLUX_ERR"].astype(float)
        m = (d["QUALITY"] == 0) & np.isfinite(t) & np.isfinite(f) & np.isfinite(e) & (f > 0)
        med = np.nanmedian(f[m])
        T.append(t[m]); F.append(f[m] / med); E.append(e[m] / med); S.append(np.full(m.sum(), d["sector"]))
        C1.append(d["MOM_CENTR1"][m]); C2.append(d["MOM_CENTR2"][m])
    return [np.concatenate(x) for x in (T, F, E, S, C1, C2)]
