// tpf_diffimg.js -- per-transit difference images from SPOC 2-min target pixel files, computed in the browser
// (run on https://mast.stsci.edu/api/v0/). Only the small derived images leave the browser; scripts/s05_pixel.py
// fits them with the TESS PRF model.
//
// For each signal (P, T0, T14 in BTJD days) and each transit in the sector:
//   in-transit  : quality == 0 cadences with |t - tc| < 0.4 T14
//   out-of-transit: quality == 0 cadences with 0.75 T14 < |t - tc| < 1.75 T14, on both sides
//   diff = mean(out) - mean(in) per pixel (FLUX column, background-subtracted), err from the cadence scatter.
// A transit is used only if in, out-before and out-after each have >= 50% of the expected cadences.
// "Null" difference images use the same windows centred on random times that avoid every listed signal
// by > 2.5 T14; they calibrate the noise of the source position fitted to the real difference image.
function parseTPF(buf) {
  const dv = new DataView(buf), td = new TextDecoder("ascii"); let pos = 0; const hdus = [];
  while (pos < buf.byteLength && hdus.length < 3) {
    const h = {}; let end = false;
    while (!end) {
      const block = td.decode(new Uint8Array(buf, pos, 2880)); pos += 2880;
      for (let i = 0; i < 36; i++) {
        const c = block.slice(i * 80, i * 80 + 80), k = c.slice(0, 8).trim();
        if (k === "END") { end = true; break; }
        if (c[8] === "=") { let v = c.slice(10).trim(); h[k] = v.startsWith("'") ? v.slice(1, v.indexOf("'", 1)).trim() : v.split("/")[0].trim(); }
      }
    }
    const naxis = +h.NAXIS || 0; let size = 0;
    if (naxis > 0) { size = Math.abs(+h.BITPIX) / 8; for (let i = 1; i <= naxis; i++) size *= +h["NAXIS" + i]; size += (+h.PCOUNT || 0); }
    hdus.push({h, dataStart: pos}); pos += Math.ceil(size / 2880) * 2880;
  }
  const H = hdus[1].h, nf = +H.TFIELDS, rowb = +H.NAXIS1, nr = +H.NAXIS2, sz = {L: 1, B: 1, I: 2, J: 4, K: 8, E: 4, D: 8, A: 1};
  let off = 0; const cols = {};
  for (let i = 1; i <= nf; i++) {
    const m = H["TFORM" + i].match(/^(\d*)([A-Z])/), rep = m[1] ? +m[1] : 1;
    cols[H["TTYPE" + i]] = {off, t: m[2], rep, idx: i}; off += rep * sz[m[2]];
  }
  const base = hdus[1].dataStart;
  const scalar = name => {
    const c = cols[name]; const a = new Float64Array(nr);
    for (let r = 0; r < nr; r++) {
      const p = base + r * rowb + c.off;
      a[r] = c.t === "D" ? dv.getFloat64(p, false) : c.t === "E" ? dv.getFloat32(p, false) : c.t === "J" ? dv.getInt32(p, false) : dv.getInt16(p, false);
    }
    return a;
  };
  const flux = cols["FLUX"];
  const pix = (r, j) => dv.getFloat32(base + r * rowb + flux.off + 4 * j, false);
  const tdim = (H["TDIM" + flux.idx] || "").match(/\((\d+),(\d+)\)/);
  const ncol = +tdim[1], nrow = +tdim[2];
  // aperture HDU: mask image and WCS
  const A = hdus[2], ap = new Int32Array(ncol * nrow);
  for (let j = 0; j < ncol * nrow; j++) ap[j] = dv.getInt32(A.dataStart + 4 * j, false);
  return {h0: hdus[0].h, h1: H, h2: A.h, nr, ncol, nrow, npix: flux.rep, scalar, pix, ap: Array.from(ap)};
}

function windowStats(T, Q, P, rows, pix, npix, tc, T14, cad) {
  // returns null if coverage is insufficient
  const inn = [], ob = [], oa = [];
  for (const r of rows) {
    const dt = T[r] - tc, a = Math.abs(dt);
    if (a < 0.4 * T14) inn.push(r);
    else if (a > 0.75 * T14 && a < 1.75 * T14) (dt < 0 ? ob : oa).push(r);
  }
  const expIn = 0.8 * T14 / cad, expOut = 1.0 * T14 / cad;
  if (inn.length < 0.5 * expIn || ob.length < 0.5 * expOut || oa.length < 0.5 * expOut) return null;
  const out = ob.concat(oa);
  const mi = new Float64Array(npix), mo = new Float64Array(npix), vi = new Float64Array(npix), vo = new Float64Array(npix);
  for (const r of inn) for (let j = 0; j < npix; j++) { const v = pix(r, j); mi[j] += v; vi[j] += v * v; }
  for (const r of out) for (let j = 0; j < npix; j++) { const v = pix(r, j); mo[j] += v; vo[j] += v * v; }
  const diff = [], err = [], img = [];
  for (let j = 0; j < npix; j++) {
    const a = mi[j] / inn.length, b = mo[j] / out.length;
    const sa = Math.max(vi[j] / inn.length - a * a, 0), sb = Math.max(vo[j] / out.length - b * b, 0);
    diff.push(+(b - a).toFixed(3)); err.push(+Math.sqrt(sa / inn.length + sb / out.length).toFixed(3)); img.push(+b.toFixed(1));
  }
  return {nin: inn.length, nout: out.length, diff, err, img};
}

async function processTPF(lcURL, signals, maskSignals, nNull = 60, seed = 1) {
  const url = lcURL.replace("_lc.fits", "_tp.fits");
  const buf = await (await fetch("/api/v0.1/Download/file?uri=" + url)).arrayBuffer();
  const p = parseTPF(buf);
  const T = p.scalar("TIME"), Q = p.scalar("QUALITY"), PC1 = p.scalar("POS_CORR1"), PC2 = p.scalar("POS_CORR2");
  const good = [];
  for (let r = 0; r < p.nr; r++) if (Q[r] === 0 && isFinite(T[r]) && isFinite(p.pix(r, Math.floor(p.npix / 2)))) good.push(r);
  const tg = good.map(r => T[r]); const cad = 2 / 1440;
  const tmin = tg[0], tmax = tg[tg.length - 1];
  // median out-of-everything image for registration
  const imgSum = new Float64Array(p.npix); let nimg = 0;
  for (let k = 0; k < good.length; k += 25) { const r = good[k]; for (let j = 0; j < p.npix; j++) imgSum[j] += p.pix(r, j); nimg++; }
  const res = {url, tic: +p.h0.TICID, sector: +p.h0.SECTOR, camera: +p.h0.CAMERA, ccd: +p.h0.CCD, ncol: p.ncol, nrow: p.nrow,
               col0: +p.h2.CRVAL1P, row0: +p.h2.CRVAL2P, tmin, tmax, ngood: good.length,
               wcs: Object.fromEntries(Object.entries(p.h2).filter(([k]) => /^(CTYPE|CRVAL|CRPIX|CDELT|PC\d_\d|CD\d_\d|CUNIT|RADESYS|EQUINOX|A_|B_|AP_|BP_)/.test(k))),
               h0: Object.fromEntries(["TICID", "SECTOR", "CAMERA", "CCD", "TESSMAG", "RA_OBJ", "DEC_OBJ", "PMRA", "PMDEC"].map(k => [k, p.h0[k]])),
               aperture: p.ap, meanimg: Array.from(imgSum, v => +(v / nimg).toFixed(1)), signals: {}};
  const near = (tc, sigs, fac) => sigs.some(s => { const ph = ((tc - s.T0) / s.P) - Math.round((tc - s.T0) / s.P); return Math.abs(ph * s.P) < fac * s.T14; });
  for (const s of signals) {
    const rows = good;
    const out = {P: s.P, T0: s.T0, T14: s.T14, transits: [], nulls: []};
    const n0 = Math.ceil((tmin - s.T0) / s.P), n1 = Math.floor((tmax - s.T0) / s.P);
    for (let n = n0; n <= n1; n++) {
      const tc = s.T0 + n * s.P;
      const sub = rows.filter(r => Math.abs(T[r] - tc) < 2 * s.T14);
      const w = windowStats(T, Q, s.P, sub, p.pix, p.npix, tc, s.T14, cad);
      if (!w) continue;
      const pin = sub.filter(r => Math.abs(T[r] - tc) < 0.4 * s.T14), pout = sub.filter(r => Math.abs(T[r] - tc) > 0.75 * s.T14);
      const mean = (A, rr) => rr.reduce((x, r) => x + A[r], 0) / Math.max(rr.length, 1);
      out.transits.push({n, tc, nin: w.nin, nout: w.nout, diff: w.diff, err: w.err,
                         dpc1: +(mean(PC1, pout) - mean(PC1, pin)).toFixed(5), dpc2: +(mean(PC2, pout) - mean(PC2, pin)).toFixed(5)});
    }
    // nulls
    let x = seed * 9973 + s.P * 1000 | 0; const rnd = () => { x = (x * 1103515245 + 12345) % 2147483648; return x / 2147483648; };
    let tries = 0;
    while (out.nulls.length < nNull && tries < 5000) {
      tries++;
      const tc = tmin + 2 * s.T14 + rnd() * (tmax - tmin - 4 * s.T14);
      if (near(tc, maskSignals, 2.5)) continue;
      const sub = rows.filter(r => Math.abs(T[r] - tc) < 2 * s.T14);
      const w = windowStats(T, Q, s.P, sub, p.pix, p.npix, tc, s.T14, cad);
      if (!w) continue;
      out.nulls.push({tc: +tc.toFixed(5), diff: w.diff});
    }
    res.signals[s.name] = out;
  }
  return res;
}

async function runTPFs(jobs) {
  // jobs: [{lcURL, signals, mask}]
  window._TPF = window._TPF || {}; window._TPFlog = window._TPFlog || [];
  for (const j of jobs) {
    const key = j.lcURL.split("/").pop();
    if (window._TPF[key]) continue;
    try { const t0 = Date.now(); window._TPF[key] = await processTPF(j.lcURL, j.signals, j.mask);
          window._TPFlog.push(key + " ok " + ((Date.now() - t0) / 1000).toFixed(0) + "s"); }
    catch (e) { window._TPFlog.push(key + " ERR " + e); }
  }
  window._TPFdone = true;
}
