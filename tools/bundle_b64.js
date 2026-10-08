// bundle_b64.js -- same bundle format as tools/fetch_bundle.js (tcemine.lightcurves.load_bundle),
// but the gzipped bundle is kept in memory as base64 (window._B64[name]) so it can be transferred as page text
// when the analysis machine cannot reach MAST and files cannot be downloaded. Run on https://mast.stsci.edu/api/v0/.
async function mastQuery(request) {
  const r = await fetch("/api/v0/invoke", {method: "POST", headers: {"Content-Type": "application/x-www-form-urlencoded"},
                                          body: "request=" + encodeURIComponent(JSON.stringify(request))});
  return (await r.json()).data;
}
function parseFits(buf) {
  const dv = new DataView(buf), td = new TextDecoder("ascii"); let pos = 0; const hdus = [];
  while (pos < buf.byteLength && hdus.length < 2) {
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
  for (let i = 1; i <= nf; i++) { const m = H["TFORM" + i].match(/^(\d*)([A-Z])/), rep = m[1] ? +m[1] : 1; cols[H["TTYPE" + i]] = {off, t: m[2]}; off += rep * sz[m[2]]; }
  const get = name => {
    const c = cols[name]; if (!c) return null; const base = hdus[1].dataStart + c.off; let a;
    if (c.t === "D") { a = new Float64Array(nr); for (let r = 0; r < nr; r++) a[r] = dv.getFloat64(base + r * rowb, false); }
    else if (c.t === "E") { a = new Float32Array(nr); for (let r = 0; r < nr; r++) a[r] = dv.getFloat32(base + r * rowb, false); }
    else if (c.t === "J") { a = new Int32Array(nr); for (let r = 0; r < nr; r++) a[r] = dv.getInt32(base + r * rowb, false); }
    else if (c.t === "I") { a = new Int32Array(nr); for (let r = 0; r < nr; r++) a[r] = dv.getInt16(base + r * rowb, false); }
    return a;
  };
  return {h0: hdus[0].h, h1: H, get};
}
async function buildBundleB64(tics, name, provenance = "SPOC", exptime = 120) {
  const files = [];
  for (const tic of tics) {
    const rows = await mastQuery({service: "Mast.Caom.Filtered", format: "json", params: {
      columns: "provenance_name,sequence_number,t_exptime,dataURL",
      filters: [{paramName: "obs_collection", values: ["TESS", "HLSP"]}, {paramName: "dataproduct_type", values: ["timeseries"]},
                {paramName: "target_name", values: [String(tic)]}]}});
    for (const x of rows) if (x.provenance_name === provenance && x.t_exptime == exptime && /_lc\.fits$/.test(x.dataURL)) files.push({tic, url: x.dataURL, sector: x.sequence_number});
  }
  const COLS = ["TIME", "PDCSAP_FLUX", "PDCSAP_FLUX_ERR", "SAP_FLUX", "SAP_BKG", "QUALITY", "MOM_CENTR1", "MOM_CENTR2", "POS_CORR1", "POS_CORR2"];
  const META0 = ["OBJECT", "TICID", "SECTOR", "CAMERA", "CCD", "TESSMAG", "TEFF", "LOGG", "MH", "RADIUS", "RA_OBJ", "DEC_OBJ", "PMRA", "PMDEC"];
  const META1 = ["CROWDSAP", "FLFRCSAP", "PDCVAR", "PDCMETHD", "CDPP0_5", "CDPP1_0", "CDPP2_0", "EXPOSURE", "TIMEDEL"];
  const items = [], chunks = []; let offset = 0;
  for (const f of [...new Map(files.map(f => [f.url, f])).values()]) {
    const p = parseFits(await (await fetch("/api/v0.1/Download/file?uri=" + f.url)).arrayBuffer());
    const meta = {}; META0.forEach(k => meta[k] = p.h0[k]); META1.forEach(k => meta[k] = p.h1[k]);
    const arrs = {};
    for (const c of COLS) {
      const a = p.get(c); if (!a) continue;
      const dt = a instanceof Float64Array ? "f8" : a instanceof Float32Array ? "f4" : "i4", pad = (8 - offset % 8) % 8;
      if (pad) { chunks.push(new Uint8Array(pad)); offset += pad; }
      arrs[c] = [offset, a.length, dt]; chunks.push(new Uint8Array(a.buffer)); offset += a.byteLength;
    }
    items.push({tic: f.tic, sector: f.sector, url: f.url, meta, arrs});
  }
  const hdr = new TextEncoder().encode(JSON.stringify({created: new Date().toISOString(), items}));
  const pre = new ArrayBuffer(8), pv = new DataView(pre); pv.setUint32(0, 0x424f5845, true); pv.setUint32(4, hdr.length, true);
  const blob = new Blob([pre, hdr, new Uint8Array((8 - (8 + hdr.length) % 8) % 8), ...chunks]);
  const gz = new Uint8Array(await new Response(blob.stream().pipeThrough(new CompressionStream("gzip"))).arrayBuffer());
  let bin = ""; for (let i = 0; i < gz.length; i += 0x8000) bin += String.fromCharCode.apply(null, gz.subarray(i, i + 0x8000));
  window._B64 = window._B64 || {}; window._B64[name] = btoa(bin);
  const dig = await crypto.subtle.digest("SHA-256", gz);
  window._SHA = window._SHA || {}; window._SHA[name] = [...new Uint8Array(dig)].map(b => b.toString(16).padStart(2, "0")).join("");
  return {name, files: items.length, sectors: items.map(i => i.tic + ":" + i.sector).join(","), gzBytes: gz.length, b64: window._B64[name].length, sha256: window._SHA[name]};
}
// Show chunk k of size n of bundle `name` as the only page text, framed by markers.
function showChunk(name, k, n) {
  const s = window._B64[name].slice(k * n, (k + 1) * n);
  document.body.innerHTML = '<pre id="xout">@@' + name + '|' + k + '|' + s.length + '@@' + s + '@@END@@</pre>';
  return s.length;
}
