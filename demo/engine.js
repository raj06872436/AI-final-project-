// Fraud detection engine: synthetic transactions, leakage-free features,
// chronological vs random splits, logistic regression, metrics.
const Engine = (() => {
  function rng(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(r) {
    let u = 0; while (u === 0) u = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r());
  }

  const STATIC = ["log_amount", "V14", "V4", "V12"];
  const TEMPORAL = ["log_gap_prev", "txn_count_1h", "amount_zscore"];

  // Generate a 48-hour transaction stream. Temporal features use ONLY past
  // transactions of the same card (no look-ahead leakage).
  function generate(seed, n, fraudRate, drift) {
    const r = rng(seed), C = 1200, SPAN = 172800;
    const cards = Array.from({ length: C }, () => ({ mu: 3 + gauss(r) * 0.8, times: [], hist: [] }));
    const rows = []; const bursts = []; let t = 0;
    while (rows.length < n) {
      t += -Math.log(1 - r()) * (SPAN / n);
      let card, y;
      if (bursts.length && bursts[0].at <= t) { card = bursts.shift().card; y = 1; }
      else {
        card = Math.floor(r() * C); y = r() < fraudRate ? 1 : 0;
        if (y && r() < 0.6) {
          const k = 1 + Math.floor(r() * 2);
          for (let j = 0; j < k; j++) bursts.push({ card, at: t + 30 + r() * 600 });
          bursts.sort((a, b) => a.at - b.at);
        }
      }
      const c = cards[card], progress = t / SPAN;
      const logAmt = y ? c.mu + 1.0 + gauss(r) * 0.8 : c.mu + gauss(r) * 0.5;
      const sep = 2.4 * (1 - drift * progress);      // fraud signature fades: concept drift
      const v14 = gauss(r) - (y ? sep : 0);
      const v4 = gauss(r) + (y ? 0.9 : 0);
      const v12 = gauss(r) - (y ? 0.5 : 0);
      const last = c.times.length ? c.times[c.times.length - 1] : t - 86400;
      let cnt = 0; for (let i = c.times.length - 1; i >= 0 && t - c.times[i] <= 3600; i--) cnt++;
      const h = c.hist.slice(-10);
      let z = 0;
      if (h.length >= 2) {
        const m = h.reduce((a, b) => a + b, 0) / h.length;
        const s = Math.sqrt(h.reduce((a, b) => a + (b - m) ** 2, 0) / h.length);
        z = (logAmt - m) / Math.max(s, 0.3);
      }
      rows.push({ id: rows.length, t, card, y, x: [logAmt, v14, v4, v12, Math.log1p(t - last), cnt, z] });
      c.times.push(t); c.hist.push(logAmt);
    }
    return rows;
  }

  function split(rows, mode, seed) {
    let idx = rows.map((_, i) => i);
    if (mode === "random") {
      const r = rng(seed + 99);
      for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    }
    const a = Math.floor(idx.length * 0.6), b = Math.floor(idx.length * 0.8);
    const pick = (s) => s.map((i) => rows[i]);
    return { train: pick(idx.slice(0, a)), val: pick(idx.slice(a, b)), test: pick(idx.slice(b)) };
  }

  // Logistic regression with balanced class weights; scaler fit on TRAIN only.
  function trainLR(train, cols, iters = 300, lr = 0.3, l2 = 1e-3) {
    const d = cols.length, n = train.length;
    const mean = Array(d).fill(0), sd = Array(d).fill(0);
    for (const r of train) cols.forEach((c, j) => (mean[j] += r.x[c] / n));
    for (const r of train) cols.forEach((c, j) => (sd[j] += (r.x[c] - mean[j]) ** 2 / n));
    for (let j = 0; j < d; j++) sd[j] = Math.sqrt(sd[j]) || 1;
    const X = train.map((r) => cols.map((c, j) => (r.x[c] - mean[j]) / sd[j]));
    const Y = train.map((r) => r.y);
    const pos = Y.reduce((a, b) => a + b, 0), wPos = n / (2 * pos), wNeg = n / (2 * (n - pos));
    const w = Array(d).fill(0); let b = 0;
    for (let it = 0; it < iters; it++) {
      const g = Array(d).fill(0); let gb = 0, W = 0;
      for (let i = 0; i < n; i++) {
        let s = b; for (let j = 0; j < d; j++) s += w[j] * X[i][j];
        const p = 1 / (1 + Math.exp(-s)), wt = Y[i] ? wPos : wNeg, e = (p - Y[i]) * wt;
        for (let j = 0; j < d; j++) g[j] += e * X[i][j];
        gb += e; W += wt;
      }
      for (let j = 0; j < d; j++) w[j] -= lr * (g[j] / W + l2 * w[j]);
      b -= lr * gb / W;
    }
    const z = (r) => cols.map((c, j) => (r.x[c] - mean[j]) / sd[j]);
    const predict = (r) => { const v = z(r); let s = b; for (let j = 0; j < d; j++) s += w[j] * v[j]; return 1 / (1 + Math.exp(-s)); };
    const explain = (r) => { const v = z(r); return cols.map((c, j) => w[j] * v[j]); };
    return { w, b, cols, predict, explain };
  }

  function curves(scores, labels) {
    const o = scores.map((s, i) => [s, labels[i]]).sort((a, b) => b[0] - a[0]);
    const P = labels.reduce((a, b) => a + b, 0), N = labels.length - P;
    let tp = 0, fp = 0, ap = 0; const pr = [{ r: 0, p: 1 }];
    for (let i = 0; i < o.length; i++) {
      if (o[i][1]) { tp++; ap += tp / (tp + fp); } else fp++;
      if (i === o.length - 1 || o[i + 1][0] !== o[i][0]) pr.push({ r: tp / P, p: tp / (tp + fp) });
    }
    // ROC-AUC (Mann-Whitney)
    const asc = o.slice().reverse(); let rankSum = 0;
    for (let i = 0; i < asc.length;) {
      let j = i; while (j < asc.length && asc[j][0] === asc[i][0]) j++;
      const avg = (i + j + 1) / 2; for (let k = i; k < j; k++) if (asc[k][1]) rankSum += avg; i = j;
    }
    const roc = (rankSum - (P * (P + 1)) / 2) / (P * N);
    return { prauc: ap / P, rocauc: roc, pr };
  }

  function confusion(scores, labels, th) {
    let tp = 0, fp = 0, tn = 0, fn = 0;
    scores.forEach((s, i) => { const p = s >= th; if (p && labels[i]) tp++; else if (p) fp++; else if (labels[i]) fn++; else tn++; });
    const precision = tp + fp ? tp / (tp + fp) : 0, recall = tp + fn ? tp / (tp + fn) : 0;
    const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
    return { tp, fp, tn, fn, precision, recall, f1, fpr: fp / (fp + tn || 1), accuracy: (tp + tn) / scores.length };
  }

  function bestThreshold(scores, labels) {
    let best = { th: 0.5, f1: -1 };
    for (let th = 0.02; th < 1; th += 0.01) { const c = confusion(scores, labels, th); if (c.f1 > best.f1) best = { th, f1: c.f1 }; }
    return best.th;
  }

  // Full experiment: model fit on train, threshold on validation, single test evaluation.
  function experiment(rows, mode, cols, seed) {
    const s = split(rows, mode, seed);
    const m = trainLR(s.train, cols);
    const vs = s.val.map(m.predict), vy = s.val.map((r) => r.y);
    const th = bestThreshold(vs, vy);
    const ts = s.test.map(m.predict), ty = s.test.map((r) => r.y);
    return { model: m, split: s, th, scores: ts, labels: ty, ...curves(ts, ty), cm: confusion(ts, ty, th) };
  }

  return { generate, experiment, curves, confusion, STATIC, TEMPORAL, FEATURES: STATIC.concat(TEMPORAL) };
})();
if (typeof module !== "undefined") module.exports = Engine;
