(() => {
  const NOV = /*NOVELTY_DATA*/;
  const $ = (id) => document.getElementById(id);
  const N = 24, ONSET = 8, H0 = 33.44, HW = (48.0 - 33.44) / N;
  const DESC = {
    real: "The validation and test periods exactly as recorded: 113,491 transactions and 131 frauds in 24 chunks of about 36 minutes. Each chunk is scored before its labels exist; labels arrive one chunk late.",
    harmless: "From chunk 8, the 10 features the model relies on least (counts, gaps, V2, V15, V27…) shift by +2 standard deviations, as after an upstream logging change. Any alarm here is a false alarm.",
    harmful: "From chunk 8, the 3 features the model relies on most (V14, V4, V12) shift by +3 standard deviations, as after a feature-pipeline or unit change. The model degrades and should retrain.",
    flip: "From chunk 8, the sign of V14, V4 and V12 flips. Fraud becomes invisible to the model, while the unlabelled data still looks normal to both drift monitors.",
  };
  let scen = "harmful", cur = ONSET - 1, timer = null;

  const hours = (c) => `hours ${(H0 + c * HW).toFixed(1)}–${(H0 + (c + 1) * HW).toFixed(1)}`;
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const smooth = (a) => a.map((_, i) => { const w = a.slice(Math.max(0, i - 2), i + 1).filter((v) => v !== null); return w.length ? w.reduce((s, v) => s + v, 0) / w.length : null; });
  const sum = (a, n) => a.slice(0, n + 1).reduce((s, v) => s + v, 0);

  function chart(svg, series, yMax, opts = {}) {
    const W = 420, H = 230, L = 36, R = 10, T = 10, B = 26, x = (i) => L + (i / (N - 1)) * (W - L - R), y = (v) => T + (1 - v / yMax) * (H - T - B);
    let s = "";
    for (let k = 0; k <= 4; k++) { const v = (yMax * k) / 4; s += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="var(--line)"/><text x="${L - 6}" y="${y(v) + 3}" text-anchor="end">${v.toFixed(yMax > 1.5 ? 1 : 2)}</text>`; }
    for (let c = 0; c < N; c += 4) s += `<text x="${x(c)}" y="${H - 8}" text-anchor="middle">${c}</text>`;
    if (scen !== "real") s += `<line x1="${x(ONSET - 0.5)}" x2="${x(ONSET - 0.5)}" y1="${T}" y2="${H - B}" stroke="var(--muted)" stroke-dasharray="2 3"/><text x="${x(ONSET - 0.5) + 4}" y="${T + 10}" text-anchor="start">shift starts</text>`;
    if (opts.alarmLine) s += `<line x1="${L}" x2="${W - R}" y1="${y(1)}" y2="${y(1)}" stroke="var(--ink)" stroke-dasharray="5 4"/><text x="${W - R}" y="${y(1) - 4}" text-anchor="end">alarm threshold</text>`;
    s += `<rect x="${x(cur) - 6}" y="${T}" width="12" height="${H - T - B}" fill="var(--accent-soft)" opacity=".7"/>`;
    for (const se of series) {
      let d = "", pen = false;
      se.v.forEach((v, i) => { if (i > cur || v === null) { pen = false; return; } const yy = y(Math.min(v, yMax)); d += `${pen ? "L" : "M"}${x(i).toFixed(1)},${yy.toFixed(1)}`; pen = true; });
      s += `<path d="${d}" fill="none" stroke="${se.c}" stroke-width="${se.w || 2}" ${se.dash ? `stroke-dasharray="${se.dash}"` : ""}/>`;
      const last = se.v.slice(0, cur + 1).map((v, i) => [v, i]).filter(([v]) => v !== null).pop();
      if (last) s += `<circle cx="${x(last[1])}" cy="${y(Math.min(last[0], yMax))}" r="3.2" fill="${se.c}"/>`;
      (se.rt || []).forEach((r, i) => { if (r && i <= cur) { const xx = x(i + 0.5); s += `<path d="M${xx - 4},${H - B} L${xx + 4},${H - B} L${xx},${H - B - 7}Z" fill="${se.c}"/>`; } });
    }
    svg.innerHTML = s;
  }

  function render() {
    const D = NOV[scen], f = D.frozen, p = D.psi, q = D.prop;
    $("cur").value = cur; $("curV").textContent = `${cur} of 23 · ${hours(cur)}`;
    $("scenDesc").textContent = DESC[scen];
    document.querySelectorAll(".segb").forEach((b) => b.setAttribute("aria-selected", b.dataset.s === scen));
    const tot = sum(f.fraud, cur);
    const lane = (id, rid, s) => { $(id).textContent = `${sum(s.hit, cur)} / ${tot}`; const r = sum(s.rt, cur); $(rid).textContent = `frauds caught · ${r} retrain${r === 1 ? "" : "s"}`; };
    lane("cFrozen", "rFrozen", f); lane("cPsi", "rPsi", p); lane("cProp", "rProp", q);
    chart($("qChart"), [
      { v: smooth(f.ap), c: css("--muted"), w: 2 },
      { v: smooth(p.ap), c: css("--accent"), w: 2, dash: "5 3", rt: p.rt },
      { v: smooth(q.ap), c: css("--prop"), w: 2.6, rt: q.rt },
    ], 1);
    chart($("dChart"), [
      { v: f.pd, c: css("--accent"), w: 2, dash: "5 3" },
      { v: f.xd, c: css("--prop"), w: 2.6 },
    ], 2, { alarmLine: true });

    const ev = [];
    for (let c = 0; c <= cur; c++) {
      if (scen !== "real" && c === ONSET) ev.push(["shift", c, "The injected shift starts."]);
      if (p.pa[c]) ev.push(["psi", c, `PSI input-drift alarm.${scen === "harmless" ? " This is a false alarm: only ignored fields changed." : ""}`]);
      if (q.xa[c]) ev.push(["prop", c, "SHAP explanation-drift alarm. Adaptation window opens."]);
      if (q.ga[c]) ev.push(["prop", c, `Label guard alarm: labels for chunk ${c - 1} show the model caught only ${q.hit[c - 1]} of ${f.fraud[c - 1]} frauds.`]);
      if (q.rt[c]) ev.push(["prop", c, `Proposed model retrains on all labels released so far (up to chunk ${c - 1}).`]);
      if (p.rt[c]) ev.push(["psi", c, "PSI strategy retrains."]);
    }
    if (!ev.length) ev.push(["none", cur, scen === "real" ? "No alarms. Both monitors stay below threshold, so no compute is spent on retraining." : "Waiting for the shift."]);
    $("log").innerHTML = ev.slice(-7).reverse().map(([k, c, t]) => `<li class="ev ${k}"><span class="mono">#${c}</span>${t}</li>`).join("");

    const A = NOV.attr[scen], tot2 = A.reduce((s, [, v]) => s + v, 0), mx = Math.max(...A.map(([, v]) => v));
    const fired = scen === "harmful";
    $("attrNote").textContent = scen === "real" ? "Explanation drift on the real stream, first chunk of the period. The biggest mover is hour_sin, the day/night cycle. It is natural and stays below the threshold."
      : scen === "harmless" ? "After the harmless shift the SHAP score barely moves (the model ignores these fields), so no alarm. PSI fires anyway."
      : scen === "harmful" ? "First chunk after the harmful shift: V4, V14 and V12 carry most of the drift score. The alarm points at the features that changed."
      : "After the flip, the attribution distributions look almost the same, so the SHAP monitor stays quiet. The label guard catches it once labels arrive.";
    $("attrBars").innerHTML = (cur >= ONSET || scen === "real" ? A : []).map(([n, v]) => `<div class="bar"><span class="mono">${n}</span><div class="track"><div class="fill ${fired ? "pos" : ""}" style="left:0;width:${(v / mx) * 100}%"></div></div><span class="v">${v.toFixed(3)}</span></div>`).join("") || `<p class="note">Play the stream past chunk ${ONSET} to see the attribution.</p>`;
  }

  function stop() { clearInterval(timer); timer = null; $("play").textContent = "▶ Play stream"; }
  $("play").onclick = () => {
    if (timer) return stop();
    if (cur >= N - 1) cur = -1;
    $("play").textContent = "❚❚ Pause";
    const tick = () => { if (cur >= N - 1) return stop(); cur++; render(); };
    tick(); timer = setInterval(tick, matchMedia("(prefers-reduced-motion: reduce)").matches ? 900 : 650);
  };
  $("stepB").onclick = () => { stop(); cur = Math.min(N - 1, cur + 1); render(); };
  $("cur").oninput = (e) => { stop(); cur = +e.target.value; render(); };
  document.querySelectorAll(".segb").forEach((b) => (b.onclick = () => { stop(); scen = b.dataset.s; cur = scen === "real" ? N - 1 : ONSET - 1; render(); }));
  matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", render);
  new MutationObserver(render).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

  const T = NOV.table, scs = ["real", "harmless", "harmful", "flip"];
  const strategies = [...new Set(T.map((r) => r.strategy))];
  const cell = (s, sc) => { const r = T.find((t) => t.strategy === s && t.scenario === sc); return r ? `${r.post_onset_pr_auc.toFixed(3)} <span class="note">(${(+r.retrains.toFixed(1)).toString()})</span>` : "<span class='note'>n/a</span>"; };
  $("novTable").innerHTML = `<thead><tr><th>Strategy</th>${scs.map((s) => `<th class="r">${s}</th>`).join("")}</tr></thead><tbody>${strategies.map((s) => `<tr class="${s.includes("label guard") ? "best" : ""}"><td>${s}</td>${scs.map((sc) => `<td class="num">${cell(s, sc)}</td>`).join("")}</tr>`).join("")}</tbody>`;
  render();
})();
