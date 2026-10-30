// Hero and background scene (three.js): a banknote-style guilloche rosette built from the transaction stream.
// Each engraved thread is a path; particles running along it are transactions (vermilion = fraud). A sweeping
// arm is the model: fraud it catches is pulled to the centre. The novelty replay drives the scene through
// window "stream:chunk" events: drift warps the threads, the silent flip hides fraud, a retrain re-engraves them.
(() => {
  const canvas = document.getElementById("scene");
  if (!canvas || !window.THREE) return;
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let renderer;
  try { renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true }); } catch (e) { canvas.remove(); return; }
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
  camera.position.set(0, 0, 9.5);
  const root = new THREE.Group(); scene.add(root);
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const col = (v) => new THREE.Color(css(v) || "#888");

  // ---------------------------------------------------------------- engraved threads
  const THREADS = 34, SEG = 420;
  const threads = [];
  const shape = (i, t, warp, time) => {             // guilloche: base circle + two harmonics, phase per thread
    const ph = (i / THREADS) * Math.PI * 2;
    const r = 2.25 + 0.42 * Math.sin(7 * t + ph) + 0.18 * Math.cos(13 * t - 2 * ph)
      + warp * 0.55 * Math.sin(3 * t + time * 0.9 + i * 0.4) * (i % 3 === 0 ? 1.0 : 0.55);
    const z = (i / THREADS - 0.5) * 1.1 + 0.25 * Math.sin(5 * t + ph) * (1 + warp);
    return [r * Math.cos(t), r * Math.sin(t), z];
  };
  for (let i = 0; i < THREADS; i++) {
    const pos = new Float32Array((SEG + 1) * 3);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    const m = new THREE.LineBasicMaterial({ transparent: true, opacity: 0.0 });
    const line = new THREE.Line(g, m);
    root.add(line);
    threads.push({ g, m, pos, reveal: 0 });
  }

  // ---------------------------------------------------------------- transactions
  const N = reduce ? 900 : 2600, FRAUD_EVERY = 46;
  const pPos = new Float32Array(N * 3), pCol = new Float32Array(N * 3), pSize = new Float32Array(N);
  const tx = [];
  for (let k = 0; k < N; k++) {
    const fraud = k % FRAUD_EVERY === 0;
    tx.push({ i: (Math.random() * THREADS) | 0, t: Math.random() * Math.PI * 2, v: 0.05 + Math.random() * 0.09, fraud, caught: 0, cx: 0, cy: 0, cz: 0 });
    pSize[k] = fraud ? 13 : 4.2 + Math.random() * 2.5;
  }
  const pg = new THREE.BufferGeometry();
  pg.setAttribute("position", new THREE.BufferAttribute(pPos, 3));
  pg.setAttribute("color", new THREE.BufferAttribute(pCol, 3));
  pg.setAttribute("size", new THREE.BufferAttribute(pSize, 1));
  const pm = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false,
    uniforms: { scale: { value: 1 }, alpha: { value: 0 } },
    vertexShader: `attribute float size; attribute vec3 color; varying vec3 vC; uniform float scale;
      void main(){ vC = color; vec4 mv = modelViewMatrix * vec4(position,1.0); gl_PointSize = size * scale / -mv.z; gl_Position = projectionMatrix * mv; }`,
    fragmentShader: `varying vec3 vC; uniform float alpha;
      void main(){ vec2 d = gl_PointCoord - .5; float r = length(d); if (r > .5) discard;
        float core = smoothstep(.5, .12, r); gl_FragColor = vec4(vC, core * alpha); }`,
  });
  root.add(new THREE.Points(pg, pm));

  // ---------------------------------------------------------------- the model: a sweeping arm
  const armG = new THREE.BufferGeometry();
  armG.setAttribute("position", new THREE.BufferAttribute(new Float32Array([0, 0, 0, 3.4, 0, 0]), 3));
  const armM = new THREE.LineBasicMaterial({ transparent: true, opacity: 0 });
  const arm = new THREE.Line(armG, armM); root.add(arm);
  const ringG = new THREE.RingGeometry(0.18, 0.2, 64);
  const ringM = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, side: THREE.DoubleSide });
  const ring = new THREE.Mesh(ringG, ringM); root.add(ring);
  // A pulse that runs outwards when the model retrains
  const pulseM = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, side: THREE.DoubleSide });
  const pulse = new THREE.Mesh(new THREE.RingGeometry(0.98, 1.0, 128), pulseM); root.add(pulse);

  // ---------------------------------------------------------------- state driven by the page
  const S = { focus: 0, warp: 0, warpT: 0, hide: 0, hideT: 0, pulse: 1, intro: 0, sweep: 0, scroll: 0, mx: 0, my: 0 };
  addEventListener("stream:chunk", (e) => {
    const d = e.detail;
    const shifted = d.scen !== "real" && d.cur >= d.onset;
    S.warpT = shifted && d.scen === "harmful" ? (d.adapted ? 0.35 : 1) : shifted && d.scen === "harmless" ? 0.25 : 0;
    S.hideT = shifted && d.scen === "flip" && !d.adapted ? 1 : 0;
    if (d.retrained) S.pulse = 0;
    if (d.live) S.focus = 1;
  });
  addEventListener("pointermove", (e) => { S.mx = e.clientX / innerWidth - 0.5; S.my = e.clientY / innerHeight - 0.5; }, { passive: true });
  addEventListener("scroll", () => { S.scroll = Math.min(scrollY / innerHeight, 1); }, { passive: true });

  let palette;
  function readPalette() {
    const hsl = {}; col("--paper").getHSL(hsl);
    palette = { boost: hsl.l < 0.3 ? 2.2 : 1, line: col("--engrave"), legit: col("--engrave"), fraud: col("--fraud"), model: col("--plum"), ink: col("--ink") };
    threads.forEach((th) => th.m.color = palette.line);
    armM.color = palette.model; ringM.color = palette.model; pulseM.color = palette.model;
  }
  readPalette();
  matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", readPalette);
  new MutationObserver(readPalette).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h; camera.position.z = w < h ? 16 : 9.5; camera.updateProjectionMatrix();
    pm.uniforms.scale.value = h * Math.min(devicePixelRatio || 1, 2) * 0.0065;
  }
  addEventListener("resize", resize); resize();

  let last = performance.now(), time = 0, visible = true;
  new IntersectionObserver(([en]) => (visible = en.isIntersecting)).observe(canvas);
  document.addEventListener("visibilitychange", () => { if (!document.hidden) last = performance.now(); });

  function frame(now) {
    requestAnimationFrame(frame);
    if (!visible || document.hidden) return;
    const dt = Math.min((now - last) / 1000, 0.05); last = now; time += dt;
    const k = 1 - Math.pow(0.02, dt);
    S.intro = Math.min(1, S.intro + dt / (reduce ? 0.01 : 2.4));
    S.warp += (S.warpT - S.warp) * k; S.hide += (S.hideT - S.hide) * k;
    S.pulse = Math.min(1, S.pulse + dt / 1.6);
    S.focus = Math.max(0, S.focus - dt / 5);
    canvas.style.opacity = Math.max(1 - 0.4 * S.scroll, 0.6 + 0.4 * S.focus).toFixed(3);
    const ease = 1 - Math.pow(1 - S.intro, 3);

    // threads: engrave in one after another, then breathe
    threads.forEach((th, i) => {
      const local = Math.max(0, Math.min(1, ease * 1.6 - i / THREADS * 0.6));
      const n = Math.max(2, Math.floor(local * SEG) + 1);
      for (let s = 0; s <= SEG; s++) {
        const p = shape(i, (s / SEG) * Math.PI * 2, S.warp, time);
        th.pos[s * 3] = p[0]; th.pos[s * 3 + 1] = p[1]; th.pos[s * 3 + 2] = p[2];
      }
      th.g.attributes.position.needsUpdate = true;
      th.g.setDrawRange(0, n);
      th.m.opacity = Math.min(1, (0.16 + 0.22 * (i % 4 === 0)) * palette.boost) * local;
    });

    // model arm
    S.sweep = (S.sweep + dt * (reduce ? 0.2 : 0.75)) % (Math.PI * 2);
    arm.rotation.z = S.sweep; armM.opacity = Math.min(1, 0.55 * palette.boost) * ease;
    ringM.opacity = 0.7 * ease; ring.rotation.z = -time;
    pulse.scale.setScalar(0.2 + S.pulse * 3.6); pulseM.opacity = (1 - S.pulse) * 0.6;

    // transactions
    const legit = palette.legit, fraud = palette.fraud;
    for (let q = 0; q < N; q++) {
      const o = tx[q];
      if (o.caught > 0) {                               // caught fraud drifts to the centre, then re-enters
        o.caught += dt;
        const a = Math.min(o.caught / 1.4, 1), e2 = a * a * (3 - 2 * a);
        pPos[q * 3] = o.cx * (1 - e2); pPos[q * 3 + 1] = o.cy * (1 - e2); pPos[q * 3 + 2] = o.cz * (1 - e2);
        if (a >= 1) { o.caught = 0; o.t = Math.random() * Math.PI * 2; o.i = (Math.random() * THREADS) | 0; }
      } else {
        o.t += o.v * dt * (1 + S.warp * 0.6);
        const p = shape(o.i, o.t, S.warp, time);
        pPos[q * 3] = p[0]; pPos[q * 3 + 1] = p[1]; pPos[q * 3 + 2] = p[2];
        if (o.fraud && S.hide < 0.5) {                  // the model catches fraud as its arm passes
          const ang = ((Math.atan2(p[1], p[0]) - S.sweep) % (Math.PI * 2) + Math.PI * 2) % (Math.PI * 2);
          if (ang < 0.05 && Math.random() < 0.85 - S.warp * 0.5) { o.caught = 1e-4; o.cx = p[0]; o.cy = p[1]; o.cz = p[2]; }
        }
      }
      const f = o.fraud ? 1 - S.hide : 0;               // silent flip: fraud looks like everything else
      const c = f > 0 ? fraud : legit;
      pCol[q * 3] = legit.r + (c.r - legit.r) * f; pCol[q * 3 + 1] = legit.g + (c.g - legit.g) * f; pCol[q * 3 + 2] = legit.b + (c.b - legit.b) * f;
      pSize[q] = o.fraud ? 4.5 + 9 * f : pSize[q];
    }
    pg.attributes.position.needsUpdate = true; pg.attributes.color.needsUpdate = true; pg.attributes.size.needsUpdate = true;
    pm.uniforms.alpha.value = 0.9 * ease;

    // camera: tilt with the pointer, slide aside as the page scrolls
    root.rotation.x += ((-0.55 + S.my * 0.25) - root.rotation.x) * k;
    root.rotation.y += ((S.mx * 0.35 + S.scroll * 0.5) - root.rotation.y) * k;
    root.rotation.z += dt * 0.025;
    const wide = innerWidth > 860;
    root.position.x = (wide ? 1.9 : 0) - S.scroll * (wide ? 1.9 : 0);
    root.position.y = wide ? 0 : 2.2;
    renderer.render(scene, camera);
  }
  requestAnimationFrame(frame);
})();
