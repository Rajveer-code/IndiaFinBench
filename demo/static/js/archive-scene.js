/* ════════════════════════════════════════════════════════════════════
   IndiaFinBench — archive-scene.js
   "The Archive Assembles": 192 regulatory documents rendered as ruled
   paper cards in WebGL. Scroll morphs the field through three states —
   drifting cloud (hero) → tangled supersession chains (§01) → ordered
   archive wall, SEBI block then RBI block (§02) — then dissolves.
   Raw WebGL + GLSL, no dependencies. Degrades silently without WebGL.
   ════════════════════════════════════════════════════════════════════ */
(function () {
  'use strict';

  var canvas = document.getElementById('archiveCanvas');
  if (!canvas) return;
  var gl = canvas.getContext('webgl', { alpha: true, antialias: true, premultipliedAlpha: false });
  if (!gl) { canvas.remove(); return; }

  var REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var N_DOCS = 192, N_SEBI = 92;
  var N_MOTES = window.innerWidth < 700 ? 110 : 240;

  /* ── Shaders ──────────────────────────────────────────────────────── */
  var DOC_VS = [
    'attribute vec3 aCloud;',
    'attribute vec3 aTangle;',
    'attribute vec3 aGrid;',
    'attribute vec2 aCorner;',
    'attribute vec2 aMeta;', // x: kind (0 SEBI / 1 RBI), y: seed
    'uniform mat4 uProj;',
    'uniform mat4 uView;',
    'uniform float uT1;',
    'uniform float uT2;',
    'uniform float uTime;',
    'uniform float uSize;',
    'varying vec2 vUv;',
    'varying float vKind;',
    'varying float vFade;',
    'void main(){',
    '  vec3 p = mix(aCloud, aTangle, uT1);',
    '  p = mix(p, aGrid, uT2);',
    '  float s = aMeta.y * 43.7;',
    '  float calm = 1.0 - uT2 * 0.85;',
    '  p += vec3(sin(uTime*0.40+s), cos(uTime*0.31+s*1.7), sin(uTime*0.23+s*2.3)) * 0.09 * calm;',
    '  vec4 mv = uView * vec4(p, 1.0);',
    '  mv.xy += aCorner * vec2(uSize, uSize*1.36);',
    '  gl_Position = uProj * mv;',
    '  vUv = aCorner * 0.5 + 0.5;',
    '  vKind = aMeta.x;',
    '  vFade = clamp(1.0 - (-mv.z - 4.0) / 16.0, 0.18, 1.0);',
    '}'
  ].join('\n');

  var DOC_FS = [
    'precision mediump float;',
    'varying vec2 vUv;',
    'varying float vKind;',
    'varying float vFade;',
    'uniform float uAlpha;',
    'void main(){',
    '  vec2 d = min(vUv, 1.0 - vUv);',
    '  float border = 1.0 - step(0.07, min(d.x, d.y));',
    // three ruled "text lines" inside the card
    '  float lines = 0.0;',
    '  if (vUv.x > 0.18 && vUv.x < 0.82 && vUv.y > 0.22 && vUv.y < 0.80) {',
    '    lines = step(fract(vUv.y * 4.6), 0.14);',
    '  }',
    '  vec3 ink   = vec3(0.110, 0.094, 0.071);',
    '  vec3 sebi  = vec3(0.122, 0.361, 0.271);',
    '  vec3 rbi   = vec3(0.639, 0.231, 0.125);',
    '  vec3 tint  = mix(sebi, rbi, vKind);',
    '  vec3 col   = mix(ink, tint, border);',
    '  float a = border * 0.62 + lines * 0.20 + 0.045;',
    '  gl_FragColor = vec4(col, a * uAlpha * vFade);',
    '}'
  ].join('\n');

  var LINE_VS = [
    'attribute vec3 aCloud;',
    'attribute vec3 aTangle;',
    'attribute vec3 aGrid;',
    'uniform mat4 uProj;',
    'uniform mat4 uView;',
    'uniform float uT1;',
    'uniform float uT2;',
    'void main(){',
    '  vec3 p = mix(aCloud, aTangle, uT1);',
    '  p = mix(p, aGrid, uT2);',
    '  gl_Position = uProj * uView * vec4(p, 1.0);',
    '}'
  ].join('\n');

  var LINE_FS = [
    'precision mediump float;',
    'uniform float uAlpha;',
    'void main(){ gl_FragColor = vec4(0.110, 0.094, 0.071, uAlpha); }'
  ].join('\n');

  var MOTE_VS = [
    'attribute vec3 aPos;',
    'attribute float aSeed;',
    'uniform mat4 uProj;',
    'uniform mat4 uView;',
    'uniform float uTime;',
    'void main(){',
    '  vec3 p = aPos;',
    '  float s = aSeed * 61.3;',
    '  p += vec3(sin(uTime*0.18+s), cos(uTime*0.14+s*1.3), 0.0) * 0.45;',
    '  vec4 mv = uView * vec4(p, 1.0);',
    '  gl_Position = uProj * mv;',
    '  gl_PointSize = clamp(36.0 / -mv.z, 1.0, 3.2);',
    '}'
  ].join('\n');

  var MOTE_FS = [
    'precision mediump float;',
    'uniform float uAlpha;',
    'void main(){ gl_FragColor = vec4(0.110, 0.094, 0.071, 0.13 * uAlpha); }'
  ].join('\n');

  function compile(type, src) {
    var sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(sh) || 'shader compile failed');
    }
    return sh;
  }
  function program(vs, fs) {
    var p = gl.createProgram();
    gl.attachShader(p, compile(gl.VERTEX_SHADER, vs));
    gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(p) || 'program link failed');
    }
    return p;
  }

  var docProg, lineProg, moteProg;
  try {
    docProg = program(DOC_VS, DOC_FS);
    lineProg = program(LINE_VS, LINE_FS);
    moteProg = program(MOTE_VS, MOTE_FS);
  } catch (e) { canvas.remove(); return; }

  /* ── Formations ───────────────────────────────────────────────────── */
  var rand = (function () { // deterministic, so the scene is identical every visit
    var s = 1337;
    return function () { s = (s * 16807) % 2147483647; return (s - 1) / 2147483646; };
  })();

  var cloud = new Float32Array(N_DOCS * 3);
  var tangle = new Float32Array(N_DOCS * 3);
  var grid = new Float32Array(N_DOCS * 3);

  // cloud: wide drifting ellipsoid behind the hero
  for (var i = 0; i < N_DOCS; i++) {
    var th = rand() * Math.PI * 2, ph = Math.acos(rand() * 2 - 1), r = Math.pow(rand(), 0.5);
    cloud[i * 3]     = Math.sin(ph) * Math.cos(th) * 8.2 * r;
    cloud[i * 3 + 1] = Math.cos(ph) * 4.0 * r;
    cloud[i * 3 + 2] = -9.0 + Math.sin(ph) * Math.sin(th) * 3.4 * r;
  }

  // tangle: 12 supersession chains — random walks knotting right of the text column
  var CHAINS = 12, perChain = Math.ceil(N_DOCS / CHAINS), idx = 0;
  for (var c = 0; c < CHAINS; c++) {
    var x = 2.6 + (rand() - 0.5) * 6.4, y = (rand() - 0.5) * 3.6, z = -9.0 + (rand() - 0.5) * 2.5;
    for (var k = 0; k < perChain && idx < N_DOCS; k++, idx++) {
      x += (rand() - 0.5) * 1.5 - (x - 2.6) * 0.07;
      y += (rand() - 0.5) * 1.1 - y * 0.07;
      z += (rand() - 0.5) * 0.8;
      tangle[idx * 3] = x; tangle[idx * 3 + 1] = y; tangle[idx * 3 + 2] = z;
    }
  }

  // grid: 16 × 12 archive wall — SEBI block fills first, RBI block after
  var COLS = 16, SX = 0.74, SY = 0.96;
  for (i = 0; i < N_DOCS; i++) {
    var col = i % COLS, row = Math.floor(i / COLS);
    grid[i * 3]     = (col - (COLS - 1) / 2) * SX + 1.2;
    grid[i * 3 + 1] = ((11 - row) - 5.5) * SY * 0.62;
    grid[i * 3 + 2] = -10.5 + (rand() - 0.5) * 0.3;
  }

  /* ── Doc quad buffers (6 verts per doc) ───────────────────────────── */
  var V = N_DOCS * 6;
  var bCloud = new Float32Array(V * 3), bTangle = new Float32Array(V * 3),
      bGrid = new Float32Array(V * 3), bCorner = new Float32Array(V * 2),
      bMeta = new Float32Array(V * 2);
  var CORNERS = [-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1];
  for (i = 0; i < N_DOCS; i++) {
    var kind = i < N_SEBI ? 0 : 1, seed = rand();
    for (var v = 0; v < 6; v++) {
      var o = i * 6 + v;
      bCloud.set([cloud[i * 3], cloud[i * 3 + 1], cloud[i * 3 + 2]], o * 3);
      bTangle.set([tangle[i * 3], tangle[i * 3 + 1], tangle[i * 3 + 2]], o * 3);
      bGrid.set([grid[i * 3], grid[i * 3 + 1], grid[i * 3 + 2]], o * 3);
      bCorner.set([CORNERS[v * 2], CORNERS[v * 2 + 1]], o * 2);
      bMeta.set([kind, seed], o * 2);
    }
  }

  /* ── Chain line buffers (consecutive docs within each chain) ──────── */
  var linePairs = [];
  idx = 0;
  for (c = 0; c < CHAINS; c++) {
    for (k = 0; k < perChain - 1 && idx + 1 < N_DOCS; k++, idx++) linePairs.push(idx, idx + 1);
    idx++;
  }
  var L = linePairs.length;
  var lCloud = new Float32Array(L * 3), lTangle = new Float32Array(L * 3), lGrid = new Float32Array(L * 3);
  for (i = 0; i < L; i++) {
    var di = linePairs[i];
    lCloud.set([cloud[di * 3], cloud[di * 3 + 1], cloud[di * 3 + 2]], i * 3);
    lTangle.set([tangle[di * 3], tangle[di * 3 + 1], tangle[di * 3 + 2]], i * 3);
    lGrid.set([grid[di * 3], grid[di * 3 + 1], grid[di * 3 + 2]], i * 3);
  }

  /* ── Motes ────────────────────────────────────────────────────────── */
  var mPos = new Float32Array(N_MOTES * 3), mSeed = new Float32Array(N_MOTES);
  for (i = 0; i < N_MOTES; i++) {
    mPos[i * 3] = (rand() - 0.5) * 18;
    mPos[i * 3 + 1] = (rand() - 0.5) * 9;
    mPos[i * 3 + 2] = -6 - rand() * 9;
    mSeed[i] = rand();
  }

  function buf(data) {
    var b = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    return b;
  }
  var docBufs = { cloud: buf(bCloud), tangle: buf(bTangle), grid: buf(bGrid), corner: buf(bCorner), meta: buf(bMeta) };
  var lineBufs = { cloud: buf(lCloud), tangle: buf(lTangle), grid: buf(lGrid) };
  var moteBufs = { pos: buf(mPos), seed: buf(mSeed) };

  /* ── Matrices ─────────────────────────────────────────────────────── */
  function perspective(fovy, aspect, near, far) {
    var f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
    return new Float32Array([
      f / aspect, 0, 0, 0,
      0, f, 0, 0,
      0, 0, (far + near) * nf, -1,
      0, 0, 2 * far * near * nf, 0
    ]);
  }
  function viewMatrix(rx, ry) {
    var cx = Math.cos(rx), sx = Math.sin(rx), cy = Math.cos(ry), sy = Math.sin(ry);
    // rotateX(rx) * rotateY(ry), column-major
    return new Float32Array([
      cy, sx * sy, -cx * sy, 0,
      0, cx, sx, 0,
      sy, -sx * cy, cx * cy, 0,
      0, 0, 0, 1
    ]);
  }

  /* ── Uniform/attribute lookups ────────────────────────────────────── */
  function locs(prog, attrs, unis) {
    var out = { a: {}, u: {} };
    attrs.forEach(function (n) { out.a[n] = gl.getAttribLocation(prog, n); });
    unis.forEach(function (n) { out.u[n] = gl.getUniformLocation(prog, n); });
    return out;
  }
  var docL = locs(docProg, ['aCloud', 'aTangle', 'aGrid', 'aCorner', 'aMeta'],
    ['uProj', 'uView', 'uT1', 'uT2', 'uTime', 'uSize', 'uAlpha']);
  var lineL = locs(lineProg, ['aCloud', 'aTangle', 'aGrid'], ['uProj', 'uView', 'uT1', 'uT2', 'uAlpha']);
  var moteL = locs(moteProg, ['aPos', 'aSeed'], ['uProj', 'uView', 'uTime', 'uAlpha']);

  function attrib(loc, b, size) {
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
  }

  /* ── State ────────────────────────────────────────────────────────── */
  var proj, W = 0, H = 0;
  var mouseX = 0, mouseY = 0, rotX = 0, rotY = 0;
  var problemEl = document.getElementById('problem');
  var corpusEl = document.getElementById('corpus');

  function resize() {
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = window.innerWidth; H = window.innerHeight;
    canvas.width = W * dpr; canvas.height = H * dpr;
    gl.viewport(0, 0, canvas.width, canvas.height);
    proj = perspective(50 * Math.PI / 180, W / H, 0.1, 60);
  }
  resize();
  window.addEventListener('resize', resize);

  if (!REDUCED) {
    window.addEventListener('pointermove', function (e) {
      mouseX = (e.clientX / W) * 2 - 1;
      mouseY = (e.clientY / H) * 2 - 1;
    }, { passive: true });
  }

  function smooth(t) { return t * t * (3 - 2 * t); }
  function sectionT(el, span) {
    if (!el) return 0;
    var r = el.getBoundingClientRect();
    return smooth(Math.max(0, Math.min(1, (H - r.top) / (H * span))));
  }

  function sceneAlpha() {
    if (!corpusEl) return 1;
    var r = corpusEl.getBoundingClientRect();
    // fade out once the corpus section's bottom rises past 70% of the viewport
    var f = Math.max(0, Math.min(1, (H * 0.7 - r.bottom) / (H * 0.45)));
    return 1 - f;
  }

  /* ── Render ───────────────────────────────────────────────────────── */
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  gl.clearColor(0, 0, 0, 0);

  function draw(time, t1, t2, alpha) {
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (alpha <= 0.005) return;
    var view = viewMatrix(rotX, rotY);

    // motes
    gl.useProgram(moteProg);
    attrib(moteL.a.aPos, moteBufs.pos, 3);
    attrib(moteL.a.aSeed, moteBufs.seed, 1);
    gl.uniformMatrix4fv(moteL.u.uProj, false, proj);
    gl.uniformMatrix4fv(moteL.u.uView, false, view);
    gl.uniform1f(moteL.u.uTime, time);
    gl.uniform1f(moteL.u.uAlpha, alpha);
    gl.drawArrays(gl.POINTS, 0, N_MOTES);

    // chain lines: visible only inside the tangle phase
    var lineAlpha = alpha * t1 * (1 - t2) * 0.16;
    if (lineAlpha > 0.004) {
      gl.useProgram(lineProg);
      attrib(lineL.a.aCloud, lineBufs.cloud, 3);
      attrib(lineL.a.aTangle, lineBufs.tangle, 3);
      attrib(lineL.a.aGrid, lineBufs.grid, 3);
      gl.uniformMatrix4fv(lineL.u.uProj, false, proj);
      gl.uniformMatrix4fv(lineL.u.uView, false, view);
      gl.uniform1f(lineL.u.uT1, t1);
      gl.uniform1f(lineL.u.uT2, t2);
      gl.uniform1f(lineL.u.uAlpha, lineAlpha);
      gl.drawArrays(gl.LINES, 0, L);
    }

    // documents
    gl.useProgram(docProg);
    attrib(docL.a.aCloud, docBufs.cloud, 3);
    attrib(docL.a.aTangle, docBufs.tangle, 3);
    attrib(docL.a.aGrid, docBufs.grid, 3);
    attrib(docL.a.aCorner, docBufs.corner, 2);
    attrib(docL.a.aMeta, docBufs.meta, 2);
    gl.uniformMatrix4fv(docL.u.uProj, false, proj);
    gl.uniformMatrix4fv(docL.u.uView, false, view);
    gl.uniform1f(docL.u.uT1, t1);
    gl.uniform1f(docL.u.uT2, t2);
    gl.uniform1f(docL.u.uTime, time);
    var small = W < 700;
    gl.uniform1f(docL.u.uSize, (small ? 0.115 : 0.155) + t2 * 0.07);
    gl.uniform1f(docL.u.uAlpha, alpha * (small ? 0.68 : 1) * (1 - t2 * 0.25));
    gl.drawArrays(gl.TRIANGLES, 0, V);
  }

  if (REDUCED) {
    // single static frame: mid-cloud, no motion, no listeners beyond resize redraw
    var staticDraw = function () { draw(0, 0, 0, 0.9); };
    staticDraw();
    window.addEventListener('resize', staticDraw);
    return;
  }

  var running = false;
  function frame(now) {
    if (!running) return;
    var t = now * 0.001;
    rotY += ((mouseX * 0.10) - rotY) * 0.04;
    rotX += ((mouseY * 0.06) - rotX) * 0.04;
    var t1 = sectionT(problemEl, 0.95);
    var t2 = sectionT(corpusEl, 0.85);
    draw(t, t1, t2, sceneAlpha());
    requestAnimationFrame(frame);
  }
  function startLoop() {
    if (running) return;
    running = true;
    requestAnimationFrame(frame);
  }

  // one immediate frame so the scene exists even before the loop starts
  draw(0, sectionT(problemEl, 0.95), sectionT(corpusEl, 0.85), sceneAlpha());

  if (document.visibilityState === 'visible') startLoop();
  document.addEventListener('visibilitychange', function () {
    if (document.hidden) running = false;
    else startLoop();
  });
})();
