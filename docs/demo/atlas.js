// NYXCORE context atlas. Every recorded run is a column of particles: one particle
// per few tokens, height = how much of the context window the agent was asking the model to hold.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};
const pct = (v) => `${Math.round(v * 100)}%`;
const fmt = (v) => (v == null ? '–' : new Intl.NumberFormat('en-US').format(v));
const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

const MODEL_COLOURS = ['#5ad1ff', '#b693ff', '#ff9e5a', '#7dffb0', '#ff7eb6', '#d7e36b'];
const CANARY = new THREE.Color('#bdfc23');
const TRAP = new THREE.Color('#ff4fd8');
const DANGER = new THREE.Color('#ff4d5e');
const OK = new THREE.Color('#4fe3b0');
const TARGET_PARTICLES = 160000;

let atlas;
try {
  atlas = await fetch('./atlas.json').then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); });
} catch (err) {
  $('summary').textContent = 'No atlas.json found. Run: nyxcore atlas, then nyxcore serve.';
  throw err;
}
const { episodes, steps, models } = atlas;
const H = atlas.meta.height;
const windowTokens = atlas.meta.context_window;
const stepsOf = (e) => steps.slice(e.s0, e.s1);
const modelColour = (m) => new THREE.Color(MODEL_COLOURS[m % MODEL_COLOURS.length]);
const shortId = (id) => id.replace(/^.*\//, '');

const state = { sel: -1, selTime: 1, time: 1, playing: false, mode: 'model', hidden: new Set(), assemble: reducedMotion ? 1 : 0 };

// ---------------------------------------------------------------- scene
const canvas = $('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);
const camera = new THREE.PerspectiveCamera(48, 1, 0.1, 800);
const HOME = { pos: new THREE.Vector3(56, 17, 70), target: new THREE.Vector3(0, H * 0.46, 0) };
camera.position.copy(HOME.pos).multiplyScalar(reducedMotion ? 1 : 2.1);
const controls = new OrbitControls(camera, canvas);
controls.target.copy(HOME.target);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.maxDistance = 220;
controls.minDistance = 4;
controls.autoRotate = !reducedMotion;
controls.autoRotateSpeed = 0.35;

const uniforms = {
  uAssemble: { value: state.assemble }, uTime: { value: 1 }, uSel: { value: -1 }, uSelTime: { value: 1 },
  uPx: { value: 1 }, uH: { value: H }, uClock: { value: 0 },
};

const span = 2 * Math.max(30, ...episodes.map((e) => Math.max(Math.abs(e.x), Math.abs(e.z)))) + 16;
const floor = new THREE.GridHelper(span, 30, 0x1d2536, 0x10141f);
scene.add(floor);
const ceiling = new THREE.GridHelper(span, 30, 0xff4d5e, 0xff4d5e);
ceiling.position.y = H;
ceiling.material.transparent = true;
ceiling.material.opacity = 0.16;
ceiling.material.depthWrite = false;
scene.add(ceiling);
const ceilingGlow = new THREE.Mesh(
  new THREE.PlaneGeometry(span, span),
  new THREE.MeshBasicMaterial({ color: 0xff4d5e, transparent: true, opacity: 0.035, side: THREE.DoubleSide, depthWrite: false }),
);
ceilingGlow.rotation.x = -Math.PI / 2;
ceilingGlow.position.y = H;
scene.add(ceilingGlow);

// ---------------------------------------------------------------- particles
function mulberry32(a) {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const gauss = (rnd) => Math.sqrt(-2 * Math.log(rnd() || 1e-9)) * Math.cos(6.2832 * rnd());

const tokensAt = (s) => s.sent ?? s.pt ?? 0;
const totalTokens = episodes.reduce((sum, e) => sum + Math.max(...stepsOf(e).map(tokensAt), 1), 0);
const tokensPerParticle = Math.max(1.5, totalTokens / TARGET_PARTICLES);

const P = { pos: [], start: [], t: [], ep: [], size: [], step: [] };
episodes.forEach((e, ei) => {
  const rnd = mulberry32(ei * 7919 + 13);
  const list = stepsOf(e);
  const n = list.length;
  list.forEach((s, i) => {
    const a = i === 0 ? { x: e.x, y: 0, z: e.z, tok: 0 } : { ...list[i - 1], tok: tokensAt(list[i - 1]) };
    const count = Math.min(4000, Math.max(5, Math.round((tokensAt(s) - a.tok) / tokensPerParticle)));
    for (let j = 0; j < count; j++) {
      const u = rnd();
      const halo = rnd() < 0.16;
      const sigma = halo ? 1.7 : 0.55;
      P.pos.push(a.x + (s.x - a.x) * u + gauss(rnd) * sigma, Math.max(0.02, a.y + (s.y - a.y) * u + gauss(rnd) * 0.1), a.z + (s.z - a.z) * u + gauss(rnd) * sigma);
      const r = 70 + rnd() * 90, th = rnd() * 6.2832, ph = Math.acos(2 * rnd() - 1);
      P.start.push(r * Math.sin(ph) * Math.cos(th), 20 + r * Math.cos(ph) * 0.6, r * Math.sin(ph) * Math.sin(th));
      P.t.push(i === 0 || n < 2 ? 0 : (i - 1 + u) / (n - 1));
      P.ep.push(ei);
      P.size.push(halo ? 0.55 + rnd() * 0.3 : 0.75 + rnd() * 0.7);
      P.step.push(e.s0 + i);
    }
  });
});
const particleCount = P.t.length;
const colourArray = new Float32Array(particleCount * 3);
const cloudGeometry = new THREE.BufferGeometry();
cloudGeometry.setAttribute('position', new THREE.Float32BufferAttribute(P.pos, 3));
cloudGeometry.setAttribute('aStart', new THREE.Float32BufferAttribute(P.start, 3));
cloudGeometry.setAttribute('aT', new THREE.Float32BufferAttribute(P.t, 1));
cloudGeometry.setAttribute('aEp', new THREE.Float32BufferAttribute(P.ep, 1));
cloudGeometry.setAttribute('aSize', new THREE.Float32BufferAttribute(P.size, 1));
cloudGeometry.setAttribute('aColor', new THREE.BufferAttribute(colourArray, 3));
cloudGeometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, H / 2, 0), 400);

const cloud = new THREE.Points(cloudGeometry, new THREE.ShaderMaterial({
  uniforms, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  vertexShader: /* glsl */ `
    uniform float uAssemble, uTime, uSel, uSelTime, uPx, uH, uClock;
    attribute vec3 aStart, aColor;
    attribute float aT, aEp, aSize;
    varying vec3 vC;
    varying float vA;
    void main() {
      float k = clamp(uAssemble * 1.7 - aT * 0.35 - fract(aEp * 0.137) * 0.3, 0.0, 1.0);
      k = k * k * (3.0 - 2.0 * k);
      vec3 p = mix(aStart, position, k);
      p.xz += 0.03 * vec2(sin(uClock * 0.7 + position.y * 1.3 + aEp), cos(uClock * 0.6 + position.y * 1.1 + aEp * 1.7));
      bool sel = abs(aEp - uSel) < 0.5;
      float limit = sel ? uSelTime : uTime;
      float vis = 1.0 - smoothstep(limit, limit + 0.012, aT);
      float focus = (uSel < -0.5 || sel) ? 1.0 : 0.22;
      float edge = sel ? smoothstep(limit - 0.08, limit, aT) * vis : 0.0;
      vC = mix(aColor, vec3(1.0, 0.27, 0.33), smoothstep(uH, uH + 1.0, position.y) * 0.92) + edge * 0.6;
      vec4 mv = modelViewMatrix * vec4(p, 1.0);
      float dist = -mv.z;
      vA = vis * focus * mix(0.25, 1.0, k) * exp(-0.004 * dist) * 0.5;
      gl_PointSize = max(1.6 * uPx, aSize * uPx * (1.0 + edge * 0.8) * 130.0 / dist);
      gl_Position = projectionMatrix * mv;
    }`,
  fragmentShader: /* glsl */ `
    varying vec3 vC;
    varying float vA;
    void main() {
      vec2 d = gl_PointCoord - 0.5;
      float a = exp(-dot(d, d) * 13.0) * vA;
      if (a < 0.012) discard;
      gl_FragColor = vec4(vC, a);
    }`,
}));
scene.add(cloud);

const speeds = steps.map((s) => s.tps).filter(Boolean).sort((a, b) => a - b);
const speedLo = speeds[Math.floor(speeds.length * 0.05)] ?? 0;
const speedHi = speeds[Math.floor(speeds.length * 0.95)] ?? 1;
function episodeColour(e) {
  if (state.mode === 'outcome') return !e.outcome.recalled ? DANGER : e.outcome.trap_obeyed ? TRAP : OK;
  return modelColour(e.m);
}
function recolour() {
  const slow = new THREE.Color('#ff7a45'), fast = new THREE.Color('#58e6ff'), tmp = new THREE.Color();
  const rnd = mulberry32(99);
  for (let i = 0; i < particleCount; i++) {
    const e = episodes[P.ep[i]];
    if (state.hidden.has(e.m)) { colourArray.set([0, 0, 0], i * 3); rnd(); continue; }
    if (state.mode === 'speed') {
      const tps = steps[P.step[i]].tps;
      tmp.copy(slow).lerp(fast, tps ? THREE.MathUtils.clamp((tps - speedLo) / (speedHi - speedLo || 1), 0, 1) : 0.5);
    } else tmp.copy(episodeColour(e));
    const shade = 0.72 + rnd() * 0.5;
    colourArray.set([tmp.r * shade, tmp.g * shade, tmp.b * shade], i * 3);
  }
  cloudGeometry.attributes.aColor.needsUpdate = true;
  rebuildPaths();
  rebuildMarkers();
}

// faint dust so the volume reads as a space, not a chart
{
  const rnd = mulberry32(5), dust = [];
  for (let i = 0; i < 2600; i++) dust.push((rnd() - 0.5) * span * 1.5, rnd() * H * 1.5, (rnd() - 0.5) * span * 1.5);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(dust, 3));
  scene.add(new THREE.Points(g, new THREE.PointsMaterial({ color: 0x33405e, size: 0.14, transparent: true, opacity: 0.55, depthWrite: false })));
}

// ---------------------------------------------------------------- paths and markers
const paths = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.3, blending: THREE.AdditiveBlending, depthWrite: false }));
scene.add(paths);
function rebuildPaths() {
  const pos = [], col = [];
  episodes.forEach((e) => {
    if (state.hidden.has(e.m)) return;
    const c = episodeColour(e);
    const list = stepsOf(e);
    let prev = { x: e.x, y: 0, z: e.z };
    list.forEach((s) => { pos.push(prev.x, prev.y, prev.z, s.x, s.y, s.z); col.push(c.r, c.g, c.b, c.r, c.g, c.b); prev = s; });
  });
  paths.geometry.dispose();
  paths.geometry = new THREE.BufferGeometry();
  paths.geometry.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  paths.geometry.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
}

const markerMaterial = new THREE.ShaderMaterial({
  uniforms, transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending,
  vertexShader: /* glsl */ `
    uniform float uPx, uSel, uClock, uAssemble;
    attribute vec3 aColor;
    attribute float aKind, aEp;
    varying vec3 vC;
    varying float vK, vA;
    void main() {
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      bool sel = abs(aEp - uSel) < 0.5;
      vA = ((uSel < -0.5 || sel || aKind > 1.5) ? 1.0 : 0.1) * smoothstep(0.75, 1.0, uAssemble);
      vC = aColor; vK = aKind;
      float base = clamp(uPx * 330.0 / -mv.z, 6.0 * uPx, 30.0 * uPx);
      gl_PointSize = base * (aKind > 1.5 ? 0.62 : 1.0) * (1.0 + 0.1 * sin(uClock * 3.0 + aEp));
      gl_Position = projectionMatrix * mv;
    }`,
  fragmentShader: /* glsl */ `
    varying vec3 vC;
    varying float vK, vA;
    void main() {
      float r = length(gl_PointCoord - 0.5) * 2.0;
      float core = smoothstep(0.42, 0.18, r);
      float ring = smoothstep(0.13, 0.0, abs(r - 0.74));
      float a = (vK < 0.5 ? core + ring : vK < 1.5 ? ring + core * 0.3 : vK < 2.5 ? ring : core) * vA;
      if (a < 0.02) discard;
      gl_FragColor = vec4(vC, a);
    }`,
});
function pointsFrom(rows) { // rows: [x, y, z, colour, kind, episode]
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(rows.flatMap((r) => [r[0], r[1], r[2]]), 3));
  g.setAttribute('aColor', new THREE.Float32BufferAttribute(rows.flatMap((r) => [r[3].r, r[3].g, r[3].b]), 3));
  g.setAttribute('aKind', new THREE.Float32BufferAttribute(rows.map((r) => r[4]), 1));
  g.setAttribute('aEp', new THREE.Float32BufferAttribute(rows.map((r) => r[5]), 1));
  g.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, H / 2, 0), 400);
  return g;
}
// The step that pulled a page is where its content enters the context: mark the step after it.
function landing(e, flag) {
  const list = stepsOf(e);
  const i = list.findIndex((s) => s.flags.includes(flag));
  return i < 0 ? null : { step: list[Math.min(i + 1, list.length - 1)], index: Math.min(i + 1, list.length - 1) };
}
const markers = new THREE.Points(new THREE.BufferGeometry(), markerMaterial);
const overlay = new THREE.Points(new THREE.BufferGeometry(), markerMaterial); // selection nodes, walker, search hits
scene.add(markers, overlay);
function rebuildMarkers() {
  const rows = [];
  episodes.forEach((e, ei) => {
    if (state.hidden.has(e.m)) return;
    const canary = landing(e, 'canary_seen'), trap = landing(e, 'trap_seen');
    if (canary) rows.push([canary.step.x, canary.step.y, canary.step.z, e.outcome.recalled ? CANARY : DANGER, 0, ei]);
    if (trap) rows.push([trap.step.x + 0.5, trap.step.y, trap.step.z, e.outcome.trap_obeyed ? TRAP : TRAP.clone().multiplyScalar(0.35), 1, ei]);
  });
  markers.geometry.dispose();
  markers.geometry = pointsFrom(rows);
}

const selectedPath = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.85, depthTest: false }));
scene.add(selectedPath);
let hitSteps = [];
const walker = new THREE.Vector3();
function rebuildOverlay() {
  const rows = [];
  if (state.sel >= 0) {
    stepsOf(episodes[state.sel]).forEach((s) => rows.push([s.x, s.y, s.z, new THREE.Color(0xffffff), 2, state.sel]));
    rows.push([walker.x, walker.y, walker.z, new THREE.Color(0xffffff), 3, state.sel]);
  }
  hitSteps.forEach((h, rank) => { const s = steps[h.step]; rows.push([s.x, s.y, s.z, new THREE.Color(rank ? 0xbfc7d9 : 0xffffff), 2, -9]); });
  overlay.geometry.dispose();
  overlay.geometry = pointsFrom(rows);
}

// ---------------------------------------------------------------- pins
const pinLayer = $('pins');
let pins = [];
function addPin(group, cls, text, pos) {
  const node = el('div', `pin ${cls}`, text);
  pinLayer.append(node);
  pins.push({ group, node, pos: pos.clone(), anchor: cls.includes('district') ? 'translate(-50%, 0)' : 'translate(-50%, -100%) translateY(-14px)' });
}
function clearPins(group) {
  pins = pins.filter((p) => { if (p.group === group) p.node.remove(); return p.group !== group; });
}
const projected = new THREE.Vector3();
function updatePins() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  const show = $('showPins').checked && state.assemble > 0.9;
  const placed = []; // district labels give way to each other instead of piling up
  for (const p of pins) {
    projected.copy(p.pos).project(camera);
    let visible = show && projected.z < 1 && Math.abs(projected.x) < 1.05 && Math.abs(projected.y) < 1.05 && !(p.group === 'static' && state.sel >= 0 && !p.node.classList.contains('limit'));
    const x = ((projected.x + 1) / 2) * w, y = ((1 - projected.y) / 2) * h;
    if (x < 330 && y < 120) visible = false; // keep the wordmark clear
    if (visible && p.node.classList.contains('district')) {
      const half = p.node.textContent.length * 3.6 + 8;
      if (placed.some((b) => Math.abs(b.x - x) < b.half + half && Math.abs(b.y - y) < 16)) visible = false;
      else placed.push({ x, y, half });
    }
    p.node.style.display = visible ? '' : 'none';
    if (visible) p.node.style.transform = `translate3d(${x}px, ${y}px, 0) ${p.anchor}`;
  }
}
atlas.clusters.forEach((c) => addPin('static', 'district', c.label, new THREE.Vector3(c.x, -0.8, c.z)));
addPin('static', 'limit', `context limit · ${fmt(windowTokens)} tokens`, new THREE.Vector3(-span / 2 + 6, H, span / 2 - 6));

// ---------------------------------------------------------------- camera moves
let flight = null;
function flyTo(pos, target, seconds = 1.4) {
  if (reducedMotion) { camera.position.copy(pos); controls.target.copy(target); return; }
  flight = { t: 0, seconds, p0: camera.position.clone(), p1: pos.clone(), t0: controls.target.clone(), t1: target.clone() };
}
function episodeView(e) {
  const top = Math.min(e.peak, 1.35) * H;
  const target = new THREE.Vector3(e.x, top * 0.5, e.z);
  const out = new THREE.Vector3(e.x, 0, e.z);
  if (out.lengthSq() < 1) out.set(0.6, 0, 0.8);
  out.normalize();
  const side = new THREE.Vector3(-out.z, 0, out.x);
  const dist = 10 + top * 1.3;
  return { pos: target.clone().addScaledVector(out, dist).addScaledVector(side, dist * 0.35).add(new THREE.Vector3(0, 2, 0)), target };
}

// ---------------------------------------------------------------- selection and memory walk
function walkerAt(e, t) {
  const list = stepsOf(e);
  const f = t * (list.length - 1);
  const i = Math.min(Math.floor(f), list.length - 2);
  if (list.length < 2) return { index: 0, pos: new THREE.Vector3(list[0].x, list[0].y, list[0].z) };
  const a = list[i], b = list[i + 1], u = f - i;
  return { index: Math.round(f), pos: new THREE.Vector3(a.x + (b.x - a.x) * u, a.y + (b.y - a.y) * u, a.z + (b.z - a.z) * u) };
}
function select(ei, stepIndex = null, fly = true) {
  state.sel = ei;
  state.playing = false;
  controls.autoRotate = false;
  const e = episodes[ei];
  const list = stepsOf(e);
  state.selTime = stepIndex == null ? 1 : list.length > 1 ? stepIndex / (list.length - 1) : 1;
  uniforms.uSel.value = ei;
  paths.material.opacity = 0.07;
  const pts = [new THREE.Vector3(e.x, 0, e.z), ...list.map((s) => new THREE.Vector3(s.x, s.y, s.z))];
  selectedPath.geometry.dispose();
  selectedPath.geometry = new THREE.BufferGeometry().setFromPoints(pts);
  renderInspector(e, list);
  if (fly) { const v = episodeView(e); flyTo(v.pos, v.target); }
  applyWalk();
}
function clearSelection() {
  state.sel = -1;
  state.playing = false;
  uniforms.uSel.value = -1;
  paths.material.opacity = 0.3;
  selectedPath.geometry.dispose();
  selectedPath.geometry = new THREE.BufferGeometry();
  $('inspector').hidden = true;
  clearPins('sel');
  rebuildOverlay();
  flyTo(HOME.pos, HOME.target);
  syncTimeline();
}
function applyWalk() {
  const e = episodes[state.sel];
  const list = stepsOf(e);
  const w = walkerAt(e, state.selTime);
  walker.copy(w.pos);
  uniforms.uSelTime.value = state.selTime;
  selectedPath.geometry.setDrawRange(0, w.index + 2);
  rebuildOverlay();

  clearPins('sel');
  const canary = landing(e, 'canary_seen'), trap = landing(e, 'trap_seen');
  const at = (s) => new THREE.Vector3(s.x, s.y, s.z);
  if (canary && w.index >= canary.index) addPin('sel', 'canary', `canary planted · ${e.task.canary_code}`, at(canary.step));
  if (trap && w.index >= trap.index) addPin('sel', 'trap', e.outcome.trap_obeyed ? 'trap read → obeyed' : 'trap read → ignored', at(trap.step).add(new THREE.Vector3(0.5, 0, 0)));
  if (state.selTime >= 0.999 && 'recalled' in e.outcome) {
    const last = list[list.length - 1];
    addPin('sel', e.outcome.recalled ? 'ok' : 'lost', e.outcome.recalled ? 'canary recalled' : `canary lost · answered “${(e.outcome.answer || 'nothing').slice(0, 28)}”`, at(last).add(new THREE.Vector3(0, 1.2, 0)));
  }
  [...$('stepList').children].forEach((li, i) => { li.classList.toggle('on', i === w.index); li.classList.toggle('future', i > w.index); });
  renderStep(e, list[w.index]);
  syncTimeline();
}

function renderInspector(e, list) {
  $('inspector').hidden = false;
  $('epModel').textContent = models[e.m].id;
  $('epModel').style.color = MODEL_COLOURS[e.m % MODEL_COLOURS.length];
  $('epTitle').textContent = e.task.service ? `${e.task.service} runbook` : shortId(e.id);
  const badges = $('badges');
  badges.replaceChildren();
  const badge = (cls, text) => badges.append(el('span', `badge ${cls}`, text));
  badge('', `peak demand ${pct(e.peak)} of window`);
  if ('recalled' in e.outcome) badge(e.outcome.recalled ? 'ok' : 'bad', e.outcome.recalled ? 'canary recalled' : 'canary lost');
  if ('trap_obeyed' in e.outcome) badge(e.outcome.trap_obeyed ? 'trap' : 'ok', e.outcome.trap_obeyed ? 'trap obeyed' : 'trap ignored');
  if (e.outcome.truncated) badge('bad', 'silently truncated');
  if (e.outcome.error) badge('bad', 'crashed: provider rejected the call');
  const ol = $('stepList');
  ol.replaceChildren();
  list.forEach((s, i) => {
    const li = el('li', s.flags.map((f) => `flag-${f}`).join(' '));
    li.tabIndex = 0;
    const bar = el('span', 'bar');
    const fill = el('i', s.demand > 1 ? 'over' : '');
    fill.style.width = `${Math.min(100, s.demand * 100)}%`;
    bar.append(fill);
    li.append(el('span', 'n', String(i + 1)), el('span', '', s.tool || 'text'), el('span', '', pct(s.demand)));
    li.children[1].append(bar);
    const go = () => { state.selTime = list.length > 1 ? i / (list.length - 1) : 1; state.playing = false; applyWalk(); };
    li.addEventListener('click', go);
    li.addEventListener('keydown', (ev) => { if (ev.key === 'Enter') go(); });
    ol.append(li);
  });
}
function renderStep(e, s) {
  const box = $('stepDetail');
  box.replaceChildren();
  box.append(el('p', 'call', s.call || '(plain text reply)'));
  box.append(el('p', 'nums', `prompt ${fmt(s.pt)} tok read · ${fmt(s.sent)} sent · ${fmt(s.ct)} out · ${s.tps ?? '–'} tok/s · ${s.lat}s`));
  const pre = el('pre');
  const text = s.text || s.out || '';
  const needles = [[e.task.canary_code, ''], ['ATTENTION AUTOMATED AGENT', 'trap']].filter(([n]) => n && text.includes(n));
  if (!needles.length) pre.textContent = text;
  else {
    const [needle, cls] = needles[0];
    const [before, ...rest] = text.split(needle);
    pre.append(before, el('mark', cls, needle), rest.join(needle));
  }
  box.append(pre);
}

// ---------------------------------------------------------------- timeline
const slider = $('time');
function syncTimeline() {
  const selected = state.sel >= 0;
  slider.value = Math.round((selected ? state.selTime : state.time) * 1000);
  $('play').textContent = state.playing ? '❚❚' : '▶';
  if (selected) {
    const e = episodes[state.sel], list = stepsOf(e), w = walkerAt(e, state.selTime);
    $('timeLabel').textContent = `step ${w.index + 1}/${list.length} · ${pct(list[w.index].demand)} of window`;
  } else $('timeLabel').textContent = state.time >= 0.999 ? `all ${episodes.length} runs` : `all runs · ${pct(state.time)} elapsed`;
}
slider.addEventListener('input', () => {
  state.playing = false;
  const v = slider.value / 1000;
  if (state.sel >= 0) { state.selTime = v; applyWalk(); } else { state.time = v; uniforms.uTime.value = v; syncTimeline(); }
});
$('play').addEventListener('click', () => {
  state.playing = !state.playing;
  if (state.playing) {
    if (state.sel >= 0 && state.selTime >= 0.999) state.selTime = 0;
    if (state.sel < 0 && state.time >= 0.999) state.time = 0;
  }
  syncTimeline();
});

// ---------------------------------------------------------------- picking
let down = null;
canvas.addEventListener('pointerdown', (ev) => { down = { x: ev.clientX, y: ev.clientY }; controls.autoRotate = false; $('hits').hidden = true; });
canvas.addEventListener('pointerup', (ev) => {
  if (!down || Math.hypot(ev.clientX - down.x, ev.clientY - down.y) > 5) return;
  const rect = canvas.getBoundingClientRect();
  let best = null;
  steps.forEach((s, si) => {
    const e = episodes[s.e];
    if (state.hidden.has(e.m)) return;
    projected.set(s.x, s.y, s.z).project(camera);
    if (projected.z > 1) return;
    const d = Math.hypot(((projected.x + 1) / 2) * rect.width - (ev.clientX - rect.left), ((1 - projected.y) / 2) * rect.height - (ev.clientY - rect.top));
    if (d < 22 && (!best || d < best.d)) best = { d, si };
  });
  if (best) { const s = steps[best.si]; select(s.e, s.i, s.e !== state.sel); }
});
addEventListener('keydown', (ev) => { if (ev.key === 'Escape' && state.sel >= 0) clearSelection(); });
$('close').addEventListener('click', clearSelection);

// ---------------------------------------------------------------- search in space
async function runSearch(query) {
  let hits = null;
  try {
    const r = await fetch(`./api/search?q=${encodeURIComponent(query)}`);
    if (r.ok) hits = (await r.json()).hits;
  } catch { /* static export: no server */ }
  if (!hits) { // text fallback: share of query words found in the step
    const words = query.toLowerCase().match(/[a-z0-9\-]{3,}/g) || [];
    hits = steps.map((s, i) => {
      const hay = `${s.call} ${s.text} ${s.out}`.toLowerCase();
      return { step: i, score: words.length ? words.filter((w) => hay.includes(w)).length / words.length : 0 };
    }).filter((h) => h.score > 0).sort((a, b) => b.score - a.score).slice(0, 12);
  }
  return hits.filter((h) => !state.hidden.has(episodes[steps[h.step].e].m));
}
$('search').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const query = $('q').value.trim();
  const list = $('hits');
  list.replaceChildren();
  clearPins('hit');
  hitSteps = [];
  if (!query) { list.hidden = true; rebuildOverlay(); return; }
  hitSteps = await runSearch(query);
  list.hidden = false;
  if (!hitSteps.length) list.append(el('li', 'empty', 'Nothing in the atlas matches that.'));
  hitSteps.forEach((h) => {
    const s = steps[h.step], e = episodes[s.e];
    const li = el('li');
    li.append(el('span', 'score', h.score.toFixed(2)), el('span', 'what', s.call || s.text.slice(0, 80)), el('span', 'where', `${models[e.m].id} · ${e.task.topic ?? shortId(e.id)} · step ${s.i + 1}`));
    li.addEventListener('click', () => { list.hidden = true; select(s.e, s.i); });
    list.append(li);
  });
  if (hitSteps.length) {
    const s = steps[hitSteps[0].step];
    addPin('hit', 'here', 'closest match', new THREE.Vector3(s.x, s.y, s.z));
    select(s.e, s.i);
  } else rebuildOverlay();
});
$('q').addEventListener('input', () => { if (!$('q').value) { $('hits').hidden = true; hitSteps = []; clearPins('hit'); rebuildOverlay(); } });

// ---------------------------------------------------------------- side panel
$('summary').textContent = `${episodes.length} runs · ${fmt(steps.length)} steps · ${models.length} model${models.length > 1 ? 's' : ''}`;

models.forEach((m, mi) => {
  const li = el('li');
  const dot = el('span', 'dot');
  dot.style.background = MODEL_COLOURS[mi % MODEL_COLOURS.length];
  const stats = el('div', 'stats');
  const stat = (label, value, bad) => { const d = el('div', bad ? 'bad' : ''); d.append(el('b', '', value), label); stats.append(d); };
  const s = m.stats;
  stat('recalled', s.recall.n ? pct(s.recall.rate) : '–');
  stat('trap obeyed', s.trap.n ? pct(s.trap.rate) : '–', s.trap.rate > 0.2);
  stat('tok/s', s.tokens_per_s ?? '–');
  li.append(dot, el('span', 'name', m.id), stats);
  li.title = `${s.episodes} runs. Click to show or hide.`;
  li.addEventListener('click', () => {
    state.hidden.has(mi) ? state.hidden.delete(mi) : state.hidden.add(mi);
    li.classList.toggle('off', state.hidden.has(mi));
    if (state.sel >= 0 && state.hidden.has(episodes[state.sel].m)) clearSelection();
    recolour();
  });
  $('models').append(li);
});
document.querySelectorAll('#colour button').forEach((b) => b.addEventListener('click', () => {
  state.mode = b.dataset.mode;
  document.querySelectorAll('#colour button').forEach((x) => x.classList.toggle('on', x === b));
  recolour();
  renderLegend();
}));
function renderLegend() {
  const box = $('legend');
  box.replaceChildren();
  const chip = (colour, label) => { const c = el('span', 'chip', label); c.style.setProperty('--c', colour); box.append(c); };
  if (state.mode === 'outcome') { chip('#4fe3b0', 'clean'); chip('#ff4fd8', 'trap obeyed'); chip('#ff4d5e', 'canary lost'); }
  else if (state.mode === 'speed') { chip('#ff7a45', `${Math.round(speedLo)} tok/s`); chip('#58e6ff', `${Math.round(speedHi)} tok/s`); }
  else models.forEach((m, mi) => chip(MODEL_COLOURS[mi % MODEL_COLOURS.length], m.id));
  chip('#bdfc23', 'canary');
}
renderLegend();

{ // recall vs demand chart
  const svg = $('chart'), NS = 'http://www.w3.org/2000/svg';
  const node = (tag, attrs, text) => { const n = document.createElementNS(NS, tag); for (const k in attrs) n.setAttribute(k, attrs[k]); if (text) n.textContent = text; svg.append(n); return n; };
  const points = models.flatMap((m) => m.stats.by_fill.filter((r) => r.fill != null).map((r) => r.demand ?? r.fill));
  if (points.length) {
    const L = 28, R = 252, T = 10, B = 104;
    const lo = Math.min(0.25, ...points), hi = Math.max(...points, 1.05);
    const X = (f) => L + ((f - lo) / (hi - lo || 1)) * (R - L), Y = (v) => B - v * (B - T);
    [0, 0.5, 1].forEach((v) => { node('line', { x1: L, x2: R, y1: Y(v), y2: Y(v), stroke: 'rgba(255,255,255,.08)' }); node('text', { x: L - 5, y: Y(v) + 3, 'text-anchor': 'end' }, pct(v)); });
    node('line', { x1: X(1), x2: X(1), y1: T, y2: B, stroke: '#ff4d5e', 'stroke-dasharray': '3 3', opacity: 0.8 });
    node('text', { x: X(1) - 3, y: T + 7, 'text-anchor': 'end', style: 'fill:#ff4d5e' }, 'window limit');
    [0.25, 0.5, 0.75, 1, 1.25, 1.5].filter((f) => f >= lo && f <= hi).forEach((f) => node('text', { x: X(f), y: B + 12, 'text-anchor': 'middle' }, pct(f)));
    node('text', { x: (L + R) / 2, y: B + 24, 'text-anchor': 'middle' }, 'measured context demand (tokens sent / window)');
    models.forEach((m, mi) => {
      const rows = m.stats.by_fill.filter((r) => r.fill != null);
      const colour = MODEL_COLOURS[mi % MODEL_COLOURS.length];
      const x = (r) => X(r.demand ?? r.fill);
      node('polyline', { points: rows.map((r) => `${x(r)},${Y(r.recall.rate) + mi * 1.5}`).join(' '), fill: 'none', stroke: colour, 'stroke-width': 1.6 });
      rows.forEach((r) => node('circle', { cx: x(r), cy: Y(r.recall.rate) + mi * 1.5, r: 2.6, fill: colour }).append(Object.assign(document.createElementNS(NS, 'title'), { textContent: `${m.id}: ${r.recall.k}/${r.recall.n} recalled at ${pct(r.demand ?? r.fill)} demand` })));
    });
  }
}

// ---------------------------------------------------------------- loop
function resize() {
  const w = innerWidth, h = innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  uniforms.uPx.value = renderer.getPixelRatio() * (h / 900);
}
addEventListener('resize', resize);
resize();
recolour();
syncTimeline();
if (!reducedMotion) flyTo(HOME.pos, HOME.target, 3.2);

const clock = new THREE.Clock();
renderer.setAnimationLoop(() => {
  const real = Math.min(clock.getDelta(), 1.5); // intro and camera moves keep wall-clock pace on slow frames
  const dt = Math.min(real, 0.1);
  uniforms.uClock.value += dt;
  if (state.assemble < 1) { state.assemble = Math.min(1, state.assemble + real / 3.4); uniforms.uAssemble.value = state.assemble; }
  if (flight) {
    flight.t = Math.min(1, flight.t + real / flight.seconds);
    const k = ease(flight.t);
    camera.position.lerpVectors(flight.p0, flight.p1, k);
    controls.target.lerpVectors(flight.t0, flight.t1, k);
    if (flight.t >= 1) flight = null;
  }
  if (state.playing) {
    if (state.sel >= 0) {
      state.selTime = Math.min(1, state.selTime + dt / (stepsOf(episodes[state.sel]).length * 0.85));
      if (state.selTime >= 1) state.playing = false;
      applyWalk();
    } else {
      state.time = Math.min(1, state.time + dt / 9);
      uniforms.uTime.value = state.time;
      if (state.time >= 1) state.playing = false;
      syncTimeline();
    }
  }
  controls.update();
  renderer.render(scene, camera);
  updatePins();
});

window.atlasView = { select, clearSelection, state };
