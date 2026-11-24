// NYXCORE loop view. Two fields of the same forests: a standard transformer on the left, a looped one on the right.
// A tree lights up as the model solves it. Tree shapes, depths and "solved" flags all come from `nyxcore loop-export`.
// Keys: space pause · left/right step · R restart · H hide all text. Add ?still to the URL for a motionless final frame.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const query = new URLSearchParams(location.search);
// ?still shows the finished state with no motion (also what reduced-motion users get); &angle=0.2 turns the camera
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches || query.has('still');
if (query.has('still')) document.body.classList.add('still');
if (parseFloat(query.get('ui')) > 0) document.documentElement.style.setProperty('--ui', parseFloat(query.get('ui'))); // ?ui=1.6 enlarges all text
const pct = (v) => Math.round(v * 100);
const pct1 = (v) => (v * 100).toFixed(1);

let data;
try {
  data = await fetch('./loop.json').then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); });
} catch (err) {
  $('caption').textContent = 'No loop.json found. Run: nyxcore loop-train, then nyxcore loop-export.';
  throw err;
}
const { view, models: summary } = data;
const N = view.nodes, T = view.ticks, xyz = view.xyz;
const STEP = 1.3; // scene units per level of depth
const GOOD = [new THREE.Color('#e6edff'), new THREE.Color('#bdfc23')];
const EMBER = new THREE.Color('#ff2748');

let halfX = 0, halfZ = 0;
for (let i = 0; i < N; i++) { halfX = Math.max(halfX, Math.abs(xyz[i * 3])); halfZ = Math.max(halfZ, Math.abs(xyz[i * 3 + 2])); }
halfX += 1.6; halfZ += 1.6;
const SHIFT = halfX + 3.2;
const TOP = view.eval_depth * STEP;
const SHEET = (view.train_depth + 0.5) * STEP;

// ---------------------------------------------------------------- scene
const canvas = $('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setClearColor(0x000000, 0);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 600);
const focus = new THREE.Vector3(0, TOP * 0.2, 0);
const controls = new OrbitControls(camera, canvas);
controls.target.copy(focus);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.minDistance = 12;
controls.maxDistance = 260;
let autoCamera = !reducedMotion;
controls.addEventListener('start', () => { autoCamera = false; });
const uniforms = { uMix: { value: 0 }, uPx: { value: 1 }, uClock: { value: 0 }, uAssemble: { value: reducedMotion ? 1 : 0 } };

function mulberry32(a) {
  return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const gauss = (rnd) => Math.sqrt(-2 * Math.log(rnd() || 1e-9)) * Math.cos(6.2832 * rnd());

const fields = view.models.map((m, mi) => {
  const ox = view.models.length > 1 ? (mi === 0 ? -SHIFT : SHIFT) : 0;
  const rnd = mulberry32(31 + mi);
  const pos = [], start = [], jit = [], rand = [], size = [], owner = [];
  for (let i = 0; i < N; i++) {
    const p = view.parent[i], root = p === i;
    const a = [xyz[p * 3] + ox, xyz[p * 3 + 1] * STEP, xyz[p * 3 + 2]];
    const b = [xyz[i * 3] + ox, xyz[i * 3 + 1] * STEP, xyz[i * 3 + 2]];
    const count = root ? 34 : 56;
    for (let j = 0; j < count; j++) {
      const onNode = root || j < 10; // a bright knot at the node, a filament along the edge to its parent
      const u = onNode ? 1 : rnd(), s = onNode ? 0.1 : 0.036;
      pos.push(a[0] + (b[0] - a[0]) * u + gauss(rnd) * s, Math.max(0, a[1] + (b[1] - a[1]) * u + gauss(rnd) * s), a[2] + (b[2] - a[2]) * u + gauss(rnd) * s);
      jit.push(gauss(rnd), gauss(rnd) * 0.6, gauss(rnd));
      const r = 60 + rnd() * 80, th = rnd() * 6.2832;
      start.push(ox * 0.4 + r * Math.cos(th), 30 + gauss(rnd) * 26, r * Math.sin(th));
      rand.push(rnd());
      size.push((onNode ? 1.2 : 0.72) * (0.75 + rnd() * 0.5));
      owner.push(i);
    }
  }
  const count = owner.length;
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('aStart', new THREE.Float32BufferAttribute(start, 3));
  g.setAttribute('aJit', new THREE.Float32BufferAttribute(jit, 3));
  g.setAttribute('aRand', new THREE.Float32BufferAttribute(rand, 1));
  g.setAttribute('aSize', new THREE.Float32BufferAttribute(size, 1));
  g.setAttribute('aFrom', new THREE.BufferAttribute(new Float32Array(count), 1));
  g.setAttribute('aTo', new THREE.BufferAttribute(new Float32Array(count), 1));
  const points = new THREE.Points(g, new THREE.ShaderMaterial({
    uniforms: { ...uniforms, uGood: { value: GOOD[mi % 2] }, uBad: { value: EMBER } },
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    vertexShader: /* glsl */ `
      uniform float uMix, uPx, uClock, uAssemble;
      uniform vec3 uGood, uBad;
      attribute vec3 aStart, aJit;
      attribute float aFrom, aTo, aRand, aSize;
      varying vec3 vC;
      varying float vA;
      void main() {
        float k = clamp(uMix * 1.4 - aRand * 0.4, 0.0, 1.0);
        k = k * k * (3.0 - 2.0 * k);
        float ok = mix(aFrom, aTo, k);
        float born = clamp(uAssemble * 1.6 - aRand * 0.3 - position.y * 0.012, 0.0, 1.0);
        born = born * born * (3.0 - 2.0 * born);
        // unsolved structure is a drifting haze around where the tree is; solving it pulls the haze into a sharp filament
        vec3 drift = 0.09 * vec3(sin(uClock * 0.9 + aRand * 40.0), cos(uClock * 0.7 + aRand * 31.0), sin(uClock * 0.8 + aRand * 23.0));
        vec3 p = mix(aStart, position + (aJit * 0.26 + drift) * (1.0 - ok), born);
        vec4 mv = modelViewMatrix * vec4(p, 1.0);
        float flash = sin(3.14159 * ok); // a brief flare while a node is being solved
        vC = mix(uBad, uGood, ok) + flash * 0.5;
        vA = mix(0.3, 0.5, ok) * born;
        gl_PointSize = max(1.2 * uPx, aSize * uPx * mix(0.9, 1.05, ok) * (1.0 + flash * 0.7) * 150.0 / -mv.z);
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: /* glsl */ `
      varying vec3 vC;
      varying float vA;
      void main() {
        vec2 d = gl_PointCoord - 0.5;
        float a = exp(-dot(d, d) * 14.0) * vA;
        if (a < 0.01) discard;
        gl_FragColor = vec4(vC, a);
      }`,
  }));
  points.frustumCulled = false;
  scene.add(points);

  // a hairline frame on the ground and a sheet at the deepest level seen in training
  const rect = (y, opacity) => {
    const line = new THREE.LineLoop(
      new THREE.BufferGeometry().setFromPoints([[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([x, z]) => new THREE.Vector3(ox + x * halfX, y, z * halfZ))),
      new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity, depthWrite: false }),
    );
    scene.add(line);
    return line;
  };
  rect(0, 0.1);
  rect(SHEET, 0.34);
  const sheet = new THREE.Mesh(new THREE.PlaneGeometry(halfX * 2, halfZ * 2), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.03, side: THREE.DoubleSide, depthWrite: false }));
  sheet.rotation.x = -Math.PI / 2;
  sheet.position.set(ox, SHEET, 0);
  scene.add(sheet);

  const flags = m.ok.map((bits) => Uint8Array.from(bits, (c) => (c === '1' ? 1 : 0)));
  return { model: m, ox, owner, geometry: g, flags, loaded: '' };
});

function load(field, a, b) {
  const m = field.model;
  a = Math.min(a, m.max_tick); b = Math.min(b, m.max_tick);
  if (field.loaded === `${a}:${b}`) return;
  field.loaded = `${a}:${b}`;
  const from = field.geometry.attributes.aFrom.array, to = field.geometry.attributes.aTo.array;
  const fa = field.flags[a], fb = field.flags[b], owner = field.owner;
  for (let i = 0; i < owner.length; i++) { from[i] = fa[owner[i]]; to[i] = fb[owner[i]]; }
  field.geometry.attributes.aFrom.needsUpdate = field.geometry.attributes.aTo.needsUpdate = true;
}

// ---------------------------------------------------------------- readouts
const fmtParams = (n) => `${(n / 1e6).toFixed(2)}M parameters`;
const [A, B] = fields;
$('aMeta').textContent = `${A.model.layers} layers · ${fmtParams(A.model.params)}`;
if (B) $('bMeta').textContent = `${B.model.layers} layers, reused · ${fmtParams(B.model.params)}`;
const unseen = (name) => {
  const m = summary.find((x) => x.name === name);
  const ds = Array.from({ length: view.eval_depth - view.train_depth }, (_, i) => view.train_depth + 1 + i);
  return ds.reduce((s, d) => s + m.acc_by_depth[d], 0) / ds.length;
};
if (B) {
  const ratio = (A.model.params / B.model.params).toFixed(1);
  $('finale').innerHTML = '';
  const line = document.createElement('b');
  line.append(`${ratio}× fewer parameters. `, Object.assign(document.createElement('em'), { textContent: `${pct1(unseen(B.model.name))}%` }), ` vs ${pct1(unseen(A.model.name))}%`);
  $('finale').append(line, Object.assign(document.createElement('span'), { textContent: 'on problems deeper than anything seen in training' }));
}
function readout(position) {
  for (const [field, prefix] of [[A, 'a'], [B, 'b']]) {
    if (!field) continue;
    const m = field.model;
    const at = Math.min(position, m.max_tick), lo = Math.floor(at), hi = Math.min(lo + 1, m.max_tick);
    $(`${prefix}Value`).textContent = pct(m.acc[lo] + (m.acc[hi] - m.acc[lo]) * (at - lo));
    const phase = $(`${prefix}Phase`);
    const step = Math.min(Math.round(position), m.max_tick);
    const done = position >= m.max_tick && m.kind !== 'loop';
    phase.classList.toggle('out', done && position > m.max_tick + 0.05);
    if (m.kind === 'loop') phase.textContent = step ? `loop ${step}` : 'reading the forest';
    else phase.textContent = done && position > m.max_tick + 0.05 ? 'out of layers' : step ? `layer ${step * view.apps_per_tick} of ${m.max_tick * view.apps_per_tick}` : 'reading the forest';
  }
}

// ---------------------------------------------------------------- sequence: lights out -> compute tick by tick -> hold
const state = { from: 0, to: 0, mix: 0, playing: !reducedMotion, phase: 'run', hold: 0 };
const TICK_SECONDS = 1.25, HOLD_SECONDS = 7, RESET_SECONDS = 1.6;
function setSegment(from, to) { state.from = from; state.to = to; state.mix = 0; fields.forEach((f) => load(f, from, to)); }
function jump(tick) { state.playing = false; state.phase = 'run'; $('finale').classList.remove('on'); setSegment(tick, tick); }
if (reducedMotion) { setSegment(T, T); $('finale').classList.add('on'); } else setSegment(0, 1);

addEventListener('keydown', (ev) => {
  if (ev.code === 'Space') { ev.preventDefault(); state.playing = !state.playing; }
  else if (ev.key === 'ArrowRight') jump(Math.min(T, Math.round(state.from + (state.to - state.from) * state.mix) + 1));
  else if (ev.key === 'ArrowLeft') jump(Math.max(0, Math.round(state.from + (state.to - state.from) * state.mix) - 1));
  else if (ev.key === 'r' || ev.key === 'R') { state.playing = true; state.phase = 'run'; $('finale').classList.remove('on'); setSegment(0, 1); }
  else if (ev.key === 'h' || ev.key === 'H') document.body.classList.toggle('clean');
});

function advance(dt) {
  if (uniforms.uAssemble.value < 1 || !state.playing) return;
  if (state.phase === 'hold') {
    state.hold += dt;
    if (state.hold >= HOLD_SECONDS) { state.phase = 'reset'; $('finale').classList.remove('on'); setSegment(T, 0); }
    return;
  }
  state.mix += dt / (state.phase === 'reset' ? RESET_SECONDS : TICK_SECONDS);
  if (state.mix < 1) return;
  if (state.phase === 'reset') { state.phase = 'run'; setSegment(0, 1); }
  else if (state.to >= T) { state.mix = 1; state.phase = 'hold'; state.hold = 0; $('finale').classList.add('on'); }
  else setSegment(state.to, state.to + 1);
}

// ---------------------------------------------------------------- loop
function resize() {
  renderer.setSize(innerWidth, innerHeight, false);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  uniforms.uPx.value = renderer.getPixelRatio() * (innerHeight / 900);
}
addEventListener('resize', resize);
resize();
function frameDistance() { // far enough to hold both fields whatever the window shape
  const halfWidth = (fields.length > 1 ? SHIFT + halfX : halfX) + 4.5;
  const horizontal = Math.atan(Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * camera.aspect);
  return Math.max(halfWidth / Math.tan(horizontal), (TOP * 1.05) / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)));
}
{
  const angle = parseFloat(query.get('angle') || '0') || 0, distance = frameDistance() * (1 + 0.7 * Math.abs(angle));
  camera.position.set(Math.sin(angle) * distance, TOP * 0.5, Math.cos(angle) * distance);
}

const clock = new THREE.Clock();
renderer.setAnimationLoop(() => {
  const real = Math.min(clock.getDelta(), 1.5), dt = Math.min(real, 0.1);
  uniforms.uClock.value += dt;
  if (uniforms.uAssemble.value < 1) uniforms.uAssemble.value = Math.min(1, uniforms.uAssemble.value + real / 3.2);
  advance(dt);
  uniforms.uMix.value = Math.min(1, state.mix);
  readout(state.from + (state.to - state.from) * Math.min(1, state.mix));
  if (autoCamera) {
    const t = uniforms.uClock.value, angle = Math.sin(t * 0.13) * 0.24, distance = frameDistance() * (1 + 0.7 * Math.abs(angle) + 0.02 * Math.sin(t * 0.09));
    camera.position.set(Math.sin(angle) * distance, TOP * (0.5 + 0.05 * Math.sin(t * 0.11)), Math.cos(angle) * distance);
    camera.lookAt(focus);
  } else controls.update();
  renderer.render(scene, camera);
});
window.nyx = { state, jump, uniforms };
