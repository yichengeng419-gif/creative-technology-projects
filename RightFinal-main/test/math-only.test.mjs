/**
 * A standalone check for src/core/math.js only.
 */
import { clamp, ease, mix, fmt, hsv2rgb, approach } from '../src/core/math.js';

let pass = 0; let fail = 0;
const check = (name, cond, detail = '') => {
  if (cond) { pass++; console.log(`  ok   ${name}`); }
  else { fail++; console.log(`  FAIL ${name} ${detail}`); }
};

const origClamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
const origEase = (v) => { v = origClamp(v); if (v === 0 || v === 1) return v; return origClamp(v * v * v * (v * (v * 6 - 15) + 10)); };
const origFmt = (ms) => { const s = Math.max(0, Math.floor(ms / 1000)); return String(Math.floor(s / 60)).padStart(2, '0') + ':' + String(s % 60).padStart(2, '0'); };
function origHsv(h, s_, v) {
  h = ((h % 1) + 1) % 1;
  const i = Math.floor(h * 6), f = h * 6 - i;
  const p = v * (1 - s_), q = v * (1 - f * s_), t = v * (1 - (1 - f) * s_);
  const m = [[v, t, p], [q, v, p], [p, v, t], [p, q, v], [t, p, v], [v, p, q]][i % 6];
  return [m[0], m[1], m[2]];
}

console.log('\nChecking math.js against the original Right9_9.html logic\n');

let mathMismatch = 0;
for (let i = -50; i <= 150; i++) {
  const v = i / 100;
  if (clamp(v) !== origClamp(v)) mathMismatch++;
  if (ease(v) !== origEase(v)) mathMismatch++;
  if (JSON.stringify(hsv2rgb(v, 0.55, 1)) !== JSON.stringify(origHsv(v, 0.55, 1))) mathMismatch++;
}
for (const ms of [0, 999, 1000, 61000, 3599000, -5]) {
  if (fmt(ms) !== origFmt(ms)) mathMismatch++;
}
check('clamp / ease / hsv2rgb / fmt match the original, across 201 test values', mathMismatch === 0,
  `${mathMismatch} mismatches — check clamp, ease, hsv2rgb and fmt one at a time`);

check('mix is linear (mix(10, 20, 0.5) === 15)', mix(10, 20, 0.5) === 15);
check('mix(10, 20, 0) returns the start', mix(10, 20, 0) === 10);
check('mix(10, 20, 1) returns the end', mix(10, 20, 1) === 20);

const a1 = approach(0.016, 7);
check('approach returns a number between 0 and 1', typeof a1 === 'number' && a1 > 0 && a1 < 1, `got ${a1}`);
check('approach(0, speed) is 0 (no time passed, no movement)', approach(0, 7) === 0);
check('a bigger dt closes more of the distance', approach(0.5, 7) > approach(0.016, 7));

console.log(`\n${pass} passed, ${fail} failed\n`);
process.exitCode = fail ? 1 : 0;