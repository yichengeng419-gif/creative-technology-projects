export function clamp(v, a = 0, b = 1) {
  return Math.min(b, Math.max(a, v));
}

export function ease(v) {
  v = clamp(v);
  if (v === 0 || v === 1) return v;
  return clamp(v * v * v * (v * (v * 6 - 15) + 10));
}

export function mix(a, b, t) {
  return a + (b - a) * t;
}

export function fmt(ms) {
  const s = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return String(m).padStart(2, '0') + ':' + String(r).padStart(2, '0');
}

export function hsv2rgb(h, s, v) {
  h = ((h % 1) + 1) % 1;
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  const m = [[v, t, p], [q, v, p], [p, v, t], [p, q, v], [t, p, v], [v, p, q]][i % 6];
  return [m[0], m[1], m[2]];
}

export function approach(dt, speed) {
  return 1 - Math.exp(-speed * dt);
}
