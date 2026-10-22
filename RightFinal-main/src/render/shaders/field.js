export const FIELD_GLSL = `
float hash(vec3 p) { return fract(sin(dot(p, vec3(91.3458, 37.156, 63.729))) * 47453.5453); }
mat2 rot(float a) { float c = cos(a), s = sin(a); return mat2(c, s, -s, c); }

void computeField(
  vec3 a_pos, vec4 a_color, float t,
  float u_fit, float u_aspect, float u_cam, float u_yaw, float u_pitch, float u_stretch,
  float u_pressure, float u_contact, float u_hold, float u_approach, float u_rhythm, float u_excess,
  float u_spin, float u_turn, float u_space, float u_hole, float u_reject, float u_dark,
  vec2 u_focus, vec2 u_nearPos, vec2 u_travel, vec2 u_offset,
  out vec2 metric, out float depth, out float layer, out float seed,
  out float strain, out float dragCore, out vec3 col, out float edge, out float reading,
  out vec2 tangentOut
) {
  seed = hash(a_pos); layer = hash(a_pos.zxy + 2.1);
  vec3 p = a_pos - vec3(-.0095, .0379, .0273);
  p.y *= u_stretch;
  p.z *= 1.55;

  vec2 advect = u_travel * (.020 + layer * .026);
  p.x += advect.x; p.y += advect.y;
  p.x = mod(p.x + 1.8, 3.6) - 1.8;
  p.y = mod(p.y + 1.45, 2.9) - 1.45;
  p.z += sin(p.x * 5.0 + p.y * 3.0 + t * .32) * (.018 + u_space * .13) * mix(1.0, 2.7, u_excess);
  if (u_rhythm > .5 && u_rhythm < 1.5) p.y += sin(p.x * 35.0 + t * .6) * .019;
  if (u_rhythm > 1.5) p += vec3(sin(t * 13.0 + seed * 172.0), cos(t * 17.0 + seed * 71.0), 0) * .018;
  p.xz = rot(u_yaw) * p.xz; p.yz = rot(u_pitch) * p.yz;

  vec2 cameraDrift = vec2(sin(t * .105), cos(t * .083)) * (.026 + u_space * .035);
  p.xy -= cameraDrift;
  depth = max(1.25, u_cam - p.z);
  float nearFactor =
  1.0 - smoothstep(
    2.45,
    5.65,
    depth
  );
  vec2 q = vec2(p.x / u_aspect, p.y) * u_fit / depth - u_offset;
  metric = vec2(q.x * u_aspect, q.y);
  vec2 focus = vec2(u_focus.x * u_aspect, u_focus.y);
  vec2 nearPos = vec2(u_nearPos.x * u_aspect, u_nearPos.y);
  vec2 attention = nearPos - metric; float nr = length(attention);
  metric += attention * exp(-nr * nr / 1.0) * u_approach * .16;

  vec2 d = metric - focus; float radius = length(d), angle = atan(d.y, d.x);
  float influence = 1.0 - smoothstep(.12, 1.48, radius);
  reading = u_contact * influence;

  dragCore = exp(-pow(radius / (.34 + .14 * layer), 2.0)) * u_hole;
  float tau = t - dragCore * (1.7 + u_hold * 3.2);
  float rotation = (u_spin * .58 + u_turn * .9 * u_hold) * reading;
  d = rot(rotation) * d;
  d *= 1.0 - reading * (.22 + .20 * u_hold) * mix(1.0, 1.20, u_excess);
  vec2 tangent = normalize(vec2(-d.y, d.x) + vec2(.00001));

  float seam = sin(angle * 4.0 + radius * 5.2 - u_spin * 2.4 + tau * .24);
  edge = (1.0 - smoothstep(.025, .15, abs(seam))) * influence;
  float shear = u_pressure * reading;
  d += tangent * seam * shear * .34 + d * shear * .15;

  float streamId = floor(fract((angle + 3.14159265) / 6.2831853) * 4.0);
  float streamSign = mod(streamId, 2.0) < 1.0 ? -1.0 : 1.0;
  float localPhase = sin(tau * (1.15 + streamId * .09) + radius * 8.0 + layer * 6.283);
  float deflection = dragCore * (.10 + .16 * abs(seam)) * (.62 + .38 * layer);
  d += tangent * deflection * (streamSign + seam * .48 + localPhase * .24);
  d += normalize(d + vec2(.00001)) * dragCore * .052 * (.25 + .75 * abs(seam));

  float streamDelay = streamId * .055 + layer * .035;
  float rejectPulse = smoothstep(streamDelay, streamDelay + .09, u_reject)
    * (1.0 - smoothstep(streamDelay + .22, streamDelay + .62, u_reject));
  float burst = rejectPulse * exp(-radius * radius / 1.05) * (.20 + .24 * abs(seam));
  d += tangent * streamSign * burst * (.65 + .35 * localPhase);
  d += normalize(d + vec2(.00001)) * burst * .12;
  metric = focus + d;

  strain = clamp(dragCore * .72 + burst * 1.8 + edge * dragCore * .38, 0.0, 1.0);

  float lum = dot(a_color.rgb, vec3(.2126, .7152, .0722));
  //*3d Model colour
  vec3 farInk = vec3(.24, .33, .40);
  vec3 midSteel = vec3(.46, .57, .64);
  vec3 nearMist = vec3(.76, .82, .85);
  vec3 neutral = mix(farInk, midSteel, smoothstep(.05, .72, nearFactor));
  neutral = mix(neutral, nearMist, smoothstep(.62, 1.0, nearFactor) * .72);
  neutral *= .95 + clamp(lum, 0.0, 1.0) * .30;
  col = mix(neutral, a_color.rgb, .05);
  float warmth = clamp(u_pressure * edge * reading + u_hold * edge * reading * .14, 0.0, 1.0);
  vec3 tearColor = mix(vec3(.76, .25, .18), vec3(.66, .43, .22), clamp(localPhase * .5 + .5, 0.0, 1.0));
  col = mix(col, tearColor, clamp(warmth * .72 + strain * .42, 0.0, 1.0));
  col *= u_dark;

  tangentOut = tangent;
}
`;
