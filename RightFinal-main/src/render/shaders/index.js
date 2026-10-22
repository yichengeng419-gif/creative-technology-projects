import { FIELD_GLSL } from './field.js';
import { ARCHIVE_GLSL } from './archive.js';
import { POINTFRAGMENT_GLSL } from './pointfragment.js';

export const FIELD_VERTEX = `#version 300 es
precision highp float;
in vec3 a_pos;
in vec4 a_color;

uniform float u_phase, u_fit, u_aspect, u_dpr, u_cam, u_yaw, u_pitch, u_stretch, u_pass;
uniform float u_pressure, u_contact, u_hold, u_still, u_approach, u_rhythm, u_excess;
uniform float u_spin, u_turn, u_space, u_hole, u_reject, u_gather, u_lift, u_dark, u_unarchived;
uniform vec2 u_focus, u_nearPos, u_travel, u_offset, u_archiveEdge;

out vec3 v_col;
out float v_alpha;
out float v_angle;
out float v_thin;
out float v_ascii;
out float v_glyph;

${FIELD_GLSL}
${ARCHIVE_GLSL}

void main() {
  vec2 metric; float depth, layer, seed, strain, dragCore, edge, reading;
  vec3 col; vec2 tangent;

  computeField(
    a_pos, a_color, u_phase,
    u_fit, u_aspect, u_cam, u_yaw, u_pitch, u_stretch,
    u_pressure, u_contact, u_hold, u_approach, u_rhythm, u_excess,
    u_spin, u_turn, u_space, u_hole, u_reject, u_dark,
    u_focus, u_nearPos, u_travel, u_offset,
    metric, depth, layer, seed, strain, dragCore, col, edge, reading, tangent
  );

  vec2 archivedMetric; float ascii, glyphIndex, absence;
  computeArchive(
    metric, layer, seed, strain, u_aspect,
    u_nearPos, u_archiveEdge, u_gather, u_lift, u_unarchived,
    archivedMetric, ascii, glyphIndex, absence
  );

  float nearPlane = .65, farPlane = 8.0;
  float clipZ = ((farPlane + nearPlane) / (farPlane - nearPlane)) * depth
    - (2.0 * farPlane * nearPlane) / (farPlane - nearPlane);
  vec2 q = vec2(archivedMetric.x / u_aspect, archivedMetric.y);
  gl_Position = vec4(q * depth, clipZ, depth);

  v_col = col;
  float nearFactor = 1.0 - smoothstep(2.45, 5.65, depth);
  //*Reduce the centre stack brightness
  float localDensity =
  1.0 + reading * .18;
  //*Transparency
  v_alpha = (.34 + nearFactor * .52) * localDensity;
  v_alpha *= 1.0 - dragCore * .20;   
  v_alpha *= 1.0 - absence * .85;
  v_alpha *= 1.0 - smoothstep(.45, 1.0, u_lift);
  v_alpha *= 1.0 - u_gather * .18;

  v_ascii = ascii;
  v_glyph = glyphIndex;

  vec2 screenTangent = normalize(vec2(tangent.x / u_aspect, -tangent.y));
  v_angle = atan(screenTangent.y, screenTangent.x);
  // Long press changes the particles gently.
float stretch =
  reading *
  (1.0 - u_still) *
  (.25 + u_pressure * .55);

// Keep particles soft instead of compressing them into hard blocks.
v_thin = mix(
  1.0,
  .38,
  clamp(stretch, 0.0, 1.0)
);

v_angle = mix(v_angle, 1.5707963, u_gather);
v_thin = mix(v_thin, .12, u_gather);

// Limit the enlargement of particles near the camera.
float perspectiveScale =
  clamp(4.35 / depth, .72, 1.35);

float size =
  (
    1.10 +
    nearFactor * .90
  ) *
  perspectiveScale *
  u_dpr;

// Only enlarge particles moderately during the final gathering.
size = mix(
  size,
  (5.0 + layer * 8.0) * u_dpr,
  u_gather
);

// Prevent individual particles from becoming large squares.
gl_PointSize = min(size, 10.0);
}`;

export const FIELD_FRAGMENT = POINTFRAGMENT_GLSL;
