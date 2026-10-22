export const VERTEX = `#version 300 es
precision highp float;

in vec3 a_pos;
in float a_glyph;
in float a_seed;
in float a_size;
in vec3 a_scat;
in float a_tone;
in vec3 a_col;

uniform mat4 u_view;
uniform float u_time;
uniform float u_dpr;
uniform float u_wake;
uniform float u_jitter;
uniform float u_scatter;
uniform float u_opacity;

out vec3 v_color;
out float v_alpha;

void main() {
  vec3 position = a_pos;

  // Scatter particles during the transition.
  float scatterDistance = mix(2.0, 9.0, a_seed);
  position += a_scat * u_scatter * scatterDistance;

  // Quiet movement while the specimen is dormant.
  float drift = sin(u_time * 0.7 + a_seed * 30.0);
  position += a_scat * drift * u_jitter;

  // Convert the vertical DNA position into a 0-1 value.
  float axis = clamp(
    (a_pos.y + 15.0) / 30.0,
    0.0,
    1.0
  );

  // Everything below the progress line becomes illuminated.
  float illuminated = smoothstep(
    axis - 0.035,
    axis + 0.035,
    u_wake
  );

  // Bright moving band at the edge of the progress.
  float wakeFront = 1.0 - smoothstep(
    0.0,
    0.09,
    abs(axis - u_wake)
  );

  wakeFront *= step(0.001, u_wake);
  wakeFront *= 1.0 - step(0.999, u_wake);

  vec3 dormantColour = vec3(0.08, 0.12, 0.18);
  vec3 lowerColour = vec3(0.48, 0.10, 0.08);
  vec3 upperColour = vec3(0.92, 0.52, 0.18);

  vec3 awakeColour = mix(
    lowerColour,
    upperColour,
    axis
  );

  v_color = mix(
    dormantColour,
    awakeColour,
    illuminated
  );

  v_color += vec3(1.0, 0.68, 0.28)
    * wakeFront
    * 1.25;

  float variation =
    0.65
    + a_tone * 0.5
    + fract(a_glyph * 0.17) * 0.04;

  v_color *= variation;
  v_color = mix(v_color, v_color * a_col, 0.08);

  v_alpha = mix(
    0.13,
    0.82,
    illuminated
  );

  v_alpha += wakeFront * 0.18;
  v_alpha *= u_opacity;

  vec4 viewPosition = u_view * vec4(
    position,
    1.0
  );

  gl_Position = viewPosition;

  float perspectiveSize = clamp(
    28.0 / max(4.0, viewPosition.w),
    0.6,
    2.2
  );

  float pointSize =
    a_size
    * perspectiveSize
    * u_dpr;

  pointSize *= 1.0 + wakeFront * 0.42;

  gl_PointSize = min(pointSize, 30.0);
}
`;

export const FRAGMENT = `#version 300 es
precision highp float;

in vec3 v_color;
in float v_alpha;

out vec4 outColor;

void main() {
  vec2 point = gl_PointCoord * 2.0 - 1.0;
  float radius = length(point);

  float softCircle = 1.0 - smoothstep(
    0.55,
    1.0,
    radius
  );

  // Remove the square edge of every WebGL point.
  if (softCircle < 0.01) {
    discard;
  }

  float glow = 1.0 - smoothstep(
    0.0,
    1.0,
    radius
  );

  vec3 finalColour =
    v_color * (0.78 + glow * 0.42);

  float finalAlpha =
    v_alpha * softCircle;

  outColor = vec4(
    finalColour,
    finalAlpha
  );
}
`;