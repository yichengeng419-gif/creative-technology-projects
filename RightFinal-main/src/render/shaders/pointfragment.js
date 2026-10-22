export const POINTFRAGMENT_GLSL = `#version 300 es
precision highp float;
uniform sampler2D u_atlas;
uniform float u_atlasN, u_pass;
in vec3 v_col;
in float v_alpha;
in float v_angle;
in float v_thin;
in float v_ascii;
in float v_glyph;
out vec4 outColor;

void main() {
  vec2 p = gl_PointCoord * 2.0 - 1.0;
  float c = cos(v_angle), s = sin(v_angle);
  vec2 q = vec2(c * p.x + s * p.y, -s * p.x + c * p.y);
  q.y /= max(.045, v_thin);
  float d = dot(q, q);
  float dotShape = 1.0 - smoothstep(.08, 1.0, d);

  vec2 glyphUv = vec2((v_glyph + gl_PointCoord.x) / u_atlasN, gl_PointCoord.y);
  float glyphShape = texture(u_atlas, glyphUv).a;

  float shape = mix(mix(dotShape, glyphShape, v_ascii), dotShape, u_pass);
  float a = shape * min(.87, v_alpha) * mix(1.0, .115, u_pass);
  if (a < .003) discard;
  outColor = vec4(v_col, a);
}`;
