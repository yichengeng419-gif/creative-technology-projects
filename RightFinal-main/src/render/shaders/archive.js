export const ARCHIVE_GLSL = `
void computeArchive(
  vec2 metricIn, float layer, float seed, float strain, float aspect,
  vec2 nearPos, vec2 archiveEdge, float u_gather, float u_lift, float u_unarchived,
  out vec2 metricOut, out float ascii, out float glyphIndex, out float absence
) {
  metricOut = metricIn;

  // Unarchived:
  absence = u_unarchived * (1.0 - smoothstep(.1, .42, length(metricOut - nearPos)));
  metricOut += normalize(metricOut - nearPos + vec2(.00001)) * absence * .10;

  vec2 q = vec2(metricOut.x / aspect, metricOut.y);

  float thread = floor(layer * 29.0);
  float threadX = mix(archiveEdge.x, archiveEdge.y, thread / 28.0);
  float tremor = sin(seed * 148.0 + u_lift * 5.0) * .002 * (1.0 - u_lift);
  q.x = mix(q.x, threadX + tremor, u_gather);
  q.y = mix(q.y, 1.25 + layer * .65, u_lift);

  metricOut = vec2(q.x * aspect, q.y);

  // Points under strain 
  float denseCore = 1.0 - smoothstep(.16, .72, length(metricOut));
  float asciiGate = step(.56, fract(seed * 41.73 + layer * 7.1));
  ascii = 0.0;
  glyphIndex = floor(fract(seed * 17.31 + layer * .73) * 12.0);
}
`;