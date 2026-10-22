/**
 * Compiles a vertex + fragment shader pair into a linked WebGL program.
 * Both DNA and field renderers call this once, at setup.
 */
export function createProgram(gl, vertexSource, fragmentSource) {
  const compile = (type, source) => {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(shader));
    }
    return shader;
  };

  const program = gl.createProgram();
  gl.attachShader(program, compile(gl.VERTEX_SHADER, vertexSource));
  gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fragmentSource));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program));
  }
  return program;
}

/**
 * Opens a WebGL2 context on a canvas with the settings both renderers share.
 * Throws with a clear message if WebGL2 isn't available, rather than
 * failing later with a confusing null-pointer error.
 */
export function getContext(canvas) {
  const gl = canvas.getContext('webgl2', {
    alpha: false,
    antialias: false,
    powerPreference: 'high-performance',
  });
  if (!gl) throw new Error('WebGL 2 is unavailable');
  return gl;
}
