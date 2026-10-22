 
import {
  getContext,
  createProgram,
} from './gl.js';

import {
  create,
  multiply,
  perspective,
  lookAt,
  rotateX,
  rotateY,
  rotateZ,
} from './mat4.js';

import {
  VERTEX,
  FRAGMENT,
} from './shaders/wakeshader.js';

function random(index, channel) {
  const value = Math.sin(
    index * 127.1 + channel * 311.7
  ) * 43758.5453;

  return value - Math.floor(value);
}

function buildDnaMesh(gl, positions, roles) {
  const count = positions.length / 3;

  const data = {
    pos: new Float32Array(positions),
    glyph: new Float32Array(count),
    seed: new Float32Array(count),
    size: new Float32Array(count),
    scat: new Float32Array(count * 3),
    tone: new Float32Array(count),
    col: new Float32Array(count * 3),
  };

  for (let i = 0; i < count; i++) {
    const seed = random(i, 1);

    data.seed[i] = seed;
    data.glyph[i] = Math.floor(
      random(i, 2) * 8
    );

    data.size[i] =
      13 + random(i, 3) * 13;

    const theta =
      random(i, 4) * Math.PI * 2;

    const phi = Math.acos(
      2 * random(i, 5) - 1
    );

    data.scat[i * 3] =
      Math.sin(phi) * Math.cos(theta);

    data.scat[i * 3 + 1] =
      Math.cos(phi) * 0.6;

    data.scat[i * 3 + 2] =
      Math.sin(phi) * Math.sin(theta);

    const role = roles[i] ?? 1;

    data.tone[i] = role
      ? 0.45 + random(i, 6) * 0.55
      : 0.22 + random(i, 6) * 0.3;

    data.col[i * 3] = 0.91;
    data.col[i * 3 + 1] = 0.89;
    data.col[i * 3 + 2] = 0.83;
  }

  const dimensions = {
    pos: 3,
    glyph: 1,
    seed: 1,
    size: 1,
    scat: 3,
    tone: 1,
    col: 3,
  };

  const buffers = {};

  for (const key of Object.keys(data)) {
    const buffer = gl.createBuffer();

    gl.bindBuffer(
      gl.ARRAY_BUFFER,
      buffer
    );

    gl.bufferData(
      gl.ARRAY_BUFFER,
      data[key],
      gl.STATIC_DRAW
    );

    buffers[key] = {
      buffer,
      dimension: dimensions[key],
    };
  }

  return {
    buffers,
    count,
  };
}

export class PointCloud {
  constructor(canvas, positions, roles) {
    const gl = getContext(canvas);

    this.gl = gl;

    this.program = createProgram(
      gl,
      VERTEX,
      FRAGMENT
    );

    gl.useProgram(this.program);

    gl.enable(gl.BLEND);
    gl.enable(gl.DEPTH_TEST);

    gl.blendFunc(
      gl.SRC_ALPHA,
      gl.ONE_MINUS_SRC_ALPHA
    );

    gl.depthFunc(gl.LEQUAL);
    gl.clearDepth(1);

    gl.clearColor(
      0.012,
      0.020,
      0.033,
      1
    );

    this.attributes = {};

    for (const name of [
      'a_pos',
      'a_glyph',
      'a_seed',
      'a_size',
      'a_scat',
      'a_tone',
      'a_col',
    ]) {
      this.attributes[name] =
        gl.getAttribLocation(
          this.program,
          name
        );
    }

    this.uniforms = {};

    for (const name of [
      'u_view',
      'u_time',
      'u_dpr',
      'u_wake',
      'u_jitter',
      'u_scatter',
      'u_opacity',
    ]) {
      this.uniforms[name] =
        gl.getUniformLocation(
          this.program,
          name
        );
    }

    this.mesh = buildDnaMesh(
      gl,
      positions,
      roles
    );

    this.camera = {
      yaw: 0,
      pitch: 0,
      distance: 28,
      fov: 0.6,
    };
    this.eye = new Float32Array(3);

    this.matrices = Array.from(
      { length: 9 },
      create
    );
  }

  #bindMesh() {
    const gl = this.gl;

    for (
      const [key, item]
      of Object.entries(this.mesh.buffers)
    ) {
      const location =
        this.attributes[`a_${key}`];

      if (location < 0) continue;

      gl.bindBuffer(
        gl.ARRAY_BUFFER,
        item.buffer
      );

      gl.enableVertexAttribArray(location);

      gl.vertexAttribPointer(
        location,
        item.dimension,
        gl.FLOAT,
        false,
        0,
        0
      );
    }
  }

  #viewMatrix(
  width,
  height,
  spin,
  distance
) {
  const camera = this.camera;

  this.eye[0] =
    Math.sin(camera.yaw) * distance;

  this.eye[1] =
    Math.sin(camera.pitch)
    * distance
    * 0.5;

  this.eye[2] =
    Math.cos(camera.yaw) * distance;

  const [
    projection,
    view,
    rotationX,
    rotationY,
    rotationZ,
    xyRotation,
    rotation,
    modelView,
    finalMatrix,
  ] = this.matrices;

  perspective(
    projection,
    camera.fov,
    width / height,
    0.5,
    300
  );

  lookAt(
    view,
    this.eye,
    [0, 0, 0],
    [0, 1, 0]
  );

  rotateX(rotationX, 0.18);
  rotateY(rotationY, spin);
  rotateZ(rotationZ, -0.68);

  multiply(
    xyRotation,
    rotationX,
    rotationY
  );

  multiply(
    rotation,
    rotationZ,
    xyRotation
  );

  multiply(
    modelView,
    view,
    rotation
  );

  return multiply(
    finalMatrix,
    projection,
    modelView
  );
}

  render({
    size,
    time,
    wake,
    scatter = 0,
    opacity = 1,
    spin = 0,
    distance = this.camera.distance,
  }) {
    const gl = this.gl;

    gl.viewport(
      0,
      0,
      size.w * size.dpr,
      size.h * size.dpr
    );

    gl.clear(
      gl.COLOR_BUFFER_BIT
      | gl.DEPTH_BUFFER_BIT
    );

    gl.useProgram(this.program);

    gl.uniform1f(
      this.uniforms.u_time,
      time
    );

    gl.uniform1f(
      this.uniforms.u_dpr,
      size.dpr
    );

    gl.uniform1f(
      this.uniforms.u_wake,
      wake
    );

    gl.uniform1f(
      this.uniforms.u_jitter,
      0.02
    );

    gl.uniform1f(
      this.uniforms.u_scatter,
      scatter
    );

    gl.uniform1f(
      this.uniforms.u_opacity,
      opacity
    );

    gl.uniformMatrix4fv(
      this.uniforms.u_view,
      false,
      this.#viewMatrix(
        size.w,
        size.h,
        spin,
        distance
      )
    );

    this.#bindMesh();

    gl.drawArrays(
      gl.POINTS,
      0,
      this.mesh.count
    );
  }
}