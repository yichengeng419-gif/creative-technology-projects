import { getContext, createProgram } from './gl.js';
import { FIELD_VERTEX, FIELD_FRAGMENT } from './shaders/index.js';

export class FieldLayer {
  constructor(canvas, positions, colors) {
    const gl = getContext(canvas);
    this.gl = gl;
    this.program = createProgram(gl, FIELD_VERTEX, FIELD_FRAGMENT);
    gl.useProgram(this.program);
    gl.enable(gl.BLEND);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.clearDepth(1);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.clearColor(0.006, 0.010, 0.017, 1);

    this.count = positions.length / 3;

    this.posBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

    this.colorBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, colors, gl.STATIC_DRAW);

    this.attrib = {
      a_pos: gl.getAttribLocation(this.program, 'a_pos'),
      a_color: gl.getAttribLocation(this.program, 'a_color'),
    };

    const uniformNames = [
      'u_phase', 'u_fit', 'u_aspect', 'u_dpr', 'u_cam', 'u_yaw', 'u_pitch', 'u_stretch', 'u_pass',
      'u_pressure', 'u_contact', 'u_hold', 'u_still', 'u_approach', 'u_rhythm', 'u_excess',
      'u_spin', 'u_turn', 'u_space', 'u_hole', 'u_reject', 'u_gather', 'u_lift', 'u_dark', 'u_unarchived',
      'u_focus', 'u_nearPos', 'u_travel', 'u_offset', 'u_archiveEdge',
      'u_atlas', 'u_atlasN',
    ];
    this.uniform = {};
    for (const name of uniformNames) {
      this.uniform[name] = gl.getUniformLocation(this.program, name);
    }
  }

  #bind() {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuffer);
    gl.enableVertexAttribArray(this.attrib.a_pos);
    gl.vertexAttribPointer(this.attrib.a_pos, 3, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.colorBuffer);
    gl.enableVertexAttribArray(this.attrib.a_color);
    gl.vertexAttribPointer(this.attrib.a_color, 4, gl.FLOAT, false, 0, 0);
  }

  render(state) {
    const gl = this.gl;
    gl.viewport(0, 0, state.size.w * state.size.dpr, state.size.h * state.size.dpr);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.useProgram(this.program);
    this.#bind();

    const u = this.uniform;
    gl.uniform1f(u.u_phase, state.time);
    gl.uniform1f(u.u_fit, state.fit ?? 1);
    gl.uniform1f(u.u_aspect, state.size.w / state.size.h);
    gl.uniform1f(u.u_dpr, state.size.dpr);
    gl.uniform1f(u.u_cam, state.camera.dist);
    gl.uniform1f(u.u_yaw, state.camera.yaw);
    gl.uniform1f(u.u_pitch, state.camera.pitch);
    gl.uniform1f(u.u_stretch, state.stretch ?? 1);
    gl.uniform1f(u.u_pass, state.pass ?? 0);

    gl.uniform1f(u.u_pressure, state.pressure ?? 0);
    gl.uniform1f(u.u_contact, state.contact ?? 0);
    gl.uniform1f(u.u_hold, state.hold ?? 0);
    gl.uniform1f(u.u_still, state.still ?? 0);
    gl.uniform1f(u.u_approach, state.approach ?? 0);
    gl.uniform1f(u.u_rhythm, state.rhythm ?? 0);
    gl.uniform1f(u.u_excess, state.excess ?? 0);

    gl.uniform1f(u.u_spin, state.spin ?? 0);
    gl.uniform1f(u.u_turn, state.turn ?? 0);
    gl.uniform1f(u.u_space, state.space ?? 0);
    gl.uniform1f(u.u_hole, state.hole ?? 0);
    gl.uniform1f(u.u_reject, state.reject ?? 0);
    gl.uniform1f(u.u_gather, state.gather ?? 0);
    gl.uniform1f(u.u_lift, state.lift ?? 0);
    gl.uniform1f(u.u_dark, state.dark ?? 1);
    gl.uniform1f(u.u_unarchived, state.unarchived ?? 0);

    gl.uniform2f(u.u_focus, state.focus?.[0] ?? 0, state.focus?.[1] ?? 0);
    gl.uniform2f(u.u_nearPos, state.nearPos?.[0] ?? 0, state.nearPos?.[1] ?? 0);
    gl.uniform2f(u.u_travel, state.travel?.[0] ?? 0, state.travel?.[1] ?? 0);
    gl.uniform2f(u.u_offset, state.offset?.[0] ?? 0, state.offset?.[1] ?? 0);
    gl.uniform2f(u.u_archiveEdge, state.archiveEdge?.[0] ?? -1, state.archiveEdge?.[1] ?? 1);

    gl.uniform1i(u.u_atlas, 0);
    gl.uniform1f(u.u_atlasN, state.atlasN ?? 12);

    gl.drawArrays(gl.POINTS, 0, this.count);
  }
}
