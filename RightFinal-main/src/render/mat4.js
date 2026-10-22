/**
 * Column-major 4x4 matrices.
 * Every operation writes into a caller-supplied output matrix.
 */

export const create = () =>
  new Float32Array(16);

export function perspective(
  out,
  fieldOfView,
  aspect,
  near,
  far
) {
  out.fill(0);

  const scale =
    1 / Math.tan(fieldOfView / 2);

  const depth =
    1 / (near - far);

  out[0] = scale / aspect;
  out[5] = scale;
  out[10] = (far + near) * depth;
  out[11] = -1;
  out[14] = 2 * far * near * depth;

  return out;
}

export function multiply(out, a, b) {
  for (let column = 0; column < 4; column++) {
    for (let row = 0; row < 4; row++) {
      let sum = 0;

      for (let index = 0; index < 4; index++) {
        sum +=
          a[index * 4 + row]
          * b[column * 4 + index];
      }

      out[column * 4 + row] = sum;
    }
  }

  return out;
}

export function lookAt(
  out,
  eye,
  target,
  up
) {
  let zx = eye[0] - target[0];
  let zy = eye[1] - target[1];
  let zz = eye[2] - target[2];

  const zLength =
    Math.hypot(zx, zy, zz) || 1;

  zx /= zLength;
  zy /= zLength;
  zz /= zLength;

  let xx = up[1] * zz - up[2] * zy;
  let xy = up[2] * zx - up[0] * zz;
  let xz = up[0] * zy - up[1] * zx;

  const xLength =
    Math.hypot(xx, xy, xz) || 1;

  xx /= xLength;
  xy /= xLength;
  xz /= xLength;

  const yx = zy * xz - zz * xy;
  const yy = zz * xx - zx * xz;
  const yz = zx * xy - zy * xx;

  out[0] = xx;
  out[1] = yx;
  out[2] = zx;
  out[3] = 0;

  out[4] = xy;
  out[5] = yy;
  out[6] = zy;
  out[7] = 0;

  out[8] = xz;
  out[9] = yz;
  out[10] = zz;
  out[11] = 0;

  out[12] = -(
    xx * eye[0]
    + xy * eye[1]
    + xz * eye[2]
  );

  out[13] = -(
    yx * eye[0]
    + yy * eye[1]
    + yz * eye[2]
  );

  out[14] = -(
    zx * eye[0]
    + zy * eye[1]
    + zz * eye[2]
  );

  out[15] = 1;

  return out;
}

export function rotateX(out, angle) {
  out.fill(0);

  const cosine = Math.cos(angle);
  const sine = Math.sin(angle);

  out[0] = 1;
  out[5] = cosine;
  out[6] = sine;
  out[9] = -sine;
  out[10] = cosine;
  out[15] = 1;

  return out;
}

export function rotateY(out, angle) {
  out.fill(0);

  const cosine = Math.cos(angle);
  const sine = Math.sin(angle);

  out[0] = cosine;
  out[2] = -sine;
  out[5] = 1;
  out[8] = sine;
  out[10] = cosine;
  out[15] = 1;

  return out;
}

export function rotateZ(out, angle) {
  out.fill(0);

  const cosine = Math.cos(angle);
  const sine = Math.sin(angle);

  out[0] = cosine;
  out[1] = sine;
  out[4] = -sine;
  out[5] = cosine;
  out[10] = 1;
  out[15] = 1;

  return out;
}