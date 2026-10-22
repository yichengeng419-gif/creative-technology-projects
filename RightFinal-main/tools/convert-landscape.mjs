import {
  readFileSync,
  writeFileSync,
} from 'node:fs';

import {
  deflateSync,
} from 'node:zlib';

const sourcePath = process.argv[2];

if (!sourcePath) {
  throw new Error(
    'Please provide a GLB file path'
  );
}

const file = readFileSync(sourcePath);

const magic = file.toString(
  'utf8',
  0,
  4
);

if (magic !== 'glTF') {
  throw new Error(
    'This file is not a valid GLB model'
  );
}

const version = file.readUInt32LE(4);
const fileLength = file.readUInt32LE(8);

let offset = 12;
let model = null;
let binary = null;

while (offset < fileLength) {
  const chunkLength =
    file.readUInt32LE(offset);

  const chunkType =
    file.readUInt32LE(offset + 4);

  const chunkStart = offset + 8;
  const chunkEnd =
    chunkStart + chunkLength;

  const chunk = file.subarray(
    chunkStart,
    chunkEnd
  );

  if (chunkType === 0x4e4f534a) {
    model = JSON.parse(
      chunk.toString('utf8')
        .replace(/\0+$/, '')
        .trim()
    );
  }

  if (chunkType === 0x004e4942) {
    binary = chunk;
  }

  offset = chunkEnd;
}

if (!model || !binary) {
  throw new Error(
    'The GLB JSON or binary chunk is missing'
  );
}

console.log('GLB version:', version);
console.log('Meshes:', model.meshes.length);
console.log('Binary bytes:', binary.length);
const componentCounts = {
  SCALAR: 1,
  VEC2: 2,
  VEC3: 3,
  VEC4: 4,
};

function readFloatAccessor(accessorIndex) {
  const accessor =
    model.accessors[accessorIndex];

  if (!accessor) {
    throw new Error(
      `Accessor ${accessorIndex} does not exist`
    );
  }

  // 5126 is glTF's code for Float32.
  if (accessor.componentType !== 5126) {
    throw new Error(
      `Accessor ${accessorIndex} is not Float32`
    );
  }

  const componentCount =
    componentCounts[accessor.type];

  if (!componentCount) {
    throw new Error(
      `Unsupported accessor type: ${accessor.type}`
    );
  }

  const bufferView =
    model.bufferViews[accessor.bufferView];

  const byteOffset =
    (bufferView.byteOffset ?? 0)
    + (accessor.byteOffset ?? 0);

  const normalStride =
    componentCount * 4;

  const byteStride =
    bufferView.byteStride
    ?? normalStride;

  const values = new Float32Array(
    accessor.count * componentCount
  );

  if (byteStride === normalStride) {
    const source = new Float32Array(
      binary.buffer,
      binary.byteOffset + byteOffset,
      accessor.count * componentCount
    );

    values.set(source);
    return values;
  }

  const view = new DataView(
    binary.buffer,
    binary.byteOffset,
    binary.byteLength
  );

  for (let i = 0; i < accessor.count; i++) {
    for (
      let component = 0;
      component < componentCount;
      component++
    ) {
      values[
        i * componentCount + component
      ] = view.getFloat32(
        byteOffset
        + i * byteStride
        + component * 4,
        true
      );
    }
  }

  return values;
}

const positionGroups = [];
const colourGroups = [];

let totalPoints = 0;

for (const mesh of model.meshes) {
  for (const primitive of mesh.primitives) {
    if (primitive.mode !== 0) {
      continue;
    }

    const positionAccessor =
      primitive.attributes.POSITION;

    const colourAccessor =
      primitive.attributes.COLOR_0;

    if (
      positionAccessor === undefined
      || colourAccessor === undefined
    ) {
      continue;
    }

    const positions =
      readFloatAccessor(positionAccessor);

    const colours =
      readFloatAccessor(colourAccessor);

    const pointCount =
      positions.length / 3;

    if (colours.length !== pointCount * 4) {
      throw new Error(
        'A mesh has mismatched position and colour data'
      );
    }

    positionGroups.push(positions);
    colourGroups.push(colours);

    totalPoints += pointCount;
  }
}

console.log(
  'Point groups:',
  positionGroups.length
);

console.log(
  'Total source points:',
  totalPoints
);
const TARGET_POINTS = 100000;

const sampledPositions = new Float32Array(
  TARGET_POINTS * 3
);

const sampledColours = new Float32Array(
  TARGET_POINTS * 4
);

const sceneIndex = model.scene ?? 0;
const rootNodeIndex =
  model.scenes[sceneIndex].nodes[0];

const rootNode =
  model.nodes[rootNodeIndex];

const rootMatrix = rootNode.matrix ?? [
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
];

function transformPoint(x, y, z) {
  return [
    rootMatrix[0] * x
      + rootMatrix[4] * y
      + rootMatrix[8] * z
      + rootMatrix[12],

    rootMatrix[1] * x
      + rootMatrix[5] * y
      + rootMatrix[9] * z
      + rootMatrix[13],

    rootMatrix[2] * x
      + rootMatrix[6] * y
      + rootMatrix[10] * z
      + rootMatrix[14],
  ];
}

const sampleStep =
  totalPoints / TARGET_POINTS;

let nextSample = 0;
let sourceIndex = 0;
let sampledCount = 0;

for (
  let groupIndex = 0;
  groupIndex < positionGroups.length;
  groupIndex++
) {
  const positions =
    positionGroups[groupIndex];

  const colours =
    colourGroups[groupIndex];

  const groupPointCount =
    positions.length / 3;

  for (
    let i = 0;
    i < groupPointCount;
    i++
  ) {
    if (
      sourceIndex >= nextSample
      && sampledCount < TARGET_POINTS
    ) {
      const transformed = transformPoint(
        positions[i * 3],
        positions[i * 3 + 1],
        positions[i * 3 + 2]
      );

      sampledPositions.set(
        transformed,
        sampledCount * 3
      );

      sampledColours.set(
        colours.subarray(
          i * 4,
          i * 4 + 4
        ),
        sampledCount * 4
      );

      sampledCount++;
      nextSample =
        sampledCount * sampleStep;
    }

    sourceIndex++;
  }
}

const positions = sampledPositions.subarray(
  0,
  sampledCount * 3
);

const colours = sampledColours.subarray(
  0,
  sampledCount * 4
);

const minimum = [
  Infinity,
  Infinity,
  Infinity,
];

const maximum = [
  -Infinity,
  -Infinity,
  -Infinity,
];

for (let i = 0; i < sampledCount; i++) {
  for (let axis = 0; axis < 3; axis++) {
    const value = positions[i * 3 + axis];

    minimum[axis] = Math.min(
      minimum[axis],
      value
    );

    maximum[axis] = Math.max(
      maximum[axis],
      value
    );
  }
}

const centre = [
  (minimum[0] + maximum[0]) / 2,
  (minimum[1] + maximum[1]) / 2,
  (minimum[2] + maximum[2]) / 2,
];

const largestExtent = Math.max(
  maximum[0] - minimum[0],
  maximum[1] - minimum[1],
  maximum[2] - minimum[2]
);

const normaliseScale =
  2 / largestExtent;

for (let i = 0; i < sampledCount; i++) {
  positions[i * 3] =
    (positions[i * 3] - centre[0])
    * normaliseScale;

  positions[i * 3 + 1] =
    (positions[i * 3 + 1] - centre[1])
    * normaliseScale;

  positions[i * 3 + 2] =
    (positions[i * 3 + 2] - centre[2])
    * normaliseScale;
}

console.log(
  'Sampled points:',
  sampledCount
);

console.log(
  'Original bounds:',
  minimum,
  maximum
);

console.log(
  'Normalise scale:',
  normaliseScale
);
const positionBuffer = Buffer.from(
  positions.buffer,
  positions.byteOffset,
  positions.byteLength
);

const colourBuffer = Buffer.from(
  colours.buffer,
  colours.byteOffset,
  colours.byteLength
);

const compressedPositions = deflateSync(
  positionBuffer,
  {
    level: 9,
  }
);

const compressedColours = deflateSync(
  colourBuffer,
  {
    level: 9,
  }
);

writeFileSync(
  'assets/landscape-pos.bin',
  compressedPositions
);

writeFileSync(
  'assets/landscape-col.bin',
  compressedColours
);

console.log(
  'Position file:',
  compressedPositions.length,
  'bytes'
);

console.log(
  'Colour file:',
  compressedColours.length,
  'bytes'
);

console.log(
  'Landscape assets written successfully'
);