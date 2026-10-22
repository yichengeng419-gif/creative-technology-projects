async function inflateFile(url) {
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(
      `Could not load ${url}`
    );
  }

  const compressed = await response.arrayBuffer();

  const stream = new Blob([compressed])
    .stream()
    .pipeThrough(
      new DecompressionStream('deflate')
    );

  const decompressed =
    await new Response(stream).arrayBuffer();

  return new Uint8Array(decompressed);
}

export async function loadWakeAssets() {
  const [
    positionBytes,
    roles,
  ] = await Promise.all([
    inflateFile('assets/dna-pos.bin'),
    inflateFile('assets/dna-role.bin'),
  ]);

  const quantizedPositions = new Uint16Array(
    positionBytes.buffer,
    positionBytes.byteOffset,
    positionBytes.byteLength / 2
  );

  const positions = new Float32Array(
    quantizedPositions.length
  );

  for (
    let i = 0;
    i < quantizedPositions.length;
    i++
  ) {
    positions[i] =
      quantizedPositions[i] / 4095 * 40 - 20;
  }

  const pointCount = positions.length / 3;

  if (roles.length !== pointCount) {
    throw new Error(
      'DNA position and role counts do not match'
    );
  }

  return {
    positions,
    roles,
    pointCount,
  };
}
export async function loadFieldAssets() {
  const [
    positionBytes,
    colourBytes,
  ] = await Promise.all([
    inflateFile(
      'assets/landscape-pos.bin'
    ),
    inflateFile(
      'assets/landscape-col.bin'
    ),
  ]);

  if (positionBytes.byteLength % 4 !== 0
    || colourBytes.byteLength % 4 !== 0
  ) {
    throw new Error(
      'Landscape data is not Float32'
    );
  }

  const positions = new Float32Array(
    positionBytes.buffer,
    positionBytes.byteOffset,
    positionBytes.byteLength / 4
  );

  const colours = new Float32Array(
    colourBytes.buffer,
    colourBytes.byteOffset,
    colourBytes.byteLength / 4
  );

  const pointCount =
    positions.length / 3;

  if (
    pointCount !== 100000
    || colours.length !== pointCount * 4
  ) {
    throw new Error(
      'Landscape asset counts do not match'
    );
  }

  return {
    positions,
    colours,
    pointCount,
  };
}