const CHARACTERS = ' .:-=+*#%@';

export class CameraAscii {
  constructor(columns = 56) {
    this.columns = columns;

    this.sampleCanvas =
      document.createElement('canvas');

    this.sampleContext =
      this.sampleCanvas.getContext(
        '2d',
        { willReadFrequently: true }
      );
  }

  draw(video, targetCanvas, time = 0) {
    if (!video.videoWidth) return;

    const aspect =
      video.videoHeight / video.videoWidth;

    const columns = this.columns;
    const rows = Math.round(
      columns * aspect * 0.55
    );

    this.sampleCanvas.width = columns;
    this.sampleCanvas.height = rows;

    this.sampleContext.drawImage(
      video,
      0,
      0,
      columns,
      rows
    );

    const pixels =
      this.sampleContext.getImageData(
        0,
        0,
        columns,
        rows
      ).data;

    const context =
      targetCanvas.getContext('2d');

    const cellWidth =
      targetCanvas.width / columns;

    const cellHeight =
      targetCanvas.height / rows;

    context.fillStyle = '#3d7257b8';
    context.fillRect(
      0,
      0,
      targetCanvas.width,
      targetCanvas.height
    );

    context.font =
      `${Math.ceil(cellHeight * 1.15)}px monospace`;

    context.textBaseline = 'top';

    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < columns; x++) {
        const index =
          (y * columns + x) * 4;

        const red = pixels[index];
        const green = pixels[index + 1];
        const blue = pixels[index + 2];

        const brightness =
          (red * 0.21 +
           green * 0.72 +
           blue * 0.07) / 255;

        const characterIndex = Math.floor(
          brightness *
          (CHARACTERS.length - 1)
        );

        const character =
          CHARACTERS[characterIndex];

        if (character === ' ') continue;

        if (brightness > 0.72) {
          context.fillStyle =
            'rgba(217, 44, 29, 0.95)';
        } else {
          const intensity =
            Math.round(90 + brightness * 165);

          context.fillStyle =
            `rgba(40, ${intensity}, 100, 0.9)`;
        }

        context.fillText(
          character,
          x * cellWidth,
          y * cellHeight
        );
      }
    }

    const scanY =
      (time * 0.04) % targetCanvas.height;

    context.fillStyle =
      'rgba(220, 45, 35, 0.18)';

    context.fillRect(
      0,
      scanY,
      targetCanvas.width,
      2
    );
  }
}