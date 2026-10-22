export function createDossierCanvas() {
  const canvas =
    document.createElement('canvas');

  canvas.width = 1200;
  canvas.height = 1600;

  const context =
    canvas.getContext('2d');

  context.fillStyle = '#060a08';
  context.fillRect(
    0,
    0,
    canvas.width,
       canvas.height
  );

  context.strokeStyle = '#476f45';
  context.lineWidth = 3;
  context.strokeRect(
    60,
    60,
    canvas.width - 120,
    canvas.height - 120
  );

  context.fillStyle = '#d8e2da';
  context.font = '32px monospace';

  context.fillText(
    'THE RIGHT TO PAUSE',
    100,
    130
  );

  context.fillStyle = '#7ca889';
  context.font = '20px monospace';

  context.fillText(
    'LOCAL SESSION',
    100,
    175
  );

  return {
    canvas,
    context,
  };
}
export function drawEvidence(
  context,
  evidenceCanvas
) {
  const x = 100;
  const y = 230;
  const width = 1000;
  const height = 750;

  context.fillStyle = '#010302';
  context.fillRect(
    x,
    y,
    width,
    height
  );

  context.drawImage(
    evidenceCanvas,
    x,
    y,
    width,
    height
  );

  context.strokeStyle = '#cf3f32';
  context.lineWidth = 3;
  context.strokeRect(
    x,
    y,
    width,
    height
  );

  context.fillStyle = '#cf3f32';
  context.font = '18px monospace';

  context.fillText(
    'VISUAL EVIDENCE / ASCII OBSERVATION',
    x,
    y + height + 35
  );
}
function drawWrappedText(
  context,
  text,
  x,
  y,
  maxWidth,
  lineHeight
) {
  const words = text.split(' ');
  let line = '';

  for (const word of words) {
    const nextLine =
      line ? `${line} ${word}` : word;

    if (
      context.measureText(nextLine).width >
      maxWidth
    ) {
      context.fillText(line, x, y);
      line = word;
      y += lineHeight;
    } else {
      line = nextLine;
    }
  }

  if (line) {
    context.fillText(line, x, y);
    y += lineHeight;
  }

  return y;
}
export function drawDossierText(
  context,
  dossier
) {
  let y = 1080;

  context.font = '18px monospace';
  context.fillStyle = '#67d58b';
  context.fillText(
    'SYSTEM REPORT',
    100,
    y
  );

  y += 40;

  context.font = '24px monospace';
  context.fillStyle = '#d7e0d9';

  y = drawWrappedText(
    context,
    dossier.report,
    100,
    y,
    1000,
    34
  );

  y += 30;

  context.font = '18px monospace';
  context.fillStyle = '#cf3f32';
  context.fillText(
    'SYSTEM INTERPRETATION',
    100,
    y
  );

  y += 40;

  context.font = '22px monospace';
  context.fillStyle = '#c8d0ca';

  y = drawWrappedText(
    context,
    dossier.interpretation,
    100,
    y,
    1000,
    32
  );

  y += 30;

  context.font = '18px monospace';
  context.fillStyle = '#67d58b';
  context.fillText(
    'COUNTER-LETTER',
    100,
    y
  );

  y += 40;

  context.font = '24px monospace';
  context.fillStyle = '#eef4ef';

  drawWrappedText(
    context,
    dossier.statement,
    100,
    y,
    1000,
    34
  );
}
export function downloadDossier({
  evidenceCanvas,
  dossier,
}) {
  const {
    canvas,
    context,
  } = createDossierCanvas();

  drawEvidence(
    context,
    evidenceCanvas
  );

  drawDossierText(
    context,
    dossier
  );

  canvas.toBlob(
    blob => {
      if (!blob) return;

      const url =
        URL.createObjectURL(blob);

      const link =
        document.createElement('a');

      const date =
        new Date()
          .toISOString()
          .slice(0, 19)
          .replaceAll(':', '-');

      link.href = url;
      link.download =
        `right-to-pause-${date}.png`;

      link.click();

      URL.revokeObjectURL(url);
    },
    'image/png'
  );
}