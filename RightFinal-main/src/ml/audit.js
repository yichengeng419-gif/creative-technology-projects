const CLASS_KEYS = [
  'early',
  'interrupted',
  'pause',
];

export const CLASS_LABELS = {
  early: 'EARLY',
  interrupted: 'INTERRUPTED',
  pause: 'PAUSE',
};

export const MIN_CLASS_SAMPLES = 5;

/**
 * Convert a completed Session outcome into
 * one of the three classes shown by the audit UI.
 *
 * @param {object} row
 * @returns {'early'|'interrupted'|'pause'}
 */
export function classForRow(row) {
  if (row.outcome === 'pause') {
    return 'pause';
  }

  return row.first_hold === null
    ? 'early'
    : 'interrupted';
}

/**
 * Produce a read-only summary of the current dataset.
 * No model is trained in this phase.
 *
 * @param {object[]} rows
 * @param {number} target
 * @returns {object}
 */
export function auditDataset(rows, target) {
  const counts = {
    early: 0,
    interrupted: 0,
    pause: 0,
  };

  for (const row of rows) {
    counts[classForRow(row)]++;
  }

  const ready = CLASS_KEYS.every(
    key => counts[key] >= MIN_CLASS_SAMPLES
  );

  return {
    total: rows.length,
    remaining: Math.max(
      0,
      target - rows.length
    ),
    counts,
    ready,
  };
}
