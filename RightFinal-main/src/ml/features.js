/**
 * Convert validated session records into
 * numeric inputs for the classifier.
 */

export const FEATURE_NAMES = [
  'touch_count',
  'mean_hold',
  'retry_delay',
  'aborted_attempts',
  'total_session_time',
  'retry_delay_missing',
];

/**
 * Return zero when an optional measurement
 * is missing from the session.
 *
 * @param {unknown} value
 * @returns {number}
 */
function numberOrZero(value) {
  if (
    value === null
    || value === undefined
    || value === ''
  ) {
    return 0;
  }

  return Number(value);
}

/**
 * Convert one validated session record into
 * a fixed-length numeric feature array.
 *
 * @param {object} row
 * @returns {number[]}
 */
export function featuresForRow(row) {
  const missingRetry =
    row.retry_delay === null
    || row.retry_delay === undefined;

  return [
    numberOrZero(row.touch_count),
    numberOrZero(row.mean_hold),
    numberOrZero(row.retry_delay),
    numberOrZero(row.aborted_attempts),
    numberOrZero(row.total_session_time),
    missingRetry ? 1 : 0,
  ];
}