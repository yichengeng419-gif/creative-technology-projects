/**
 * Validate and normalise every session record.
 *
 * This module has no DOM or storage dependency.
 * Invalid records throw instead of entering the dataset.
 */
import {
  PROTOCOL_VERSION,
} from '../core/constants.js';
const ID_PATTERN =
  /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,120}$/;

const SYNTHETIC =
  /synthetic|generated|test|artificial/i;

const OUTCOMES = [
  'pause',
  'timeout',
];

const POLICIES = [
  'strict',
  'standard',
  'human',
];

const SOURCES = [
  'pointer',
  'keyboard',
  'serial',
  'mixed',
];

const MAX_DURATIONS = 1000;

const SUMMARY_TOLERANCE = 0.025;

/**
 * Convert an optional value to a non-negative number.
 *
 * @param {unknown} value
 * @returns {number|null}
 */
export function numberOrNull(value) {
  if (
    value === null
    || value === undefined
    || String(value).trim() === ''
  ) {
    return null;
  }

  const number = Number(value);

  if (
    !Number.isFinite(number)
    || number < 0
  ) {
    throw new Error(
      'Expected a non-negative number'
    );
  }

  return number;
}

/**
 * Convert holds or gaps into a validated duration array.
 *
 * Arrays are accepted directly. JSON strings are accepted
 * so the same validator can process imported CSV rows.
 *
 * @param {unknown} value
 * @returns {number[]}
 */
export function durationArray(value) {
  let durations;

  try {
    durations = Array.isArray(value)
      ? value
      : JSON.parse(value || '[]');
  } catch {
    throw new Error(
      'Invalid holds or gaps JSON'
    );
  }

  if (
    !Array.isArray(durations)
    || durations.length > MAX_DURATIONS
  ) {
    throw new Error(
      'Invalid holds or gaps array'
    );
  }

  return durations.map((item) => {
    const duration = numberOrNull(item);

    if (duration === null) {
      throw new Error(
        'Missing duration'
      );
    }

    return duration;
  });
}

/**
 * Calculate the mean of a duration array.
 *
 * @param {number[]} durations
 * @returns {number}
 */
function meanDuration(durations) {
  if (durations.length === 0) {
    return 0;
  }

  const total = durations.reduce(
    (sum, duration) =>
      sum + duration,
    0
  );

  return total / durations.length;
}

/**
 * Validate and normalise one session record.
 *
 * @param {object} raw
 * @param {'live'|'imported'} origin
 * @returns {object}
 */
export function normaliseRow(
  raw,
  origin = 'imported'
) {
  if (
    raw === null
    || typeof raw !== 'object'
    || Array.isArray(raw)
  ) {
    throw new Error(
      'Session record must be an object'
    );
  }

  const sessionId =
    String(raw.session_id || '').trim();

  if (!ID_PATTERN.test(sessionId)) {
    throw new Error(
      'Invalid session ID'
    );
  }

  const outcome =
    String(raw.outcome || '')
      .trim()
      .toLowerCase();

  if (!OUTCOMES.includes(outcome)) {
    throw new Error(
      'Only completed pause or timeout records can be used'
    );
  }

  const provenance =
    String(raw.provenance || '').trim();

  if (SYNTHETIC.test(provenance)) {
    throw new Error(
      'Synthetic or test records are not allowed'
    );
  }

  const holds =
    durationArray(raw.holds);

  const gaps =
    durationArray(raw.gaps);

  const touchCount =
    numberOrNull(raw.touch_count);

  const rawContactCount =
    numberOrNull(
      raw.raw_contact_count
    );

  const abortedAttempts =
    numberOrNull(
      raw.aborted_attempts
    );

  const totalSessionTime =
    numberOrNull(
      raw.total_session_time
    );

  if (
    touchCount === null
    || !Number.isInteger(touchCount)
    || touchCount !== holds.length
  ) {
    throw new Error(
      'Touch count does not match holds'
    );
  }

  if (
    abortedAttempts === null
    || !Number.isInteger(abortedAttempts)
    || abortedAttempts > touchCount
  ) {
    throw new Error(
      'Invalid aborted attempt count'
    );
  }

  if (totalSessionTime === null) {
    throw new Error(
      'Missing total session time'
    );
  }

  const contactCount =
    rawContactCount ?? touchCount;

  if (
    !Number.isInteger(contactCount)
    || contactCount < touchCount
  ) {
    throw new Error(
      'Invalid raw contact count'
    );
  }

  const firstHold =
    numberOrNull(raw.first_hold);

  const secondHold =
    numberOrNull(raw.second_hold);

  const retryDelay =
    numberOrNull(raw.retry_delay);

  if (
    outcome === 'pause'
    && (
      firstHold === null
      || secondHold === null
    )
  ) {
    throw new Error(
      'A successful pause needs both holds'
    );
  }

  const calculatedMean =
    meanDuration(holds);

  const calculatedMax =
    holds.length > 0
      ? Math.max(...holds)
      : 0;

  const suppliedMean =
    numberOrNull(raw.mean_hold);

  const suppliedMax =
    numberOrNull(raw.max_hold);

  if (
    suppliedMean !== null
    && Math.abs(
      suppliedMean - calculatedMean
    ) > SUMMARY_TOLERANCE
  ) {
    throw new Error(
      'Mean hold does not match holds'
    );
  }

  if (
    suppliedMax !== null
    && Math.abs(
      suppliedMax - calculatedMax
    ) > SUMMARY_TOLERANCE
  ) {
    throw new Error(
      'Maximum hold does not match holds'
    );
  }

  const policy =
    POLICIES.includes(
      raw.observation_policy
    )
      ? raw.observation_policy
      : 'standard';

  const inputSource =
    SOURCES.includes(raw.input_source)
      ? raw.input_source
      : 'mixed';

  return {
    session_id: sessionId,
    outcome,
    touch_count: touchCount,
    raw_contact_count: contactCount,
    holds,
    gaps,
    first_hold: firstHold,
    second_hold: secondHold,

    mean_hold: Number(
      calculatedMean.toFixed(3)
    ),

    max_hold: Number(
      calculatedMax.toFixed(3)
    ),

    retry_delay: retryDelay,
    aborted_attempts: abortedAttempts,
    total_session_time: Number(
      totalSessionTime.toFixed(3)
    ),

    observation_policy: policy,
    input_source: inputSource,

    protocol_version: String(
      raw.protocol_version || 'R16'
    ).slice(0, 32),

    provenance:
      origin === 'live'
        ? 'live'
        : 'imported',
  };
}