/**
 * Shared configuration values.
 *
 * Interaction rules and storage names are defined here
 * so every module uses the same source of truth.
 */

/** Hold durations required by the two pause stages. */
export const FIRST_HOLD_MS = 5000;
export const SECOND_HOLD_MS = 8000;

/** Wake and transition timings. */
export const WAKE_HOLD_MS = 5000;
export const WAKE_DECAY_S = 0.8;
export const TRANSITION_MS = 5200;

/** Verification windows. */
export const SESSION_LIMIT_MS = 35000;
export const SECOND_HOLD_WINDOW_MS = 10000;

/**
 * Contacts shorter than this are treated as sensor noise.
 * Value is measured in seconds.
 */
export const MIN_HOLD_S = 0.2;

/**
 * Three contacts separated by less than this duration
 * are interpreted as behavioural pressure.
 */
export const PRESSURE_GAP_MS = 900;

/** Mouse proximity to the field, in CSS pixels. */
export const APPROACH_RADIUS = 80;
export const APPROACH_EXIT_RADIUS = 104;
export const APPROACH_DWELL_MS = 180;

/** Serial connection settings. */
export const SERIAL_BAUD = 115200;
export const SERIAL_TIMEOUT_MS = 1600;
export const SERIAL_BUFFER_LIMIT = 4096;

/** Temporary interface-event timings. */
export const EVENT_MS = 2600;
export const STORY_MS = 2400;
export const PRESSURE_FLASH_MS = 2000;
export const REJECT_FLASH_MS = 1900;
export const MOTION_FLASH_MS = 1800;

/** Archive and closing-sequence timings. */
export const ARCHIVE_REPORT_MS = 4400;
export const LETTER_RISE_MS = 2800;
export const ACTIONS_READY_MS = 4500;
export const RESET_FADE_MS = 1200;

/** Frame-loop and rendering limits. */
export const MAX_DT_S = 0.06;
export const MAX_DPR = 1.75;
export const DOSSIER_INTERVAL_MS = 100;
export const PREDICT_INTERVAL_MS = 400;
export const CAMERA_SAMPLE_MS = 83;

/**
 * Version of the interaction rules used
 * to collect each session.
 */
export const PROTOCOL_VERSION =
  'RTP-PILOT-1';

/**
 * Browser storage keys.
 *
 * Changing these names creates a new local dataset
 * without deleting data saved under older keys.
 */
export const DATA_KEY =
  'rightToPausePilot1Sessions';

export const MODEL_KEY =
  'rightToPausePilot1Model';

/** Dataset and classifier configuration. */
export const TARGET_SAMPLES = 80;
export const TREE_COUNT = 48;

export const DISPLAY_CLASSES = [
  'EARLY WITHDRAWAL',
  'COMPLIANCE INTERRUPTED',
  'PAUSE ACHIEVED',
];

/**
 * Column order used when session records
 * are exported to CSV.
 */
export const CSV_COLUMNS = [
  'session_id',
  'outcome',
  'touch_count',
  'raw_contact_count',
  'holds',
  'gaps',
  'first_hold',
  'second_hold',
  'mean_hold',
  'max_hold',
  'retry_delay',
  'aborted_attempts',
  'total_session_time',
  'observation_policy',
  'input_source',
  'protocol_version',
  'provenance',
];

/** Imported dataset safety limits. */
export const MAX_IMPORT_BYTES =
  5 * 1024 * 1024;

export const MAX_IMPORT_ROWS = 2000;

/** Camera observation policies. */
export const POLICY_RULES = {
  strict: {
    threshold: 0.035,
    dwell: 150,
    grace: 0,

    detail:
      'Optional strict mode: even small movement may interrupt a hold.',
  },

  standard: {
    threshold: 0.095,
    dwell: 1200,
    grace: 900,

    detail:
      'Small movements are allowed. Sustained large movement can interrupt a hold.',
  },

  human: {
    threshold: Infinity,
    dwell: Infinity,
    grace: 0,

    detail:
      'Natural movement is accepted. Contact still determines the hold.',
  },
};