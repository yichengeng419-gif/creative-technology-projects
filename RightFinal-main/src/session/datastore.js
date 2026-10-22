import {
  DATA_KEY,
  TARGET_SAMPLES,
} from '../core/constants.js';

import {
  normaliseRow,
} from './schema.js';

/**
 * Owns the locally collected session dataset.
 *
 * Validation belongs to schema.js.
 * Session behaviour belongs to session.js.
 * This class only manages storage and collection state.
 */
export class DataStore {
  constructor({
    storage = globalThis.localStorage,
  } = {}) {
    this.storage = storage;
    this.rows = [];

    this.#load();
  }

  /**
   * Number of valid sessions currently stored.
   *
   * @returns {number}
   */
  get size() {
    return this.rows.length;
  }

  /**
   * Number of additional sessions needed
   * to reach the research target.
   *
   * @returns {number}
   */
  get remaining() {
    return Math.max(
      0,
      TARGET_SAMPLES - this.size
    );
  }

  /**
   * Return a detached copy of every record.
   *
   * Callers cannot accidentally mutate
   * the DataStore's internal data.
   *
   * @returns {object[]}
   */
  all() {
    return this.rows.map((row) => ({
      ...row,
      holds: [...row.holds],
      gaps: [...row.gaps],
    }));
  }

  /**
   * Find one session by ID.
   *
   * @param {string} sessionId
   * @returns {object|null}
   */
  find(sessionId) {
    return this.rows.find(
      (row) =>
        row.session_id === sessionId
    ) ?? null;
  }

  /**
   * Validate and store one completed session.
   *
   * Returns false when the same session
   * has already been stored.
   *
   * @param {object} rawRecord
   * @returns {boolean}
   */
  add(rawRecord) {
    const row = normaliseRow(
      rawRecord,
      'live'
    );

    if (this.find(row.session_id)) {
      return false;
    }

    this.rows.push(row);
    this.#persist();

    return true;
  }

  /**
   * Read previously collected sessions.
   *
   * Invalid rows are ignored so one damaged
   * record does not break the whole website.
   */
  #load() {
    if (!this.storage) {
      return;
    }

    let storedRows;

    try {
      const storedText =
        this.storage.getItem(DATA_KEY);

      storedRows =
        JSON.parse(storedText || '[]');
    } catch (error) {
      console.warn(
        'Stored sessions could not be read.',
        error
      );

      return;
    }

    if (!Array.isArray(storedRows)) {
      return;
    }

    for (const rawRow of storedRows) {
  try {
    const origin =
      rawRow.provenance === 'live'
        ? 'live'
        : 'imported';

    const row = normaliseRow(
      rawRow,
      origin
    );

    if (!this.find(row.session_id)) {
      this.rows.push(row);
    }
      } catch (error) {
        console.warn(
          'An invalid stored session was skipped.',
          error
        );
      }
    }
  }

  /**
   * Persist the current dataset.
   */
  #persist() {
    if (!this.storage) {
      return;
    }

    try {
      this.storage.setItem(
        DATA_KEY,
        JSON.stringify(this.rows)
      );
    } catch (error) {
      console.warn(
        'Sessions could not be saved.',
        error
      );
    }
  }
}