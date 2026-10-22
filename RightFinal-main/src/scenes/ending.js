import {
  clamp,
  ease,
} from '../core/math.js';

const TRUE_PAUSE_MS = 3000;
const REPORT_ENTER_MS = 1800;
const LETTER_ENTER_MS = 1800;

export class Ending {
  constructor() {
    this.active = false;
    this.startedAt = null;
  }

  start(now) {
    if (this.active) return;

    this.active = true;
    this.startedAt = now;
  }

  update(now) {
    if (!this.active) {
      return {
        phase: 'inactive',
        still: 0,
        report: 0,
        letter: 0,
      };
    }

    const elapsed =
      now - this.startedAt;

    if (elapsed < TRUE_PAUSE_MS) {
      return {
        phase: 'pause',
        still: ease(
          elapsed / TRUE_PAUSE_MS
        ),
        report: 0,
        letter: 0,
      };
    }

    const reportElapsed =
      elapsed - TRUE_PAUSE_MS;

    const report = ease(
      clamp(
        reportElapsed
        / REPORT_ENTER_MS
      )
    );

    const letterElapsed =
      reportElapsed - 1200;

    const letter = ease(
      clamp(
        letterElapsed
        / LETTER_ENTER_MS
      )
    );

    return {
      phase: 'archive',
      still: 1,
      report,
      letter,
    };
  }
}