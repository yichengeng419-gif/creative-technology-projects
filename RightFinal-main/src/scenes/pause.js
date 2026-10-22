import {
  FIRST_HOLD_MS,
  SECOND_HOLD_MS,
} from '../core/constants.js';

import {
  clamp,
} from '../core/math.js';

export class Pause {
  constructor() {
    // Stage 1 requires 5 seconds.
    // Stage 2 requires 8 seconds.
    // Stage 3 means the pause succeeded.
    this.stage = 1;

    this.holding = false;
    this.startedAt = null;
    this.progress = 0;

    this.rejectedAt = null;
    this.completed = false;
  }

  get duration() {
    if (this.stage === 1) {
      return FIRST_HOLD_MS;
    }

    return SECOND_HOLD_MS;
  }

  press(now) {
    if (this.holding) return;
    if (this.completed) return;

    this.holding = true;
    this.startedAt = now;
  }

  release() {
    if (!this.holding) return;

    this.holding = false;
    this.startedAt = null;
    this.progress = 0;
  }

  update(now) {
    let rejectedNow = false;
    let completedNow = false;

    if (
      this.holding
      && this.startedAt !== null
    ) {
      const elapsed =
        now - this.startedAt;

      this.progress = clamp(
        elapsed / this.duration
      );

      if (this.progress >= 1) {
        this.holding = false;
        this.startedAt = null;
        this.progress = 0;

        if (this.stage === 1) {
          this.stage = 2;
          this.rejectedAt = now;
          rejectedNow = true;
        } else {
          this.stage = 3;
          this.completed = true;
          completedNow = true;
        }
      }
    }

    const reject = this.rejectedAt === null
      ? 0
      : clamp(
          (now - this.rejectedAt) / 900
        );

    return {
      stage: this.stage,
      holding: this.holding,
      progress: this.progress,
      reject,
      rejectedNow,
      completedNow,
    };
  }
}