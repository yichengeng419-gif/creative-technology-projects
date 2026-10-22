import {
  WAKE_HOLD_MS,
  WAKE_DECAY_S,
} from '../core/constants.js';

import { clamp } from '../core/math.js';

export class Wake {
  constructor() {
    this.progress = 0;
    this.holding = false;
    this.completed = false;

    this.holdStartedAt = 0;
    this.lastFrameAt = 0;
  }

  press(now) {
    if (this.completed || this.holding) return;

    // A new attempt always starts from zero.
    this.progress = 0;
    this.holding = true;
    this.holdStartedAt = now;
    this.lastFrameAt = now;
  }

  release() {
    if (this.completed) return;

    this.holding = false;
  }

  update(now) {
    const deltaMs = this.lastFrameAt
      ? now - this.lastFrameAt
      : 0;

    this.lastFrameAt = now;

    if (this.holding) {
      const heldFor = now - this.holdStartedAt;

      this.progress = clamp(
        heldFor / WAKE_HOLD_MS
      );

      if (this.progress >= 1) {
        this.progress = 1;
        this.holding = false;
        this.completed = true;

        return {
          progress: this.progress,
          completedNow: true,
        };
      }
    } else if (!this.completed && this.progress > 0) {
      const decayMs = WAKE_DECAY_S * 1000;

      this.progress = clamp(
        this.progress - deltaMs / decayMs
      );
    }

    return {
      progress: this.progress,
      completedNow: false,
    };
  }

  reset() {
    this.progress = 0;
    this.holding = false;
    this.completed = false;
    this.holdStartedAt = 0;
    this.lastFrameAt = 0;
  }
}