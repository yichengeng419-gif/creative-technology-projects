import {
  clamp,
  ease,
} from '../core/math.js';

export class Unarchived {
  constructor(duration = 2200) {
    this.duration = duration;
    this.active = false;
    this.startedAt = 0;
  }

  start(now) {
    if (this.active) return;

    this.active = true;
    this.startedAt = now;
  }

  update(now) {
    if (!this.active) {
      return {
        active: false,
        progress: 0,
        completed: false,
      };
    }

    const rawProgress =
      (now - this.startedAt) /
      this.duration;

    const progress =
      ease(clamp(rawProgress));

    return {
      active: true,
      progress,
      completed: rawProgress >= 1,
    };
  }
}