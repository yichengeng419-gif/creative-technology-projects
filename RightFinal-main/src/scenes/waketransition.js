import {
  TRANSITION_MS,
} from '../core/constants.js';

import {
  clamp,
  ease,
  mix,
} from '../core/math.js';

export class WakeTransition {
  constructor() {
    this.active = false;
    this.finished = false;
    this.startedAt = 0;

    this.startSpin = 0;
    this.startDistance = 28;
  }

  start(now, spin, distance) {
    if (this.active || this.finished) return;

    this.active = true;
    this.startedAt = now;
    this.startSpin = spin;
    this.startDistance = distance;
  }

  update(now) {
    if (this.finished) {
      return this.#finalValues(false);
    }

    if (!this.active) {
      return {
        progress: 0,
        scatter: 0,
        opacity: 1,
        spin: this.startSpin,
        distance: this.startDistance,
        finishedNow: false,
      };
    }

    const progress = clamp(
      (now - this.startedAt)
      / TRANSITION_MS
    );

    const zoom = ease(
      progress / 0.5
    );

    const scatter = ease(
      (progress - 0.2) / 0.8
    );

    const fade = ease(
      (progress - 0.48) / 0.52
    );

    if (progress >= 1) {
      this.active = false;
      this.finished = true;

      return this.#finalValues(true);
    }

    return {
      progress,
      scatter: scatter * 1.15,
      opacity: 1 - fade,

      spin:
        this.startSpin
        + ease(progress) * Math.PI * 1.35,

      distance: mix(
        this.startDistance,
        this.startDistance - 7,
        zoom
      ),

      finishedNow: false,
    };
  }

  #finalValues(finishedNow) {
    return {
      progress: 1,
      scatter: 1.15,
      opacity: 0,

      spin:
        this.startSpin
        + Math.PI * 1.35,

      distance:
        this.startDistance - 7,

      finishedNow,
    };
  }

  reset() {
    this.active = false;
    this.finished = false;
    this.startedAt = 0;
    this.startSpin = 0;
    this.startDistance = 28;
  }
}