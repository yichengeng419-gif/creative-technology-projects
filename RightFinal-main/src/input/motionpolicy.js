// Motion policies define sensitivity and the time
// motion must continue before interrupting a hold.
export const MOTION_POLICIES = Object.freeze({
  strict: {
    threshold: 0.025,
    sustainMs: 350,
    detail:
      'Small sustained movements can interrupt a hold.',
  },

  standard: {
    threshold: 0.055,
    sustainMs: 550,
    detail:
      'Sustained large movement can interrupt a hold.',
  },

  human: {
    threshold: 0.1,
    sustainMs: 800,
    detail:
      'Normal body movement is tolerated.',
  },
});

export class MotionPolicy {
  constructor(name = 'standard') {
    this.name = name;
    this.activeSince = null;

    // Prevent repeated interrupts during one movement.
    this.latched = false;
  }

  get config() {
    return MOTION_POLICIES[this.name];
  }

  set(name) {
    if (!MOTION_POLICIES[name]) {
      throw new Error(
        `Unknown camera policy: ${name}`
      );
    }

    this.name = name;
    this.reset();

    return this.config;
  }

  reset() {
    this.activeSince = null;
    this.latched = false;
  }

  update(score, now) {
    const { threshold, sustainMs } =
      this.config;

    // Stillness resets the motion timer.
    if (score < threshold) {
      this.reset();
      return false;
    }

    if (this.activeSince === null) {
      this.activeSince = now;
      return false;
    }

    // Interrupt only after sustained movement.
    if (
      !this.latched
      && now - this.activeSince >= sustainMs
    ) {
      this.latched = true;
      return true;
    }

    return false;
  }
}