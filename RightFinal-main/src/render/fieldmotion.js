import { clamp, mix, approach } from '../core/math.js';

export class FieldMotion {
  constructor() {
    this.contact = 0;
    this.hold = 0;
    this.pressure = 0;
    this.reject = 0;

    this._lastPauseStage = 0;
    this._rejectAt = null;
    this._lastPressureCount = 0;
    this._pressureAt = null;
  }

  update(session, now, dt) {
    if (session.pauseStage === 1 && this._lastPauseStage === 0) {
      this._rejectAt = now;
    }
    this._lastPauseStage = session.pauseStage;

    if (session.pressure > this._lastPressureCount) {
      this._pressureAt = now;
    }
    this._lastPressureCount = session.pressure;

    const targetContact = session.touching ? 1 : 0;
    this.contact = mix(this.contact, targetContact, approach(dt, 10));

    const targetHold = session.touching ? clamp(session.heldMs(now) / session.targetMs) : 0;
    this.hold = mix(this.hold, targetHold, approach(dt, 14));

    this.reject = this._rejectAt === null ? 0 : clamp((now - this._rejectAt) / 900);

    if (this._pressureAt !== null) {
      const age = (now - this._pressureAt) / 1800;
      this.pressure = age >= 1 ? 0 : 1 - age;
      if (age >= 1) this._pressureAt = null;
    } else {
      this.pressure = 0;
    }
  }

  toRenderState(extra = {}) {
    return {
      contact: this.contact,
      hold: this.hold,
      pressure: this.pressure,
      reject: this.reject,
      dark: 1,
      unarchived: 0,
      ...extra,
    };
  }
}
