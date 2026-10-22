import {
  FIRST_HOLD_MS,
  SECOND_HOLD_MS,
  SESSION_LIMIT_MS,
  SECOND_HOLD_WINDOW_MS,
  PROTOCOL_VERSION,
} from '../core/constants.js';

function sessionId() {
  const bytes = crypto.getRandomValues(new Uint32Array(2));
  return 'RT-' + Array.from(bytes, v => v.toString(16).padStart(8, '0')).join('').toUpperCase();
}
//Pause request protocol as a purely status thing.
export class Session {
  constructor(now) {
    this.id = sessionId();
    this.startedAt = now;
    this.decisionAt = null;
    this.deadlineAt = null;
    this.endedAt = null;
    this.outcome = null;

    this.contacts = 0;
    this.pressure = 0;
    this.attempts = 0;
    this.aborted = 0;

    this.touching = false;
    this.rejected = false;
    this.pauseStage = 0; // 0 = first attempt (5s), 1 = waiting for the second hold (8s)

    this.contactsAt = [];
    this.holds = [];
    this.gaps = [];
    this.activeGap = null;
    this.lastHoldEnd = null;
    this.firstHold = null;
    this.secondHold = null;
    this.retryDelay = null;
    this.inputSources = new Set();
    this.saved = false;

    this.rhythm = 'UNRESOLVED';
    this.grade = 'PENDING';

    this._contactStartedAt = null;
  }

  //How long the current hold needs to last: 5s the first time, 8s the second. 
  get targetMs() {
    return this.pauseStage ? SECOND_HOLD_MS : FIRST_HOLD_MS;
  }
  get over() {
  return this.endedAt !== null;
}
  //How long the current, in-progress contact has lasted so far
  heldMs(now) {
    if (this._contactStartedAt === null) return 0;
    return Math.max(0, now - this._contactStartedAt);
  }

  //Returns false if a contact can't start right now.
  beginContact(now, source = 'pointer') {
    if (this.endedAt !== null || this.rejected) return false;
    if (this.deadlineAt !== null && now >= this.deadlineAt) {
      this.finish('timeout', this.deadlineAt);
      return false;
    }

    this._contactStartedAt = now;
    if (this.decisionAt === null) {
      this.decisionAt = now;
      this.deadlineAt = now + SESSION_LIMIT_MS;
    }
    // Each retry of the second stage gets a fresh, full window.
    if (this.pauseStage) this.deadlineAt = now + SECOND_HOLD_WINDOW_MS;

    this.activeGap = this.lastHoldEnd === null ? null : Number(((now - this.lastHoldEnd) / 1000).toFixed(3));
    this.contacts++;
    this.attempts++;
    this.touching = true;
    this.contactsAt.push(now);
    this.inputSources.add(source);

    const recent = this.contactsAt.slice(-3);
    if (recent.length === 3 && recent[2] - recent[1] < 900 && recent[1] - recent[0] < 900) {
      this.pressure++;
    }

    this.#classifyRhythm();
    return true;
  }

  //Call every frame while holding, so a target reached mid-hold is caught immediately.
  tick(now) {
    if (!this.touching) return;
    if (this.heldMs(now) >= this.targetMs) {
      if (!this.pauseStage) this.#reject(now);
      else this.finish('pause', now);
    }
  }

  //Call when the user releases. `cancel` skips the "did they hold long enough" check.
    endContact(now, cancel = false) {
    if (this.touching) {
      if (!cancel && this.heldMs(now) >= this.targetMs) {
        if (!this.pauseStage) this.#reject(now);
        else this.finish('pause', now);
      }
      if (this.touching) {
        this.#recordHold(this.heldMs(now), null, now);
        this.touching = false;
      }
    }
    this._contactStartedAt = null;
    if (this.endedAt === null) {
      if (this.pauseStage) this.deadlineAt = now + SESSION_LIMIT_MS;
      this.rejected = false;
    }
  }

  //Call when camera motion detection interrupts an in-progress hold.
  interrupt(now) {
    if (!this.touching) return;
    this.#recordHold(this.heldMs(now), null, now);
    this.touching = false;
    this.rejected = true;
    this._contactStartedAt = null;
    this.pressure++;
    if (this.pauseStage) this.deadlineAt = now + SESSION_LIMIT_MS;
    this.#classifyRhythm();
  }

  //Call every frame (even while not touching) to catch the session simply running out of time.
  checkTimeout(now) {
    if (this.endedAt !== null || this.touching) return;
    if (this.deadlineAt !== null && now >= this.deadlineAt) {
      this.finish('timeout', now);
    }
  }

  finish(outcome, now) {
    if (this.endedAt !== null) return false;
    if (this.touching) {
      this.#recordHold(
        this.heldMs(now),
        outcome === 'pause' ? 'second' : null,
        now
      );
    }
    this.outcome = outcome;
    this.endedAt = now;
    this.touching = false;
    this.rejected = false;
    this._contactStartedAt = null;
    return true;
  }

  #reject(now) {
    this.#recordHold(this.heldMs(now), 'first', now);
    this.pauseStage = 1;
    this.touching = false;
    this.rejected = true;
    this.deadlineAt = now + SESSION_LIMIT_MS;
    this._contactStartedAt = null;
  }

  #recordHold(ms, milestone, now) {
    const v = Number((Math.max(0, ms) / 1000).toFixed(3));
    if (v < 0.2) { this.activeGap = null; return; }this.holds.push(v);
    if (this.activeGap !== null) {
        this.gaps.push(this.activeGap);
      if (this.pauseStage && this.retryDelay === null) this.retryDelay = this.activeGap;
    }
    this.activeGap = null;
    this.lastHoldEnd = now;
    if (milestone === 'first') this.firstHold = v;else if (milestone === 'second') this.secondHold = v;else this.aborted++;
  }

  #classifyRhythm() {
    const intervals = this.contactsAt.slice(1).map((v, i) => v - this.contactsAt[i]);
    this.grade = this.contacts >= 10 || this.pressure >= 3 ? 'EXCESSIVE'
      : this.contacts >= 6 || this.pressure ? 'ELEVATED'
      : this.contacts ? 'MEASURED' : 'PENDING';
    if (intervals.length < 3) { this.rhythm = 'UNRESOLVED'; return; }
    const recent = intervals.slice(-7);
    const mean = recent.reduce((a, b) => a + b, 0) / recent.length;
    const cv = Math.sqrt(recent.reduce((a, b) => a + (b - mean) ** 2, 0) / recent.length) / Math.max(1, mean);
    this.rhythm = cv < 0.09 ? 'MECHANICAL' : cv > 0.65 ? 'ERRATIC' : 'VARIABLE';
  }

  toRecord(policy = 'standard') {
    if (this.endedAt === null) throw Error('Cannot record an unfinished session');
    const mean = this.holds.length
      ? this.holds.reduce((a, b) => a + b, 0) / this.holds.length
      : 0;
    const source = this.inputSources.size === 1
      ? [...this.inputSources][0]
      : 'mixed';

    return {
      session_id: this.id,
      outcome: this.outcome,
      touch_count: this.holds.length,
      raw_contact_count: this.contacts,
      holds: [...this.holds],
      gaps: [...this.gaps],
      first_hold: this.firstHold,
      second_hold: this.secondHold,
      mean_hold: Number(mean.toFixed(3)),
      max_hold: this.holds.length ? Math.max(...this.holds) : 0,
      retry_delay: this.retryDelay,
      aborted_attempts: this.aborted,
      total_session_time: Number(((this.endedAt - (this.decisionAt ?? this.startedAt)) / 1000).toFixed(3)),
      observation_policy: policy,
      input_source: source,
      protocol_version: PROTOCOL_VERSION,
      provenance: 'live',
    };
  }
}