import test from 'node:test';
import assert from 'node:assert/strict';

import { MOTION_POLICIES,MotionPolicy,} from '../src/input/motionpolicy.js';

test('policies become more tolerant', () => {
  assert.ok(
    MOTION_POLICIES.strict.threshold
    < MOTION_POLICIES.standard.threshold
  );

  assert.ok(
    MOTION_POLICIES.standard.threshold
    < MOTION_POLICIES.human.threshold
  );
});

test('one movement frame does not interrupt', () => {
  const policy =
    new MotionPolicy('strict');

  assert.equal(
    policy.update(0.2, 0),
    false
  );
});

test('sustained movement interrupts', () => {
  const policy =
    new MotionPolicy('strict');

  assert.equal(
    policy.update(0.2, 0),
    false
  );

  assert.equal(
    policy.update(0.2, 349),
    false
  );

  assert.equal(
    policy.update(0.2, 350),
    true
  );
});

test('one movement produces one interrupt', () => {
  const policy =
    new MotionPolicy('strict');

  policy.update(0.2, 0);

  assert.equal(
    policy.update(0.2, 350),
    true
  );

  // Continued movement must not trigger again.
  assert.equal(
    policy.update(0.2, 500),
    false
  );
});

test('stillness resets the detector', () => {
  const policy =
    new MotionPolicy('strict');

  policy.update(0.2, 0);
  policy.update(0.2, 350);

  // Stillness prepares the detector for a new event.
  assert.equal(
    policy.update(0, 400),
    false
  );

  assert.equal(
    policy.update(0.2, 500),
    false
  );

  assert.equal(
    policy.update(0.2, 850),
    true
  );
});