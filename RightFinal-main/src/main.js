import {
  AudioEngine,
} from './audio/audioengine.js';
import { Input } from './input/input.js';
import { SerialLink } from './input/serial.js';
import { Camera } from './input/camera.js';
import {
  loadWakeAssets,
  loadFieldAssets,
} from './assets.js';

import { Wake } from './scenes/wake.js';
import { WakeTransition } from './scenes/waketransition.js';
import { Pause } from './scenes/pause.js';
import { Ending } from './scenes/ending.js';
import { Unarchived } from './scenes/unarchived.js';
import { createDossier } from './scenes/dossier.js';
import { Session } from './session/session.js';
import { DataStore } from './session/datastore.js';
import {
  toCSV,
} from './session/csv.js';

import {
  auditDataset,
  CLASS_LABELS,
  MIN_CLASS_SAMPLES,
} from './ml/audit.js';

import {
  loadForest,
  predictSession,
} from './ml/forest.js';
import { PointCloud } from './render/PointCloud.js';
import { FieldLayer } from './render/fieldlayer.js';
import { Viewport } from './render/viewport.js';
import {
  FIRST_HOLD_MS,
  SECOND_HOLD_MS,
  TARGET_SAMPLES,
} from './core/constants.js';
import {
  fmt,
} from './core/math.js';
import { FieldMotion } from './render/fieldmotion.js';
import {
  downloadDossier,
} from './dossierexport.js';
const SERIAL_COPY = Object.freeze({
  offline:
    'MOUSE SIMULATION / SENSOR OFFLINE',

  connecting:
    'SELECT MPR121 SERIAL PORT',

  connected:
    'MPR121 CONNECTED / WAITING FOR TOUCH',

  cancelled:
    'SENSOR SELECTION CANCELLED',

  unsupported:
    'WEB SERIAL REQUIRES CHROME OR EDGE',

  overflow:
    'SERIAL DATA OVERFLOW / RECONNECT',
});

const FIELD_TRAVEL = [0, 0];
const FIELD_OFFSET = [0, -0.16];
const ARCHIVE_EDGE = [-1, 1];

const scene =
  document.querySelector('#scene');

const sessionIdLabel =
  document.querySelector('#session-id');

const sampleProgress =
  document.querySelector('#sample-progress');

const sampleFill =
  document.querySelector('#sample-fill');

const exportButton =
  document.querySelector('#exportButton');

const auditButton =
  document.querySelector('#auditButton');

const auditDialog =
  document.querySelector('#audit-dialog');

const auditClose =
  document.querySelector('#audit-close');

const auditSummary =
  document.querySelector('#audit-summary');

const auditStatus =
  document.querySelector('#audit-status');

const auditMetrics =
  document.querySelector('#audit-metrics');

const auditMatrix =
  document.querySelector('#audit-matrix');

const auditCounts =
  document.querySelector('#audit-counts');

const auditTarget =
  document.querySelector('#audit-target');

const auditExportButton =
  document.querySelector('#audit-export');

const importButton =
  document.querySelector('#importButton');

const retrainButton =
  document.querySelector('#retrainButton');

const importMessage =
  document.querySelector('#import-message');

const auditLimitations =
  document.querySelector('#audit-limitations');
//Get the six elements of the Body Dossier
const dossierContacts =
  document.querySelector('#dos-contacts');

const dossierPressure =
  document.querySelector('#dos-pressure');

const dossierHolds =
  document.querySelector('#dos-holds');

const dossierRhythm =
  document.querySelector('#dos-rhythm');

const dossierGrade =
  document.querySelector('#dos-grade');

const dossierClass =
  document.querySelector('#dos-class');

const dossierConfidence =
  document.querySelector(
    '#dos-confidence'
  );

const dossierTime =
  document.querySelector('#dos-time');

const mainCanvas =
  document.querySelector('#mainCanvas');

const fieldCanvas =
  document.querySelector('#fieldCanvas');

const prelude =
  document.querySelector('#prelude-overlay');

const wakeCopy =
  document.querySelector('#wake-copy');

const wakeFill =
  document.querySelector('#wake-fill');

const status =
  document.querySelector('#status');

const holdTitle =
  document.querySelector('#hold-title');

const holdLabel =
  document.querySelector('#holdLabel');

const meterFill =
  document.querySelector('#meterFill');

const holdMeter =
  document.querySelector('#hold-meter');

const finalOverlay =
  document.querySelector('#final-overlay');

const finalReport =
  document.querySelector('#fo-report');

const finalLetter =
  document.querySelector('#fo-letter');

const finalActions =
  document.querySelector('#fo-actions');

const reportText =
  document.querySelector('#fo-report-text');

const reportInterpretation =
  document.querySelector(
    '#fo-report-interpretation'
  );

const statement =
  document.querySelector('#fo-statement');
const resetButton =
  document.querySelector('#fo-reset');
const downloadButton =
  document.querySelector('#fo-download');
const continueButton =
  document.querySelector('#fo-continue');

const unarchivedCopy =
  document.querySelector('#unarchived-copy');

const unarchivedResetButton =
  document.querySelector('#unarchived-reset');
const cameraButton =
  document.querySelector('#cameraButton');

const cameraPanel =
  document.querySelector('#obs-panel');

const cameraClose =
  document.querySelector('#obs-close');

const cameraStatus =
  document.querySelector('#obs-status');

const cameraVideo =
  document.querySelector('#motionVideo');

const cameraCanvas =
  document.querySelector('#cameraCanvas');

const motionCanvas =
  document.querySelector('#motionCanvas');

const motionReadout =
  document.querySelector('#motionReadoutPanel');

const policyDetail =
  document.querySelector('#policy-detail');

const policyButtons =
  document.querySelectorAll(
    '[data-policy]'
  );

const dossierPolicy =
  document.querySelector('#dos-policy');

const evidenceCanvas =
  document.querySelector('#fo-evidence-canvas');

const evidenceCaption =
  document.querySelector('#fo-evidence-caption');

const serialButton =
  document.querySelector('#serialButton');

const serialLabel =
  document.querySelector('#serialLabel');
const input = new Input(scene);
const serial = new SerialLink();
const session = new Session(performance.now());
const audio = new AudioEngine();
const dataStore = new DataStore();
const wake = new Wake();
const transition = new WakeTransition();
const pause = new Pause();
const fieldMotion = new FieldMotion();
const ending = new Ending();
const unarchived =
  new Unarchived();
const viewport = new Viewport([
  mainCanvas,
  fieldCanvas,
]);
//const camera =
  //new Camera(cameraVideo, cameraCanvas);
const camera = new Camera(cameraVideo,cameraCanvas, motionCanvas);
// Translate camera motion into a session interruption.
camera.subscribe(({score,interrupted,time,}) => {
  motionReadout.textContent =
    `${Math.round(score * 100)}%`;

  // Motion affects only an active hold in the Field.
  if (!interrupted) return;
  if (!transition.finished) return;
  if (!session.touching) return;
  if (session.over) return;

  session.interrupt(time);
  pause.release();

  audio.playReject();

  status.textContent =
    'HOLD INTERRUPTED / MOVEMENT DETECTED';
});

let previousFrameAt = performance.now();
let pointCloud = null;
let fieldLayer = null;
let forest = null;
let dossierWritten = false;
let currentDossier = null;
let unarchivedShown = false;
document.body.dataset.stage = 'loading';

wakeCopy.textContent =
  'LOADING SPECIMEN';
  /**
 * Send normalised input to the active scene.
 * The scene does not need to know whether the source
 * is the mouse, keyboard or MPR121.
 */
input.subscribe(({
  touching,
  time,
  source,
  cancelled,
}) => { 
    if (touching) {
    void audio.unlock();
  }
  if (!pointCloud) return;

  // Before Field, input controls the DNA wake stage.
  if (!transition.finished) {
    if (transition.active) return;

    if (touching) {wake.press(time);} else {wake.release();}
    return;
  }
  // Inside Field, the same input controls Pause.
  if (touching) {
    if (session.beginContact(time, source)) {
      pause.press(time);
    }
  } else {
    pause.release();
    session.endContact(time, cancelled);
  }
});

function updateDataUi() {
  sessionIdLabel.textContent = session.id;
  sampleProgress.textContent =
    `${dataStore.size} / ${TARGET_SAMPLES} SESSIONS`;
  sampleFill.style.width =
    `${Math.min(1, dataStore.size / TARGET_SAMPLES) * 100}%`;
}
function updateDossierUi(now) {
  const endTime =
    session.endedAt ?? now;

  const elapsed =
    session.decisionAt === null
      ? 0
      : Math.max(
          0,
          endTime - session.decisionAt
        );

  dossierContacts.textContent =
    String(session.contacts)
      .padStart(2, '0');

  dossierPressure.textContent =
    String(session.pressure)
      .padStart(2, '0');

  dossierHolds.textContent =
    String(session.holds.length)
      .padStart(2, '0');

  dossierRhythm.textContent =
    session.rhythm;

  dossierGrade.textContent =
    session.grade;

  dossierTime.textContent =
    fmt(elapsed);
}
function saveCompletedSession() {
  if (!session.over || session.saved) return;

  const record =
    session.toRecord(camera.policy);

  dataStore.add(record);
  updatePrediction(record);

  session.saved = true;
  updateDataUi();

  if (session.outcome === 'timeout') {
  status.textContent =
    'SESSION EXPIRED / RECORD SAVED';

  holdMeter.hidden = true;

  unarchivedResetButton.textContent =
    'RESET SESSION';

  unarchivedResetButton.hidden = false;

  if (camera.running) {
    stopCamera();
  }
}
}

function exportSessions() {
  const rows = dataStore.all();

  if (rows.length === 0) {
    status.textContent =
      'NO SESSION DATA TO EXPORT';

    return;
  }

  const blob = new Blob(
    [toCSV(rows)],
    {
      type:
        'text/csv;charset=utf-8',
    }
  );

  const url =
    URL.createObjectURL(blob);

  const link =
    document.createElement('a');

  const date =
    new Date()
      .toISOString()
      .slice(0, 10);

  link.href = url;
  link.download =
    `right-to-pause-${date}.csv`;

  link.click();

  setTimeout(() => {
    URL.revokeObjectURL(url);
  }, 0);
}

function addAuditMetric(label, value) {
  const metric =
    document.createElement('div');

  const name =
    document.createElement('span');

  const result =
    document.createElement('b');

  name.textContent = label;
  result.textContent = String(value);

  metric.append(name, result);
  auditMetrics.append(metric);
}

/**
 * Show the supplied model's confusion matrix.
 *
 * @param {number[][]|null} matrix
 */
function renderAuditMatrix(
  matrix = null
) {
  auditMatrix.replaceChildren();

  const keys = [
    'early',
    'interrupted',
    'pause',
  ];

  keys.forEach((key, rowIndex) => {
    const row =
      document.createElement('tr');

    const label =
      document.createElement('th');

    label.scope = 'row';
    label.textContent =
      CLASS_LABELS[key];

    row.append(label);

    for (
      let column = 0;
      column < keys.length;
      column++
    ) {
      const cell =
        document.createElement('td');

      cell.textContent =
        matrix?.[rowIndex]?.[column]
        ?? '–';

      row.append(cell);
    }

    auditMatrix.append(row);
  });
}

/**
 * Show the supplied pilot model and
 * the locally collected dataset.
 */
function renderAudit() {
  const summary =
    auditDataset(
      dataStore.all(),
      TARGET_SAMPLES
    );

  const modelAudit =
    forest?.audit;

  if (modelAudit) {
    auditSummary.textContent =
      `${forest.sample_count} TRAINING SESSIONS / `
      + `${forest.trees.length} TREES`;

    auditStatus.textContent =
      'SUPPLIED PILOT MODEL / ACTIVE';
  } else {
    auditSummary.textContent =
      `${summary.total} LOCAL SESSIONS`;

    auditStatus.textContent =
      'MODEL STATUS / LOADING';
  }

  auditMetrics.replaceChildren();

  addAuditMetric(
    'ACCURACY',
    modelAudit
      ? `${Math.round(
          modelAudit.accuracy * 100
        )}%`
      : '–'
  );

  addAuditMetric(
    'MACRO F1',
    modelAudit
      ? `${Math.round(
          modelAudit.f1_macro * 100
        )}%`
      : '–'
  );

  addAuditMetric(
    'TRAINING DATA',
    forest?.sample_count ?? '–'
  );

  addAuditMetric(
    'LOCAL DATA',
    summary.total
  );

  auditCounts.textContent =
    `LOCAL / EARLY ${summary.counts.early} / `
    + `INTERRUPTED ${summary.counts.interrupted} / `
    + `PAUSE ${summary.counts.pause}`;

  auditTarget.textContent =
    `${summary.total} / ${TARGET_SAMPLES} SESSIONS / `
    + `${summary.remaining} REMAINING`;

  auditLimitations.textContent =
    modelAudit?.limitations
    ?? (
      'The supplied pilot model is still loading. '
      + `Local training would require at least `
      + `${MIN_CLASS_SAMPLES} sessions in every class.`
    );

  importMessage.textContent =
    'LIVE SESSIONS ARE STORED IN THIS BROWSER.';

  importButton.disabled = true;
  retrainButton.disabled = true;

  renderAuditMatrix(
    modelAudit?.confusion_matrix
  );
}
function openAudit() {
  renderAudit();

  if (!auditDialog.open) {auditDialog.showModal();}}

function closeAudit() {
  if (auditDialog.open) {auditDialog.close();}}

/**
 * Update all serial interface elements from one place.
 */
function updateSerialUi({
  buttonText,
  label,
  connected = false,
  disabled = false,
}) {
  serialButton.textContent =
    buttonText;

  serialButton.disabled =
    disabled;

  serialButton.classList.toggle(
    'is-connected',
    connected
  );

  serialLabel.textContent =
    label;
}

/**
 * Restore mouse and keyboard simulation.
 */
function restoreLocalInput() {
  input.setTouching(
    false,
    'serial'
  );

  input.setPointerEnabled(true);

  updateSerialUi({
    buttonText: 'CONNECT SENSOR',
    label: SERIAL_COPY.offline,
  });
}
  /**
 * Classify one completed session.
 *
 * The supplied model is descriptive and does not
 * affect whether the pause is accepted.
 *
 * @param {object} record
 */
function updatePrediction(record) {
  if (!forest) {
    dossierClass.textContent =
      'MODEL PENDING';
    dossierConfidence.textContent =
      '-';
    return;
  }

  try {
    const prediction =
      predictSession(
        forest,
        record
      );

    dossierClass.textContent =
      prediction.label;
    dossierConfidence.textContent =
      `${Math.round(
        prediction.confidence * 100
      )}%`;
  } catch (error) {
    dossierClass.textContent =
      'UNAVAILABLE';

    dossierConfidence.textContent =
      '-';

    console.warn(
      'Session prediction failed',
      error
    );
  }
}
/**
 * Close an invalid connection and show its reason.
 */
function handleSerialFailure(message) {
  restoreLocalInput();

  serialLabel.textContent =
    message;

  void serial.disconnect(true);
}

/**
 * Translate SerialLink messages into Input state.
 */
function handleSerialEvent(event) {
  switch (event.type) {
    case 'touch':
      input.setTouching(
        true,
        'serial'
      );
      break;

    case 'release':
      input.setTouching(
        false,
        'serial'
      );
      break;

    case 'connecting':
      updateSerialUi({
        buttonText: 'CONNECTING...',
        label: SERIAL_COPY.connecting,
        disabled: true,
      });
      break;

    case 'connected':
      // Hardware owns input while its port is open.
      input.setPointerEnabled(false);

      updateSerialUi({
        buttonText:
          'DISCONNECT SENSOR',

        label:
          SERIAL_COPY.connected,

        connected: true,
      });
      break;

    case 'ready':
      serialLabel.textContent =
        `${event.device} READY / ELECTRODE 0`;
      break;

    case 'cancelled':
      updateSerialUi({
        buttonText: 'CONNECT SENSOR',
        label: SERIAL_COPY.cancelled,
      });
      break;

    case 'unsupported':
      updateSerialUi({
        buttonText: 'CONNECT SENSOR',
        label: SERIAL_COPY.unsupported,
        disabled: true,
      });
      break;

    case 'buffer-error':
      handleSerialFailure(
        SERIAL_COPY.overflow
      );
      break;

    case 'device-error':
      handleSerialFailure(
        `MPR121 ERROR / ${event.message}`
      );
      break;

    case 'error':
      handleSerialFailure(
        'SERIAL CONNECTION ERROR'
      );
      break;

    case 'disconnected':
      restoreLocalInput();
      break;

    default:
      break;
  }
}
// SerialLink produces hardware events;
// Input remains the single owner of contact state.
serial.subscribe(handleSerialEvent);

window.addEventListener('resize', () => {
  viewport.resize();
});
function updateWakeText(state) {
  const percent = Math.round(
    state.progress * 100
  );

  wakeFill.style.width =
    `${percent}%`;

  document.documentElement
    .style
    .setProperty(
      '--hold',
      state.progress
    );

  if (state.completedNow) {
    wakeCopy.textContent =
      'SPECIMEN AWAKE';
  } else if (wake.holding) {
    wakeCopy.textContent =
      `WAKING / ${percent}%`;
  } else if (state.progress > 0) {
    wakeCopy.textContent =
      'SIGNAL RECEDING';
  } else {
    wakeCopy.textContent =
      'HOLD TO WAKE';
  }
}

function updatePauseText(state) {
  const targetSeconds =
    pause.stage === 1
      ? FIRST_HOLD_MS / 1000
      : SECOND_HOLD_MS / 1000;

  const progress =
    pause.completed
      ? 1
      : state.progress;

  const elapsed =
    progress * targetSeconds;

  holdTitle.textContent =
    pause.stage === 1
      ? 'PAUSE / 01'
      : 'PAUSE / 02';

  holdLabel.textContent =
    `${elapsed.toFixed(1)} / ${targetSeconds.toFixed(1)} S`;

  meterFill.style.width =
    `${progress * 100}%`;

  holdMeter.setAttribute(
    'aria-valuemax',
    String(targetSeconds)
  );

  holdMeter.setAttribute(
    'aria-valuenow',
    elapsed.toFixed(1)
  );

  if (state.holding) {
    status.textContent =
      pause.stage === 1
        ? 'READING FIRST PAUSE REQUEST'
        : 'READING SECOND PAUSE REQUEST';
  }
}

function updateArchive(state) {
  if (state.phase !== 'archive') {
    return;
  }

  document.body.dataset.stage =
    'final';

  finalOverlay.classList.add(
    'visible'
  );

  finalOverlay.setAttribute(
    'aria-hidden',
    'false'
  );

  if (!dossierWritten) {
    const dossier = createDossier({
      firstHold: session.firstHold,
      secondHold: session.secondHold,
    });
  currentDossier = dossier;
  const captured =
  camera.capture(evidenceCanvas);

evidenceCaption.textContent =
  captured
    ? 'VISUAL EVIDENCE / ASCII FRAME'
    : 'VISUAL EVIDENCE / NOT SUBMITTED';

if (captured) {
  stopCamera();
}
    reportText.textContent =
      dossier.report;

    reportInterpretation.textContent =
      dossier.interpretation;

    statement.textContent =
      dossier.statement;

    dossierWritten = true;
  }

  if (state.report > 0.02) {
    finalReport.classList.add(
      'arrived'
    );
  }

  if (state.letter > 0.02) {
    finalLetter.classList.add(
      'rising'
    );
  }

  if (state.letter >= 1) {
    finalActions.classList.add(
      'ready'
    );
  }
}
//Landscape rendering
function renderField(
  now,
  fieldState,
  endingState,
  unarchivedState
) {
  if (!fieldLayer) return;

  fieldLayer.render({
    size: viewport,
    time: now / 1000,

    fit: 11.0,
    stretch: 1.12,

    camera: {
      dist: 4.25,
      yaw: now * 0.000025,
      pitch: 0.72,
    },

    spin: now * 0.00004,
    turn: 1,
    space: 0.08,

    pressure: fieldState.pressure,

    contact: fieldState.contact,

    hold: fieldState.hold,
    still:
       unarchivedState.active
    ? 0
    : endingState.still,
    approach: 1,
    rhythm: 0,
    excess: 0,

    hole:
      unarchivedState.active
        ? 0
        : fieldState.hold* 0.28,
    reject: fieldState.reject,
    gather: 0,
    lift: 0,
    unarchived:
      unarchivedState.progress,

    dark:
      1 - endingState.still * 0.35,

    focus: input.position,
    nearPos: input.position,
    travel: FIELD_TRAVEL,
    offset: FIELD_OFFSET,
    archiveEdge: ARCHIVE_EDGE,
  });
}
// Apply one policy to sensing, interface and session data.
function selectCameraPolicy(name) {
  const config = camera.setPolicy(name);

  for (const button of policyButtons) {
    const active =
      button.dataset.policy === name;

    button.classList.toggle('active',active);
    button.setAttribute('aria-pressed',String(active));
  }
  policyDetail.textContent =
    config.detail;
  dossierPolicy.textContent =
    name.toUpperCase();
}
function stopCamera() {
  camera.stop();
  cameraPanel.hidden = true;
  cameraButton.textContent = 'ENABLE CAMERA';
  motionReadout.textContent = 'OFF';
}

async function toggleCamera() {
  if (camera.running) {
    stopCamera();
    return;
  }

  try {
    cameraPanel.hidden = false;
    cameraStatus.textContent =
      'REQUESTING LOCAL CAMERA';

    await camera.start();
motionReadout.textContent = '0%';

cameraStatus.textContent =
  'ASCII OBSERVATION / LOCAL ONLY';

    cameraStatus.textContent =
      'ASCII OBSERVATION / LOCAL ONLY';

    cameraButton.textContent =
      'DISABLE CAMERA';
  } catch (error) {
    cameraStatus.textContent =
      'CAMERA ACCESS WAS NOT AVAILABLE';

    console.error(error);
  }
}

function frame(now) {
  const dt = Math.min(
    0.05,
    Math.max(
      0,
      (now - previousFrameAt) / 1000
    )
  );

  previousFrameAt = now;

  const wakeState =
    wake.update(now);

  const pauseState =
    pause.update(now);

  if (transition.finished) {
    session.tick(now);
    session.checkTimeout(now);
    saveCompletedSession();
  }

  fieldMotion.update(
    session,
    now,
    dt
  );

  const fieldState =
    fieldMotion.toRenderState();

  const endingState =
    ending.update(now);
  audio.update({
  stage: document.body.dataset.stage,
  wakeProgress: wakeState.progress,
  holding: pauseState.holding,
  holdProgress: pauseState.progress,
  pauseStage: pauseState.stage,
  endingPhase: endingState.phase,
  timedOut:
    session.outcome === 'timeout',
});
  const unarchivedState =
    unarchived.update(now);
if (
  unarchivedState.completed &&
  !unarchivedShown
) {
  finalOverlay.classList.remove('visible');
  finalOverlay.setAttribute('aria-hidden', 'true');

  unarchivedCopy.hidden = false;
  unarchivedResetButton.hidden = false;

  status.textContent =
    'UNARCHIVED / STILL IN MOTION';

  unarchivedShown = true;
}
  const idleSpin =
    now * 0.00008;

  updateWakeText(wakeState);

  if (transition.finished) {
    updatePauseText(pauseState);
    updateDossierUi(now);
  }
  if (pauseState.rejectedNow) {
    status.textContent =
      'PAUSE REFUSED / RELEASE AND TRY AGAIN';
  }
  if (pauseState.rejectedNow) {
  audio.playReject();

  status.textContent =
    'PAUSE REFUSED / RELEASE AND TRY AGAIN';
}

  //if (pauseState.completedNow) {ending.start(now);status.textContent ='PAUSE ACCEPTED';}
  if (pauseState.completedNow) {
    ending.start(now);
    audio.accept();

    status.textContent =
      'PAUSE ACCEPTED';
  }

  if (endingState.phase === 'pause') {
    status.textContent =
      'TRUE PAUSE / SYSTEM COUNTING SUSPENDED';
  }

  if (endingState.phase === 'archive') {
    status.textContent =
      'ARCHIVE READY';
  }

  if (!unarchivedState.active) {
  updateArchive(endingState);
}
//if (wakeState.completedNow) {transition.start(now,idleSpin, pointCloud.camera.distance;
  //document.body.dataset.stage ='transition';status.textContent ='SPECIMEN RELEASE / TRANSITION';}
  if (wakeState.completedNow) {
    transition.start(
      now,
      idleSpin,
      pointCloud.camera.distance
    );
    audio.playTransition();
    document.body.dataset.stage =
      'transition';

    status.textContent =
      'SPECIMEN RELEASE / TRANSITION';
  }

  if (
    transition.active
    || transition.finished
  ) {
    const transitionState =
      transition.update(now);
  // Field only starts fading in after the transition is 35% complete,
  // so the DNA dissolve reads before the landscape appears.
    const fieldReveal = Math.max(
      0,
      Math.min(
        1,
        (
          transitionState.progress
          - 0.35
        ) / 0.65
      )
    );

    renderField(
      now,
      fieldState,
      endingState,
      unarchivedState
   );

    fieldCanvas.style.opacity =
      String(fieldReveal);

    mainCanvas.style.opacity =
      String(transitionState.opacity);

    pointCloud.render({
      size: viewport,
      time: now / 1000,
      wake: 1,
      scatter:
        transitionState.scatter,
      opacity:
        transitionState.opacity,
      spin:
        transitionState.spin,
      distance:
        transitionState.distance,
    });

    prelude.style.opacity =
      String(transitionState.opacity);

    if (transitionState.finishedNow) {
      document.body.dataset.stage =
        'field';

      status.textContent =
        'VERIFICATION FIELD / ACTIVE';

      prelude.hidden = true;

      mainCanvas.style.opacity = '0';
      fieldCanvas.style.opacity = '1';
    }
  } else {
    mainCanvas.style.opacity = '1';
    fieldCanvas.style.opacity = '0';

    pointCloud.render({
      size: viewport,
      time: now / 1000,
      wake: wakeState.progress,
      scatter: 0,
      opacity: 1,
      spin: idleSpin,
      distance:
        pointCloud.camera.distance,
    });
  }

  requestAnimationFrame(frame);
}

async function start() {
  try {
    updateDataUi();
  
    void loadForest()
      .then((model) => {
        forest = model;
      })
      .catch((error) => {
        console.warn(
          'ML model unavailable',
          error
        );
      });
    const [
      wakeAssets,
      fieldAssets,
    ] = await Promise.all([
      loadWakeAssets(),
      loadFieldAssets(),
    ]);
    
    pointCloud = new PointCloud(
      mainCanvas,
      wakeAssets.positions,
      wakeAssets.roles
    );

    fieldLayer = new FieldLayer(
      fieldCanvas,
      fieldAssets.positions,
      fieldAssets.colours
    );

    viewport.resize();

    document.body.dataset.stage =
      'dna';

    status.textContent =
      'DORMANT SPECIMEN / HOLD TO WAKE';

    wakeCopy.textContent =
      'HOLD TO WAKE';

    input.connect();
    requestAnimationFrame(frame);
  } catch (error) {
    document.body.dataset.stage =
      'loading';

    wakeCopy.textContent =
      'RENDERER UNAVAILABLE';

    status.textContent =
      'MODEL COULD NOT BE LOADED';

    console.error(error);
  }
}

resetButton.addEventListener('click', () => {
  window.location.reload();
});
serialButton.addEventListener(
  'click',
  async () => {
    if (serial.connected) {
      await serial.disconnect();
      return;
    }

    await serial.connect();
  }
);
cameraButton.addEventListener(
  'click',
  toggleCamera
);

cameraClose.addEventListener('click',stopCamera);
for (const button of policyButtons) {
  button.addEventListener(
    'click',
    () => {
      selectCameraPolicy(
        button.dataset.policy
      );
    }
  );
}
exportButton.addEventListener(
  'click',
  exportSessions
);

auditButton.addEventListener(
  'click',
  openAudit
);

auditClose.addEventListener(
  'click',
  closeAudit
);

auditExportButton.addEventListener(
  'click',
  exportSessions
);
// Prevent clicks on UI controls (buttons, panels) from being
// mistaken for contact with the verification field itself.
for (const button of [
  serialButton,
  cameraButton,
  cameraClose,
  ...policyButtons,
  exportButton,
  auditButton,
  auditClose,
  auditExportButton,
  importButton,
  retrainButton,
  downloadButton,
  resetButton,
  continueButton,
  unarchivedResetButton,
]) {
  button.addEventListener(
    'pointerdown',
    event => {
      event.stopPropagation();
    }
  );
}
downloadButton.addEventListener(
  'click',
  () => {
    if (!currentDossier) return;

    downloadDossier({
      evidenceCanvas,
      dossier: currentDossier,
    });
  }
  
);
continueButton.addEventListener('click', () => {
  unarchived.start(performance.now());

  document.body.dataset.stage = 'unarchived';
  finalOverlay.classList.add('exiting');

  status.textContent ='LEAVING CLASSIFICATION';});

unarchivedResetButton.addEventListener(
  'click',
  () => window.location.reload()
);
//window.addEventListener(
  //'pagehide',
  //() => {
    //void serial.disconnect(true);
 //}
//);
window.addEventListener(
  'pagehide',
  () => {
    void serial.disconnect(true);
    audio.dispose();
  }
);
selectCameraPolicy('standard');
start();