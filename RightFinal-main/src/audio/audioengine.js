function clamp(value, min = 0, max = 1) {
  return Math.max(min, Math.min(max, value));
}

export class AudioEngine {
  constructor() {
    this.context = null;
    this.unlocked = false;
    this.transitionPlayed = false;
    this.accepted = false;

// Four audio layers.
    this.opening = new Audio(
      'assets/audio/opening-bass.wav'
    );

    this.machine = new Audio(
      'assets/audio/machine-hum.wav'
    );

    this.transition = new Audio(
      'assets/audio/transition-brass.wav'
    );

    this.reject = new Audio(
      'assets/audio/reject.wav'
    );

    this.opening.preload = 'auto';
    this.machine.preload = 'auto';
    this.transition.preload = 'auto';
    this.reject.preload = 'auto';

    this.machine.loop = true;

    this.openingGain = null;
    this.machineGain = null;
    this.transitionGain = null;
    this.filter = null;
    this.masterGain = null;
  }

  async unlock() {
    if (this.unlocked) {
      if (this.context?.state === 'suspended') {
        await this.context.resume();
      }

      return;
    }

    const AudioContextClass =
      window.AudioContext
      || window.webkitAudioContext;

    if (!AudioContextClass) {
      console.warn('Web Audio is not supported.');
      return;
    }

    this.context = new AudioContextClass();

    this.openingGain =
      this.context.createGain();

    this.machineGain =
      this.context.createGain();

    this.transitionGain =
      this.context.createGain();

    this.filter =
      this.context.createBiquadFilter();

    this.masterGain =
      this.context.createGain();

    this.openingGain.gain.value = 0;
    this.machineGain.gain.value = 0;
    this.transitionGain.gain.value = 0;

    this.filter.type = 'lowpass';
    //this.filter.frequency.value = 200;
    this.filter.frequency.value = 250;

    this.masterGain.gain.value = 0.65;

// Connect the Web Audio signal chain.
    const openingSource =
      this.context.createMediaElementSource(
        this.opening
      );

    const machineSource =
      this.context.createMediaElementSource(
        this.machine
      );

    const transitionSource =
      this.context.createMediaElementSource(
        this.transition
      );

    openingSource
      .connect(this.openingGain)
      .connect(this.masterGain);

    machineSource
      .connect(this.machineGain)
      .connect(this.filter)
      .connect(this.masterGain);

    transitionSource
      .connect(this.transitionGain)
      .connect(this.masterGain);

    this.masterGain.connect(
      this.context.destination
    );

    this.unlocked = true;

    await this.context.resume();

    const results =
      await Promise.allSettled([
        this.opening.play(),
        this.machine.play(),
      ]);

    for (const result of results) {
      if (result.status === 'rejected') {
        console.warn(
          'Audio could not start.',
          result.reason
        );
      }
    }
  }

  change(
    parameter,
    value,
    //speed = 0.1
    speed = 0.15
  ) {
    if (!this.context || !parameter) {
      return;
    }

    parameter.setTargetAtTime(
      value,
      this.context.currentTime,
      speed
    );
  }

  update({
    stage,
    wakeProgress,
    holding,
    holdProgress,
    pauseStage,
    endingPhase,
    timedOut,
  }) {
    if (!this.unlocked) return;

// Normalised live interaction data.
    const wake =
      clamp(wakeProgress);

    const hold =
      clamp(holdProgress);

    const tension =
      hold * hold;

    let openingVolume = 0;
    let machineVolume = 0;
    let filterFrequency = 250;

// Wake progress controls the opening sound.
    if (stage === 'dna') {
      openingVolume =
        wake * 0.3;
    }

// The machine hum represents system observation.
    if (
      stage === 'transition'
      || stage === 'field'
    ) {
      machineVolume = 0.16;
      filterFrequency = 650;
    }
// First pause request.
    if (
      holding
      && pauseStage === 1
    ) {
      machineVolume =
        0.16 + tension * 0.16;

      filterFrequency =
        650 + tension * 1150;
    }
// The second request creates more pressure.
    if (
      holding
      && pauseStage === 2
    ) {
      machineVolume =
        0.2 + tension * 0.18;

      filterFrequency =
        800 + tension * 1800;
    }

// Silence becomes the result of a successful pause.
    if (
      endingPhase === 'pause'
      || endingPhase === 'archive'
      || timedOut
      || stage === 'unarchived'
    ) {
      openingVolume = 0;
      machineVolume = 0.005;
      filterFrequency = 150;
    }

    this.change(
      this.openingGain.gain,
      openingVolume
    );

    this.change(
      this.machineGain.gain,
      machineVolume
    );

    this.change(
      this.filter.frequency,
      filterFrequency,
      0.2
    );
  }

  playTransition() {
    if (
      !this.unlocked
      || this.transitionPlayed
    ) {
      return;
    }

    this.transitionPlayed = true;
    this.transition.currentTime = 0;

    this.change(
      this.transitionGain.gain,
      0.16,
      0.1
    );

    void this.transition
      .play()
      .catch((error) => {
        console.warn(
          'Transition audio failed.',
          error
        );
      });

    window.setTimeout(() => {
      this.change(
        this.transitionGain.gain,
        0,
        0.5
      );
    }, 3800);

    window.setTimeout(() => {
      this.transition.pause();
    }, 6000);
  }

  playReject() {
    if (!this.unlocked) return;

    this.reject.currentTime = 0;
    this.reject.volume = 0.55;

    void this.reject
      .play()
      .catch((error) => {
        console.warn(
          'Reject audio is unavailable.',
          error
        );
      });

    this.change(
      this.machineGain.gain,
      0.03,
      0.03
    );
  }

  accept() {
    if (
      !this.unlocked
      || this.accepted
    ) {
      return;
    }

    this.accepted = true;
// Remove the machine sound instead of playing a reward.
    this.change(
      this.openingGain.gain,
      0,
      0.12
    );

    this.change(
      this.transitionGain.gain,
      0,
      0.12
    );

    this.change(
      this.machineGain.gain,
      0.005,
      0.18
    );

    this.change(
      this.filter.frequency,
      120,
      0.18
    );
  }

  dispose() {
    this.opening.pause();
    this.machine.pause();
    this.transition.pause();
    this.reject.pause();

    if (this.context) {
      void this.context.close();
    }
  }
}