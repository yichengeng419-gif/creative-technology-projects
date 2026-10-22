import { CameraAscii } from './cameraascii.js';
import { MotionPolicy } from './motionpolicy.js';

export class Camera {
  constructor(
    video,
    previewCanvas,
    motionCanvas
  ) {
    this.video = video;
    this.canvas = previewCanvas;
    this.motionCanvas = motionCanvas;
    this.ascii = new CameraAscii(56);
    // Frequent pixel reads are required for motion analysis.
    this.motionContext =
      motionCanvas.getContext(
        '2d',
        { willReadFrequently: true }
      );

    this.stream = null;
    this.frameId = null;
    this.previousFrame = null;
    this.lastSampleAt = 0;
    this.startedAt = 0;

    this.motionPolicy =
      new MotionPolicy('standard');

    // Main.js can subscribe without owning camera logic.
    this.listeners = new Set();
  }

  get running() {
    return this.stream !== null;
  }

  get policy() {
    return this.motionPolicy.name;
  }

  setPolicy(name) {
    this.previousFrame = null;
    return this.motionPolicy.set(name);
  }

  subscribe(listener) {
    this.listeners.add(listener);

    return () => {
      this.listeners.delete(listener);
    };
  }

  emit(message) {
    for (const listener of this.listeners) {
      listener(message);
    }
  }

  async start() {
    if (this.running) return;

    this.stream =
      await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user',
          width: 640,
          height: 480,
        },
        audio: false,
      });

    this.video.srcObject = this.stream;
    await this.video.play();

    this.startedAt = performance.now();
    this.previousFrame = null;
    this.motionPolicy.reset();

    this.drawPreview();
  }

  drawPreview() {
    if (!this.running) return;

    const now = performance.now();

    this.ascii.draw(
      this.video,
      this.canvas,
      now
    );

    // Analyse motion at about 12 frames per second.
    if (now - this.lastSampleAt >= 80) {
      this.lastSampleAt = now;
      this.measureMotion(now);
    }

    this.frameId = requestAnimationFrame(
      () => this.drawPreview()
    );
  }

  measureMotion(now) {
    if (!this.video.videoWidth) return;

    const width = this.motionCanvas.width;
    const height = this.motionCanvas.height;

    // Use a small hidden canvas to reduce processing cost.
    this.motionContext.drawImage(
      this.video,
      0,
      0,
      width,
      height
    );

    const pixels =
      this.motionContext.getImageData(
        0,
        0,
        width,
        height
      ).data;

    const frame =
      new Uint8Array(width * height);

    let difference = 0;

    for (
      let pixel = 0, index = 0;
      pixel < pixels.length;
      pixel += 4, index++
    ) {
      // Convert RGB pixels into one brightness value.
      const brightness =
        pixels[pixel] * 0.21
        + pixels[pixel + 1] * 0.72
        + pixels[pixel + 2] * 0.07;

      frame[index] = brightness;

      if (this.previousFrame) {
        difference += Math.abs(
          brightness
          - this.previousFrame[index]
        );
      }
    }

    // Normalise motion to a value between 0 and 1.
    const score = this.previousFrame
      ? difference / (frame.length * 255)
      : 0;

    this.previousFrame = frame;

    // Ignore initial camera exposure changes.
    const cameraReady =
      now - this.startedAt > 1200;

    const interrupted =
      cameraReady
      && this.motionPolicy.update(
        score,
        now
      );

    this.emit({
      score,
      interrupted,
      time: now,
      policy: this.policy,
    });
  }

  capture(targetCanvas) {
    if (!this.running) return false;

    this.ascii.draw(
      this.video,
      targetCanvas,
      performance.now()
    );

    return true;
  }

  stop() {
    if (this.frameId !== null) {
      cancelAnimationFrame(this.frameId);
      this.frameId = null;
    }

    // Release the physical camera when the panel closes.
    if (this.stream) {
      for (const track of this.stream.getTracks()) {
        track.stop();
      }
    }

    this.video.srcObject = null;
    this.stream = null;
    this.previousFrame = null;
    this.motionPolicy.reset();
  }
}