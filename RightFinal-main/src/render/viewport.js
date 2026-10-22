import { MAX_DPR } from '../core/constants.js';

/**
 * Owns the shared screen geometry (CSS size + device pixel ratio) and keeps
 * every canvas sized to match it. Nothing about cameras or content lives
 * here — only "how big is the screen right now".
 */
export class Viewport {
  constructor(canvases) {
    this.canvases = canvases;
    this.w = 1;
    this.h = 1;
    this.dpr = 1;
    this.resize();
  }

  resize() {
    this.w = window.innerWidth;
    this.h = window.innerHeight;
    this.dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR);

    for (const canvas of this.canvases) {
      canvas.width = Math.round(this.w * this.dpr);
      canvas.height = Math.round(this.h * this.dpr);
    }
  }
}