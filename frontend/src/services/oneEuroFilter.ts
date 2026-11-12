/**
 * Precision Anatomical Filter Engine
 * Implements speed-adaptive One Euro filtering with anatomical joint tuning,
 * outlier jump rejection, and explicit landmark provenance tracking.
 *
 * Grounded in: Casiez, G., Roussel, N. and Vogel, F. (CHI '12).
 */

import { getAnatomicalFilterConfig } from './anatomicalHandModel';

export type LandmarkProvenance = 'OBSERVED' | 'PREDICTED' | 'FILTERED';

export interface ProvenancedLandmark extends Point3D {
  provenance: LandmarkProvenance;
  rawX?: number;
  rawY?: number;
  rawZ?: number;
  confidence?: number;
}

class LowPassFilter {
  private s: number;
  private isInitialized: boolean = false;

  constructor(initVal: number = 0) {
    this.s = initVal;
  }

  public filter(val: number, alpha: number): number {
    if (!this.isInitialized) {
      this.s = val;
      this.isInitialized = true;
      return val;
    }
    this.s = alpha * val + (1.0 - alpha) * this.s;
    return this.s;
  }

  public last(): number {
    return this.s;
  }

  public reset(val: number = 0) {
    this.s = val;
    this.isInitialized = false;
  }
}

export class OneEuroFilter1D {
  private minCutoff: number;
  private beta: number;
  private dCutoff: number;
  private xFilter: LowPassFilter;
  private dxFilter: LowPassFilter;
  private lastTime: number = -1;

  constructor(minCutoff: number = 0.85, beta: number = 0.009, dCutoff: number = 1.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.xFilter = new LowPassFilter();
    this.dxFilter = new LowPassFilter();
  }

  private alpha(rate: number, cutoff: number): number {
    const tau = 1.0 / (2.0 * Math.PI * cutoff);
    const te = 1.0 / rate;
    return 1.0 / (1.0 + tau / te);
  }

  public filter(val: number, timestamp: number): number {
    if (this.lastTime < 0) {
      this.lastTime = timestamp;
      return this.xFilter.filter(val, 1.0);
    }

    const dt = Math.max(1e-4, (timestamp - this.lastTime) / 1000.0);
    this.lastTime = timestamp;
    const rate = 1.0 / dt;

    // Filtered derivative
    const dx = (val - this.xFilter.last()) / dt;
    const edx = this.dxFilter.filter(dx, this.alpha(rate, this.dCutoff));

    // Dynamic cutoff frequency
    const cutoff = this.minCutoff + this.beta * Math.abs(edx);

    // Filter value
    return this.xFilter.filter(val, this.alpha(rate, cutoff));
  }

  public reset() {
    this.lastTime = -1;
    this.xFilter.reset();
    this.dxFilter.reset();
  }
}

export class OneEuroFilter3D {
  private xf: OneEuroFilter1D;
  private yf: OneEuroFilter1D;
  private zf: OneEuroFilter1D;
  private lastFiltered: Point3D | null = null;

  constructor(minCutoff: number = 0.85, beta: number = 0.009, dCutoff: number = 1.0) {
    this.xf = new OneEuroFilter1D(minCutoff, beta, dCutoff);
    this.yf = new OneEuroFilter1D(minCutoff, beta, dCutoff);
    this.zf = new OneEuroFilter1D(minCutoff, beta, dCutoff);
  }

  public filter(p: Point3D, timestamp: number, maxJumpThreshold: number = 0.35): Point3D {
    let target = p;

    // Outlier jump rejection: clamp non-physical spatial spikes (e.g. sensor/detector glitch)
    if (this.lastFiltered) {
      const dx = target.x - this.lastFiltered.x;
      const dy = target.y - this.lastFiltered.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist > maxJumpThreshold) {
        const factor = maxJumpThreshold / dist;
        target = {
          x: this.lastFiltered.x + dx * factor,
          y: this.lastFiltered.y + dy * factor,
          z: target.z
        };
      }
    }

    const out: Point3D = {
      x: this.xf.filter(target.x, timestamp),
      y: this.yf.filter(target.y, timestamp),
      z: this.zf.filter(target.z ?? 0, timestamp)
    };
    this.lastFiltered = out;
    return out;
  }

  public reset() {
    this.lastFiltered = null;
    this.xf.reset();
    this.yf.reset();
    this.zf.reset();
  }
}

/**
 * Anatomically differentiated 21-joint filter bank for hand skeletons.
 * Each joint is assigned filter parameters matching its biomechanical rigidity.
 */
export class HandSkeletonFilter {
  private landmarkFilters: OneEuroFilter3D[];

  constructor() {
    this.landmarkFilters = Array.from({ length: 21 }, (_, jointIdx) => {
      const cfg = getAnatomicalFilterConfig(jointIdx);
      return new OneEuroFilter3D(cfg.minCutoff, cfg.beta, cfg.dCutoff);
    });
  }

  public filter(
    landmarks: Point3D[],
    timestamp: number,
    palmScale: number = 0.15
  ): ProvenancedLandmark[] {
    if (!landmarks || landmarks.length !== 21) {
      return (landmarks || []).map((pt) => ({ ...pt, provenance: 'OBSERVED' }));
    }

    // Dynamic jump threshold scaled to the hand's pixel dimensions
    const maxJump = Math.max(0.12, palmScale * 2.8);

    return landmarks.map((pt, idx) => {
      const filtered = this.landmarkFilters[idx].filter(pt, timestamp, maxJump);
      return {
        x: filtered.x,
        y: filtered.y,
        z: filtered.z,
        rawX: pt.x,
        rawY: pt.y,
        rawZ: pt.z,
        provenance: 'FILTERED'
      };
    });
  }

  public reset() {
    this.landmarkFilters.forEach((f) => f.reset());
  }
}

export interface Point3D {
  x: number;
  y: number;
  z: number;
}
