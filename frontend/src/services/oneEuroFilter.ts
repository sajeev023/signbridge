/**
 * One Euro Filter (1€ Filter) for adaptive low-pass filtering of noisy signals.
 * Casiez, G., Roussel, N. and Vogel, F. (2012).
 * 1 € Filter: A Simple Speed-based Low-pass Filter for Noisy and Lag-sensitive Everyday Interaction.
 * ACM Conference on Human Factors in Computing Systems (CHI '12).
 *
 * Provides near-zero lag during rapid movement while achieving jitter-free stability at low velocity.
 */

export interface Point3D {
  x: number;
  y: number;
  z: number;
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

  /**
   * @param minCutoff Minimum cutoff frequency in Hz (smaller = more filtering at low speed, less jitter). Default: 1.0
   * @param beta Speed coefficient (larger = less lag during rapid movement). Default: 0.007
   * @param dCutoff Cutoff frequency for derivative calculation in Hz. Default: 1.0
   */
  constructor(minCutoff: number = 0.9, beta: number = 0.008, dCutoff: number = 1.0) {
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

  constructor(minCutoff: number = 0.9, beta: number = 0.008, dCutoff: number = 1.0) {
    this.xf = new OneEuroFilter1D(minCutoff, beta, dCutoff);
    this.yf = new OneEuroFilter1D(minCutoff, beta, dCutoff);
    this.zf = new OneEuroFilter1D(minCutoff, beta, dCutoff);
  }

  public filter(p: Point3D, timestamp: number): Point3D {
    return {
      x: this.xf.filter(p.x, timestamp),
      y: this.yf.filter(p.y, timestamp),
      z: this.zf.filter(p.z ?? 0, timestamp)
    };
  }

  public reset() {
    this.xf.reset();
    this.yf.reset();
    this.zf.reset();
  }
}

/**
 * Manages 21 3D One Euro Filters for an entire hand skeleton.
 */
export class HandSkeletonFilter {
  private landmarkFilters: OneEuroFilter3D[];

  constructor(minCutoff: number = 0.85, beta: number = 0.009, dCutoff: number = 1.0) {
    this.landmarkFilters = Array.from({ length: 21 }, () => new OneEuroFilter3D(minCutoff, beta, dCutoff));
  }

  public filter(landmarks: Point3D[], timestamp: number): Point3D[] {
    if (!landmarks || landmarks.length !== 21) return landmarks;

    return landmarks.map((pt, idx) => {
      return this.landmarkFilters[idx].filter(pt, timestamp);
    });
  }

  public reset() {
    this.landmarkFilters.forEach((f) => f.reset());
  }
}
