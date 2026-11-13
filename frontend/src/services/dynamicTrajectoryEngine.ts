/**
 * SIGNBRIDGE Dynamic J/Z Point-History Trajectory Engine
 * 
 * Implements circular point-history temporal buffering, palm-scale distance-invariant
 * normalization, variable-speed arc-length spatial resampling, and anatomical handshape
 * context gating for dynamic ASL alphabet signs 'J' and 'Z'.
 * 
 * Grounded in:
 * - Point-history circular buffer architecture by Kazuhito Takahashi (@Kazuhito00)
 *   and Nikita Kiselov (@kinivi) (Apache License 2.0).
 * - Upgraded for SIGNBRIDGE with palm-scale Euclidean normalization, arc-length resampling,
 *   and anatomical handshape context gating.
 */

import dynamicModel from './dynamic_alphabet_model.json';
import { Point3D } from './oneEuroFilter';

export interface DynamicTrajectoryResult {
  sign: 'J' | 'Z' | 'NEUTRAL';
  confidence: number;
  isDynamic: boolean;
  pathLength: number;
  allProbabilities: Record<string, number>;
}

interface BufferEntry {
  x: number;
  y: number;
  z: number;
  palmScale: number;
  timestamp: number;
  handId: number;
}

export const TRAJECTORY_BUFFER_LENGTH = 16;
export const TRAJECTORY_FEATURE_SIZE = 32;
export const MIN_PATH_LENGTH_THRESHOLD = 0.70; // 0.70x palm-scale minimum stroke displacement
export const DYNAMIC_CONFIDENCE_THRESHOLD = 0.85;

/**
 * Spatially resamples a 2D trajectory of points to exactly targetPoints
 * uniform arc-length samples, achieving invariance to drawing speed.
 */
export function resampleTrajectory2D(
  points: Array<[number, number]>,
  targetPoints: number = TRAJECTORY_BUFFER_LENGTH
): Array<[number, number]> {
  const n = points.length;
  if (n === 0) {
    return Array.from({ length: targetPoints }, () => [0, 0]);
  }
  if (n === 1 || n === targetPoints) {
    if (n === 1) {
      return Array.from({ length: targetPoints }, () => [points[0][0], points[0][1]]);
    }
    return points.map((p) => [p[0], p[1]]);
  }

  // Calculate cumulative Euclidean distance along path
  const cumDist: number[] = new Array(n).fill(0);
  for (let i = 1; i < n; i++) {
    const dx = points[i][0] - points[i - 1][0];
    const dy = points[i][1] - points[i - 1][1];
    cumDist[i] = cumDist[i - 1] + Math.sqrt(dx * dx + dy * dy);
  }
  const totalLen = cumDist[n - 1];

  if (totalLen < 1e-4) {
    return Array.from({ length: targetPoints }, () => [points[0][0], points[0][1]]);
  }

  const resampled: Array<[number, number]> = [];
  for (let k = 0; k < targetPoints; k++) {
    const targetD = (k / (targetPoints - 1)) * totalLen;
    let idx = 0;
    while (idx < n - 2 && cumDist[idx + 1] < targetD) {
      idx++;
    }
    const segDist = cumDist[idx + 1] - cumDist[idx];
    const t = segDist > 1e-6 ? (targetD - cumDist[idx]) / segDist : 0;
    const x = points[idx][0] + t * (points[idx + 1][0] - points[idx][0]);
    const y = points[idx][1] + t * (points[idx + 1][1] - points[idx][1]);
    resampled.push([x, y]);
  }

  return resampled;
}

function dist2D(p1: { x: number; y: number }, p2: { x: number; y: number }): number {
  const dx = p1.x - p2.x;
  const dy = p1.y - p2.y;
  return Math.sqrt(dx * dx + dy * dy);
}

/**
 * Validates anatomical handshape context for ASL 'J':
 * Pinky extended (I-shape), index and middle curled.
 */
export function isJHandshapeContext(landmarks: Point3D[], palmScale: number): boolean {
  if (!landmarks || landmarks.length < 21) return false;
  const scale = Math.max(0.035, palmScale);

  const pinkyExt =
    (dist2D(landmarks[20], landmarks[0]) - dist2D(landmarks[17], landmarks[0])) / scale;
  const idxCurl = dist2D(landmarks[8], landmarks[5]) / scale;
  const midCurl = dist2D(landmarks[12], landmarks[9]) / scale;

  return pinkyExt > 0.40 && idxCurl < 0.65 && midCurl < 0.65;
}

/**
 * Validates anatomical handshape context for ASL 'Z':
 * Index finger extended (D-shape), middle, ring and pinky curled.
 */
export function isZHandshapeContext(landmarks: Point3D[], palmScale: number): boolean {
  if (!landmarks || landmarks.length < 21) return false;
  const scale = Math.max(0.035, palmScale);

  const idxExt =
    (dist2D(landmarks[8], landmarks[0]) - dist2D(landmarks[5], landmarks[0])) / scale;
  const midCurl = dist2D(landmarks[12], landmarks[9]) / scale;
  const pkyCurl = dist2D(landmarks[20], landmarks[17]) / scale;

  return idxExt > 0.45 && midCurl < 0.65 && pkyCurl < 0.65;
}

/**
 * Circular point-history buffer maintaining fingertip positions with
 * palm-scale distance invariance, continuity checking, and speed normalization.
 */
export class PointHistoryBuffer {
  private buffer: BufferEntry[] = [];
  private maxlen: number;

  constructor(maxlen: number = 32) {
    this.maxlen = maxlen;
  }

  public append(point: Point3D, palmScale: number, timestampMs: number, handId: number = 0) {
    if (this.buffer.length >= this.maxlen) {
      this.buffer.shift();
    }
    this.buffer.push({
      x: point.x,
      y: point.y,
      z: point.z ?? 0,
      palmScale: Math.max(0.035, palmScale),
      timestamp: timestampMs,
      handId
    });
  }

  public reset() {
    this.buffer = [];
  }

  public length(): number {
    return this.buffer.length;
  }

  public isContinuous(maxGapMs: number = 300): boolean {
    if (this.buffer.length < 8) return false;
    const firstId = this.buffer[0].handId;
    for (let i = 1; i < this.buffer.length; i++) {
      if (this.buffer[i].handId !== firstId) return false;
      if (this.buffer[i].timestamp - this.buffer[i - 1].timestamp > maxGapMs) {
        return false;
      }
    }
    return true;
  }

  public calculatePathLength(): number {
    if (this.buffer.length < 2) return 0;
    let sumDist = 0;
    let sumScale = 0;
    for (let i = 0; i < this.buffer.length; i++) {
      sumScale += this.buffer[i].palmScale;
      if (i > 0) {
        const dx = this.buffer[i].x - this.buffer[i - 1].x;
        const dy = this.buffer[i].y - this.buffer[i - 1].y;
        sumDist += Math.sqrt(dx * dx + dy * dy);
      }
    }
    const avgScale = sumScale / this.buffer.length;
    return sumDist / avgScale;
  }

  public extractNormalizedFeatures(): number[] {
    if (this.buffer.length < 4) {
      return new Array<number>(TRAJECTORY_FEATURE_SIZE).fill(0);
    }

    const firstPt = this.buffer[0];
    let sumScale = 0;
    for (let i = 0; i < this.buffer.length; i++) {
      sumScale += this.buffer[i].palmScale;
    }
    const avgScale = sumScale / this.buffer.length;

    // Origin shift: relative to starting point, normalized by avg palm scale
    const relPts: Array<[number, number]> = this.buffer.map((p) => [
      (p.x - firstPt.x) / avgScale,
      (p.y - firstPt.y) / avgScale
    ]);

    // Resample to exactly 16 uniform arc-length steps
    const resampled = resampleTrajectory2D(relPts, TRAJECTORY_BUFFER_LENGTH);

    // Flatten to 32 continuous features
    const feat = new Array<number>(TRAJECTORY_FEATURE_SIZE);
    for (let i = 0; i < TRAJECTORY_BUFFER_LENGTH; i++) {
      feat[i * 2] = resampled[i][0];
      feat[i * 2 + 1] = resampled[i][1];
    }
    return feat;
  }
}

/**
 * Production Dynamic Trajectory Engine
 * Integrates Kinivi-inspired point-history buffer with SIGNBRIDGE's
 * palm-scale invariants and dynamic MLP classifier.
 */
export class DynamicTrajectoryEngine {
  private weights = dynamicModel.weights;
  private classes = dynamicModel.classes as Array<'NEUTRAL' | 'J' | 'Z'>;
  private pinkyBuffer = new PointHistoryBuffer(32);
  private indexBuffer = new PointHistoryBuffer(32);
  private cooldownUntil = 0;
  private lastTriggeredSign: 'J' | 'Z' | null = null;

  public reset() {
    this.pinkyBuffer.reset();
    this.indexBuffer.reset();
    this.cooldownUntil = 0;
    this.lastTriggeredSign = null;
  }

  private runMLP(x: number[]): { sign: 'NEUTRAL' | 'J' | 'Z'; confidence: number; allProbs: Record<string, number> } {
    // Layer 1 (32 -> 32)
    const z1 = new Array<number>(this.weights.w1.length);
    for (let i = 0; i < this.weights.w1.length; i++) {
      let sum = this.weights.b1[i];
      const row = this.weights.w1[i];
      for (let j = 0; j < row.length; j++) {
        sum += row[j] * x[j];
      }
      z1[i] = Math.max(0, sum);
    }

    // Layer 2 (32 -> 16)
    const z2 = new Array<number>(this.weights.w2.length);
    for (let i = 0; i < this.weights.w2.length; i++) {
      let sum = this.weights.b2[i];
      const row = this.weights.w2[i];
      for (let j = 0; j < row.length; j++) {
        sum += row[j] * z1[j];
      }
      z2[i] = Math.max(0, sum);
    }

    // Layer 3 (16 -> 3)
    const logits = new Array<number>(this.weights.w3.length);
    let maxLogit = -Infinity;
    for (let i = 0; i < this.weights.w3.length; i++) {
      let sum = this.weights.b3[i];
      const row = this.weights.w3[i];
      for (let j = 0; j < row.length; j++) {
        sum += row[j] * z2[j];
      }
      logits[i] = sum;
      if (sum > maxLogit) maxLogit = sum;
    }

    // Softmax
    let sumExp = 0;
    const probs = new Array<number>(logits.length);
    for (let i = 0; i < logits.length; i++) {
      probs[i] = Math.exp(logits[i] - maxLogit);
      sumExp += probs[i];
    }

    const allProbs: Record<string, number> = {};
    let bestSign: 'NEUTRAL' | 'J' | 'Z' = 'NEUTRAL';
    let bestConf = 0;

    for (let i = 0; i < logits.length; i++) {
      probs[i] /= sumExp;
      const c = this.classes[i] || 'NEUTRAL';
      allProbs[c] = probs[i];
      if (probs[i] > bestConf) {
        bestConf = probs[i];
        bestSign = c;
      }
    }

    return { sign: bestSign, confidence: bestConf, allProbs };
  }

  /**
   * Updates rolling buffers and evaluates dynamic signs.
   * Returns dynamic result if a valid J or Z gesture is detected, null otherwise.
   */
  public update(
    landmarks: Point3D[],
    palmScale: number,
    timestampMs: number,
    handId: number = 0
  ): DynamicTrajectoryResult | null {
    if (!landmarks || landmarks.length < 21) {
      this.reset();
      return null;
    }

    // Handle post-trigger cooldown
    if (timestampMs < this.cooldownUntil) {
      return null;
    }

    const inJContext = isJHandshapeContext(landmarks, palmScale);
    const inZContext = isZHandshapeContext(landmarks, palmScale);

    // Track active fingertip
    const pinkyTip = landmarks[20];
    const indexTip = landmarks[8];

    this.pinkyBuffer.append(pinkyTip, palmScale, timestampMs, handId);
    this.indexBuffer.append(indexTip, palmScale, timestampMs, handId);

    // Dynamic 'J' evaluation (gated on pinky handshape context)
    if (inJContext && this.pinkyBuffer.isContinuous()) {
      const pathLen = this.pinkyBuffer.calculatePathLength();
      if (pathLen >= MIN_PATH_LENGTH_THRESHOLD) {
        const feat = this.pinkyBuffer.extractNormalizedFeatures();
        const pred = this.runMLP(feat);
        if (pred.sign === 'J' && pred.confidence >= DYNAMIC_CONFIDENCE_THRESHOLD) {
          this.cooldownUntil = timestampMs + 750;
          this.pinkyBuffer.reset();
          this.indexBuffer.reset();
          return {
            sign: 'J',
            confidence: pred.confidence,
            isDynamic: true,
            pathLength: pathLen,
            allProbabilities: pred.allProbs
          };
        }
      }
    }

    // Dynamic 'Z' evaluation (gated on index handshape context)
    if (inZContext && this.indexBuffer.isContinuous()) {
      const pathLen = this.indexBuffer.calculatePathLength();
      if (pathLen >= MIN_PATH_LENGTH_THRESHOLD) {
        const feat = this.indexBuffer.extractNormalizedFeatures();
        const pred = this.runMLP(feat);
        if (pred.sign === 'Z' && pred.confidence >= DYNAMIC_CONFIDENCE_THRESHOLD) {
          this.cooldownUntil = timestampMs + 750;
          this.pinkyBuffer.reset();
          this.indexBuffer.reset();
          return {
            sign: 'Z',
            confidence: pred.confidence,
            isDynamic: true,
            pathLength: pathLen,
            allProbabilities: pred.allProbs
          };
        }
      }
    }

    return null;
  }
}

export const dynamicTrajectoryEngine = new DynamicTrajectoryEngine();
