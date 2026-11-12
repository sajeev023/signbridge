import islDualHandModel from './isl_alphabet_model.json';
import singleHandModel from './single_hand_alphabet_model.json';

export interface LandmarkPoint {
  x: number;
  y: number;
  z: number;
}

export type AlphabetConvention = 'auto' | 'islrtc_bimanual' | 'single_hand_chart';

export interface InferenceResult {
  sign: string;
  confidence: number;
  isDynamic: boolean;
  handCount: number;
  convention: 'ISLRTC (2-Hand)' | 'Single-Hand (Chart)';
  allProbabilities: Record<string, number>;
  rawLandmarks?: LandmarkPoint[][];
  smoothedLandmarks?: LandmarkPoint[][];
}

export interface TrajectoryPoint {
  x: number;
  y: number;
  time: number;
}

class AlphabetInferenceEngine {
  // Dual-hand ISLRTC weights & metadata
  private dualHandWeights = islDualHandModel.weights;
  private dualHandClasses = islDualHandModel.classes;

  // Single-hand Chart weights & metadata
  private singleHandWeights = singleHandModel.weights;
  private singleHandClasses = singleHandModel.classes;

  // Anti-jitter landmark smoothing state (EMA filter)
  private prevSmoothedHands: LandmarkPoint[][] = [];
  private readonly SMOOTHING_ALPHA = 0.72; // Balance between zero lag and jitter-free rendering

  // Temporal buffer for dynamic sign tracking ('J', 'Z')
  private trajectoryBuffer: TrajectoryPoint[] = [];

  private dist3D(p1: LandmarkPoint, p2: LandmarkPoint): number {
    const dx = p1.x - p2.x;
    const dy = p1.y - p2.y;
    const dz = (p1.z ?? 0) - (p2.z ?? 0);
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  private dist2D(p1: LandmarkPoint, p2: LandmarkPoint): number {
    const dx = p1.x - p2.x;
    const dy = p1.y - p2.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  private cosineAngle(v1: [number, number], v2: [number, number]): number {
    const dot = v1[0] * v2[0] + v1[1] * v2[1];
    const mag1 = Math.sqrt(v1[0] * v1[0] + v1[1] * v1[1]) || 1e-6;
    const mag2 = Math.sqrt(v2[0] * v2[0] + v2[1] * v2[1]) || 1e-6;
    return Math.max(-1, Math.min(1, dot / (mag1 * mag2)));
  }

  /**
   * Applies Exponential Moving Average (EMA) smoothing to 21 landmarks
   * to eliminate camera sensor jitter and landmark flicker.
   */
  public smoothLandmarks(rawHands: LandmarkPoint[][]): LandmarkPoint[][] {
    if (!rawHands || rawHands.length === 0) {
      this.prevSmoothedHands = [];
      return [];
    }

    const smoothed: LandmarkPoint[][] = [];

    for (let h = 0; h < rawHands.length; h++) {
      const rawHand = rawHands[h];
      const prevHand = this.prevSmoothedHands[h];

      if (!prevHand || prevHand.length !== 21) {
        smoothed.push(rawHand.map((p) => ({ ...p })));
        continue;
      }

      // Check for abrupt hand relocation / tracking re-acquisition
      const wristJump = this.dist2D(rawHand[0], prevHand[0]);
      const alpha = wristJump > 0.25 ? 1.0 : this.SMOOTHING_ALPHA;

      const smoothedHand: LandmarkPoint[] = [];
      for (let j = 0; j < 21; j++) {
        smoothedHand.push({
          x: alpha * rawHand[j].x + (1 - alpha) * prevHand[j].x,
          y: alpha * rawHand[j].y + (1 - alpha) * prevHand[j].y,
          z: alpha * (rawHand[j].z ?? 0) + (1 - alpha) * (prevHand[j].z ?? 0)
        });
      }
      smoothed.push(smoothedHand);
    }

    this.prevSmoothedHands = smoothed;
    return smoothed;
  }

  /**
   * Extracts 28 invariant features for single-hand manual alphabet.
   */
  public extractSingleHandFeatures(hand: LandmarkPoint[]): number[] {
    const feat = new Array<number>(28).fill(0);
    if (!hand || hand.length < 21) return feat;

    const s = Math.max(0.04, this.dist3D(hand[0], hand[9])); // Palm scale

    // 1. Extensions
    feat[0] = (this.dist3D(hand[4], hand[0]) - this.dist3D(hand[2], hand[0])) / s;
    feat[1] = (this.dist3D(hand[8], hand[0]) - this.dist3D(hand[6], hand[0])) / s;
    feat[2] = (this.dist3D(hand[12], hand[0]) - this.dist3D(hand[10], hand[0])) / s;
    feat[3] = (this.dist3D(hand[16], hand[0]) - this.dist3D(hand[14], hand[0])) / s;
    feat[4] = (this.dist3D(hand[20], hand[0]) - this.dist3D(hand[18], hand[0])) / s;

    // 2. Tip to wrist distances
    feat[5] = this.dist3D(hand[4], hand[0]) / s;
    feat[6] = this.dist3D(hand[8], hand[0]) / s;
    feat[7] = this.dist3D(hand[12], hand[0]) / s;
    feat[8] = this.dist3D(hand[16], hand[0]) / s;
    feat[9] = this.dist3D(hand[20], hand[0]) / s;

    // 3. Curls
    feat[10] = 0.5;
    feat[11] = Math.max(0, 1.0 - this.dist3D(hand[8], hand[5]) / s);
    feat[12] = Math.max(0, 1.0 - this.dist3D(hand[12], hand[9]) / s);
    feat[13] = Math.max(0, 1.0 - this.dist3D(hand[16], hand[13]) / s);
    feat[14] = Math.max(0, 1.0 - this.dist3D(hand[20], hand[17]) / s);

    // 4. Spreads
    feat[15] = this.dist3D(hand[4], hand[8]) / s;
    feat[16] = this.dist3D(hand[8], hand[12]) / s;
    feat[17] = this.dist3D(hand[12], hand[16]) / s;
    feat[18] = this.dist3D(hand[16], hand[20]) / s;

    // 5. Handshape Invariants
    feat[19] = feat[15]; // loop thumb-index
    feat[20] = this.dist3D(hand[4], hand[12]) / s; // loop thumb-mid

    // L-angle
    const thumbVec: [number, number] = [hand[4].x - hand[2].x, hand[4].y - hand[2].y];
    const indexVec: [number, number] = [hand[8].x - hand[5].x, hand[8].y - hand[5].y];
    const dot = Math.abs(this.cosineAngle(thumbVec, indexVec));
    feat[21] = feat[15] > 0.6 && feat[1] > 0.4 ? 1.0 - dot : 0.0;

    // Y-span (thumb tip to pinky tip)
    feat[22] = this.dist3D(hand[4], hand[20]) / s;

    // Crossed index-middle (R)
    feat[23] = feat[1] > 0.5 && feat[2] > 0.5 && feat[16] < 0.25 ? 0.9 : 0.0;

    // Fist compactness
    const avgTipDist =
      (this.dist3D(hand[4], hand[9]) +
        this.dist3D(hand[8], hand[9]) +
        this.dist3D(hand[12], hand[9]) +
        this.dist3D(hand[16], hand[9]) +
        this.dist3D(hand[20], hand[9])) /
      (5 * s);
    feat[24] = Math.max(0, 1.2 - avgTipDist);

    // Horizontal alignment (G, H)
    const idxDx = Math.abs(hand[8].x - hand[5].x);
    const idxDy = Math.abs(hand[8].y - hand[5].y);
    feat[25] = idxDx > 1.5 * idxDy ? 0.9 : 0.1;

    // C-curve metric
    feat[26] = feat[15] > 0.3 && feat[11] > 0.2 && feat[12] > 0.2 ? 0.92 : 0.1;

    // U-parallel metric
    feat[27] = feat[1] > 0.7 && feat[2] > 0.7 && feat[16] < 0.15 ? 0.94 : 0.1;

    return feat;
  }

  /**
   * Extracts 44 invariant features for dual-hand ISLRTC manual alphabet.
   */
  public extractDualHandFeatures(
    landmarks: LandmarkPoint[][],
    handedness?: Array<Array<{ categoryName: string; score: number }>>
  ): number[] {
    const feat = new Array<number>(44).fill(0);
    if (!landmarks || landmarks.length === 0) return feat;

    let h1 = landmarks[0];
    let h2: LandmarkPoint[] | null = landmarks.length > 1 ? landmarks[1] : null;

    if (landmarks.length > 1 && handedness && handedness.length >= 2) {
      const h0Name = handedness[0]?.[0]?.categoryName;
      if (h0Name === 'Left' && handedness[1]?.[0]?.categoryName === 'Right') {
        h1 = landmarks[1];
        h2 = landmarks[0];
      }
    } else if (landmarks.length > 1) {
      const iso0 = this.dist2D(landmarks[0][8], landmarks[0][0]) / (this.dist2D(landmarks[0][12], landmarks[0][0]) || 1);
      const iso1 = this.dist2D(landmarks[1][8], landmarks[1][0]) / (this.dist2D(landmarks[1][12], landmarks[1][0]) || 1);
      if (iso1 > iso0) {
        h1 = landmarks[1];
        h2 = landmarks[0];
      }
    }

    const s1 = Math.max(0.04, this.dist3D(h1[0], h1[9]));

    feat[0] = (this.dist3D(h1[4], h1[0]) - this.dist3D(h1[2], h1[0])) / s1;
    feat[1] = (this.dist3D(h1[8], h1[0]) - this.dist3D(h1[6], h1[0])) / s1;
    feat[2] = (this.dist3D(h1[12], h1[0]) - this.dist3D(h1[10], h1[0])) / s1;
    feat[3] = (this.dist3D(h1[16], h1[0]) - this.dist3D(h1[14], h1[0])) / s1;
    feat[4] = (this.dist3D(h1[20], h1[0]) - this.dist3D(h1[18], h1[0])) / s1;
    feat[5] = 0.5;
    feat[6] = Math.max(0, 1.0 - this.dist3D(h1[8], h1[5]) / s1);
    feat[7] = Math.max(0, 1.0 - this.dist3D(h1[12], h1[9]) / s1);
    feat[8] = Math.max(0, 1.0 - this.dist3D(h1[16], h1[13]) / s1);
    feat[9] = Math.max(0, 1.0 - this.dist3D(h1[20], h1[17]) / s1);
    feat[10] = this.dist3D(h1[4], h1[8]) / s1;
    feat[11] = this.dist3D(h1[8], h1[12]) / s1;
    feat[12] = this.dist3D(h1[12], h1[16]) / s1;
    feat[13] = this.dist3D(h1[16], h1[20]) / s1;

    const thumbVec: [number, number] = [h1[4].x - h1[2].x, h1[4].y - h1[2].y];
    const indexVec: [number, number] = [h1[8].x - h1[5].x, h1[8].y - h1[5].y];
    const dotThumbIndex = Math.abs(this.cosineAngle(thumbVec, indexVec));
    feat[14] = feat[10] > 0.6 && feat[1] > 0.4 ? 1.0 - dotThumbIndex : 0.0;
    feat[15] = feat[10] > 0.3 && feat[6] > 0.2 ? 0.9 : 0.1;
    feat[16] = feat[10];

    if (!h2) {
      feat[17] = 0.0;
      feat[24] = 3.0;
      feat[27] = 3.0;
      feat[28] = 3.0;
      feat[29] = 3.0;
      feat[30] = 3.0;
      feat[31] = 3.0;
      return feat;
    }

    feat[17] = 1.0;
    const s2 = Math.max(0.04, this.dist3D(h2[0], h2[9]));

    feat[18] = (this.dist3D(h2[4], h2[0]) - this.dist3D(h2[2], h2[0])) / s2;
    feat[19] = (this.dist3D(h2[8], h2[0]) - this.dist3D(h2[6], h2[0])) / s2;
    feat[20] = (this.dist3D(h2[12], h2[0]) - this.dist3D(h2[10], h2[0])) / s2;
    feat[21] = (this.dist3D(h2[16], h2[0]) - this.dist3D(h2[14], h2[0])) / s2;
    feat[22] = (this.dist3D(h2[20], h2[0]) - this.dist3D(h2[18], h2[0])) / s2;
    feat[23] = this.dist3D(h2[4], h2[8]) / s2;

    feat[24] = (h1[0].x - h2[0].x) / s2;
    feat[25] = (h1[0].y - h2[0].y) / s2;
    feat[26] = ((h1[0].z ?? 0) - (h2[0].z ?? 0)) / s2;

    feat[27] = this.dist3D(h1[8], h2[4]) / s2; // Vowel A
    feat[28] = this.dist3D(h1[8], h2[8]) / s2; // Vowel E
    feat[29] = this.dist3D(h1[8], h2[12]) / s2; // Vowel I
    feat[30] = this.dist3D(h1[8], h2[16]) / s2; // Vowel O
    feat[31] = this.dist3D(h1[8], h2[20]) / s2; // Vowel U
    feat[32] = this.dist3D(h1[8], h2[5]) / s2; // T
    feat[33] = this.dist3D(h1[8], h2[9]) / s2; // M, N, V, H

    const h2Cleft: LandmarkPoint = {
      x: (h2[2].x + h2[5].x) / 2,
      y: (h2[2].y + h2[5].y) / 2,
      z: ((h2[2].z ?? 0) + (h2[5].z ?? 0)) / 2
    };
    feat[34] = this.dist3D(h1[8], h2Cleft) / s2; // Y
    feat[35] = this.dist3D(h1[8], h2[6]) / s2; // K

    const h1LoopMid: LandmarkPoint = {
      x: (h1[4].x + h1[8].x) / 2,
      y: (h1[4].y + h1[8].y) / 2,
      z: ((h1[4].z ?? 0) + (h1[8].z ?? 0)) / 2
    };
    const h2LoopMid: LandmarkPoint = {
      x: (h2[4].x + h2[8].x) / 2,
      y: (h2[4].y + h2[8].y) / 2,
      z: ((h2[4].z ?? 0) + (h2[8].z ?? 0)) / 2
    };
    feat[36] = this.dist3D(h1LoopMid, h2LoopMid) / s2; // B
    feat[37] = this.dist3D(h1[4], h2[8]) / s2; // D
    feat[38] = this.dist3D(h1LoopMid, h2[8]) / s2; // P
    feat[39] = this.dist3D(h1[20], h2[20]) / s2; // S
    feat[40] = this.dist3D(h1[9], h2[9]) / s2; // G
    feat[41] = this.dist3D(h1[12], h2[9]) / s2; // M, N
    feat[42] = this.dist3D(h1[16], h2[9]) / s2; // M
    feat[43] = this.dist3D(h1[12], h2[8]) / s2; // F, X

    return feat;
  }

  /**
   * Forward pass through 3-layer MLP.
   */
  private runMLP(weights: any, classes: string[], x: number[]): { sign: string; confidence: number; allProbs: Record<string, number> } {
    // Layer 1
    const z1 = new Array<number>(weights.w1.length);
    for (let i = 0; i < weights.w1.length; i++) {
      let sum = weights.b1[i];
      const row = weights.w1[i];
      for (let j = 0; j < row.length; j++) {
        sum += row[j] * x[j];
      }
      z1[i] = Math.max(0, sum);
    }

    // Layer 2
    const z2 = new Array<number>(weights.w2.length);
    for (let i = 0; i < weights.w2.length; i++) {
      let sum = weights.b2[i];
      const row = weights.w2[i];
      for (let j = 0; j < row.length; j++) {
        sum += row[j] * z1[j];
      }
      z2[i] = Math.max(0, sum);
    }

    // Layer 3 (logits)
    const logits = new Array<number>(weights.w3.length);
    let maxLogit = -Infinity;
    for (let i = 0; i < weights.w3.length; i++) {
      let sum = weights.b3[i];
      const row = weights.w3[i];
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
    let bestSign = 'NEUTRAL';
    let bestConf = 0;

    for (let i = 0; i < logits.length; i++) {
      probs[i] /= sumExp;
      const className = classes[i] || `Class_${i}`;
      allProbs[className] = probs[i];
      if (probs[i] > bestConf) {
        bestConf = probs[i];
        bestSign = className;
      }
    }

    return { sign: bestSign, confidence: bestConf, allProbs };
  }

  /**
   * Evaluates dynamic trajectory buffer for ISLRTC dynamic letters 'J' and 'Z'.
   */
  private checkDynamicSigns(
    landmarks: LandmarkPoint[][],
    now: number
  ): { sign: string; confidence: number } | null {
    if (!landmarks || landmarks.length === 0) {
      this.trajectoryBuffer = [];
      return null;
    }

    const tip = landmarks[0][8];
    this.trajectoryBuffer.push({ x: tip.x, y: tip.y, time: now });

    while (this.trajectoryBuffer.length > 0 && now - this.trajectoryBuffer[0].time > 1200) {
      this.trajectoryBuffer.shift();
    }
    if (this.trajectoryBuffer.length < 12) return null;

    // Check letter 'J'
    const firstPoint = this.trajectoryBuffer[0];
    const midPoint = this.trajectoryBuffer[Math.floor(this.trajectoryBuffer.length / 2)];
    const lastPoint = this.trajectoryBuffer[this.trajectoryBuffer.length - 1];

    const deltaYDown = midPoint.y - firstPoint.y;
    const deltaXHook = lastPoint.x - midPoint.x;
    const deltaYUp = midPoint.y - lastPoint.y;

    if (deltaYDown > 0.08 && deltaXHook < -0.04 && deltaYUp > 0.02) {
      return { sign: 'J', confidence: 0.92 };
    }

    // Check letter 'Z'
    if (this.trajectoryBuffer.length >= 18) {
      const p1 = this.trajectoryBuffer[0];
      const p2 = this.trajectoryBuffer[Math.floor(this.trajectoryBuffer.length / 3)];
      const p3 = this.trajectoryBuffer[Math.floor((this.trajectoryBuffer.length * 2) / 3)];
      const p4 = this.trajectoryBuffer[this.trajectoryBuffer.length - 1];

      const stroke1Right = p2.x - p1.x > 0.05;
      const stroke2DownLeft = p3.y - p2.y > 0.05 && p3.x - p2.x < -0.04;
      const stroke3Right = p4.x - p3.x > 0.05;

      if (stroke1Right && stroke2DownLeft && stroke3Right) {
        return { sign: 'Z', confidence: 0.94 };
      }
    }

    return null;
  }

  /**
   * Runs end-to-end forward neural inference on hand landmarks.
   */
  public predict(
    rawLandmarks: LandmarkPoint[][],
    handedness?: Array<Array<{ categoryName: string; score: number }>>,
    convention: AlphabetConvention = 'auto',
    timestamp: number = performance.now()
  ): InferenceResult {
    if (!rawLandmarks || rawLandmarks.length === 0) {
      this.prevSmoothedHands = [];
      return {
        sign: 'NEUTRAL',
        confidence: 0,
        isDynamic: false,
        handCount: 0,
        convention: 'ISLRTC (2-Hand)',
        allProbabilities: {}
      };
    }

    // 1. Anti-jitter smoothing
    const smoothed = this.smoothLandmarks(rawLandmarks);

    // 2. Check for dynamic signs
    const dynamicMatch = this.checkDynamicSigns(smoothed, timestamp);
    if (dynamicMatch && dynamicMatch.confidence >= 0.85) {
      return {
        sign: dynamicMatch.sign,
        confidence: dynamicMatch.confidence,
        isDynamic: true,
        handCount: smoothed.length,
        convention: 'ISLRTC (2-Hand)',
        allProbabilities: { [dynamicMatch.sign]: dynamicMatch.confidence },
        rawLandmarks,
        smoothedLandmarks: smoothed
      };
    }

    // 3. Determine execution mode
    const useSingleHand =
      convention === 'single_hand_chart' || (convention === 'auto' && smoothed.length === 1);

    if (useSingleHand) {
      const features = this.extractSingleHandFeatures(smoothed[0]);
      const res = this.runMLP(this.singleHandWeights, this.singleHandClasses, features);
      return {
        sign: res.sign,
        confidence: res.confidence,
        isDynamic: false,
        handCount: 1,
        convention: 'Single-Hand (Chart)',
        allProbabilities: res.allProbs,
        rawLandmarks,
        smoothedLandmarks: smoothed
      };
    } else {
      const features = this.extractDualHandFeatures(smoothed, handedness);
      const res = this.runMLP(this.dualHandWeights, this.dualHandClasses, features);
      return {
        sign: res.sign,
        confidence: res.confidence,
        isDynamic: false,
        handCount: smoothed.length,
        convention: 'ISLRTC (2-Hand)',
        allProbabilities: res.allProbs,
        rawLandmarks,
        smoothedLandmarks: smoothed
      };
    }
  }
}

export const alphabetInferenceEngine = new AlphabetInferenceEngine();
