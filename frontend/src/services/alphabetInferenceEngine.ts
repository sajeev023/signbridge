import islDualHandModel from './isl_alphabet_model.json';
import singleHandModel from './single_hand_alphabet_model.json';
import { Point3D } from './oneEuroFilter';
import { persistentHandTracker, PersistentHandTrack } from './persistentHandTracker';
import { trackingQualityEngine, TrackingQualityReport } from './trackingQualityEngine';

export interface LandmarkPoint extends Point3D {}

export type AlphabetConvention = 'auto' | 'islrtc_bimanual' | 'single_hand_chart';

export interface InferenceResult {
  sign: string;
  confidence: number;
  isDynamic: boolean;
  handCount: number;
  convention: 'ISLRTC (2-Hand)' | 'Single-Hand (Chart)';
  allProbabilities: Record<string, number>;
  tracks: PersistentHandTrack[];
  qualityReport: TrackingQualityReport;
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

  // Temporal buffer for dynamic sign tracking ('J', 'Z')
  private trajectoryBuffer: TrajectoryPoint[] = [];

  private dist3D(p1: Point3D, p2: Point3D): number {
    const dx = p1.x - p2.x;
    const dy = p1.y - p2.y;
    const dz = (p1.z ?? 0) - (p2.z ?? 0);
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  private cosineAngle(v1: [number, number], v2: [number, number]): number {
    const dot = v1[0] * v2[0] + v1[1] * v2[1];
    const mag1 = Math.sqrt(v1[0] * v1[0] + v1[1] * v1[1]) || 1e-6;
    const mag2 = Math.sqrt(v2[0] * v2[0] + v2[1] * v2[1]) || 1e-6;
    return Math.max(-1, Math.min(1, dot / (mag1 * mag2)));
  }

  /**
   * Extracts 42 continuous, scale-, translation-, and 3D rotation-invariant
   * biomechanical features from 21 MediaPipe hand landmarks.
   */
  public extractSingleHandFeatures(hand: Point3D[]): number[] {
    const feat = new Array<number>(42).fill(0);
    if (!hand || hand.length < 21) return feat;

    const wrist = hand[0];
    const midMCP = hand[9];
    const idxMCP = hand[5];
    const pkyMCP = hand[17];

    const scale = Math.max(0.035, this.dist3D(midMCP, wrist));

    // Longitudinal palm axis (wrist to middle MCP)
    const vecY: [number, number, number] = [
      (midMCP.x - wrist.x) / scale,
      (midMCP.y - wrist.y) / scale,
      ((midMCP.z ?? 0) - (wrist.z ?? 0)) / scale
    ];

    // Palm normal Z: (idxMCP - wrist) x (pkyMCP - wrist)
    const vIdx: [number, number, number] = [
      idxMCP.x - wrist.x,
      idxMCP.y - wrist.y,
      (idxMCP.z ?? 0) - (wrist.z ?? 0)
    ];
    const vPky: [number, number, number] = [
      pkyMCP.x - wrist.x,
      pkyMCP.y - wrist.y,
      (pkyMCP.z ?? 0) - (wrist.z ?? 0)
    ];

    const normX = vIdx[1] * vPky[2] - vIdx[2] * vPky[1];
    const normY = vIdx[2] * vPky[0] - vIdx[0] * vPky[2];
    const normZ = vIdx[0] * vPky[1] - vIdx[1] * vPky[0];
    const normMag = Math.sqrt(normX * normX + normY * normY + normZ * normZ);
    const vecZ: [number, number, number] = normMag > 1e-6
      ? [normX / normMag, normY / normMag, normZ / normMag]
      : [0, 0, 1];

    // Lateral axis X: vecY x vecZ
    const crossX_x = vecY[1] * vecZ[2] - vecY[2] * vecZ[1];
    const crossX_y = vecY[2] * vecZ[0] - vecY[0] * vecZ[2];
    const crossX_z = vecY[0] * vecZ[1] - vecY[1] * vecZ[0];
    const crossXMag = Math.sqrt(crossX_x * crossX_x + crossX_y * crossX_y + crossX_z * crossX_z);
    const vecX: [number, number, number] = crossXMag > 1e-6
      ? [crossX_x / crossXMag, crossX_y / crossXMag, crossX_z / crossXMag]
      : [1, 0, 0];

    // Transform landmarks to local canonical coordinates
    const localPts: Array<[number, number, number]> = new Array(21);
    for (let i = 0; i < 21; i++) {
      const rx = (hand[i].x - wrist.x) / scale;
      const ry = (hand[i].y - wrist.y) / scale;
      const rz = ((hand[i].z ?? 0) - (wrist.z ?? 0)) / scale;
      localPts[i] = [
        rx * vecX[0] + ry * vecX[1] + rz * vecX[2],
        rx * vecY[0] + ry * vecY[1] + rz * vecY[2],
        rx * vecZ[0] + ry * vecZ[1] + rz * vecZ[2]
      ];
    }

    const tips = [4, 8, 12, 16, 20];
    const mcps = [2, 5, 9, 13, 17];

    const palmCenterX = (hand[0].x + hand[5].x + hand[9].x + hand[13].x + hand[17].x) / 5;
    const palmCenterY = (hand[0].y + hand[5].y + hand[9].y + hand[13].y + hand[17].y) / 5;
    const palmCenterZ = ((hand[0].z ?? 0) + (hand[5].z ?? 0) + (hand[9].z ?? 0) + (hand[13].z ?? 0) + (hand[17].z ?? 0)) / 5;
    const palmCenter: Point3D = { x: palmCenterX, y: palmCenterY, z: palmCenterZ };

    // 0-4: 5 Finger Extensions
    for (let i = 0; i < 5; i++) {
      const tipDist = this.dist3D(hand[tips[i]], wrist);
      const mcpDist = this.dist3D(hand[mcps[i]], wrist);
      feat[i] = (tipDist - mcpDist) / scale;
    }

    // 5-9: 5 Fingertip to Palm Center Distances
    for (let i = 0; i < 5; i++) {
      feat[5 + i] = this.dist3D(hand[tips[i]], palmCenter) / scale;
    }

    // 10-14: 5 Finger Curls
    for (let i = 0; i < 5; i++) {
      feat[10 + i] = this.dist3D(hand[tips[i]], hand[mcps[i]]) / scale;
    }

    const cosAngle3D = (u: [number, number, number], v: [number, number, number]): number => {
      const dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
      const magU = Math.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
      const magV = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
      if (magU < 1e-6 || magV < 1e-6) return 0;
      return Math.max(-1, Math.min(1, dot / (magU * magV)));
    };

    const vThumb: [number, number, number] = [hand[4].x - hand[2].x, hand[4].y - hand[2].y, (hand[4].z ?? 0) - (hand[2].z ?? 0)];
    const vIndex: [number, number, number] = [hand[8].x - hand[5].x, hand[8].y - hand[5].y, (hand[8].z ?? 0) - (hand[5].z ?? 0)];
    const vMid:   [number, number, number] = [hand[12].x - hand[9].x, hand[12].y - hand[9].y, (hand[12].z ?? 0) - (hand[9].z ?? 0)];
    const vRing:  [number, number, number] = [hand[16].x - hand[13].x, hand[16].y - hand[13].y, (hand[16].z ?? 0) - (hand[13].z ?? 0)];
    const vPinky: [number, number, number] = [hand[20].x - hand[17].x, hand[20].y - hand[17].y, (hand[20].z ?? 0) - (hand[17].z ?? 0)];

    // 15-18: Inter-finger Spread Cosines
    feat[15] = cosAngle3D(vThumb, vIndex);
    feat[16] = cosAngle3D(vIndex, vMid);
    feat[17] = cosAngle3D(vMid, vRing);
    feat[18] = cosAngle3D(vRing, vPinky);

    // 19-23: Inter-fingertip Distances
    feat[19] = this.dist3D(hand[4], hand[8]) / scale;
    feat[20] = this.dist3D(hand[4], hand[12]) / scale;
    feat[21] = this.dist3D(hand[4], hand[20]) / scale;
    feat[22] = this.dist3D(hand[8], hand[12]) / scale;
    feat[23] = this.dist3D(hand[12], hand[16]) / scale;

    // 24-28: Canonical 3D Thumb Coordinates & Opposition
    feat[24] = localPts[4][0];
    feat[25] = localPts[4][1];
    feat[26] = localPts[4][2];
    feat[27] = this.dist3D(hand[4], hand[5]) / scale;
    feat[28] = this.dist3D(hand[4], hand[13]) / scale;

    // 29-31: Relative Finger Crossing Metrics
    const deltaXTips = localPts[8][0] - localPts[12][0];
    const deltaXMcps = localPts[5][0] - localPts[9][0];
    feat[29] = deltaXTips / (Math.abs(deltaXMcps) > 1e-4 ? deltaXMcps : 1);
    feat[30] = localPts[8][2] - localPts[12][2];
    const cross2D = vIndex[0] * vMid[1] - vIndex[1] * vMid[0];
    feat[31] = Math.abs(cross2D) > 1e-5 ? Math.sign(cross2D) : 0;

    // 32-34: Canonical Index Direction Vectors
    const idxDirX = localPts[8][0] - localPts[5][0];
    const idxDirY = localPts[8][1] - localPts[5][1];
    const idxDirZ = localPts[8][2] - localPts[5][2];
    const idxMag = Math.sqrt(idxDirX * idxDirX + idxDirY * idxDirY + idxDirZ * idxDirZ);
    if (idxMag > 1e-6) {
      feat[32] = idxDirX / idxMag;
      feat[33] = idxDirY / idxMag;
      feat[34] = idxDirZ / idxMag;
    }

    // 35-37: PIP Curl Angles & Fist Compactness
    const vIdxPip1: [number, number, number] = [hand[5].x - hand[6].x, hand[5].y - hand[6].y, (hand[5].z ?? 0) - (hand[6].z ?? 0)];
    const vIdxPip2: [number, number, number] = [hand[8].x - hand[6].x, hand[8].y - hand[6].y, (hand[8].z ?? 0) - (hand[6].z ?? 0)];
    feat[35] = cosAngle3D(vIdxPip1, vIdxPip2);

    const vMidPip1: [number, number, number] = [hand[9].x - hand[10].x, hand[9].y - hand[10].y, (hand[9].z ?? 0) - (hand[10].z ?? 0)];
    const vMidPip2: [number, number, number] = [hand[12].x - hand[10].x, hand[12].y - hand[10].y, (hand[12].z ?? 0) - (hand[10].z ?? 0)];
    feat[36] = cosAngle3D(vMidPip1, vMidPip2);

    let sumTipDist = 0;
    for (let i = 0; i < 5; i++) {
      sumTipDist += this.dist3D(hand[tips[i]], hand[9]);
    }
    feat[37] = sumTipDist / (5 * scale);

    // 38-41: Structural Invariant Ratios
    feat[38] = this.dist3D(hand[8], hand[12]) / Math.max(0.01, this.dist3D(hand[5], hand[9]));
    feat[39] = this.dist3D(hand[4], hand[20]) / scale;
    feat[40] = cosAngle3D(vIndex, vMid);
    feat[41] = this.dist3D(hand[4], hand[8]) / scale;

    return feat;
  }

  /**
   * Extracts 44 invariant features for dual-hand ISLRTC manual alphabet.
   */
  public extractDualHandFeatures(h1: Point3D[], h2: Point3D[] | null): number[] {
    const feat = new Array<number>(44).fill(0);
    if (!h1 || h1.length < 21) return feat;

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

    if (!h2 || h2.length < 21) {
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

    const h2Cleft: Point3D = {
      x: (h2[2].x + h2[5].x) / 2,
      y: (h2[2].y + h2[5].y) / 2,
      z: ((h2[2].z ?? 0) + (h2[5].z ?? 0)) / 2
    };
    feat[34] = this.dist3D(h1[8], h2Cleft) / s2; // Y
    feat[35] = this.dist3D(h1[8], h2[6]) / s2; // K

    const h1LoopMid: Point3D = {
      x: (h1[4].x + h1[8].x) / 2,
      y: (h1[4].y + h1[8].y) / 2,
      z: ((h1[4].z ?? 0) + (h1[8].z ?? 0)) / 2
    };
    const h2LoopMid: Point3D = {
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
    tracks: PersistentHandTrack[],
    now: number
  ): { sign: string; confidence: number } | null {
    if (!tracks || tracks.length === 0) {
      this.trajectoryBuffer = [];
      return null;
    }

    const tip = tracks[0].smoothedLandmarks[8];
    this.trajectoryBuffer.push({ x: tip.x, y: tip.y, time: now });

    while (this.trajectoryBuffer.length > 0 && now - this.trajectoryBuffer[0].time > 1200) {
      this.trajectoryBuffer.shift();
    }
    if (this.trajectoryBuffer.length < 12) return null;

    // Check letter 'J': Starts high, moves down, hooks left/up
    const firstPoint = this.trajectoryBuffer[0];
    const midPoint = this.trajectoryBuffer[Math.floor(this.trajectoryBuffer.length / 2)];
    const lastPoint = this.trajectoryBuffer[this.trajectoryBuffer.length - 1];

    const deltaYDown = midPoint.y - firstPoint.y;
    const deltaXHook = lastPoint.x - midPoint.x;
    const deltaYUp = midPoint.y - lastPoint.y;

    if (deltaYDown > 0.08 && deltaXHook < -0.04 && deltaYUp > 0.02) {
      return { sign: 'J', confidence: 0.92 };
    }

    // Check letter 'Z': 3-stroke zigzag
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
   * End-to-end multi-stage pipeline:
   * 1. Persistent Hand Identity Association
   * 2. Adaptive One Euro Landmark Filtering
   * 3. Optical Tracking Quality Assessment
   * 4. Kinematic Geometry & Classifier Inference
   */
  public predict(
    rawLandmarks: Point3D[][],
    handednessData?: Array<Array<{ categoryName: string; score: number }>>,
    convention: AlphabetConvention = 'auto',
    videoWidth: number = 640,
    videoHeight: number = 480,
    timestamp: number = performance.now()
  ): InferenceResult {
    // 1. Stage C & D: Persistent Hand Tracking + One Euro Filtering
    const tracks = persistentHandTracker.update(rawLandmarks, handednessData, timestamp);

    // 2. Stage E: Tracking Quality Engine Evaluation
    const qualityReport = trackingQualityEngine.evaluate(tracks, videoWidth, videoHeight);

    if (tracks.length === 0) {
      return {
        sign: 'NEUTRAL',
        confidence: 0,
        isDynamic: false,
        handCount: 0,
        convention: 'ISLRTC (2-Hand)',
        allProbabilities: {},
        tracks: [],
        qualityReport
      };
    }

    // 3. Dynamic Sign Check ('J', 'Z')
    const dynamicMatch = this.checkDynamicSigns(tracks, timestamp);
    if (dynamicMatch && dynamicMatch.confidence >= 0.85) {
      return {
        sign: dynamicMatch.sign,
        confidence: dynamicMatch.confidence,
        isDynamic: true,
        handCount: tracks.length,
        convention: 'ISLRTC (2-Hand)',
        allProbabilities: { [dynamicMatch.sign]: dynamicMatch.confidence },
        tracks,
        qualityReport
      };
    }

    // 4. Determine Active Convention
    const useSingleHand =
      convention === 'single_hand_chart' || (convention === 'auto' && tracks.length === 1);

    if (useSingleHand) {
      const dominantTrack = tracks[0];

      // Phase 6: Strict tracking-confidence integration.
      // Suppress classification during coasting or when tracking quality is compromised.
      if (dominantTrack.isCoasting || !qualityReport.isReliableForInference) {
        return {
          sign: 'NEUTRAL',
          confidence: 0,
          isDynamic: false,
          handCount: 1,
          convention: 'Single-Hand (Chart)',
          allProbabilities: { NEUTRAL: 1.0 },
          tracks,
          qualityReport
        };
      }

      const features = this.extractSingleHandFeatures(dominantTrack.smoothedLandmarks);
      const res = this.runMLP(this.singleHandWeights, this.singleHandClasses, features);
      return {
        sign: res.sign,
        confidence: res.confidence,
        isDynamic: false,
        handCount: 1,
        convention: 'Single-Hand (Chart)',
        allProbabilities: res.allProbs,
        tracks,
        qualityReport
      };
    } else {
      const dominantTrack = tracks.find((t) => t.role === 'Dominant') || tracks[0];
      const baseTrack = tracks.find((t) => t.role === 'Base') || (tracks.length > 1 ? tracks[1] : null);

      // Phase 6: Strict tracking-confidence integration.
      if (dominantTrack.isCoasting || !qualityReport.isReliableForInference) {
        return {
          sign: 'NEUTRAL',
          confidence: 0,
          isDynamic: false,
          handCount: tracks.length,
          convention: 'ISLRTC (2-Hand)',
          allProbabilities: { NEUTRAL: 1.0 },
          tracks,
          qualityReport
        };
      }

      const features = this.extractDualHandFeatures(
        dominantTrack.smoothedLandmarks,
        baseTrack ? baseTrack.smoothedLandmarks : null
      );
      const res = this.runMLP(this.dualHandWeights, this.dualHandClasses, features);
      return {
        sign: res.sign,
        confidence: res.confidence,
        isDynamic: false,
        handCount: tracks.length,
        convention: 'ISLRTC (2-Hand)',
        allProbabilities: res.allProbs,
        tracks,
        qualityReport
      };
    }
  }

  public reset() {
    persistentHandTracker.reset();
    trackingQualityEngine.reset();
    this.trajectoryBuffer = [];
  }
}

export const alphabetInferenceEngine = new AlphabetInferenceEngine();
