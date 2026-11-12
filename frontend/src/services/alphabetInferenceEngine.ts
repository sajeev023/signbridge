import modelData from './isl_alphabet_model.json';

export interface LandmarkPoint {
  x: number;
  y: number;
  z: number;
}

export interface InferenceResult {
  sign: string;
  confidence: number;
  isDynamic: boolean;
  handCount: number;
  allProbabilities: Record<string, number>;
}

export interface TrajectoryPoint {
  x: number;
  y: number;
  time: number;
}

class AlphabetInferenceEngine {
  private weights: {
    w1: number[][];
    b1: number[];
    w2: number[][];
    b2: number[];
    w3: number[][];
    b3: number[];
  };
  private classes: string[];
  private numFeatures: number;

  // Temporal buffer for dynamic sign tracking ('J', 'Z')
  private trajectoryBuffer: TrajectoryPoint[] = [];
  private readonly MAX_TRAJECTORY_POINTS = 30;

  constructor() {
    this.weights = modelData.weights;
    this.classes = modelData.classes;
    this.numFeatures = modelData.num_features;
  }

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
   * Extracts scale and position invariant 44-dimensional geometric features
   * matching the trained SignBridge ISL Alphabet model specification.
   */
  public extractFeatures(
    landmarks: LandmarkPoint[][],
    handedness?: Array<Array<{ categoryName: string; score: number }>>
  ): number[] {
    const feat = new Array<number>(this.numFeatures).fill(0);
    if (!landmarks || landmarks.length === 0) return feat;

    // Determine Hand 1 (Dominant) and Hand 2 (Base)
    let h1 = landmarks[0];
    let h2: LandmarkPoint[] | null = landmarks.length > 1 ? landmarks[1] : null;

    if (landmarks.length > 1 && handedness && handedness.length >= 2) {
      const h0Name = handedness[0]?.[0]?.categoryName;
      // In mirrored selfie camera: MediaPipe 'Right' hand appears on right of viewer
      // Default right hand as dominant pointer if both exist
      if (h0Name === 'Left' && handedness[1]?.[0]?.categoryName === 'Right') {
        h1 = landmarks[1];
        h2 = landmarks[0];
      }
    } else if (landmarks.length > 1) {
      // Heuristic: The hand with higher index-finger isolation is the pointer H1
      const iso0 = this.dist2D(landmarks[0][8], landmarks[0][0]) / (this.dist2D(landmarks[0][12], landmarks[0][0]) || 1);
      const iso1 = this.dist2D(landmarks[1][8], landmarks[1][0]) / (this.dist2D(landmarks[1][12], landmarks[1][0]) || 1);
      if (iso1 > iso0) {
        h1 = landmarks[1];
        h2 = landmarks[0];
      }
    }

    // Hand 1 Palm scale (wrist to middle MCP)
    const s1 = Math.max(0.04, this.dist3D(h1[0], h1[9]));

    // H1 Finger Extensions (tip distance from wrist vs pip distance from wrist)
    const h1ThumbExt = (this.dist3D(h1[4], h1[0]) - this.dist3D(h1[2], h1[0])) / s1;
    const h1IndexExt = (this.dist3D(h1[8], h1[0]) - this.dist3D(h1[6], h1[0])) / s1;
    const h1MiddleExt = (this.dist3D(h1[12], h1[0]) - this.dist3D(h1[10], h1[0])) / s1;
    const h1RingExt = (this.dist3D(h1[16], h1[0]) - this.dist3D(h1[14], h1[0])) / s1;
    const h1PinkyExt = (this.dist3D(h1[20], h1[0]) - this.dist3D(h1[18], h1[0])) / s1;

    // H1 Curls (cosine between MCP-PIP and PIP-TIP)
    const h1ThumbCurl = 0.5;
    const h1IndexCurl = 1.0 - Math.max(0, this.dist3D(h1[8], h1[5]) / s1);
    const h1MiddleCurl = 1.0 - Math.max(0, this.dist3D(h1[12], h1[9]) / s1);
    const h1RingCurl = 1.0 - Math.max(0, this.dist3D(h1[16], h1[13]) / s1);
    const h1PinkyCurl = 1.0 - Math.max(0, this.dist3D(h1[20], h1[17]) / s1);

    // H1 Spreads
    const h1SpreadThumbIdx = this.dist3D(h1[4], h1[8]) / s1;
    const h1SpreadIdxMid = this.dist3D(h1[8], h1[12]) / s1;
    const h1SpreadMidRing = this.dist3D(h1[12], h1[16]) / s1;
    const h1SpreadRingPinky = this.dist3D(h1[16], h1[20]) / s1;

    // H1 L-angle (cosine between thumb vector and index vector)
    const thumbVec: [number, number] = [h1[4].x - h1[2].x, h1[4].y - h1[2].y];
    const indexVec: [number, number] = [h1[8].x - h1[5].x, h1[8].y - h1[5].y];
    const dotThumbIndex = Math.abs(this.cosineAngle(thumbVec, indexVec));
    // When thumb and index are perpendicular (L sign), dotThumbIndex is near 0
    const h1LAngle = h1SpreadThumbIdx > 0.6 && h1IndexExt > 0.4 ? 1.0 - dotThumbIndex : 0.0;

    // H1 C-curve curvature
    const isCurved = h1SpreadThumbIdx > 0.3 && h1IndexCurl > 0.2 && h1MiddleCurl > 0.2;
    const h1CCurve = isCurved ? 0.9 : 0.1;

    // H1 loop closure (thumb tip to index tip)
    const h1LoopClosure = h1SpreadThumbIdx;

    // Populate Hand 1 features
    feat[0] = Math.max(-1, Math.min(2, h1ThumbExt));
    feat[1] = Math.max(-1, Math.min(2, h1IndexExt));
    feat[2] = Math.max(-1, Math.min(2, h1MiddleExt));
    feat[3] = Math.max(-1, Math.min(2, h1RingExt));
    feat[4] = Math.max(-1, Math.min(2, h1PinkyExt));
    feat[5] = h1ThumbCurl;
    feat[6] = Math.max(0, Math.min(1, h1IndexCurl));
    feat[7] = Math.max(0, Math.min(1, h1MiddleCurl));
    feat[8] = Math.max(0, Math.min(1, h1RingCurl));
    feat[9] = Math.max(0, Math.min(1, h1PinkyCurl));
    feat[10] = Math.min(3, h1SpreadThumbIdx);
    feat[11] = Math.min(3, h1SpreadIdxMid);
    feat[12] = Math.min(3, h1SpreadMidRing);
    feat[13] = Math.min(3, h1SpreadRingPinky);
    feat[14] = h1LAngle;
    feat[15] = h1CCurve;
    feat[16] = h1LoopClosure;

    if (!h2) {
      feat[17] = 0.0; // H2 not present
      // Defaults for single hand
      feat[24] = 3.0; // High relative distance
      feat[27] = 3.0;
      feat[28] = 3.0;
      feat[29] = 3.0;
      feat[30] = 3.0;
      feat[31] = 3.0;
      return feat;
    }

    // Hand 2 is present
    feat[17] = 1.0;
    const s2 = Math.max(0.04, this.dist3D(h2[0], h2[9]));

    feat[18] = Math.max(-1, Math.min(2, (this.dist3D(h2[4], h2[0]) - this.dist3D(h2[2], h2[0])) / s2));
    feat[19] = Math.max(-1, Math.min(2, (this.dist3D(h2[8], h2[0]) - this.dist3D(h2[6], h2[0])) / s2));
    feat[20] = Math.max(-1, Math.min(2, (this.dist3D(h2[12], h2[0]) - this.dist3D(h2[10], h2[0])) / s2));
    feat[21] = Math.max(-1, Math.min(2, (this.dist3D(h2[16], h2[0]) - this.dist3D(h2[14], h2[0])) / s2));
    feat[22] = Math.max(-1, Math.min(2, (this.dist3D(h2[20], h2[0]) - this.dist3D(h2[18], h2[0])) / s2));
    feat[23] = this.dist3D(h2[4], h2[8]) / s2; // H2 loop closure

    // Relative wrist displacement
    feat[24] = (h1[0].x - h2[0].x) / s2;
    feat[25] = (h1[0].y - h2[0].y) / s2;
    feat[26] = ((h1[0].z ?? 0) - (h2[0].z ?? 0)) / s2;

    // Contact points: H1 Index Tip (landmark 8) to H2 landmarks
    feat[27] = this.dist3D(h1[8], h2[4]) / s2;  // To Thumb Tip ('A')
    feat[28] = this.dist3D(h1[8], h2[8]) / s2;  // To Index Tip ('E')
    feat[29] = this.dist3D(h1[8], h2[12]) / s2; // To Middle Tip ('I')
    feat[30] = this.dist3D(h1[8], h2[16]) / s2; // To Ring Tip ('O')
    feat[31] = this.dist3D(h1[8], h2[20]) / s2; // To Pinky Tip ('U')
    feat[32] = this.dist3D(h1[8], h2[5]) / s2;  // To Index Base ('T')
    feat[33] = this.dist3D(h1[8], h2[9]) / s2;  // To Palm Center ('M', 'N', 'V', 'H')

    // Midpoint of H2 thumb and index (cleft for 'Y')
    const h2Cleft: LandmarkPoint = {
      x: (h2[2].x + h2[5].x) / 2,
      y: (h2[2].y + h2[5].y) / 2,
      z: ((h2[2].z ?? 0) + (h2[5].z ?? 0)) / 2
    };
    feat[34] = this.dist3D(h1[8], h2Cleft) / s2;
    feat[35] = this.dist3D(h1[8], h2[6]) / s2;  // To Index PIP ('K')

    // Loop & Structure Contacts
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
    feat[36] = this.dist3D(h1LoopMid, h2LoopMid) / s2; // ('B')
    feat[37] = this.dist3D(h1[4], h2[8]) / s2;         // ('D')
    feat[38] = this.dist3D(h1LoopMid, h2[8]) / s2;     // ('P')
    feat[39] = this.dist3D(h1[20], h2[20]) / s2;       // ('S')
    feat[40] = this.dist3D(h1[9], h2[9]) / s2;         // Knuckles ('G')
    feat[41] = this.dist3D(h1[12], h2[9]) / s2;        // Middle to Palm ('M', 'N')
    feat[42] = this.dist3D(h1[16], h2[9]) / s2;        // Ring to Palm ('M')
    feat[43] = this.dist3D(h1[12], h2[8]) / s2;        // Cross ('F', 'X')

    return feat;
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

    const tip = landmarks[0][8]; // Dominant index fingertip
    this.trajectoryBuffer.push({ x: tip.x, y: tip.y, time: now });

    // Clean buffer older than 1200ms
    while (this.trajectoryBuffer.length > 0 && now - this.trajectoryBuffer[0].time > 1200) {
      this.trajectoryBuffer.shift();
    }
    if (this.trajectoryBuffer.length < 12) return null;

    // Check letter 'J': Begins around I-position (touches middle finger), moves DOWN, then curves LEFT/UP
    const firstPoint = this.trajectoryBuffer[0];
    const midIdx = Math.floor(this.trajectoryBuffer.length / 2);
    const midPoint = this.trajectoryBuffer[midIdx];
    const lastPoint = this.trajectoryBuffer[this.trajectoryBuffer.length - 1];

    const deltaYDown = midPoint.y - firstPoint.y; // Positive in screen coords means moving down
    const deltaXHook = lastPoint.x - midPoint.x;
    const deltaYUp = midPoint.y - lastPoint.y;

    if (deltaYDown > 0.08 && deltaXHook < -0.04 && deltaYUp > 0.02) {
      return { sign: 'J', confidence: 0.92 };
    }

    // Check letter 'Z': 3-stroke zigzag (Right -> Down-Left -> Right)
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
    landmarks: LandmarkPoint[][],
    handedness?: Array<Array<{ categoryName: string; score: number }>>,
    timestamp: number = performance.now()
  ): InferenceResult {
    if (!landmarks || landmarks.length === 0) {
      return {
        sign: 'NEUTRAL',
        confidence: 0,
        isDynamic: false,
        handCount: 0,
        allProbabilities: {}
      };
    }

    // 1. Check for dynamic signs ('J', 'Z')
    const dynamicMatch = this.checkDynamicSigns(landmarks, timestamp);
    if (dynamicMatch && dynamicMatch.confidence >= 0.85) {
      return {
        sign: dynamicMatch.sign,
        confidence: dynamicMatch.confidence,
        isDynamic: true,
        handCount: landmarks.length,
        allProbabilities: { [dynamicMatch.sign]: dynamicMatch.confidence }
      };
    }

    // 2. Extract 44-D kinematic features
    const x = this.extractFeatures(landmarks, handedness);

    // 3. Layer 1: ReLU(W1 * x + b1)
    const z1 = new Array<number>(this.weights.w1.length);
    for (let i = 0; i < this.weights.w1.length; i++) {
      let sum = this.weights.b1[i];
      const row = this.weights.w1[i];
      for (let j = 0; j < row.length; j++) {
        sum += row[j] * x[j];
      }
      z1[i] = Math.max(0, sum);
    }

    // 4. Layer 2: ReLU(W2 * z1 + b2)
    const z2 = new Array<number>(this.weights.w2.length);
    for (let i = 0; i < this.weights.w2.length; i++) {
      let sum = this.weights.b2[i];
      const row = this.weights.w2[i];
      for (let j = 0; j < row.length; j++) {
        sum += row[j] * z1[j];
      }
      z2[i] = Math.max(0, sum);
    }

    // 5. Layer 3: Logits = W3 * z2 + b3
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

    // 6. Softmax
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
      const className = this.classes[i] || `Class_${i}`;
      allProbs[className] = probs[i];
      if (probs[i] > bestConf) {
        bestConf = probs[i];
        bestSign = className;
      }
    }

    return {
      sign: bestSign,
      confidence: bestConf,
      isDynamic: false,
      handCount: landmarks.length,
      allProbabilities: allProbs
    };
  }

  public getSupportedVocabulary(): string[] {
    return this.classes.filter(c => c !== 'NEUTRAL');
  }
}

export const alphabetInferenceEngine = new AlphabetInferenceEngine();
