/**
 * SIGNBRIDGE Coordinate AI — Client-Side Coordinate Inference Engine
 * 
 * Provides real-time, browser-native inference for 26-letter ASL alphabet recognition
 * using 21 MediaPipe hand landmark coordinates:
 * 1. Vectorized 94-dimensional invariant coordinate feature extraction.
 * 2. Deep neural network forward pass (CoordinateMLP) with LayerNorm and temperature scaling.
 * 3. Top-3 candidate probability ranking.
 * 4. Temporal trajectory integration for dynamic letters J and Z.
 * 5. Low-confidence rejection and tracking loss detection.
 */

import coordinateModel from './coordinate_ai_model.json';
import { Point3D } from './oneEuroFilter';
import { dynamicTrajectoryEngine, DynamicTrajectoryResult } from './dynamicTrajectoryEngine';

export interface CoordinateCandidate {
  letter: string;
  confidence: number;
}

export interface CoordinateRecognitionResult {
  predictedLetter: string;
  bestCandidate: string | null;
  confidence: number;
  isAccepted: boolean;
  isDynamic: boolean;
  topCandidates: CoordinateCandidate[];
  latencyMs: number;
  status: 'TRACKING' | 'REJECTED' | 'NO_HAND';
  probabilities: Record<string, number>;
}

// MediaPipe Landmark Indices
const WRIST = 0;
const THUMB_CMC = 1, THUMB_MCP = 2, THUMB_IP = 3, THUMB_TIP = 4;
const INDEX_MCP = 5, INDEX_PIP = 6, INDEX_DIP = 7, INDEX_TIP = 8;
const MID_MCP = 9, MID_PIP = 10, MID_DIP = 11, MID_TIP = 12;
const RING_MCP = 13, RING_PIP = 14, RING_DIP = 15, RING_TIP = 16;
const PINKY_MCP = 17, PINKY_PIP = 18, PINKY_DIP = 19, PINKY_TIP = 20;

const FINGERTIPS = [THUMB_TIP, INDEX_TIP, MID_TIP, RING_TIP, PINKY_TIP];

const PAIRWISE_FINGERTIPS: Array<[number, number]> = [
  [THUMB_TIP, INDEX_TIP],
  [THUMB_TIP, MID_TIP],
  [THUMB_TIP, RING_TIP],
  [THUMB_TIP, PINKY_TIP],
  [INDEX_TIP, MID_TIP],
  [INDEX_TIP, RING_TIP],
  [INDEX_TIP, PINKY_TIP],
  [MID_TIP, RING_TIP],
  [MID_TIP, PINKY_TIP],
  [RING_TIP, PINKY_TIP],
];

const JOINT_ANGLE_TRIPLETS: Array<[number, number, number]> = [
  [THUMB_CMC, THUMB_MCP, THUMB_IP],
  [THUMB_MCP, THUMB_IP, THUMB_TIP],
  [INDEX_MCP, INDEX_PIP, INDEX_DIP],
  [INDEX_PIP, INDEX_DIP, INDEX_TIP],
  [MID_MCP, MID_PIP, MID_DIP],
  [MID_PIP, MID_DIP, MID_TIP],
  [RING_MCP, RING_PIP, RING_DIP],
  [RING_PIP, RING_DIP, RING_TIP],
  [PINKY_MCP, PINKY_PIP, PINKY_DIP],
  [PINKY_PIP, PINKY_DIP, PINKY_TIP],
];

export class CoordinateAiEngine {
  private classes: string[] = coordinateModel.classes;
  private temperature: number = (coordinateModel as any).temperature || 1.0;
  private weights = (coordinateModel as any).weights;
  private confidenceThreshold: number = 0.70;

  private dist3D(a: Point3D, b: Point3D): number {
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    const dz = (a.z ?? 0) - (b.z ?? 0);
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  private compute3DAngle(a: Point3D, b: Point3D, c: Point3D): number {
    const v1x = a.x - b.x;
    const v1y = a.y - b.y;
    const v1z = (a.z ?? 0) - (b.z ?? 0);

    const v2x = c.x - b.x;
    const v2y = c.y - b.y;
    const v2z = (c.z ?? 0) - (b.z ?? 0);

    const norm1 = Math.sqrt(v1x * v1x + v1y * v1y + v1z * v1z);
    const norm2 = Math.sqrt(v2x * v2x + v2y * v2y + v2z * v2z);
    if (norm1 < 1e-7 || norm2 < 1e-7) return 0.0;

    const dot = v1x * v2x + v1y * v2y + v1z * v2z;
    const cos = Math.max(-1.0, Math.min(1.0, dot / (norm1 * norm2)));
    return Math.acos(cos);
  }

  /**
   * Extracts the full 94-dimensional invariant coordinate feature vector:
   * 63 normalized coordinates + 10 joint angles + 5 curl ratios + 10 pairwise distances + 6 palm vectors.
   */
  public extractCoordinateFeatures(hand: Point3D[]): number[] {
    const feat = new Array<number>(94).fill(0);
    if (!hand || hand.length < 21) return feat;

    const wrist = hand[WRIST];
    const midMCP = hand[MID_MCP];
    const scale = Math.max(0.035, this.dist3D(midMCP, wrist));

    // 1. 63 Raw Normalized Coordinates
    let idx = 0;
    for (let i = 0; i < 21; i++) {
      feat[idx++] = (hand[i].x - wrist.x) / scale;
      feat[idx++] = (hand[i].y - wrist.y) / scale;
      feat[idx++] = ((hand[i].z ?? 0) - (wrist.z ?? 0)) / scale;
    }

    // 2. 10 Articulation Angles
    for (const [a, b, c] of JOINT_ANGLE_TRIPLETS) {
      feat[idx++] = this.compute3DAngle(hand[a], hand[b], hand[c]);
    }

    // 3. 5 Fingertip Curl Ratios
    for (const tip of FINGERTIPS) {
      feat[idx++] = this.dist3D(hand[tip], wrist) / scale;
    }

    // 4. 10 Pairwise Fingertip Distances
    for (const [i, j] of PAIRWISE_FINGERTIPS) {
      feat[idx++] = this.dist3D(hand[i], hand[j]) / scale;
    }

    // 5. 6 Palm Orientation Vectors
    const idxMCP = hand[INDEX_MCP];
    const pkyMCP = hand[PINKY_MCP];

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
    if (normMag > 1e-7) {
      feat[idx++] = normX / normMag;
      feat[idx++] = normY / normMag;
      feat[idx++] = normZ / normMag;
    } else {
      feat[idx++] = 0; feat[idx++] = 0; feat[idx++] = 1;
    }

    const midVec: [number, number, number] = [
      midMCP.x - wrist.x,
      midMCP.y - wrist.y,
      (midMCP.z ?? 0) - (wrist.z ?? 0)
    ];
    const midMag = Math.sqrt(midVec[0] * midVec[0] + midVec[1] * midVec[1] + midVec[2] * midVec[2]);
    if (midMag > 1e-7) {
      feat[idx++] = midVec[0] / midMag;
      feat[idx++] = midVec[1] / midMag;
      feat[idx++] = midVec[2] / midMag;
    } else {
      feat[idx++] = 0; feat[idx++] = 1; feat[idx++] = 0;
    }

    return feat;
  }

  private layerNorm(vec: number[], gamma: number[], beta: number[], eps: number = 1e-5): number[] {
    const n = vec.length;
    let sum = 0;
    for (let i = 0; i < n; i++) sum += vec[i];
    const mean = sum / n;

    let varSum = 0;
    for (let i = 0; i < n; i++) {
      const diff = vec[i] - mean;
      varSum += diff * diff;
    }
    const std = Math.sqrt(varSum / n + eps);

    const out = new Array<number>(n);
    for (let i = 0; i < n; i++) {
      out[i] = ((vec[i] - mean) / std) * gamma[i] + beta[i];
    }
    return out;
  }

  /**
   * Forward pass through CoordinateMLP:
   * 94 -> 256 (LN + ReLU) -> 128 (LN + ReLU) -> 64 (LN + ReLU) -> 24 / T -> Softmax
   */
  public forward(features: number[]): { bestLetter: string; confidence: number; top3: CoordinateCandidate[]; probs: Record<string, number> } {
    const w1 = this.weights.w1;
    const b1 = this.weights.b1;
    const g1 = this.weights.ln1_gamma;
    const bet1 = this.weights.ln1_beta;

    // Layer 1
    const h1 = new Array<number>(w1.length);
    for (let i = 0; i < w1.length; i++) {
      let sum = b1[i];
      const row = w1[i];
      for (let j = 0; j < row.length; j++) {
        sum += row[j] * features[j];
      }
      h1[i] = sum;
    }
    const a1 = this.layerNorm(h1, g1, bet1);
    for (let i = 0; i < a1.length; i++) {
      if (a1[i] < 0) a1[i] = 0;
    }

    // Layer 2
    const w2 = this.weights.w2;
    const b2 = this.weights.b2;
    const g2 = this.weights.ln2_gamma;
    const bet2 = this.weights.ln2_beta;

    const h2 = new Array<number>(w2.length);
    for (let i = 0; i < w2.length; i++) {
      let sum = b2[i];
      const row = w2[i];
      for (let j = 0; j < row.length; j++) {
        sum += row[j] * a1[j];
      }
      h2[i] = sum;
    }
    const a2 = this.layerNorm(h2, g2, bet2);
    for (let i = 0; i < a2.length; i++) {
      if (a2[i] < 0) a2[i] = 0;
    }

    // Layer 3
    const w3 = this.weights.w3;
    const b3 = this.weights.b3;
    const g3 = this.weights.ln3_gamma;
    const bet3 = this.weights.ln3_beta;

    const h3 = new Array<number>(w3.length);
    for (let i = 0; i < w3.length; i++) {
      let sum = b3[i];
      const row = w3[i];
      for (let j = 0; j < row.length; j++) {
        sum += row[j] * a2[j];
      }
      h3[i] = sum;
    }
    const a3 = this.layerNorm(h3, g3, bet3);
    for (let i = 0; i < a3.length; i++) {
      if (a3[i] < 0) a3[i] = 0;
    }

    // Layer 4 (Output logits)
    const w4 = this.weights.w4;
    const b4 = this.weights.b4;
    const temp = Math.max(0.1, this.temperature);

    const logits = new Array<number>(w4.length);
    let maxLogit = -Infinity;
    for (let i = 0; i < w4.length; i++) {
      let sum = b4[i];
      const row = w4[i];
      for (let j = 0; j < row.length; j++) {
        sum += row[j] * a3[j];
      }
      const scaled = sum / temp;
      logits[i] = scaled;
      if (scaled > maxLogit) maxLogit = scaled;
    }

    // Softmax
    let sumExp = 0;
    const probs = new Array<number>(logits.length);
    for (let i = 0; i < logits.length; i++) {
      probs[i] = Math.exp(logits[i] - maxLogit);
      sumExp += probs[i];
    }

    const classProbPairs: Array<{ letter: string; confidence: number }> = [];
    const probMap: Record<string, number> = {};

    for (let i = 0; i < logits.length; i++) {
      const p = probs[i] / sumExp;
      const letter = this.classes[i] || `CLASS_${i}`;
      probMap[letter] = p;
      classProbPairs.push({ letter, confidence: p });
    }

    classProbPairs.sort((a, b) => b.confidence - a.confidence);
    const top3 = classProbPairs.slice(0, 3);

    return {
      bestLetter: top3[0].letter,
      confidence: top3[0].confidence,
      top3,
      probs: probMap
    };
  }

  /**
   * Main inference entrypoint: Evaluates hand landmarks, checks dynamic trajectories,
   * enforces confidence thresholds, and returns full diagnostics.
   */
  public recognizeHand(
    hand: Point3D[] | null | undefined,
    timestampMs: number = Date.now()
  ): CoordinateRecognitionResult {
    const tStart = performance.now();

    if (!hand || hand.length < 21) {
      return {
        predictedLetter: 'NO_HAND',
        bestCandidate: null,
        confidence: 0.0,
        isAccepted: false,
        isDynamic: false,
        topCandidates: [],
        latencyMs: performance.now() - tStart,
        status: 'NO_HAND',
        probabilities: {}
      };
    }

    // 1. Check for dynamic signs J and Z via temporal trajectory engine
    const palmScale = Math.max(0.035, this.dist3D(hand[MID_MCP], hand[WRIST]));
    const dynamicRes: DynamicTrajectoryResult | null = dynamicTrajectoryEngine.update(
      hand,
      palmScale,
      timestampMs
    );

    if (dynamicRes && dynamicRes.sign !== 'NEUTRAL' && dynamicRes.confidence >= 0.85) {
      const latencyMs = performance.now() - tStart;
      return {
        predictedLetter: dynamicRes.sign,
        bestCandidate: dynamicRes.sign,
        confidence: dynamicRes.confidence,
        isAccepted: true,
        isDynamic: true,
        topCandidates: [
          { letter: dynamicRes.sign, confidence: dynamicRes.confidence },
          { letter: 'NEUTRAL', confidence: 1.0 - dynamicRes.confidence }
        ],
        latencyMs,
        status: 'TRACKING',
        probabilities: dynamicRes.allProbabilities
      };
    }

    // 2. Static Coordinate Recognition
    const features = this.extractCoordinateFeatures(hand);
    const { bestLetter, confidence, top3, probs } = this.forward(features);
    const latencyMs = performance.now() - tStart;

    const isAccepted = confidence >= this.confidenceThreshold;

    return {
      predictedLetter: isAccepted ? bestLetter : 'REJECTED',
      bestCandidate: bestLetter,
      confidence,
      isAccepted,
      isDynamic: false,
      topCandidates: top3,
      latencyMs,
      status: isAccepted ? 'TRACKING' : 'REJECTED',
      probabilities: probs
    };
  }
}

export const coordinateAiEngine = new CoordinateAiEngine();
