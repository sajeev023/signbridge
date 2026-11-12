/**
 * Anatomical Hand Model & Biomechanical Invariants
 * Evaluates physiological plausibility, segment lengths, joint-level kinematics,
 * and provides joint-specific filtering parameters for 21-landmark hand skeletons.
 */

import { Point3D } from './oneEuroFilter';

export interface BoneSegment {
  name: string;
  startJoint: number;
  endJoint: number;
  nominalRatio: number; // Ratio relative to palm baseline (Wrist [0] -> Mid MCP [9])
  minTolerance: number;
  maxTolerance: number;
}

export interface FingerTelemetry {
  name: 'Thumb' | 'Index' | 'Middle' | 'Ring' | 'Pinky';
  joints: [number, number, number, number];
  tipJoint: number;
  extensionRatio: number; // 0 (curled) to 1 (extended)
  jitterVariance: number;
  stabilityScore: number; // 0 - 100
  isOccluded: boolean;
}

export interface AnatomicalIntegrityReport {
  isBiologicallyPlausible: boolean;
  integrityScore: number; // 0 - 100
  deviantBonesCount: number;
  maxBoneRatioDeviation: number;
  perFingerTelemetry: FingerTelemetry[];
  overallPalmScale: number;
}

export const ANATOMICAL_BONES: BoneSegment[] = [
  // Thumb chain
  { name: 'Thumb CMC', startJoint: 0, endJoint: 1, nominalRatio: 0.38, minTolerance: 0.20, maxTolerance: 0.65 },
  { name: 'Thumb MCP', startJoint: 1, endJoint: 2, nominalRatio: 0.40, minTolerance: 0.22, maxTolerance: 0.68 },
  { name: 'Thumb IP', startJoint: 2, endJoint: 3, nominalRatio: 0.34, minTolerance: 0.18, maxTolerance: 0.58 },
  { name: 'Thumb Tip', startJoint: 3, endJoint: 4, nominalRatio: 0.26, minTolerance: 0.14, maxTolerance: 0.48 },

  // Index chain
  { name: 'Index Metacarpal', startJoint: 0, endJoint: 5, nominalRatio: 0.94, minTolerance: 0.65, maxTolerance: 1.25 },
  { name: 'Index Proximal', startJoint: 5, endJoint: 6, nominalRatio: 0.48, minTolerance: 0.28, maxTolerance: 0.72 },
  { name: 'Index Middle', startJoint: 6, endJoint: 7, nominalRatio: 0.30, minTolerance: 0.16, maxTolerance: 0.48 },
  { name: 'Index Distal', startJoint: 7, endJoint: 8, nominalRatio: 0.24, minTolerance: 0.12, maxTolerance: 0.40 },

  // Middle chain
  { name: 'Middle Metacarpal', startJoint: 0, endJoint: 9, nominalRatio: 1.00, minTolerance: 0.95, maxTolerance: 1.05 }, // Baseline = 1.0
  { name: 'Middle Proximal', startJoint: 9, endJoint: 10, nominalRatio: 0.52, minTolerance: 0.30, maxTolerance: 0.76 },
  { name: 'Middle Middle', startJoint: 10, endJoint: 11, nominalRatio: 0.34, minTolerance: 0.18, maxTolerance: 0.52 },
  { name: 'Middle Distal', startJoint: 11, endJoint: 12, nominalRatio: 0.25, minTolerance: 0.13, maxTolerance: 0.42 },

  // Ring chain
  { name: 'Ring Metacarpal', startJoint: 0, endJoint: 13, nominalRatio: 0.92, minTolerance: 0.62, maxTolerance: 1.22 },
  { name: 'Ring Proximal', startJoint: 13, endJoint: 14, nominalRatio: 0.46, minTolerance: 0.26, maxTolerance: 0.70 },
  { name: 'Ring Middle', startJoint: 14, endJoint: 15, nominalRatio: 0.31, minTolerance: 0.17, maxTolerance: 0.49 },
  { name: 'Ring Distal', startJoint: 15, endJoint: 16, nominalRatio: 0.23, minTolerance: 0.12, maxTolerance: 0.39 },

  // Pinky chain
  { name: 'Pinky Metacarpal', startJoint: 0, endJoint: 17, nominalRatio: 0.85, minTolerance: 0.55, maxTolerance: 1.15 },
  { name: 'Pinky Proximal', startJoint: 17, endJoint: 18, nominalRatio: 0.38, minTolerance: 0.20, maxTolerance: 0.60 },
  { name: 'Pinky Middle', startJoint: 18, endJoint: 19, nominalRatio: 0.24, minTolerance: 0.12, maxTolerance: 0.40 },
  { name: 'Pinky Distal', startJoint: 19, endJoint: 20, nominalRatio: 0.21, minTolerance: 0.10, maxTolerance: 0.36 },

  // Palm transverse arches
  { name: 'Transverse Palm 5-17', startJoint: 5, endJoint: 17, nominalRatio: 0.75, minTolerance: 0.45, maxTolerance: 1.05 }
];

export const FINGER_DEFINITIONS: Array<{
  name: 'Thumb' | 'Index' | 'Middle' | 'Ring' | 'Pinky';
  joints: [number, number, number, number];
  mcpJoint: number;
  tipJoint: number;
}> = [
  { name: 'Thumb', joints: [1, 2, 3, 4], mcpJoint: 1, tipJoint: 4 },
  { name: 'Index', joints: [5, 6, 7, 8], mcpJoint: 5, tipJoint: 8 },
  { name: 'Middle', joints: [9, 10, 11, 12], mcpJoint: 9, tipJoint: 12 },
  { name: 'Ring', joints: [13, 14, 15, 16], mcpJoint: 13, tipJoint: 16 },
  { name: 'Pinky', joints: [17, 18, 19, 20], mcpJoint: 17, tipJoint: 20 }
];

/**
 * Returns customized One Euro Filter parameters tuned to anatomical joint rigidity.
 * Rigid palm/wrist joints receive heavier low-pass smoothing.
 * Dynamic fingertips receive higher beta and cutoff to avoid phase lag during rapid finger snapping.
 */
export function getAnatomicalFilterConfig(jointIndex: number): { minCutoff: number; beta: number; dCutoff: number } {
  // Wrist & Palm anchor joints (0, 1, 5, 9, 13, 17): high stability, zero drift
  if (jointIndex === 0 || jointIndex === 1 || jointIndex === 5 || jointIndex === 9 || jointIndex === 13 || jointIndex === 17) {
    return { minCutoff: 0.75, beta: 0.014, dCutoff: 1.0 };
  }
  // Fingertips (4, 8, 12, 16, 20): high agility, zero phase lag during rapid finger articulation
  if (jointIndex === 4 || jointIndex === 8 || jointIndex === 12 || jointIndex === 16 || jointIndex === 20) {
    return { minCutoff: 1.05, beta: 0.026, dCutoff: 1.0 };
  }
  // PIP & DIP intermediate articulation joints
  return { minCutoff: 0.85, beta: 0.018, dCutoff: 1.0 };
}

export class AnatomicalHandModel {
  private static dist3D(p1: Point3D, p2: Point3D): number {
    const dx = p1.x - p2.x;
    const dy = p1.y - p2.y;
    const dz = (p1.z ?? 0) - (p2.z ?? 0);
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  public static evaluateIntegrity(
    landmarks: Point3D[],
    previousLandmarks?: Point3D[]
  ): AnatomicalIntegrityReport {
    if (!landmarks || landmarks.length < 21) {
      return {
        isBiologicallyPlausible: false,
        integrityScore: 0,
        deviantBonesCount: 0,
        maxBoneRatioDeviation: 0,
        perFingerTelemetry: [],
        overallPalmScale: 0
      };
    }

    // 1. Compute Palm Baseline Scale (Wrist [0] -> Mid MCP [9])
    const wrist = landmarks[0];
    const midMCP = landmarks[9];
    const palmScale = Math.max(0.035, this.dist3D(wrist, midMCP));

    // 2. Evaluate Bone Segments against Anatomical Constraints
    let deviantBones = 0;
    let maxRatioDeviation = 0;

    for (const bone of ANATOMICAL_BONES) {
      const p1 = landmarks[bone.startJoint];
      const p2 = landmarks[bone.endJoint];
      if (!p1 || !p2) continue;

      const len = this.dist3D(p1, p2);
      const ratio = len / palmScale;

      if (ratio < bone.minTolerance || ratio > bone.maxTolerance) {
        deviantBones++;
        const dev = Math.max(bone.minTolerance - ratio, ratio - bone.maxTolerance);
        if (dev > maxRatioDeviation) maxRatioDeviation = dev;
      }
    }

    // 3. Evaluate Per-Finger Kinematic Telemetry
    const perFinger: FingerTelemetry[] = FINGER_DEFINITIONS.map((def) => {
      const tip = landmarks[def.tipJoint];
      const mcp = landmarks[def.mcpJoint];
      const tipDistWrist = this.dist3D(tip, wrist);
      const mcpDistWrist = this.dist3D(mcp, wrist);

      // Extension ratio: extended if tip is far beyond MCP relative to palm scale
      const extension = Math.max(0, Math.min(1.0, (tipDistWrist - mcpDistWrist) / (palmScale * 0.95)));

      // Calculate instantaneous jitter / displacement vs previous frame if available
      let jitter = 0;
      if (previousLandmarks && previousLandmarks.length >= 21) {
        const prevTip = previousLandmarks[def.tipJoint];
        const dx = tip.x - prevTip.x;
        const dy = tip.y - prevTip.y;
        jitter = Math.sqrt(dx * dx + dy * dy);
      }

      // Stability rating (100 = steady, 0 = jittering violently)
      const stability = Math.max(0, Math.min(100, Math.round(100 - jitter * 3500)));

      // Occlusion flag: if tip is unusually close to palm center and intermediate bones compress
      const isOccluded = tipDistWrist / palmScale < 0.45 && extension < 0.15;

      return {
        name: def.name,
        joints: def.joints,
        tipJoint: def.tipJoint,
        extensionRatio: Number(extension.toFixed(2)),
        jitterVariance: Number(jitter.toFixed(4)),
        stabilityScore: stability,
        isOccluded
      };
    });

    // 4. Calculate Aggregate Anatomical Integrity Score (0 - 100)
    const boneRatioPenalties = Math.min(60, deviantBones * 8 + maxRatioDeviation * 40);
    const integrityScore = Math.max(10, Math.round(100 - boneRatioPenalties));
    const isBiologicallyPlausible = deviantBones <= 3 && maxRatioDeviation < 0.55;

    return {
      isBiologicallyPlausible,
      integrityScore,
      deviantBonesCount: deviantBones,
      maxBoneRatioDeviation: Number(maxRatioDeviation.toFixed(3)),
      perFingerTelemetry: perFinger,
      overallPalmScale: Number(palmScale.toFixed(4))
    };
  }
}
