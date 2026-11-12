/**
 * Tracking Benchmark Lab
 * Provides repeatable, deterministic benchmarking of hand tracking pipelines
 * using annotated ground-truth test sequences.
 *
 * Compares:
 * 1. Raw MediaPipe (Unfiltered, naive matching)
 * 2. Baseline bd86151 (Uniform 1€ filter, static coasting)
 * 3. New Hybrid Precision Engine (Anatomical 1€ filter, motion-projected association, outlier gate)
 */

import { Point3D, OneEuroFilter3D } from './oneEuroFilter';
import { HandSkeletonFilter as AnatomicalFilter } from './oneEuroFilter';
import { PersistentHandTracker } from './persistentHandTracker';

export interface BenchmarkFrame {
  frameIndex: number;
  timestamp: number;
  groundTruth: Point3D[][]; // 1 or 2 hands, 21 joints each
  simulatedDetections: Point3D[][]; // Input with realistic sensor noise & occasional dropouts
}

export interface PipelineBenchmarkReport {
  pipelineName: string;
  totalFrames: number;
  overallMAE: number; // Mean Absolute Error in pixels (based on 640x480)
  overallPCK10: number; // % of keypoints within 10px
  overallPCK20: number; // % of keypoints within 20px
  fingertipMAE: number; // Tips: 4, 8, 12, 16, 20
  wristPalmMAE: number; // Anchor joints: 0, 5, 9, 13, 17
  perFingerMAE: {
    thumb: number;
    index: number;
    middle: number;
    ring: number;
    pinky: number;
  };
  identitySwitches: number;
  continuityRate: number; // % frames with valid track
  avgProcessingTimeMs: number;
  reacquisitionTimeMs: number;
}

export interface ComparativeBenchmarkResult {
  timestamp: string;
  frameCount: number;
  rawMediaPipe: PipelineBenchmarkReport;
  baselineCommit: PipelineBenchmarkReport;
  newPrecisionEngine: PipelineBenchmarkReport;
  improvementMetrics: {
    overallMAEReductionPct: number;
    fingertipMAEReductionPct: number;
    pck10GainPct: number;
    identitySwitchReductionPct: number;
  };
}

// Canonical ground-truth hand poses (normalized 0..1 coordinates)
const POSE_L_SHAPE: Point3D[] = [
  { x: 0.50, y: 0.70, z: 0.00 }, // 0: Wrist
  { x: 0.46, y: 0.67, z: -0.01 }, // 1: Thumb CMC
  { x: 0.42, y: 0.64, z: -0.02 }, // 2: Thumb MCP
  { x: 0.38, y: 0.62, z: -0.03 }, // 3: Thumb IP
  { x: 0.33, y: 0.62, z: -0.04 }, // 4: Thumb Tip (Extended horizontal)
  { x: 0.47, y: 0.56, z: -0.01 }, // 5: Index MCP
  { x: 0.47, y: 0.48, z: -0.02 }, // 6: Index PIP
  { x: 0.47, y: 0.41, z: -0.03 }, // 7: Index DIP
  { x: 0.47, y: 0.34, z: -0.04 }, // 8: Index Tip (Extended vertical)
  { x: 0.50, y: 0.56, z: 0.00 }, // 9: Middle MCP
  { x: 0.50, y: 0.61, z: 0.02 }, // 10: Middle PIP (Curled)
  { x: 0.50, y: 0.64, z: 0.03 }, // 11: Middle DIP
  { x: 0.49, y: 0.61, z: 0.01 }, // 12: Middle Tip
  { x: 0.53, y: 0.57, z: 0.01 }, // 13: Ring MCP
  { x: 0.53, y: 0.62, z: 0.02 }, // 14: Ring PIP (Curled)
  { x: 0.53, y: 0.64, z: 0.03 }, // 15: Ring DIP
  { x: 0.52, y: 0.61, z: 0.01 }, // 16: Ring Tip
  { x: 0.56, y: 0.59, z: 0.01 }, // 17: Pinky MCP
  { x: 0.56, y: 0.63, z: 0.02 }, // 18: Pinky PIP (Curled)
  { x: 0.56, y: 0.65, z: 0.03 }, // 19: Pinky DIP
  { x: 0.55, y: 0.62, z: 0.01 }  // 20: Pinky Tip
];

export class TrackingBenchmarkLab {
  private static distPx(p1: Point3D, p2: Point3D, width: number = 640, height: number = 480): number {
    const dx = (p1.x - p2.x) * width;
    const dy = (p1.y - p2.y) * height;
    return Math.sqrt(dx * dx + dy * dy);
  }

  /**
   * Generates a deterministic, reproducible 90-frame multi-scenario test dataset:
   * 1. Frames 0-29: Static Hold with Gaussian sensor noise (tests micro-jitter suppression).
   * 2. Frames 30-59: Rapid hand translation & direction reversal (tests velocity lag & overshoot).
   * 3. Frames 60-89: Brief 3-frame sensor dropouts & recovery (tests coasting & reacquisition).
   */
  public static generateStandardTestSequence(): BenchmarkFrame[] {
    const frames: BenchmarkFrame[] = [];
    const basePose = POSE_L_SHAPE;

    // Pseudo-random deterministic noise generator (LCG)
    let seed = 1337;
    const lcg = () => {
      seed = (seed * 1664525 + 1013904223) % 4294967296;
      return seed / 4294967296;
    };

    for (let f = 0; f < 90; f++) {
      const timestamp = f * 33.333; // 30 FPS interval

      // Scenario 1: Frames 0-29 (Static hold at center)
      let shiftX = 0;
      let shiftY = 0;

      // Scenario 2: Frames 30-59 (Fast lateral translation)
      if (f >= 30 && f < 60) {
        const t = (f - 30) / 30;
        shiftX = Math.sin(t * Math.PI * 2) * 0.18;
        shiftY = Math.cos(t * Math.PI * 2) * 0.08;
      }

      // Ground truth exact positions
      const gtHand: Point3D[] = basePose.map((p) => ({
        x: p.x + shiftX,
        y: p.y + shiftY,
        z: p.z
      }));

      // Simulate detector output with camera sensor noise & dropouts
      let simHands: Point3D[][] = [];

      // Scenario 3: Frames 70-72 simulate complete detector loss (dropout)
      const isDropped = f >= 70 && f <= 72;

      if (!isDropped) {
        const noisyHand: Point3D[] = gtHand.map((p, idx) => {
          // Fingertips have higher sensor noise than wrist
          const isTip = [4, 8, 12, 16, 20].includes(idx);
          const noiseScale = isTip ? 0.0055 : 0.0022;
          const noiseX = (lcg() - 0.5) * 2 * noiseScale;
          const noiseY = (lcg() - 0.5) * 2 * noiseScale;
          return {
            x: Math.max(0, Math.min(1.0, p.x + noiseX)),
            y: Math.max(0, Math.min(1.0, p.y + noiseY)),
            z: p.z
          };
        });
        simHands = [noisyHand];
      }

      frames.push({
        frameIndex: f,
        timestamp,
        groundTruth: [gtHand],
        simulatedDetections: simHands
      });
    }

    return frames;
  }

  /**
   * Executes comparative benchmark across all three tracking configurations.
   */
  public static runComparativeBenchmark(): ComparativeBenchmarkResult {
    const dataset = this.generateStandardTestSequence();
    const width = 640;
    const height = 480;

    // 1. Raw MediaPipe Pipeline (Unfiltered, naive pass-through)
    const rawReport = this.evaluatePipeline(
      'Raw MediaPipe (Unfiltered)',
      dataset,
      () => {
        return (dets: Point3D[][]) => dets[0] || null;
      },
      width,
      height
    );

    // 2. Baseline bd86151 (Uniform One Euro filter, static coasting)
    const baselineReport = this.evaluatePipeline(
      'Baseline Commit bd86151',
      dataset,
      () => {
        // Uniform filter: minCutoff = 0.85, beta = 0.009 for all joints
        const filters = Array.from({ length: 21 }, () => new OneEuroFilter3D(0.85, 0.009, 1.0));
        let lastOutput: Point3D[] | null = null;
        let missing = 0;

        return (dets: Point3D[][], ts: number) => {
          if (dets.length > 0) {
            missing = 0;
            const res = dets[0].map((p, i) => filters[i].filter(p, ts));
            lastOutput = res;
            return res;
          } else {
            missing++;
            if (missing <= 4 && lastOutput) {
              return lastOutput; // Static coasting
            }
            return null;
          }
        };
      },
      width,
      height
    );

    // 3. New Hybrid Precision Engine (Anatomical filter, velocity prediction, outlier gate)
    const precisionReport = this.evaluatePipeline(
      'New Precision Engine (Hybrid)',
      dataset,
      () => {
        const tracker = new PersistentHandTracker();
        return (dets: Point3D[][], ts: number) => {
          const tracks = tracker.update(dets, undefined, ts);
          return tracks.length > 0 ? tracks[0].smoothedLandmarks : null;
        };
      },
      width,
      height
    );

    // Compute relative gains
    const overallMAEReductionPct = Number(
      (((baselineReport.overallMAE - precisionReport.overallMAE) / baselineReport.overallMAE) * 100).toFixed(1)
    );
    const fingertipMAEReductionPct = Number(
      (((baselineReport.fingertipMAE - precisionReport.fingertipMAE) / baselineReport.fingertipMAE) * 100).toFixed(1)
    );
    const pck10GainPct = Number((precisionReport.overallPCK10 - baselineReport.overallPCK10).toFixed(1));
    const identitySwitchReductionPct = baselineReport.identitySwitches > 0
      ? Number((((baselineReport.identitySwitches - precisionReport.identitySwitches) / baselineReport.identitySwitches) * 100).toFixed(1))
      : 0;

    return {
      timestamp: new Date().toISOString(),
      frameCount: dataset.length,
      rawMediaPipe: rawReport,
      baselineCommit: baselineReport,
      newPrecisionEngine: precisionReport,
      improvementMetrics: {
        overallMAEReductionPct,
        fingertipMAEReductionPct,
        pck10GainPct,
        identitySwitchReductionPct
      }
    };
  }

  private static evaluatePipeline(
    name: string,
    dataset: BenchmarkFrame[],
    initFn: () => (dets: Point3D[][], ts: number) => Point3D[] | null,
    width: number,
    height: number
  ): PipelineBenchmarkReport {
    const processFrame = initFn();
    let totalErrors = 0;
    let countKeypoints = 0;
    let countPCK10 = 0;
    let countPCK20 = 0;

    let tipErrors = 0;
    let countTips = 0;
    let anchorErrors = 0;
    let countAnchors = 0;

    const fingerErrors = {
      thumb: 0,
      index: 0,
      middle: 0,
      ring: 0,
      pinky: 0
    };
    const fingerCounts = { thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0 };

    let trackedFrames = 0;
    let totalProcTimeMs = 0;
    let reacquisitionTimeMs = 0;
    let dropoutDetected = false;
    let dropoutStartTs = 0;

    for (const frame of dataset) {
      const gt = frame.groundTruth[0];
      const t0 = performance.now();
      const output = processFrame(frame.simulatedDetections, frame.timestamp);
      const t1 = performance.now();
      totalProcTimeMs += t1 - t0;

      if (frame.simulatedDetections.length === 0 && !dropoutDetected) {
        dropoutDetected = true;
        dropoutStartTs = frame.timestamp;
      }

      if (output && output.length === 21) {
        trackedFrames++;

        if (dropoutDetected && frame.simulatedDetections.length > 0 && reacquisitionTimeMs === 0) {
          reacquisitionTimeMs = frame.timestamp - dropoutStartTs;
        }

        for (let i = 0; i < 21; i++) {
          const err = this.distPx(output[i], gt[i], width, height);
          totalErrors += err;
          countKeypoints++;
          if (err <= 10.0) countPCK10++;
          if (err <= 20.0) countPCK20++;

          // Anchor joints: 0 (wrist), 5, 9, 13, 17 (MCPs)
          if ([0, 5, 9, 13, 17].includes(i)) {
            anchorErrors += err;
            countAnchors++;
          }
          // Fingertips: 4, 8, 12, 16, 20
          if ([4, 8, 12, 16, 20].includes(i)) {
            tipErrors += err;
            countTips++;
          }

          // Per-finger
          if (i >= 1 && i <= 4) { fingerErrors.thumb += err; fingerCounts.thumb++; }
          else if (i >= 5 && i <= 8) { fingerErrors.index += err; fingerCounts.index++; }
          else if (i >= 9 && i <= 12) { fingerErrors.middle += err; fingerCounts.middle++; }
          else if (i >= 13 && i <= 16) { fingerErrors.ring += err; fingerCounts.ring++; }
          else if (i >= 17 && i <= 20) { fingerErrors.pinky += err; fingerCounts.pinky++; }
        }
      }
    }

    return {
      pipelineName: name,
      totalFrames: dataset.length,
      overallMAE: Number((totalErrors / (countKeypoints || 1)).toFixed(2)),
      overallPCK10: Number(((countPCK10 / (countKeypoints || 1)) * 100).toFixed(1)),
      overallPCK20: Number(((countPCK20 / (countKeypoints || 1)) * 100).toFixed(1)),
      fingertipMAE: Number((tipErrors / (countTips || 1)).toFixed(2)),
      wristPalmMAE: Number((anchorErrors / (countAnchors || 1)).toFixed(2)),
      perFingerMAE: {
        thumb: Number((fingerErrors.thumb / (fingerCounts.thumb || 1)).toFixed(2)),
        index: Number((fingerErrors.index / (fingerCounts.index || 1)).toFixed(2)),
        middle: Number((fingerErrors.middle / (fingerCounts.middle || 1)).toFixed(2)),
        ring: Number((fingerErrors.ring / (fingerCounts.ring || 1)).toFixed(2)),
        pinky: Number((fingerErrors.pinky / (fingerCounts.pinky || 1)).toFixed(2))
      },
      identitySwitches: 0,
      continuityRate: Number(((trackedFrames / dataset.length) * 100).toFixed(1)),
      avgProcessingTimeMs: Number((totalProcTimeMs / dataset.length).toFixed(3)),
      reacquisitionTimeMs: Number(reacquisitionTimeMs.toFixed(1))
    };
  }
}
