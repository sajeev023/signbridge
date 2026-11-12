/**
 * Persistent Hand Tracker with Motion-Aware Velocity Prediction,
 * Scale-Normalized Association, and Identity Continuity Preservation.
 *
 * Implements Phase 2 & 3 hybrid architecture:
 * - Distinguishes OBSERVED, PREDICTED, and FILTERED landmark states.
 * - Velocity-extrapolates coasting tracks during temporary occlusion.
 * - Prevents identity swaps during hand crossing and proximity interaction.
 * - Logs identity-switch events and reacquisition timing.
 */

import { HandSkeletonFilter, Point3D, ProvenancedLandmark } from './oneEuroFilter';
import { AnatomicalHandModel, AnatomicalIntegrityReport } from './anatomicalHandModel';

export interface PersistentHandTrack {
  trackId: number;
  role: 'Dominant' | 'Base';
  handedness: 'Left' | 'Right';
  handednessConfidence: number;
  rawLandmarks: Point3D[];
  observedLandmarks: Point3D[];
  predictedLandmarks: Point3D[];
  smoothedLandmarks: ProvenancedLandmark[];
  wrist: Point3D;
  palmCenter: Point3D;
  palmScale: number;
  velocity: Point3D; // Normalized units per second
  framesTracked: number;
  framesMissing: number;
  isCoasting: boolean;
  filter: HandSkeletonFilter;
  integrityReport: AnatomicalIntegrityReport;
  lastDetectionTimestamp: number;
}

export interface TrackerDiagnosticTelemetry {
  activeTracksCount: number;
  identitySwitchesCount: number;
  reacquisitionsCount: number;
  totalFramesProcessed: number;
  coastingTracksCount: number;
}

export class PersistentHandTracker {
  private activeTracks: PersistentHandTrack[] = [];
  private nextTrackId: number = 0;
  private readonly MAX_COASTING_FRAMES = 5; // Survives up to ~160ms occlusion at 30 FPS
  private readonly BASE_MATCHING_DISTANCE_THRESHOLD = 0.28; // Screen normalized max movement per frame

  // Diagnostic counters
  private identitySwitchesCount: number = 0;
  private reacquisitionsCount: number = 0;
  private totalFramesProcessed: number = 0;

  private dist2D(p1: Point3D, p2: Point3D): number {
    const dx = p1.x - p2.x;
    const dy = p1.y - p2.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  private computePalmCenter(landmarks: Point3D[]): Point3D {
    // Robust geometric centroid of wrist (0) and 4 metacarpal bases (5, 9, 13, 17)
    return {
      x: (landmarks[0].x + landmarks[5].x + landmarks[9].x + landmarks[13].x + landmarks[17].x) / 5.0,
      y: (landmarks[0].y + landmarks[5].y + landmarks[9].y + landmarks[13].y + landmarks[17].y) / 5.0,
      z: ((landmarks[0].z ?? 0) + (landmarks[5].z ?? 0) + (landmarks[9].z ?? 0) + (landmarks[13].z ?? 0) + (landmarks[17].z ?? 0)) / 5.0
    };
  }

  private computePalmScale(landmarks: Point3D[]): number {
    const wrist = landmarks[0];
    const midMCP = landmarks[9];
    const dx = wrist.x - midMCP.x;
    const dy = wrist.y - midMCP.y;
    const dz = (wrist.z ?? 0) - (midMCP.z ?? 0);
    return Math.max(0.035, Math.sqrt(dx * dx + dy * dy + dz * dz));
  }

  /**
   * Updates tracker state with incoming MediaPipe detection frames.
   */
  public update(
    rawDetections: Point3D[][],
    handednessData?: Array<Array<{ categoryName: string; score: number }>>,
    timestamp: number = performance.now()
  ): PersistentHandTrack[] {
    this.totalFramesProcessed++;

    const incomingDetections = rawDetections.map((lm, idx) => {
      const palmCenter = this.computePalmCenter(lm);
      const palmScale = this.computePalmScale(lm);
      const hData = handednessData?.[idx]?.[0];
      const handedness: 'Left' | 'Right' = (hData?.categoryName === 'Left' ? 'Left' : 'Right');
      const handednessConfidence = hData?.score ?? 0.8;
      return {
        landmarks: lm,
        palmCenter,
        palmScale,
        wrist: lm[0],
        handedness,
        handednessConfidence
      };
    });

    const matchedIncoming = new Set<number>();
    const matchedTracks = new Set<number>();

    // 1. Solve Association Problem with Motion-Projected Bipartite Matching
    if (this.activeTracks.length > 0 && incomingDetections.length > 0) {
      const candidates: Array<{ trackIdx: number; detIdx: number; cost: number }> = [];

      for (let t = 0; t < this.activeTracks.length; t++) {
        const track = this.activeTracks[t];
        const dt = Math.max(0.016, (timestamp - track.lastDetectionTimestamp) / 1000.0);

        // Motion-aware Kalman/velocity projected position
        const projectedPalmCenter: Point3D = {
          x: track.palmCenter.x + track.velocity.x * dt,
          y: track.palmCenter.y + track.velocity.y * dt,
          z: (track.palmCenter.z ?? 0) + track.velocity.z * dt
        };
        const projectedWrist: Point3D = {
          x: track.wrist.x + track.velocity.x * dt,
          y: track.wrist.y + track.velocity.y * dt,
          z: (track.wrist.z ?? 0) + track.velocity.z * dt
        };

        for (let d = 0; d < incomingDetections.length; d++) {
          const det = incomingDetections[d];
          const distPalm = this.dist2D(projectedPalmCenter, det.palmCenter);
          const distWrist = this.dist2D(projectedWrist, det.wrist);

          // Scale-normalized distance (accounts for depth/distance differences)
          const normDistPalm = distPalm / track.palmScale;
          const normDistWrist = distWrist / track.palmScale;
          let cost = 0.6 * normDistPalm + 0.4 * normDistWrist;

          // Handedness consistency penalty
          if (track.handedness !== det.handedness) {
            cost += 0.85; // Heavy penalty against swapping Left/Right hands
          }

          // Dynamic matching threshold scaled to palm size
          const threshold = (this.BASE_MATCHING_DISTANCE_THRESHOLD / track.palmScale) * 1.35;
          if (cost <= threshold) {
            candidates.push({ trackIdx: t, detIdx: d, cost });
          }
        }
      }

      candidates.sort((a, b) => a.cost - b.cost);

      for (const c of candidates) {
        if (!matchedTracks.has(c.trackIdx) && !matchedIncoming.has(c.detIdx)) {
          matchedTracks.add(c.trackIdx);
          matchedIncoming.add(c.detIdx);

          // Update active track
          const track = this.activeTracks[c.trackIdx];
          const det = incomingDetections[c.detIdx];

          if (track.isCoasting) {
            this.reacquisitionsCount++;
          }

          const dt = Math.max(0.016, (timestamp - track.lastDetectionTimestamp) / 1000.0);
          track.velocity = {
            x: (det.palmCenter.x - track.palmCenter.x) / dt,
            y: (det.palmCenter.y - track.palmCenter.y) / dt,
            z: ((det.palmCenter.z ?? 0) - (track.palmCenter.z ?? 0)) / dt
          };

          // Filter with anatomical joint tuning
          const smoothed = track.filter.filter(det.landmarks, timestamp, det.palmScale);

          track.rawLandmarks = det.landmarks;
          track.observedLandmarks = det.landmarks;
          track.predictedLandmarks = [];
          track.smoothedLandmarks = smoothed;
          track.wrist = smoothed[0];
          track.palmCenter = this.computePalmCenter(smoothed);
          track.palmScale = det.palmScale;
          track.handedness = det.handedness;
          track.handednessConfidence = det.handednessConfidence;
          track.framesTracked++;
          track.framesMissing = 0;
          track.isCoasting = false;
          track.lastDetectionTimestamp = timestamp;

          // Anatomical integrity assessment
          track.integrityReport = AnatomicalHandModel.evaluateIntegrity(
            det.landmarks,
            track.smoothedLandmarks
          );
        }
      }
    }

    // 2. Handle unmatched incoming detections -> spawn new tracks
    for (let d = 0; d < incomingDetections.length; d++) {
      if (!matchedIncoming.has(d)) {
        const det = incomingDetections[d];
        const filter = new HandSkeletonFilter();
        const smoothed = filter.filter(det.landmarks, timestamp, det.palmScale);
        const integrity = AnatomicalHandModel.evaluateIntegrity(det.landmarks);

        const newTrack: PersistentHandTrack = {
          trackId: this.nextTrackId++,
          role: 'Dominant', // Assessed below
          handedness: det.handedness,
          handednessConfidence: det.handednessConfidence,
          rawLandmarks: det.landmarks,
          observedLandmarks: det.landmarks,
          predictedLandmarks: [],
          smoothedLandmarks: smoothed,
          wrist: smoothed[0],
          palmCenter: this.computePalmCenter(smoothed),
          palmScale: det.palmScale,
          velocity: { x: 0, y: 0, z: 0 },
          framesTracked: 1,
          framesMissing: 0,
          isCoasting: false,
          filter,
          integrityReport: integrity,
          lastDetectionTimestamp: timestamp
        };

        this.activeTracks.push(newTrack);
      }
    }

    // 3. Handle unmatched active tracks -> Motion-assisted coasting or prune
    const survivingTracks: PersistentHandTrack[] = [];
    for (let t = 0; t < this.activeTracks.length; t++) {
      const track = this.activeTracks[t];
      if (matchedTracks.has(t)) {
        survivingTracks.push(track);
      } else {
        track.framesMissing++;
        if (track.framesMissing <= this.MAX_COASTING_FRAMES) {
          track.isCoasting = true;

          // Motion-Aware Kalman/Linear Extrapolation for PREDICTED landmarks
          const dt = Math.max(0.016, (timestamp - track.lastDetectionTimestamp) / 1000.0);
          const damping = Math.pow(0.85, track.framesMissing); // Damp velocity over time
          const dx = track.velocity.x * dt * damping;
          const dy = track.velocity.y * dt * damping;
          const dz = track.velocity.z * dt * damping;

          // Project all 21 landmarks according to hand velocity
          const predicted = track.smoothedLandmarks.map((pt) => ({
            x: Math.max(0, Math.min(1.0, pt.x + dx)),
            y: Math.max(0, Math.min(1.0, pt.y + dy)),
            z: (pt.z ?? 0) + dz,
            provenance: 'PREDICTED' as const
          }));

          track.predictedLandmarks = predicted;
          track.smoothedLandmarks = predicted;
          track.wrist = predicted[0];
          track.palmCenter = this.computePalmCenter(predicted);

          survivingTracks.push(track);
        } else {
          track.filter.reset();
        }
      }
    }

    // 4. Stable Role Arbitration (Dominant vs Base)
    if (survivingTracks.length === 1) {
      survivingTracks[0].role = 'Dominant';
    } else if (survivingTracks.length >= 2) {
      const t0 = survivingTracks[0];
      const t1 = survivingTracks[1];

      const prevT0Role = t0.role;
      const prevT1Role = t1.role;

      // In mirrored selfie camera:
      // The hand with more active finger articulation or Right hand is Dominant
      if (t0.handedness === 'Right' && t1.handedness === 'Left') {
        t0.role = 'Dominant';
        t1.role = 'Base';
      } else if (t0.handedness === 'Left' && t1.handedness === 'Right') {
        t0.role = 'Base';
        t1.role = 'Dominant';
      } else {
        // Evaluate kinematic finger extension activity
        const act0 = t0.integrityReport?.perFingerTelemetry?.[1]?.extensionRatio ?? 0.5;
        const act1 = t1.integrityReport?.perFingerTelemetry?.[1]?.extensionRatio ?? 0.5;
        if (act1 > act0 + 0.25) {
          t0.role = 'Base';
          t1.role = 'Dominant';
        } else if (act0 > act1 + 0.25) {
          t0.role = 'Dominant';
          t1.role = 'Base';
        }
      }

      if (t0.role !== prevT0Role && t0.framesTracked > 15) {
        this.identitySwitchesCount++;
      }
    }

    this.activeTracks = survivingTracks;
    return survivingTracks;
  }

  public reset() {
    this.activeTracks.forEach((t) => t.filter.reset());
    this.activeTracks = [];
    this.identitySwitchesCount = 0;
    this.reacquisitionsCount = 0;
    this.totalFramesProcessed = 0;
  }

  public getTracks(): PersistentHandTrack[] {
    return this.activeTracks;
  }

  public getDiagnosticTelemetry(): TrackerDiagnosticTelemetry {
    return {
      activeTracksCount: this.activeTracks.length,
      identitySwitchesCount: this.identitySwitchesCount,
      reacquisitionsCount: this.reacquisitionsCount,
      totalFramesProcessed: this.totalFramesProcessed,
      coastingTracksCount: this.activeTracks.filter((t) => t.isCoasting).length
    };
  }
}

export const persistentHandTracker = new PersistentHandTracker();
