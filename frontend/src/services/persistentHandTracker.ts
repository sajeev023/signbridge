import { HandSkeletonFilter, Point3D } from './oneEuroFilter';

export interface PersistentHandTrack {
  trackId: number;
  role: 'Dominant' | 'Base';
  handedness: 'Left' | 'Right';
  handednessConfidence: number;
  rawLandmarks: Point3D[];
  smoothedLandmarks: Point3D[];
  wrist: Point3D;
  palmCenter: Point3D;
  palmScale: number;
  velocity: Point3D;
  framesTracked: number;
  framesMissing: number;
  isCoasting: boolean;
  filter: HandSkeletonFilter;
}

export class PersistentHandTracker {
  private activeTracks: PersistentHandTrack[] = [];
  private nextTrackId: number = 0;
  private readonly MAX_COASTING_FRAMES = 4; // Grace period for temporary finger occlusion
  private readonly MATCHING_DISTANCE_THRESHOLD = 0.28; // Screen normalized max movement per frame

  private dist2D(p1: Point3D, p2: Point3D): number {
    const dx = p1.x - p2.x;
    const dy = p1.y - p2.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  private computePalmCenter(landmarks: Point3D[]): Point3D {
    // Center of MCP joints (5, 9, 13, 17) and wrist (0)
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
    return Math.max(0.04, Math.sqrt(dx * dx + dy * dy + dz * dz));
  }

  /**
   * Associates new MediaPipe detections with persistent tracks using spatial distance
   * and handedness continuity to prevent identity swapping and landmark jumps.
   */
  public update(
    rawDetections: Point3D[][],
    handednessData?: Array<Array<{ categoryName: string; score: number }>>,
    timestamp: number = performance.now()
  ): PersistentHandTrack[] {
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

    // 1. Solve Association Problem (Greedy bipartite matching by Euclidean distance)
    if (this.activeTracks.length > 0 && incomingDetections.length > 0) {
      const candidates: Array<{ trackIdx: number; detIdx: number; cost: number }> = [];

      for (let t = 0; t < this.activeTracks.length; t++) {
        const track = this.activeTracks[t];
        for (let d = 0; d < incomingDetections.length; d++) {
          const det = incomingDetections[d];
          const distPalm = this.dist2D(track.palmCenter, det.palmCenter);
          const distWrist = this.dist2D(track.wrist, det.wrist);
          let cost = 0.6 * distPalm + 0.4 * distWrist;

          // Penalize unexpected handedness flip
          if (track.handedness !== det.handedness && track.handednessConfidence > 0.85 && det.handednessConfidence > 0.85) {
            cost += 0.25;
          }

          if (cost <= this.MATCHING_DISTANCE_THRESHOLD) {
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
          const smoothed = track.filter.filter(det.landmarks, timestamp);

          const dt = Math.max(0.016, (timestamp - (track as any)._lastTimestamp || 33) / 1000);
          (track as any)._lastTimestamp = timestamp;

          track.velocity = {
            x: (det.palmCenter.x - track.palmCenter.x) / dt,
            y: (det.palmCenter.y - track.palmCenter.y) / dt,
            z: ((det.palmCenter.z ?? 0) - (track.palmCenter.z ?? 0)) / dt
          };

          track.rawLandmarks = det.landmarks;
          track.smoothedLandmarks = smoothed;
          track.wrist = smoothed[0];
          track.palmCenter = this.computePalmCenter(smoothed);
          track.palmScale = det.palmScale;
          track.handedness = det.handedness;
          track.handednessConfidence = det.handednessConfidence;
          track.framesTracked++;
          track.framesMissing = 0;
          track.isCoasting = false;
        }
      }
    }

    // 2. Handle unmatched incoming detections -> spawn new tracks
    for (let d = 0; d < incomingDetections.length; d++) {
      if (!matchedIncoming.has(d)) {
        const det = incomingDetections[d];
        const filter = new HandSkeletonFilter(0.85, 0.009, 1.0);
        const smoothed = filter.filter(det.landmarks, timestamp);

        const newTrack: PersistentHandTrack = {
          trackId: this.nextTrackId++,
          role: 'Dominant', // Assessed below
          handedness: det.handedness,
          handednessConfidence: det.handednessConfidence,
          rawLandmarks: det.landmarks,
          smoothedLandmarks: smoothed,
          wrist: smoothed[0],
          palmCenter: this.computePalmCenter(smoothed),
          palmScale: det.palmScale,
          velocity: { x: 0, y: 0, z: 0 },
          framesTracked: 1,
          framesMissing: 0,
          isCoasting: false,
          filter
        };
        (newTrack as any)._lastTimestamp = timestamp;
        this.activeTracks.push(newTrack);
      }
    }

    // 3. Handle unmatched active tracks -> coast or prune
    const survivingTracks: PersistentHandTrack[] = [];
    for (let t = 0; t < this.activeTracks.length; t++) {
      const track = this.activeTracks[t];
      if (matchedTracks.has(t)) {
        survivingTracks.push(track);
      } else {
        track.framesMissing++;
        if (track.framesMissing <= this.MAX_COASTING_FRAMES) {
          track.isCoasting = true;
          survivingTracks.push(track);
        } else {
          track.filter.reset();
        }
      }
    }

    // 4. Assign Roles (Dominant vs Base) stably
    if (survivingTracks.length === 1) {
      survivingTracks[0].role = 'Dominant';
    } else if (survivingTracks.length >= 2) {
      // In mirrored selfie camera: Right hand of signer appears on right side of viewer
      // The hand with more isolated/extended index finger or Right hand is Dominant
      const t0 = survivingTracks[0];
      const t1 = survivingTracks[1];

      const iso0 = this.dist2D(t0.smoothedLandmarks[8], t0.smoothedLandmarks[0]) / (this.dist2D(t0.smoothedLandmarks[12], t0.smoothedLandmarks[0]) || 1);
      const iso1 = this.dist2D(t1.smoothedLandmarks[8], t1.smoothedLandmarks[0]) / (this.dist2D(t1.smoothedLandmarks[12], t1.smoothedLandmarks[0]) || 1);

      if (t0.handedness === 'Right' && t1.handedness === 'Left') {
        t0.role = 'Dominant';
        t1.role = 'Base';
      } else if (t0.handedness === 'Left' && t1.handedness === 'Right') {
        t0.role = 'Base';
        t1.role = 'Dominant';
      } else if (iso1 > iso0) {
        t0.role = 'Base';
        t1.role = 'Dominant';
      } else {
        t0.role = 'Dominant';
        t1.role = 'Base';
      }
    }

    this.activeTracks = survivingTracks;
    return survivingTracks;
  }

  public reset() {
    this.activeTracks.forEach((t) => t.filter.reset());
    this.activeTracks = [];
  }

  public getTracks(): PersistentHandTrack[] {
    return this.activeTracks;
  }
}

export const persistentHandTracker = new PersistentHandTracker();
