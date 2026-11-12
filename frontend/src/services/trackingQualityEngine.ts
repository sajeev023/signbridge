import { PersistentHandTrack } from './persistentHandTracker';

export type TrackingStatusState =
  | 'SEARCHING'
  | 'HAND DETECTED'
  | 'TRACKING'
  | 'TRACKING DEGRADED'
  | 'HAND LOST'
  | 'REACQUIRING';

export type FramingQuality = 'OPTIMAL' | 'TOO_CLOSE' | 'TOO_FAR' | 'EDGE_CLIPPED';

export interface TrackingQualityReport {
  status: TrackingStatusState;
  qualityScore: number; // 0 - 100
  stabilityScore: number; // 0 - 100
  framing: FramingQuality;
  jitterVariance: number;
  trackedHandsCount: number;
  continuityFrames: number;
  isReliableForInference: boolean;
  recommendation: string;
}

export class TrackingQualityEngine {
  private previousTrackCount: number = 0;
  private consecutiveTrackingFrames: number = 0;
  private consecutiveLostFrames: number = 0;
  private jitterBuffer: number[] = [];
  private readonly JITTER_BUFFER_SIZE = 15;

  public evaluate(tracks: PersistentHandTrack[], videoWidth: number = 640, videoHeight: number = 480): TrackingQualityReport {
    const handCount = tracks.length;

    // 1. Evaluate Detection State Transitions
    let status: TrackingStatusState = 'SEARCHING';
    if (handCount === 0) {
      this.consecutiveLostFrames++;
      this.consecutiveTrackingFrames = 0;
      if (this.previousTrackCount > 0) {
        status = 'HAND LOST';
      } else {
        status = 'SEARCHING';
      }
    } else {
      this.consecutiveLostFrames = 0;
      this.consecutiveTrackingFrames++;
      if (this.previousTrackCount === 0) {
        status = 'HAND DETECTED';
      } else if (this.previousTrackCount > 0 && this.consecutiveTrackingFrames < 5) {
        status = 'REACQUIRING';
      } else {
        status = 'TRACKING';
      }
    }
    this.previousTrackCount = handCount;

    if (handCount === 0) {
      return {
        status,
        qualityScore: 0,
        stabilityScore: 0,
        framing: 'TOO_FAR',
        jitterVariance: 0,
        trackedHandsCount: 0,
        continuityFrames: 0,
        isReliableForInference: false,
        recommendation: 'Position one or both hands within the camera viewfinder.'
      };
    }

    // 2. Evaluate Framing & Scale
    let isEdgeClipped = false;
    let isTooClose = false;
    let isTooFar = false;
    let avgPalmScale = 0;

    for (const track of tracks) {
      avgPalmScale += track.palmScale;
      // Check if any landmark is within 3% of the image boundary
      for (const p of track.smoothedLandmarks) {
        if (p.x < 0.03 || p.x > 0.97 || p.y < 0.03 || p.y > 0.97) {
          isEdgeClipped = true;
          break;
        }
      }
    }
    avgPalmScale /= handCount;

    if (avgPalmScale > 0.38) {
      isTooClose = true;
    } else if (avgPalmScale < 0.06) {
      isTooFar = true;
    }

    let framing: FramingQuality = 'OPTIMAL';
    if (isEdgeClipped) framing = 'EDGE_CLIPPED';
    else if (isTooClose) framing = 'TOO_CLOSE';
    else if (isTooFar) framing = 'TOO_FAR';

    // 3. Evaluate Spatial Stability & Jitter Variance
    let frameDisplacement = 0;
    for (const track of tracks) {
      const v = track.velocity;
      frameDisplacement += Math.sqrt(v.x * v.x + v.y * v.y);
    }
    frameDisplacement /= handCount;

    this.jitterBuffer.push(frameDisplacement);
    if (this.jitterBuffer.length > this.JITTER_BUFFER_SIZE) {
      this.jitterBuffer.shift();
    }

    // Mean and variance
    const meanDisp = this.jitterBuffer.reduce((a, b) => a + b, 0) / this.jitterBuffer.length;
    const jitterVariance =
      this.jitterBuffer.reduce((sum, v) => sum + (v - meanDisp) * (v - meanDisp), 0) / this.jitterBuffer.length;

    // Stability score (100 = completely motionless, 0 = violent shaking)
    const stabilityScore = Math.max(0, Math.min(100, Math.round(100 - Math.min(100, jitterVariance * 4000 + meanDisp * 20))));

    // 4. Evaluate Coasting / Occlusion Degradation
    let hasCoastingTrack = false;
    for (const track of tracks) {
      if (track.isCoasting) {
        hasCoastingTrack = true;
        break;
      }
    }

    if (hasCoastingTrack || isEdgeClipped || framing !== 'OPTIMAL' || stabilityScore < 35) {
      if (status === 'TRACKING') {
        status = 'TRACKING DEGRADED';
      }
    }

    // 5. Compute Aggregate Optical Quality Score (0 - 100)
    let qualityScore = 90;
    if (framing === 'OPTIMAL') qualityScore += 10;
    if (isEdgeClipped) qualityScore -= 30;
    if (isTooClose) qualityScore -= 20;
    if (isTooFar) qualityScore -= 25;
    if (hasCoastingTrack) qualityScore -= 25;
    qualityScore = Math.max(10, Math.min(100, Math.round(qualityScore * 0.6 + stabilityScore * 0.4)));

    // 6. Generate Contextual Guidance
    let recommendation = 'Optical tracking optimal.';
    if (isEdgeClipped) {
      recommendation = 'Hand partially clipped at frame boundary. Shift towards center.';
    } else if (isTooClose) {
      recommendation = 'Hand too close to camera sensor. Step back slightly.';
    } else if (isTooFar) {
      recommendation = 'Hand too far from camera. Move closer to viewfinder.';
    } else if (hasCoastingTrack) {
      recommendation = 'Temporary joint occlusion detected. Hold fingers visible.';
    } else if (stabilityScore < 40) {
      recommendation = 'High kinetic motion. Hold sign steady to lock recognition.';
    }

    const isReliableForInference =
      status === 'TRACKING' && framing !== 'EDGE_CLIPPED' && !isTooFar && !hasCoastingTrack;

    return {
      status,
      qualityScore,
      stabilityScore,
      framing,
      jitterVariance: Number(jitterVariance.toFixed(5)),
      trackedHandsCount: handCount,
      continuityFrames: this.consecutiveTrackingFrames,
      isReliableForInference,
      recommendation
    };
  }

  public reset() {
    this.previousTrackCount = 0;
    this.consecutiveTrackingFrames = 0;
    this.consecutiveLostFrames = 0;
    this.jitterBuffer = [];
  }
}

export const trackingQualityEngine = new TrackingQualityEngine();
