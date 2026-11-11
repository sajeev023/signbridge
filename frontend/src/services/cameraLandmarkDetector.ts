import { Landmark3D } from '../types';

export interface HandLandmark {
  x: number;
  y: number;
  z: number;
}

export interface PoseLandmark {
  x: number;
  y: number;
  z: number;
}

export interface FrameLandmarks {
  leftHand: HandLandmark[] | null;
  rightHand: HandLandmark[] | null;
  pose: PoseLandmark[] | null;
}

// Hand bone connections (pairs of landmark indices)
export const HAND_CONNECTIONS: [number, number][] = [
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle
  [9, 10], [10, 11], [11, 12],
  // Ring
  [13, 14], [14, 15], [15, 16],
  // Pinky
  [0, 17], [17, 18], [18, 19], [19, 20],
  // Palm base
  [5, 9], [9, 13], [13, 17]
];

export const POSE_CONNECTIONS: [number, number][] = [
  [11, 12], // Left shoulder to Right shoulder
  [11, 13], [13, 15], // Left arm: shoulder -> elbow -> wrist
  [12, 14], [14, 16], // Right arm: shoulder -> elbow -> wrist
  [11, 23], [12, 24], [23, 24] // Torso to hips
];

export class LandmarkRenderer {
  public static drawLandmarks(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    landmarks: FrameLandmarks,
    confidence: number = 0.8
  ) {
    ctx.save();

    // 1. Draw Upper Body Pose
    if (landmarks.pose && landmarks.pose.length >= 25) {
      ctx.lineWidth = 3;
      ctx.strokeStyle = 'rgba(59, 130, 246, 0.65)'; // Electric blue

      // Bones
      for (const [startIdx, endIdx] of POSE_CONNECTIONS) {
        if (startIdx < landmarks.pose.length && endIdx < landmarks.pose.length) {
          const p1 = landmarks.pose[startIdx];
          const p2 = landmarks.pose[endIdx];
          if (p1 && p2) {
            ctx.beginPath();
            ctx.moveTo(p1.x * width, p1.y * height);
            ctx.lineTo(p2.x * width, p2.y * height);
            ctx.stroke();
          }
        }
      }

      // Key Joints
      for (const idx of [0, 11, 12, 13, 14, 15, 16]) {
        const p = landmarks.pose[idx];
        if (p) {
          ctx.beginPath();
          ctx.arc(p.x * width, p.y * height, 5, 0, 2 * Math.PI);
          ctx.fillStyle = idx === 0 ? '#38bdf8' : '#60a5fa';
          ctx.fill();
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }
      }
    }

    // 2. Draw Left Hand
    if (landmarks.leftHand && landmarks.leftHand.length === 21) {
      this.drawHand(ctx, width, height, landmarks.leftHand, 'rgba(168, 85, 247, 0.85)', '#c084fc');
    }

    // 3. Draw Right Hand
    if (landmarks.rightHand && landmarks.rightHand.length === 21) {
      this.drawHand(ctx, width, height, landmarks.rightHand, 'rgba(34, 197, 94, 0.85)', '#4ade80');
    }

    ctx.restore();
  }

  private static drawHand(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    points: HandLandmark[],
    boneColor: string,
    jointColor: string
  ) {
    // Bones
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = boneColor;
    for (const [start, end] of HAND_CONNECTIONS) {
      const p1 = points[start];
      const p2 = points[end];
      if (p1 && p2) {
        ctx.beginPath();
        ctx.moveTo(p1.x * width, p1.y * height);
        ctx.lineTo(p2.x * width, p2.y * height);
        ctx.stroke();
      }
    }

    // Finger Joints
    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      const isTip = [4, 8, 12, 16, 20].includes(i);
      const radius = isTip ? 4.5 : 3;

      ctx.beginPath();
      ctx.arc(p.x * width, p.y * height, radius, 0, 2 * Math.PI);
      ctx.fillStyle = isTip ? '#ffffff' : jointColor;
      ctx.fill();
      ctx.strokeStyle = '#0f172a';
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }
}
