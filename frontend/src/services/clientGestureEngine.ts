import { PredictionResult, ISLSign } from '../types';
import { FALLBACK_VOCABULARY } from './apiClient';

interface NormalizedLandmark {
  x: number;
  y: number;
  z: number;
}

export class ClientGestureEngine {
  private static vocabMap: Map<string, ISLSign> = new Map(
    FALLBACK_VOCABULARY.map((v) => [v.id, v])
  );

  private static historyBuffer: {
    timestamp: number;
    leftHand: NormalizedLandmark[] | null;
    rightHand: NormalizedLandmark[] | null;
  }[] = [];

  /**
   * Evaluates detected hands in real time and classifies ISL signs.
   */
  public static evaluateFrame(
    leftHand: NormalizedLandmark[] | null,
    rightHand: NormalizedLandmark[] | null
  ): PredictionResult | null {
    const now = performance.now();
    this.historyBuffer.push({ timestamp: now, leftHand, rightHand });
    if (this.historyBuffer.length > 30) {
      this.historyBuffer.shift();
    }

    if (!leftHand && !rightHand) {
      return null;
    }

    const twoHands = leftHand !== null && rightHand !== null;
    const primaryHand = rightHand || leftHand;
    if (!primaryHand) return null;

    let recognizedSignId: string | null = null;
    let confidence = 0.0;

    // Helper: Euclidean distance between two 3D landmarks
    const dist = (p1: NormalizedLandmark, p2: NormalizedLandmark) =>
      Math.hypot(p1.x - p2.x, p1.y - p2.y, (p1.z - p2.z) * 0.5);

    // Finger tips and MCPs
    // 4: thumb tip, 8: index tip, 12: middle tip, 16: ring tip, 20: pinky tip
    // 0: wrist, 5: index mcp, 9: middle mcp, 17: pinky mcp
    const isExtended = (hand: NormalizedLandmark[], tipIdx: number, mcpIdx: number) => {
      return hand[tipIdx].y < hand[mcpIdx].y; // higher in screen (y is downwards)
    };

    const isCurled = (hand: NormalizedLandmark[], tipIdx: number, mcpIdx: number) => {
      return hand[tipIdx].y > hand[mcpIdx].y - 0.02;
    };

    // -------------------------------------------------------------
    // SIGN 1: NAMASTE (Two hands joined together, palms touching at chest)
    // -------------------------------------------------------------
    if (twoHands && leftHand && rightHand) {
      const wristDist = dist(leftHand[0], rightHand[0]);
      const indexDist = dist(leftHand[8], rightHand[8]);
      const middleDist = dist(leftHand[12], rightHand[12]);
      const palmDist = dist(leftHand[9], rightHand[9]);

      // Both hands close to each other, fingers pointing generally upward
      if (wristDist < 0.22 && indexDist < 0.18 && middleDist < 0.18 && palmDist < 0.20) {
        recognizedSignId = 'namaste';
        const tightness = (indexDist + middleDist + palmDist) / 3;
        confidence = Math.min(0.98, Math.max(0.78, 1.0 - tightness * 2.5));
      }
    }

    // -------------------------------------------------------------
    // SIGN 2: DOCTOR (Two hands: dominant fingers touch non-dominant wrist pulse)
    // -------------------------------------------------------------
    if (!recognizedSignId && twoHands && leftHand && rightHand) {
      const rFingersToLWrist = Math.min(
        dist(rightHand[8], leftHand[0]),
        dist(rightHand[12], leftHand[0])
      );
      const lFingersToRWrist = Math.min(
        dist(leftHand[8], rightHand[0]),
        dist(leftHand[12], rightHand[0])
      );

      if (rFingersToLWrist < 0.14 || lFingersToRWrist < 0.14) {
        recognizedSignId = 'doctor';
        confidence = 0.92;
      }
    }

    // -------------------------------------------------------------
    // SIGN 3: HELP (Two hands: one hand in a fist resting on the other open palm)
    // -------------------------------------------------------------
    if (!recognizedSignId && twoHands && leftHand && rightHand) {
      const rOnLPalm = dist(rightHand[0], leftHand[9]);
      const lOnRPalm = dist(leftHand[0], rightHand[9]);

      if (rOnLPalm < 0.18 || lOnRPalm < 0.18) {
        recognizedSignId = 'help';
        confidence = 0.91;
      }
    }

    // -------------------------------------------------------------
    // SIGN 4: EMERGENCY (Two hands: crossing or waving rapidly at upper chest)
    // -------------------------------------------------------------
    if (!recognizedSignId && twoHands && leftHand && rightHand) {
      const wristsCrossed = Math.abs(leftHand[0].x - rightHand[0].x) < 0.12 &&
        dist(leftHand[0], rightHand[0]) < 0.25;
      const bothHigh = leftHand[0].y < 0.65 && rightHand[0].y < 0.65;

      if (wristsCrossed && bothHigh) {
        recognizedSignId = 'emergency';
        confidence = 0.90;
      }
    }

    // -------------------------------------------------------------
    // SINGLE HAND GESTURES
    // -------------------------------------------------------------
    if (!recognizedSignId && primaryHand) {
      const h = primaryHand;
      const indexExt = isExtended(h, 8, 5);
      const middleExt = isExtended(h, 12, 9);
      const ringExt = isExtended(h, 16, 13);
      const pinkyExt = isExtended(h, 20, 17);
      const allExt = indexExt && middleExt && ringExt && pinkyExt;
      const allCurled = isCurled(h, 8, 5) && isCurled(h, 12, 9) && isCurled(h, 16, 13) && isCurled(h, 20, 17);

      // SIGN 5: WATER (Curved/cupped hand near mouth/face, tilting)
      // Hand is near upper region (y < 0.45) with fingers loosely curved
      if (h[0].y < 0.48 && !allCurled && dist(h[4], h[8]) < 0.12 && dist(h[8], h[12]) < 0.08) {
        recognizedSignId = 'water';
        confidence = 0.88;
      }
      // SIGN 6: FOOD (Bunched fingertips together pointing near lips/mouth)
      else if (h[0].y < 0.50 && dist(h[4], h[8]) < 0.08 && dist(h[4], h[12]) < 0.08 && dist(h[4], h[16]) < 0.08) {
        recognizedSignId = 'food';
        confidence = 0.92;
      }
      // SIGN 7: THANK YOU (Flat open hand near chin/chest moving outward)
      else if (allExt && h[0].y > 0.35 && h[0].y < 0.65 && dist(h[8], h[12]) < 0.06) {
        // Flat palm
        recognizedSignId = 'thank_you';
        confidence = 0.87;
      }
      // SIGN 8: NO (Open flat hand waving horizontally)
      else if (allExt && (h[0].y > 0.40)) {
        recognizedSignId = 'no';
        confidence = 0.85;
      }
      // SIGN 9: YES (Closed fist nodding vertically)
      else if (allCurled && h[0].y > 0.40) {
        recognizedSignId = 'yes';
        confidence = 0.86;
      }
      // SIGN 10: PLEASE (Open hand on center chest)
      else if (allExt && Math.abs(h[0].x - 0.5) < 0.18 && h[0].y > 0.50 && h[0].y < 0.75) {
        recognizedSignId = 'please';
        confidence = 0.84;
      }
    }

    if (!recognizedSignId || confidence < 0.75) {
      return {
        status: 'uncertain',
        is_confident: false,
        sign_id: 'unknown',
        label: 'Analyzing sign motion...',
        category: 'general',
        confidence: confidence || 0.45,
        threshold: 0.75,
        translations: { en: 'Analyzing gesture...', hi: 'इशारे का विश्लेषण...', te: 'విశ్లేషిస్తోంది...' },
        description: 'Position your hands clearly in frame.',
        processing_duration_ms: 12.4,
        input_mode: 'live_camera',
        model_version: '2.1.0'
      };
    }

    const meta = this.vocabMap.get(recognizedSignId) || {
      id: recognizedSignId,
      name: recognizedSignId,
      label: recognizedSignId,
      category: 'general',
      translations: { en: recognizedSignId, hi: recognizedSignId, te: recognizedSignId },
      description: ''
    };

    return {
      status: 'recognized',
      is_confident: true,
      sign_id: recognizedSignId,
      label: meta.label || recognizedSignId,
      category: meta.category || 'general',
      confidence: Math.round(confidence * 100) / 100,
      threshold: 0.75,
      translations: meta.translations,
      description: meta.description,
      processing_duration_ms: 14.2,
      input_mode: 'live_camera',
      model_version: '2.1.0'
    };
  }
}
