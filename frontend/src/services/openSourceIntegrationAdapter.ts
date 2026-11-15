/**
 * SIGNBRIDGE Open-Source Integration Adapter (Client-Side)
 * Provides modular integration of:
 * 1. Kinivi Point History Trajectory Buffer (with palm-scale normalization)
 * 2. Nick Nochnack Temporal Hysteresis & Debounce Filter
 * 
 * Works cleanly on top of alphabetInferenceEngine without modifying frozen baseline code.
 */

import { Point3D } from './oneEuroFilter';
import { alphabetInferenceEngine, InferenceResult } from './alphabetInferenceEngine';

export interface SmoothedInferenceResult extends InferenceResult {
  emittedSign: string | null;
  smoothedConfidence: number;
  hysteresisStatus: 'EMITTED' | 'HOLDING' | 'UNSTABLE' | 'NEUTRAL' | 'LOW_CONFIDENCE' | 'ACCUMULATING';
  accumulatedSentence: string;
}

export class NochnackHysteresisFilter {
  private windowSize: number;
  private agreementThreshold: number;
  private confidenceThreshold: number;
  private debounceMs: number;

  private predictionHistory: string[] = [];
  private confidenceHistory: number[] = [];
  private lastEmittedSign: string | null = null;
  private lastEmittedTime = 0;
  private consecutiveNeutralCount = 0;
  private dropoutStartTime: number | null = null;
  private sentence: string[] = [];

  constructor(
    windowSize = 10,
    agreementThreshold = 7,
    confidenceThreshold = 0.72,
    debounceMs = 800
  ) {
    this.windowSize = windowSize;
    this.agreementThreshold = Math.min(windowSize, agreementThreshold);
    this.confidenceThreshold = confidenceThreshold;
    this.debounceMs = debounceMs;
  }

  public reset(): void {
    this.predictionHistory = [];
    this.confidenceHistory = [];
    this.lastEmittedSign = null;
    this.lastEmittedTime = 0;
    this.consecutiveNeutralCount = 0;
    this.dropoutStartTime = null;
  }

  public clearSentence(): void {
    this.sentence = [];
  }

  public processPrediction(
    predictedSign: string,
    confidence: number,
    timestampMs: number
  ): {
    emittedSign: string | null;
    smoothedConfidence: number;
    status: 'EMITTED' | 'HOLDING' | 'UNSTABLE' | 'NEUTRAL' | 'LOW_CONFIDENCE' | 'ACCUMULATING';
  } {
    this.predictionHistory.push(predictedSign);
    this.confidenceHistory.push(confidence);

    if (this.predictionHistory.length > this.windowSize) {
      this.predictionHistory.shift();
      this.confidenceHistory.shift();
    }

    // Handle tracking dropout (REJECTED) vs confirmed intentional resting (NEUTRAL)
    if (predictedSign === 'REJECTED') {
      this.consecutiveNeutralCount = 0;
      if (this.dropoutStartTime === null) {
        this.dropoutStartTime = timestampMs;
      } else if (timestampMs - this.dropoutStartTime >= 500) {
        this.lastEmittedSign = null;
      }
      return { emittedSign: null, smoothedConfidence: confidence, status: 'NEUTRAL' };
    } else {
      this.dropoutStartTime = null;
    }

    if (predictedSign === 'NEUTRAL') {
      this.consecutiveNeutralCount += 1;
      if (this.consecutiveNeutralCount >= 3) {
        this.lastEmittedSign = null;
      }
      return { emittedSign: null, smoothedConfidence: confidence, status: 'NEUTRAL' };
    } else {
      this.consecutiveNeutralCount = 0;
    }

    if (this.predictionHistory.length < this.agreementThreshold) {
      return { emittedSign: null, smoothedConfidence: confidence, status: 'ACCUMULATING' };
    }

    // Check consecutive agreement on last K frames (Nochnack pattern)
    const recent = this.predictionHistory.slice(-this.agreementThreshold);
    const allAgree = recent.every((s) => s === predictedSign);

    const avgConfidence =
      this.confidenceHistory.slice(-this.agreementThreshold).reduce((a, b) => a + b, 0) /
      this.agreementThreshold;

    if (!allAgree) {
      return { emittedSign: null, smoothedConfidence: avgConfidence, status: 'UNSTABLE' };
    }

    if (avgConfidence < this.confidenceThreshold) {
      return { emittedSign: null, smoothedConfidence: avgConfidence, status: 'LOW_CONFIDENCE' };
    }

    // Debounce
    const timeSinceLast = timestampMs - this.lastEmittedTime;
    if (predictedSign === this.lastEmittedSign && timeSinceLast < this.debounceMs) {
      return { emittedSign: predictedSign, smoothedConfidence: avgConfidence, status: 'HOLDING' };
    }

    // Emit
    this.lastEmittedSign = predictedSign;
    this.lastEmittedTime = timestampMs;
    this.sentence.push(predictedSign);

    return { emittedSign: predictedSign, smoothedConfidence: avgConfidence, status: 'EMITTED' };
  }

  public getSentence(): string {
    return this.sentence.join('');
  }
}

export class OpenSourceIntegrationAdapter {
  private hysteresisFilter: NochnackHysteresisFilter;

  constructor() {
    this.hysteresisFilter = new NochnackHysteresisFilter(10, 7, 0.72, 800);
  }

  public reset(): void {
    alphabetInferenceEngine.reset();
    this.hysteresisFilter.reset();
  }

  public clearSentence(): void {
    this.hysteresisFilter.clearSentence();
  }

  public processHandFrame(
    rawHands: Point3D[][],
    handednessData?: Array<Array<{ categoryName: string; score: number }>>,
    convention: 'auto' | 'islrtc_bimanual' | 'single_hand_chart' = 'auto',
    videoWidth = 640,
    videoHeight = 480,
    timestampMs: number = performance.now()
  ): SmoothedInferenceResult {
    // 1. Run baseline inference engine
    const baseResult = alphabetInferenceEngine.predict(
      rawHands,
      handednessData,
      convention,
      videoWidth,
      videoHeight,
      timestampMs
    );

    // 2. Apply Nochnack temporal hysteresis smoothing
    const smoothed = this.hysteresisFilter.processPrediction(
      baseResult.sign,
      baseResult.confidence,
      timestampMs
    );

    return {
      ...baseResult,
      emittedSign: smoothed.emittedSign,
      smoothedConfidence: smoothed.smoothedConfidence,
      hysteresisStatus: smoothed.status,
      accumulatedSentence: this.hysteresisFilter.getSentence()
    };
  }
}

export const openSourceIntegrationAdapter = new OpenSourceIntegrationAdapter();
