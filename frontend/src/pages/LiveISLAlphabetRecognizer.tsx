import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Camera,
  CameraOff,
  Volume2,
  VolumeX,
  RotateCcw,
  Delete,
  Space,
  Copy,
  Check,
  AlertCircle,
  Activity,
  Layers,
  ChevronDown,
  ChevronUp,
  Sliders,
  CheckCircle2,
  Gauge,
  Terminal,
  Play,
  FlaskConical,
  TrendingDown,
  Sparkles,
  BookOpen,
  X,
  Hand,
  Settings2,
  HelpCircle,
  Download
} from 'lucide-react';
import { webMediaPipe } from '../services/webMediaPipeService';
import {
  alphabetInferenceEngine,
  AlphabetConvention,
  InferenceResult,
  LandmarkPoint
} from '../services/alphabetInferenceEngine';
import { PersistentHandTrack } from '../services/persistentHandTracker';
import { TrackingQualityReport } from '../services/trackingQualityEngine';
import { speechService } from '../services/speechService';
import { HAND_CONNECTIONS } from '../services/cameraLandmarkDetector';
import { TrackingBenchmarkLab, ComparativeBenchmarkResult } from '../services/trackingBenchmarkLab';
import { titanInferenceEngine, TitanInferenceResult } from '../services/titanInferenceEngine';
import { titanASLInferenceEngine, TitanASLResult } from '../services/titanASLInferenceEngine';

// Comprehensive reference guide for supported alphabet signs
const ALPHABET_REFERENCE: Record<string, { desc: string; standard: string; category: 'single' | 'dual' | 'dynamic' }> = {
  A: { desc: 'Make a fist with fingers closed and thumb resting straight along the side of index finger.', standard: 'Unified', category: 'single' },
  B: { desc: 'Hold four fingers straight up together with thumb folded flat across palm.', standard: 'Unified', category: 'single' },
  C: { desc: 'Curve all fingers and thumb to form an open "C" shape facing inward.', standard: 'Single-Hand', category: 'single' },
  D: { desc: 'Point index finger straight up while other fingers and thumb touch to form an "O".', standard: 'Unified', category: 'single' },
  E: { desc: 'Curl all four fingertips tightly down to touch the thumb tucked underneath.', standard: 'Unified', category: 'single' },
  F: { desc: 'Touch index tip to thumb tip to make a circle; keep middle, ring, and pinky straight up.', standard: 'Unified', category: 'single' },
  G: { desc: 'Point index finger horizontally forward with thumb extended parallel to it.', standard: 'Unified', category: 'single' },
  H: { desc: 'Extend index and middle fingers together horizontally with thumb tucked in.', standard: 'Unified', category: 'single' },
  I: { desc: 'Make a fist and extend only the pinky finger straight up.', standard: 'Unified', category: 'single' },
  J: { desc: 'Start with the "I" sign (pinky up) and draw a downward "J" curve in the air.', standard: 'Dynamic', category: 'dynamic' },
  K: { desc: 'Index points straight up, middle finger angles forward, thumb touches middle knuckle.', standard: 'Unified', category: 'single' },
  L: { desc: 'Extend thumb and index finger at a 90° right angle to form an "L"; curl other fingers.', standard: 'Single-Hand', category: 'single' },
  M: { desc: 'ISLRTC: Place dominant 3 fingers (index, middle, ring) facing down onto non-dominant palm.', standard: 'ISLRTC Dual', category: 'dual' },
  N: { desc: 'ISLRTC: Place dominant 2 fingers (index, middle) facing down onto non-dominant palm.', standard: 'ISLRTC Dual', category: 'dual' },
  O: { desc: 'Touch all four fingertips to thumb tip to form an open circular "O" shape.', standard: 'Unified', category: 'single' },
  P: { desc: 'ISLRTC: Hold base index vertical; place dominant thumb-index loop against top knuckle.', standard: 'ISLRTC Dual', category: 'dual' },
  Q: { desc: 'ISLRTC: Form a circle with base hand; place dominant index at the bottom edge.', standard: 'ISLRTC Dual', category: 'dual' },
  R: { desc: 'Cross the middle finger over the back of the extended index finger.', standard: 'Single-Hand', category: 'single' },
  S: { desc: 'ISLRTC: Hook dominant and base pinky fingers together tightly.', standard: 'ISLRTC Dual', category: 'dual' },
  T: { desc: 'ISLRTC: Touch dominant index finger to the outer side base of non-dominant index.', standard: 'ISLRTC Dual', category: 'dual' },
  U: { desc: 'Extend index and middle fingers straight up together; keep thumb folded across ring finger.', standard: 'Unified', category: 'single' },
  V: { desc: 'Extend index and middle fingers straight up spread apart in a clear "V" peace sign.', standard: 'Unified', category: 'single' },
  W: { desc: 'Extend index, middle, and ring fingers straight up spread apart in a "W" shape.', standard: 'Unified', category: 'single' },
  X: { desc: 'Curl index finger into a hook shape like a pirate hook; fold other fingers in.', standard: 'Single-Hand', category: 'single' },
  Y: { desc: 'Extend thumb and pinky finger outward while folding middle three fingers down (shaka sign).', standard: 'Unified', category: 'single' },
  Z: { desc: 'Dynamic: Trace a 3-stroke "Z" in the air using your extended index finger.', standard: 'Dynamic', category: 'dynamic' }
};

interface BenchmarkMetrics {
  targetLetter: string;
  totalFrames: number;
  avgFps: number;
  avgLatencyMs: number;
  continuityRate: number;
  accuracyRate: number;
  avgJitter: number;
}

export interface AcceptanceTestRecord {
  id: string;
  timestamp: string;
  signerId: string;
  targetLetter: string;
  predictedLetter: string;
  confidence: number;
  trackingQualityScore: number;
  fps: number;
  latencyMs: number;
  accepted: boolean;
  lighting: string;
  distance: string;
}

export function LiveISLAlphabetRecognizer() {
  // Telemetry & Camera State
  const [isCameraActive, setIsCameraActive] = useState<boolean>(false);
  const [isInitializing, setIsInitializing] = useState<boolean>(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [facingMode, setFacingMode] = useState<'user' | 'environment'>('user');
  const [fps, setFps] = useState<number>(0);
  const [latencyMs, setLatencyMs] = useState<number>(0);
  const [trackedHandCount, setTrackedHandCount] = useState<number>(0);

  // UI Modal & Drawer Toggles
  const [showGuideModal, setShowGuideModal] = useState<boolean>(false);
  const [guideFilter, setGuideFilter] = useState<'all' | 'single' | 'dual'>('all');
  const [showDiagnostics, setShowDiagnostics] = useState<boolean>(false);

  // Benchmarking State
  const [benchmarkReport, setBenchmarkReport] = useState<ComparativeBenchmarkResult | null>(null);
  const [isRunningBenchmark, setIsRunningBenchmark] = useState<boolean>(false);
  const [isCalibrating, setIsCalibrating] = useState<boolean>(false);
  const [calibrationLetter, setCalibrationLetter] = useState<string>('L');
  const [calibrationResults, setCalibrationResults] = useState<BenchmarkMetrics | null>(null);
  const calibrationFramesRef = useRef<Array<{ predicted: string; latency: number; jitter: number; isTracked: boolean }>>([]);
  const calibrationStartTimeRef = useRef<number>(0);

  // Real-Device Acceptance Testing Mode State
  const [acceptanceTesterId, setAcceptanceTesterId] = useState<string>('Signer_Alpha');
  const [acceptanceLighting, setAcceptanceLighting] = useState<string>('Natural Daylight (450 lux)');
  const [acceptanceDistance, setAcceptanceDistance] = useState<string>('Arm Length (~0.65m)');
  const [acceptanceTargetLetter, setAcceptanceTargetLetter] = useState<string>('A');
  const [acceptanceLog, setAcceptanceLog] = useState<AcceptanceTestRecord[]>([]);

  // TITAN Research Engine State
  const [showTitanMonitor, setShowTitanMonitor] = useState<boolean>(false);
  const [titanResult, setTitanResult] = useState<TitanInferenceResult | null>(null);

  // TITAN ASL Visual Reference Engine State
  const [showASLReferenceDiagnostics, setShowASLReferenceDiagnostics] = useState<boolean>(false);
  const [selectedASLReferenceLetter, setSelectedASLReferenceLetter] = useState<string>('A');
  const [titanASLResult, setTitanASLResult] = useState<TitanASLResult | null>(null);
  const [showFullASLChartModal, setShowFullASLChartModal] = useState<boolean>(false);

  // Convention Selection
  const [convention, setConvention] = useState<AlphabetConvention>('auto');

  // Multi-Stage Tracking & Quality Telemetry
  const [trackingReport, setTrackingReport] = useState<TrackingQualityReport>({
    status: 'SEARCHING',
    qualityScore: 0,
    stabilityScore: 0,
    anatomicalIntegrityScore: 0,
    framing: 'TOO_FAR',
    jitterVariance: 0,
    trackedHandsCount: 0,
    continuityFrames: 0,
    isReliableForInference: false,
    recommendation: 'Position hand within camera frame.',
    perFingerTelemetry: [],
    identitySwitchesCount: 0,
    reacquisitionsCount: 0
  });

  // Neural Recognition & Stability Lock
  const [currentResult, setCurrentResult] = useState<InferenceResult | null>(null);
  const [stableLockedLetter, setStableLockedLetter] = useState<string | null>(null);
  const [stabilityProgress, setStabilityProgress] = useState<number>(0);
  const [statusMessage, setStatusMessage] = useState<string>('Ready to start camera');

  // Text Composition Terminal State
  const [textBuffer, setTextBuffer] = useState<string>('');
  const [isMuted, setIsMuted] = useState<boolean>(false);
  const [copied, setCopied] = useState<boolean>(false);

  // DOM & Animation Refs
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Real-time loop throttling & stability refs
  const lastVideoTimeRef = useRef<number>(-1);
  const lastFrameTimestampRef = useRef<number>(performance.now());
  const frameCounterRef = useRef<number>(0);
  const candidateLetterRef = useRef<string | null>(null);
  const candidateStreakRef = useRef<number>(0);
  const lastAcceptedLetterRef = useRef<string | null>(null);
  const neutralFramesRef = useRef<number>(0);

  const STABILITY_REQUIRED_FRAMES = 7;
  const CONFIDENCE_MINIMUM = 0.72;

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  const stopCamera = useCallback(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    alphabetInferenceEngine.reset();
    setIsCameraActive(false);
    setIsInitializing(false);
    setTrackedHandCount(0);
    setFps(0);
    setLatencyMs(0);
    setCurrentResult(null);
    setStableLockedLetter(null);
    setStabilityProgress(0);
    setStatusMessage('Camera stopped');
  }, []);

  const startCamera = async () => {
    setCameraError(null);
    setIsInitializing(true);
    setStatusMessage('Connecting AI vision model...');

    try {
      const mpReady = await webMediaPipe.init();
      if (!mpReady) {
        throw new Error('MediaPipe HandLandmarker initialization failed. WebAssembly or GPU delegate unavailable.');
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: facingMode,
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30 }
        },
        audio: false
      });
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          if (videoRef.current) {
            videoRef.current.play().catch((err) => console.warn('Autoplay error:', err));
            setIsCameraActive(true);
            setIsInitializing(false);
            setStatusMessage('Camera active — show hand sign');
            runVisionPipeline();
          }
        };
      }
    } catch (err: unknown) {
      console.error('Camera startup exception:', err);
      setIsInitializing(false);
      setIsCameraActive(false);

      if (err instanceof DOMException && (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError')) {
        setCameraError('Camera access was blocked by your browser. Please tap the lock icon in your URL bar and allow camera access.');
      } else if (err instanceof DOMException && err.name === 'NotFoundError') {
        setCameraError('No camera found on this device.');
      } else {
        setCameraError(err instanceof Error ? err.message : 'Unknown camera initialization error.');
      }
    }
  };

  const toggleFacingMode = () => {
    const next = facingMode === 'user' ? 'environment' : 'user';
    setFacingMode(next);
    if (isCameraActive) {
      stopCamera();
      setTimeout(startCamera, 150);
    }
  };

  // Modern sleek skeletal rendering
  const renderPrecisionViewfinder = (
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    tracks: PersistentHandTrack[],
    isLocked: boolean
  ) => {
    ctx.clearRect(0, 0, width, height);

    if (!tracks || tracks.length === 0) return;

    tracks.forEach((track) => {
      const lm = track.smoothedLandmarks;
      const isDominant = track.role === 'Dominant';
      const isCoasting = track.isCoasting;

      // Glow & stroke colors
      const boneStroke = isLocked
        ? 'rgba(52, 211, 153, 0.95)' // emerald
        : isCoasting
        ? 'rgba(251, 191, 36, 0.85)' // amber
        : isDominant
        ? 'rgba(96, 165, 250, 0.95)' // bright cyan-blue
        : 'rgba(167, 139, 250, 0.95)'; // violet

      const jointFill = isLocked ? '#10b981' : isCoasting ? '#f59e0b' : isDominant ? '#38bdf8' : '#a855f7';

      // Palm Membrane Polygon (clean subtle fill)
      ctx.beginPath();
      ctx.moveTo(lm[0].x * width, lm[0].y * height);
      ctx.lineTo(lm[1].x * width, lm[1].y * height);
      ctx.lineTo(lm[2].x * width, lm[2].y * height);
      ctx.lineTo(lm[5].x * width, lm[5].y * height);
      ctx.lineTo(lm[9].x * width, lm[9].y * height);
      ctx.lineTo(lm[13].x * width, lm[13].y * height);
      ctx.lineTo(lm[17].x * width, lm[17].y * height);
      ctx.closePath();
      ctx.fillStyle = isDominant ? 'rgba(56, 189, 248, 0.12)' : 'rgba(168, 85, 247, 0.12)';
      ctx.fill();

      // Finger Bone Chains with smooth anti-aliased lines
      for (const [start, end] of HAND_CONNECTIONS) {
        const p1 = lm[start];
        const p2 = lm[end];
        if (p1 && p2) {
          const isMetacarpal = [0, 1, 5, 9, 13, 17].includes(start) && [1, 2, 5, 9, 13, 17].includes(end);
          ctx.lineWidth = isMetacarpal ? 3.5 : 2.4;
          ctx.strokeStyle = boneStroke;
          ctx.lineCap = 'round';
          ctx.lineJoin = 'round';

          if (isCoasting) {
            ctx.setLineDash([4, 4]);
          } else {
            ctx.setLineDash([]);
          }

          ctx.beginPath();
          ctx.moveTo(p1.x * width, p1.y * height);
          ctx.lineTo(p2.x * width, p2.y * height);
          ctx.stroke();
        }
      }
      ctx.setLineDash([]);

      // Joints & Fingertips
      for (let i = 0; i < lm.length; i++) {
        const p = lm[i];
        const isTip = [4, 8, 12, 16, 20].includes(i);
        const radius = isTip ? 5.5 : 3.0;

        ctx.beginPath();
        ctx.arc(p.x * width, p.y * height, radius, 0, 2 * Math.PI);
        ctx.fillStyle = isTip ? '#ffffff' : jointFill;
        ctx.fill();
        ctx.strokeStyle = '#0f172a';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    });
  };

  // Real-Time Vision Pipeline Loop
  const runVisionPipeline = () => {
    if (!videoRef.current || !canvasRef.current || videoRef.current.readyState < 2) {
      animationFrameRef.current = requestAnimationFrame(runVisionPipeline);
      return;
    }

    const t0 = performance.now();
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    // Duplicate frame skipping to conserve mobile battery
    if (video.currentTime === lastVideoTimeRef.current) {
      animationFrameRef.current = requestAnimationFrame(runVisionPipeline);
      return;
    }
    lastVideoTimeRef.current = video.currentTime;

    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
    }

    // FPS computation
    frameCounterRef.current++;
    if (frameCounterRef.current % 15 === 0) {
      const dt = t0 - lastFrameTimestampRef.current;
      if (dt > 0) setFps(Math.round((15 / dt) * 1000));
      lastFrameTimestampRef.current = t0;
    }

    // MediaPipe Hand Detection
    const mpResult = webMediaPipe.detect(video, t0);
    const rawHands = (mpResult?.landmarks as LandmarkPoint[][]) || [];
    setTrackedHandCount(rawHands.length);

    let result: InferenceResult = {
      sign: 'NEUTRAL',
      confidence: 0,
      isDynamic: false,
      handCount: rawHands.length,
      convention: 'ISLRTC (2-Hand)',
      allProbabilities: {},
      tracks: [],
      qualityReport: {
        status: 'SEARCHING',
        qualityScore: 0,
        stabilityScore: 0,
        anatomicalIntegrityScore: 0,
        framing: 'TOO_FAR',
        jitterVariance: 0,
        trackedHandsCount: 0,
        continuityFrames: 0,
        isReliableForInference: false,
        recommendation: 'Show your hand to the camera',
        perFingerTelemetry: [],
        identitySwitchesCount: 0,
        reacquisitionsCount: 0
      }
    };

    if (rawHands.length > 0) {
      result = alphabetInferenceEngine.predict(
        rawHands,
        mpResult?.handedness,
        convention,
        canvas.width,
        canvas.height,
        t0
      );
    }
    setCurrentResult(result);
    setTrackingReport(result.qualityReport);

    // Run independent TITAN backup engine if diagnostics monitor enabled
    if (rawHands.length > 0 && showTitanMonitor) {
      try {
        const tTitan = titanInferenceEngine.predict(rawHands[0]);
        setTitanResult(tTitan);
      } catch (err) {
        console.warn('TITAN engine inference error:', err);
      }
    } else if (rawHands.length === 0) {
      setTitanResult(null);
    }

    // Run independent TITAN ASL reference engine if diagnostics monitor enabled
    if (rawHands.length > 0 && showASLReferenceDiagnostics) {
      try {
        const tASL = titanASLInferenceEngine.predict(rawHands[0]);
        setTitanASLResult(tASL);
      } catch (err) {
        console.warn('TITAN ASL engine inference error:', err);
      }
    } else if (rawHands.length === 0) {
      setTitanASLResult(null);
    }

    const t1 = performance.now();
    const procLatency = Number((t1 - t0).toFixed(1));
    setLatencyMs(procLatency);

    // Stability filtering & automatic letter acceptance
    const sign = result.sign;
    const conf = result.confidence;
    let isLockedNow = false;

    if (sign === 'NEUTRAL' || conf < CONFIDENCE_MINIMUM || rawHands.length === 0 || !result.qualityReport.isReliableForInference) {
      neutralFramesRef.current++;
      if (neutralFramesRef.current >= 5) {
        lastAcceptedLetterRef.current = null;
        setStableLockedLetter(null);
      }
      candidateStreakRef.current = 0;
      setStabilityProgress(0);
      setStatusMessage(rawHands.length === 0 ? 'Position hand in front of camera' : result.qualityReport.recommendation);
    } else {
      neutralFramesRef.current = 0;

      if (sign === candidateLetterRef.current) {
        candidateStreakRef.current++;
      } else {
        candidateLetterRef.current = sign;
        candidateStreakRef.current = 1;
      }

      if (sign === lastAcceptedLetterRef.current) {
        setStabilityProgress(100);
        setStatusMessage(`Holding "${sign}"`);
        isLockedNow = true;
      } else {
        const progress = Math.min(100, Math.round((candidateStreakRef.current / STABILITY_REQUIRED_FRAMES) * 100));
        setStabilityProgress(progress);
        setStatusMessage(`Recognizing "${sign}"...`);

        if (candidateStreakRef.current >= STABILITY_REQUIRED_FRAMES) {
          isLockedNow = true;
          setStableLockedLetter(sign);
          commitLetter(sign);
          lastAcceptedLetterRef.current = sign;
          candidateStreakRef.current = 0;
          setStabilityProgress(100);
          setStatusMessage(`Added "${sign}" to text!`);
        }
      }
    }

    // Calibration recording if active
    if (isCalibrating) {
      calibrationFramesRef.current.push({
        predicted: sign,
        latency: procLatency,
        jitter: result.qualityReport.jitterVariance,
        isTracked: rawHands.length > 0
      });

      if (t0 - calibrationStartTimeRef.current >= 10000) {
        concludeCalibration();
      }
    }

    // Viewfinder Canvas Render
    if (ctx) {
      renderPrecisionViewfinder(
        ctx,
        canvas.width,
        canvas.height,
        result.tracks,
        isLockedNow
      );
    }

    animationFrameRef.current = requestAnimationFrame(runVisionPipeline);
  };

  const commitLetter = (char: string) => {
    setTextBuffer((prev) => prev + char);
    if (!isMuted) {
      speechService.speak(char, 'en', 1.1);
    }
  };

  const handleManualAddLetter = () => {
    if (stableLockedLetter) {
      commitLetter(stableLockedLetter);
    } else if (currentResult && currentResult.sign !== 'NEUTRAL' && currentResult.confidence >= CONFIDENCE_MINIMUM) {
      commitLetter(currentResult.sign);
    }
  };

  const handleSpace = () => {
    setTextBuffer((prev) => (prev.endsWith(' ') ? prev : prev + ' '));
    lastAcceptedLetterRef.current = null;
  };

  const handleBackspace = () => {
    setTextBuffer((prev) => prev.slice(0, -1));
    lastAcceptedLetterRef.current = null;
  };

  const handleClear = () => {
    setTextBuffer('');
    lastAcceptedLetterRef.current = null;
    candidateStreakRef.current = 0;
    setStabilityProgress(0);
  };

  const handleSpeakText = () => {
    if (!textBuffer.trim()) return;
    speechService.speak(textBuffer, 'en', 0.95);
  };

  const handleCopy = () => {
    if (!textBuffer) return;
    navigator.clipboard.writeText(textBuffer);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Calibration Protocol
  const startCalibrationSession = (target: string) => {
    setCalibrationLetter(target);
    calibrationFramesRef.current = [];
    calibrationStartTimeRef.current = performance.now();
    setIsCalibrating(true);
  };

  const concludeCalibration = () => {
    setIsCalibrating(false);
    const frames = calibrationFramesRef.current;
    if (frames.length === 0) return;

    const correct = frames.filter((f) => f.predicted === calibrationLetter).length;
    const tracked = frames.filter((f) => f.isTracked).length;
    const total = frames.length;
    const avgLat = frames.reduce((a, b) => a + b.latency, 0) / total;
    const avgJit = frames.reduce((a, b) => a + b.jitter, 0) / total;

    setCalibrationResults({
      targetLetter: calibrationLetter,
      totalFrames: total,
      avgFps: Math.round(total / 10),
      avgLatencyMs: Number(avgLat.toFixed(1)),
      continuityRate: Math.round((tracked / total) * 100),
      accuracyRate: Math.round((correct / total) * 100),
      avgJitter: Number(avgJit.toFixed(5))
    });
  };

  const runBenchmarkSuite = () => {
    setIsRunningBenchmark(true);
    setTimeout(() => {
      const res = TrackingBenchmarkLab.runComparativeBenchmark();
      setBenchmarkReport(res);
      setIsRunningBenchmark(false);
    }, 50);
  };

  const logAcceptanceTrial = () => {
    if (!currentResult) return;
    const isMatch = currentResult.sign === acceptanceTargetLetter;
    const isConfHigh = currentResult.confidence >= CONFIDENCE_MINIMUM;
    const isAccepted = isMatch && isConfHigh && trackingReport.isReliableForInference;

    const newRecord: AcceptanceTestRecord = {
      id: `TRIAL_${Date.now()}`,
      timestamp: new Date().toISOString(),
      signerId: acceptanceTesterId,
      targetLetter: acceptanceTargetLetter,
      predictedLetter: currentResult.sign,
      confidence: Number((currentResult.confidence * 100).toFixed(1)),
      trackingQualityScore: trackingReport.qualityScore,
      fps,
      latencyMs,
      accepted: isAccepted,
      lighting: acceptanceLighting,
      distance: acceptanceDistance
    };

    setAcceptanceLog((prev) => [newRecord, ...prev]);

    // Advance to next letter in sequence (A -> B -> C -> D ...)
    const supported = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'];
    const curIdx = supported.indexOf(acceptanceTargetLetter);
    if (curIdx >= 0 && curIdx < supported.length - 1) {
      setAcceptanceTargetLetter(supported[curIdx + 1]);
    }
  };

  const downloadAcceptanceReport = () => {
    if (acceptanceLog.length === 0) return;
    const reportPayload = {
      report_title: 'SIGNBRIDGE Real-Device Acceptance Testing Session',
      generated_at: new Date().toISOString(),
      signer_id: acceptanceTesterId,
      environment_conditions: {
        lighting: acceptanceLighting,
        distance: acceptanceDistance
      },
      total_trials: acceptanceLog.length,
      accepted_trials: acceptanceLog.filter((t) => t.accepted).length,
      accuracy_rate_pct: Number(
        ((acceptanceLog.filter((t) => t.accepted).length / acceptanceLog.length) * 100).toFixed(1)
      ),
      trials: acceptanceLog
    };

    const blob = new Blob([JSON.stringify(reportPayload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `acceptance_test_${acceptanceTesterId}_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const activeSign =
    currentResult?.sign !== 'NEUTRAL' && (currentResult?.confidence ?? 0) >= CONFIDENCE_MINIMUM
      ? currentResult?.sign
      : null;

  const confidencePct = Math.round((currentResult?.confidence ?? 0) * 100);

  // Filtered reference guide
  const filteredGuide = Object.entries(ALPHABET_REFERENCE).filter(([_, info]) => {
    if (guideFilter === 'single') return info.category === 'single';
    if (guideFilter === 'dual') return info.category === 'dual';
    return true;
  });

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-indigo-500 selection:text-white">
      {/* 1. Header Navigation Bar */}
      <header className="sticky top-0 z-40 bg-slate-900/90 backdrop-blur-md border-b border-slate-800/80 px-4 py-3">
        <div className="max-w-6xl mx-auto flex items-center justify-between gap-3">
          {/* Logo & Brand */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-cyan-400 flex items-center justify-center text-white shadow-md shadow-indigo-500/20">
              <Hand className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-base tracking-tight text-white">SIGNBRIDGE</span>
                <span className={`text-[11px] px-2 py-0.5 rounded-full font-medium flex items-center gap-1.5 ${
                  isCameraActive 
                    ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' 
                    : 'bg-slate-800 text-slate-400 border border-slate-700/60'
                }`}>
                  <span className={`w-1.5 h-1.5 rounded-full ${isCameraActive ? 'bg-emerald-400 animate-pulse' : 'bg-slate-500'}`} />
                  {isCameraActive ? 'Live Camera' : 'Standby'}
                </span>
              </div>
              <p className="text-xs text-slate-400 hidden sm:block">Real-Time Indian Sign Language Recognition</p>
            </div>
          </div>

          {/* Quick Header Actions */}
          <div className="flex items-center gap-2">
            {/* Sign Language Guide Button */}
            <button
              onClick={() => setShowGuideModal(true)}
              className="px-3 py-1.5 rounded-xl border border-slate-700/80 bg-slate-800/80 hover:bg-slate-700 text-slate-200 text-xs font-medium flex items-center gap-1.5 transition-all shadow-sm active:scale-95 cursor-pointer"
            >
              <BookOpen className="w-4 h-4 text-cyan-400" />
              <span className="hidden sm:inline">Sign Guide</span>
            </button>

            {/* Audio Voice Toggle */}
            <button
              onClick={() => {
                const n = !isMuted;
                setIsMuted(n);
                speechService.setMuted(n);
              }}
              className={`px-3 py-1.5 rounded-xl border text-xs font-medium flex items-center gap-1.5 transition-all shadow-sm active:scale-95 cursor-pointer ${
                isMuted
                  ? 'border-rose-900/40 bg-rose-950/30 text-rose-300'
                  : 'border-slate-700/80 bg-slate-800/80 hover:bg-slate-700 text-slate-200'
              }`}
              title={isMuted ? 'Unmute speech' : 'Mute speech'}
            >
              {isMuted ? <VolumeX className="w-4 h-4 text-rose-400" /> : <Volume2 className="w-4 h-4 text-emerald-400" />}
              <span className="hidden sm:inline">{isMuted ? 'Muted' : 'Voice'}</span>
            </button>

            {/* Diagnostics / Science Fair Lab Toggle */}
            <button
              onClick={() => setShowDiagnostics(!showDiagnostics)}
              className={`p-2 rounded-xl border text-xs font-medium transition-all shadow-sm active:scale-95 cursor-pointer ${
                showDiagnostics
                  ? 'border-indigo-500/60 bg-indigo-500/20 text-indigo-300'
                  : 'border-slate-700/80 bg-slate-800/80 hover:bg-slate-700 text-slate-400 hover:text-slate-200'
              }`}
              title="Diagnostics & Benchmarks"
            >
              <Settings2 className="w-4 h-4" />
            </button>
          </div>
        </div>
      </header>

      {/* 2. Main Content Container */}
      <main className="flex-1 max-w-6xl w-full mx-auto p-4 sm:p-6 space-y-6">
        {/* Camera Hardware / Permission Error Notice */}
        {cameraError && (
          <div className="p-4 rounded-2xl border border-rose-800/60 bg-rose-950/40 text-rose-200 text-sm flex items-start gap-3 shadow-lg">
            <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <p className="font-semibold text-rose-100">Camera Access Notice</p>
              <p className="text-xs text-rose-300/90 leading-relaxed">{cameraError}</p>
            </div>
          </div>
        )}

        {/* Primary Viewport & Recognition Split Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Left Column: Camera Viewport (7 Cols) */}
          <div className="lg:col-span-7 space-y-3">
            <div className="relative aspect-[4/3] sm:aspect-video w-full rounded-3xl border border-slate-800/90 bg-slate-900 overflow-hidden shadow-2xl flex items-center justify-center">
              {/* Video Element */}
              <video
                ref={videoRef}
                playsInline
                muted
                autoPlay
                className={`w-full h-full object-cover transform -scale-x-100 ${
                  isCameraActive ? 'opacity-100' : 'opacity-0'
                }`}
              />

              {/* Landmark Canvas Overlay */}
              <canvas
                ref={canvasRef}
                className={`absolute inset-0 w-full h-full object-cover transform -scale-x-100 pointer-events-none ${
                  isCameraActive ? 'opacity-100' : 'opacity-0'
                }`}
              />

              {/* Viewport Inactive Welcome Hero State */}
              {!isCameraActive && (
                <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center space-y-5 bg-gradient-to-br from-slate-900 via-slate-950 to-slate-900">
                  <div className="w-16 h-16 rounded-2xl bg-indigo-600/10 border border-indigo-500/30 flex items-center justify-center text-indigo-400 shadow-inner">
                    <Camera className="w-8 h-8" />
                  </div>

                  <div className="space-y-2 max-w-md">
                    <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-white">
                      Real-Time Sign Language
                    </h2>
                    <p className="text-xs sm:text-sm text-slate-400 leading-relaxed">
                      Show your hand signs to the camera. The AI tracks 21 joints, recognizes letters, and speaks your sentences.
                    </p>
                  </div>

                  <button
                    onClick={startCamera}
                    disabled={isInitializing}
                    className="px-7 py-3 rounded-full bg-gradient-to-r from-indigo-500 to-cyan-500 hover:from-indigo-600 hover:to-cyan-600 text-white font-semibold text-sm shadow-lg shadow-indigo-500/25 active:scale-95 transition-all flex items-center gap-2 cursor-pointer disabled:opacity-50"
                  >
                    {isInitializing ? (
                      <>
                        <Activity className="w-4 h-4 animate-spin" />
                        <span>Initializing AI Engine...</span>
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4 fill-white" />
                        <span>Start Camera</span>
                      </>
                    )}
                  </button>

                  {/* 3 Quick Tips */}
                  <div className="grid grid-cols-3 gap-2 pt-2 border-t border-slate-800/80 max-w-sm w-full text-slate-400 text-[11px]">
                    <div className="text-center p-1">
                      <span className="block font-medium text-slate-300">1-2 Feet</span>
                      <span>Distance</span>
                    </div>
                    <div className="text-center p-1 border-x border-slate-800/80">
                      <span className="block font-medium text-slate-300">Clear Light</span>
                      <span>Environment</span>
                    </div>
                    <div className="text-center p-1">
                      <span className="block font-medium text-slate-300">Hold Steady</span>
                      <span>To Auto-Type</span>
                    </div>
                  </div>
                </div>
              )}

              {/* Viewport Top Floating Controls (Active State) */}
              {isCameraActive && (
                <div className="absolute top-3 left-3 right-3 flex items-center justify-between text-xs pointer-events-none select-none">
                  <div className="flex items-center gap-2 bg-slate-900/80 backdrop-blur-md px-3 py-1.5 rounded-full border border-slate-700/60 shadow-md text-slate-300">
                    <span className="text-emerald-400 font-semibold">{fps} FPS</span>
                    <span className="text-slate-600">•</span>
                    <span className="text-cyan-400">{latencyMs}ms</span>
                    <span className="text-slate-600">•</span>
                    <span className="text-slate-300">{trackedHandCount} {trackedHandCount === 1 ? 'Hand' : 'Hands'}</span>
                  </div>

                  <div className="pointer-events-auto flex items-center gap-2">
                    {/* Flip Front/Rear Camera (Vital for Mobile) */}
                    <button
                      onClick={toggleFacingMode}
                      className="p-2 rounded-full bg-slate-900/80 hover:bg-slate-800 backdrop-blur-md border border-slate-700/60 text-slate-200 transition-all active:scale-90 cursor-pointer shadow-md"
                      title="Flip Camera"
                    >
                      <RotateCcw className="w-4 h-4" />
                    </button>

                    {/* Stop Camera Button */}
                    <button
                      onClick={stopCamera}
                      className="px-3.5 py-1.5 rounded-full bg-rose-500/20 hover:bg-rose-500/30 border border-rose-500/40 text-rose-300 backdrop-blur-md text-xs font-semibold tracking-wide transition-all active:scale-95 cursor-pointer shadow-md flex items-center gap-1.5"
                    >
                      <CameraOff className="w-3.5 h-3.5" />
                      <span>Stop</span>
                    </button>
                  </div>
                </div>
              )}

              {/* Viewport Bottom Floating Status Prompt */}
              {isCameraActive && (
                <div className="absolute bottom-3 left-3 right-3 flex items-center justify-between text-xs pointer-events-none">
                  <div className="bg-slate-900/85 backdrop-blur-md px-3 py-1.5 rounded-full border border-slate-700/60 text-slate-200 flex items-center gap-2 shadow-md">
                    <span className={`w-2 h-2 rounded-full ${activeSign ? 'bg-emerald-400 animate-ping' : 'bg-cyan-400'}`} />
                    <span className="font-medium">{statusMessage}</span>
                  </div>
                </div>
              )}
            </div>

            {/* Quick Sign Convention Selector */}
            <div className="flex items-center justify-between p-3 rounded-2xl border border-slate-800/80 bg-slate-900/60 backdrop-blur text-xs">
              <span className="text-slate-400 font-medium">Recognition Mode:</span>
              <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-xl border border-slate-800">
                {(['auto', 'single_hand_chart', 'islrtc_bimanual'] as AlphabetConvention[]).map((m) => (
                  <button
                    key={m}
                    onClick={() => setConvention(m)}
                    className={`px-3 py-1 rounded-lg text-xs font-medium transition-all cursor-pointer ${
                      convention === m
                        ? 'bg-indigo-600 text-white shadow'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {m === 'auto' ? 'Auto Detect' : m === 'single_hand_chart' ? '1-Hand' : '2-Hand (ISLRTC)'}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Right Column: Hero Recognition & Text Composer (5 Cols) */}
          <div className="lg:col-span-5 space-y-4">
            {/* 1. Detected Letter Hero Card */}
            <div className="p-5 rounded-3xl border border-slate-800/80 bg-slate-900/70 backdrop-blur-xl shadow-xl space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800/80 pb-3 text-xs">
                <span className="text-slate-400 uppercase tracking-wider font-semibold flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-cyan-400" />
                  Detected Sign
                </span>
                <span className={`px-2.5 py-0.5 rounded-full text-xs font-semibold ${
                  confidencePct >= 75
                    ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                    : confidencePct > 0
                    ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                    : 'bg-slate-800 text-slate-500'
                }`}>
                  {confidencePct > 0 ? `${confidencePct}% Match` : 'No Gesture'}
                </span>
              </div>

              {/* Hero Letter Display Box */}
              <div className="flex items-center gap-5">
                <div className="w-24 h-24 rounded-2xl bg-gradient-to-br from-slate-800 to-slate-950 border border-slate-700/80 flex items-center justify-center shadow-inner shrink-0 relative overflow-hidden">
                  {activeSign ? (
                    <span className="text-5xl font-black text-transparent bg-clip-text bg-gradient-to-br from-cyan-400 via-indigo-400 to-emerald-400 animate-scaleUp">
                      {activeSign}
                    </span>
                  ) : (
                    <span className="text-3xl font-light text-slate-700">--</span>
                  )}
                  {activeSign && (
                    <div className="absolute bottom-1 right-2 text-[10px] text-slate-500 font-mono">
                      {confidencePct}%
                    </div>
                  )}
                </div>

                <div className="flex-1 space-y-1">
                  <h3 className="text-lg font-bold text-white">
                    {activeSign ? `Letter ${activeSign}` : 'Awaiting Sign'}
                  </h3>
                  <p className="text-xs text-slate-400 leading-relaxed line-clamp-3">
                    {activeSign && ALPHABET_REFERENCE[activeSign]
                      ? ALPHABET_REFERENCE[activeSign].desc
                      : 'Show an ISL or single-hand alphabet sign to the camera to detect.'}
                  </p>
                </div>
              </div>

              {/* Auto-Accept Progress Bar */}
              <div className="space-y-1.5 pt-1">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span>Hold steady to auto-type</span>
                  <span className="font-semibold text-cyan-400">{stabilityProgress}%</span>
                </div>
                <div className="w-full h-2 rounded-full bg-slate-950 overflow-hidden border border-slate-800">
                  <div
                    className="h-full bg-gradient-to-r from-cyan-500 to-emerald-400 transition-all duration-100 rounded-full"
                    style={{ width: `${stabilityProgress}%` }}
                  />
                </div>
              </div>

              {/* Force Accept Button */}
              <div className="flex items-center justify-between pt-1">
                <span className="text-xs text-slate-400">
                  {stableLockedLetter ? (
                    <span className="text-emerald-400 font-medium flex items-center gap-1.5">
                      <CheckCircle2 className="w-4 h-4" />
                      Letter accepted into text!
                    </span>
                  ) : (
                    'Holds for ~0.8s'
                  )}
                </span>

                <button
                  onClick={handleManualAddLetter}
                  disabled={!activeSign}
                  className="px-3.5 py-1.5 rounded-xl border border-indigo-500/40 bg-indigo-500/10 hover:bg-indigo-500/20 disabled:opacity-30 text-indigo-300 text-xs font-semibold transition-all active:scale-95 cursor-pointer"
                >
                  + Add Now
                </button>
              </div>
            </div>

            {/* 2. Text Composer & Sentence Builder */}
            <div className="p-5 rounded-3xl border border-slate-800/80 bg-slate-900/70 backdrop-blur-xl shadow-xl space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800/80 pb-3 text-xs">
                <span className="text-slate-400 uppercase tracking-wider font-semibold flex items-center gap-2">
                  <Terminal className="w-4 h-4 text-emerald-400" />
                  Composed Text
                </span>
                <span className="text-slate-500 text-xs">
                  {textBuffer.length} {textBuffer.length === 1 ? 'char' : 'chars'}
                </span>
              </div>

              {/* Text Box Display */}
              <div className="min-h-[80px] p-4 rounded-2xl bg-slate-950/90 border border-slate-800 text-lg font-medium tracking-wide text-white break-words select-all shadow-inner">
                {textBuffer ? (
                  <span>
                    {textBuffer}
                    <span className="inline-block w-2 h-5 ml-1 bg-cyan-400 animate-pulse align-middle" />
                  </span>
                ) : (
                  <span className="text-slate-600 text-sm font-normal">
                    Recognized letters will form words here...
                  </span>
                )}
              </div>

              {/* Mobile-Friendly Tactile Controls */}
              <div className="grid grid-cols-4 gap-2 text-xs">
                <button
                  onClick={handleSpace}
                  className="py-2.5 rounded-xl border border-slate-700/80 bg-slate-800/90 hover:bg-slate-700 text-white font-semibold transition-all active:scale-95 cursor-pointer flex items-center justify-center gap-1.5 shadow-sm"
                >
                  <Space className="w-3.5 h-3.5 text-cyan-400" />
                  <span>Space</span>
                </button>

                <button
                  onClick={handleBackspace}
                  className="py-2.5 rounded-xl border border-slate-700/80 bg-slate-800/90 hover:bg-slate-700 text-white font-semibold transition-all active:scale-95 cursor-pointer flex items-center justify-center gap-1.5 shadow-sm"
                >
                  <Delete className="w-3.5 h-3.5 text-amber-400" />
                  <span>Del</span>
                </button>

                <button
                  onClick={handleClear}
                  className="py-2.5 rounded-xl border border-slate-700/80 bg-slate-800/90 hover:bg-slate-700 text-rose-300 font-semibold transition-all active:scale-95 cursor-pointer flex items-center justify-center gap-1.5 shadow-sm"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  <span>Clear</span>
                </button>

                <button
                  onClick={handleSpeakText}
                  disabled={!textBuffer.trim()}
                  className="py-2.5 rounded-xl bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-600 hover:to-teal-600 disabled:opacity-40 text-white font-bold transition-all active:scale-95 cursor-pointer flex items-center justify-center gap-1.5 shadow-md shadow-emerald-500/20"
                >
                  <Volume2 className="w-3.5 h-3.5" />
                  <span>Speak</span>
                </button>
              </div>

              {/* Copy Full Sentence Button */}
              {textBuffer && (
                <button
                  onClick={handleCopy}
                  className="w-full py-2.5 rounded-xl border border-slate-700/80 bg-slate-800/60 hover:bg-slate-800 text-slate-200 text-xs font-semibold transition-all active:scale-95 cursor-pointer flex items-center justify-center gap-2 shadow-sm"
                >
                  {copied ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4 text-slate-400" />}
                  <span>{copied ? 'Copied to Clipboard!' : 'Copy Text'}</span>
                </button>
              )}
            </div>
          </div>
        </div>

        {/* 3. Collapsible Diagnostics & Science Fair Lab Modal/Section */}
        {showDiagnostics && (
          <div className="p-5 sm:p-6 rounded-3xl border border-indigo-500/30 bg-slate-900/90 backdrop-blur-xl shadow-2xl space-y-6">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2 text-indigo-400 font-bold text-sm tracking-wide">
                <Gauge className="w-5 h-5" />
                <span>DIAGNOSTICS & SCIENCE FAIR BENCHMARK LAB</span>
              </div>
              <button
                onClick={() => setShowDiagnostics(false)}
                className="p-1 rounded-lg text-slate-400 hover:text-white"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Dedicated Real-Device Acceptance Testing Mode (Section 10) */}
            <div className="p-4 sm:p-5 rounded-2xl bg-slate-950/80 border border-cyan-500/30 space-y-4 shadow-inner">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800/80 pb-3">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse" />
                    <h4 className="text-sm font-bold text-white tracking-wide">
                      Real-Device Acceptance Testing Mode
                    </h4>
                  </div>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Step-by-step physical mobile evaluation protocol with zero-video local telemetry logging.
                  </p>
                </div>

                <div className="flex items-center gap-2">
                  {acceptanceLog.length > 0 && (
                    <button
                      onClick={downloadAcceptanceReport}
                      className="px-3 py-1.5 rounded-xl bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-300 border border-cyan-500/40 text-xs font-semibold flex items-center gap-1.5 transition-all cursor-pointer shadow-sm"
                    >
                      <Download className="w-3.5 h-3.5" />
                      <span>Export Session JSON ({acceptanceLog.length})</span>
                    </button>
                  )}
                </div>
              </div>

              {/* Protocol Config: Signer ID & Camera Environment */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div>
                  <label className="text-[11px] font-medium text-slate-400 block mb-1">
                    Tester / Signer ID
                  </label>
                  <input
                    type="text"
                    value={acceptanceTesterId}
                    onChange={(e) => setAcceptanceTesterId(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-700/80 rounded-xl px-3 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-400"
                    placeholder="e.g. Signer_Phone_1"
                  />
                </div>

                <div>
                  <label className="text-[11px] font-medium text-slate-400 block mb-1">
                    Illumination Condition
                  </label>
                  <select
                    value={acceptanceLighting}
                    onChange={(e) => setAcceptanceLighting(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-700/80 rounded-xl px-3 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-400"
                  >
                    <option value="Natural Daylight (450 lux)">Natural Daylight (450 lux)</option>
                    <option value="Indoor Fluorescent (320 lux)">Indoor Fluorescent (320 lux)</option>
                    <option value="Low-Light Ambient (45 lux, High ISO)">Low-Light Ambient (45 lux, High ISO)</option>
                    <option value="Dim Backlit Condition">Dim Backlit Condition</option>
                  </select>
                </div>

                <div>
                  <label className="text-[11px] font-medium text-slate-400 block mb-1">
                    Camera Optical Distance
                  </label>
                  <select
                    value={acceptanceDistance}
                    onChange={(e) => setAcceptanceDistance(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-700/80 rounded-xl px-3 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-400"
                  >
                    <option value="Arm Length (~0.65m)">Arm Length (~0.65m)</option>
                    <option value="Close-Up (~0.45m)">Close-Up (~0.45m)</option>
                    <option value="Desk Mounted (~0.90m)">Desk Mounted (~0.90m)</option>
                  </select>
                </div>
              </div>

              {/* Target Sign Card & Step Execution */}
              <div className="flex flex-col sm:flex-row items-center justify-between gap-4 p-4 rounded-xl bg-slate-900/60 border border-slate-800">
                <div className="flex items-center gap-4">
                  <div className="w-16 h-16 rounded-2xl bg-cyan-500/10 border-2 border-cyan-500/40 flex flex-col items-center justify-center">
                    <span className="text-[10px] text-cyan-400 font-bold uppercase tracking-wider">Target</span>
                    <span className="text-2xl font-black text-white">{acceptanceTargetLetter}</span>
                  </div>

                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-slate-400">Live Model Prediction:</span>
                      <span className="text-xs font-bold text-white px-2 py-0.5 rounded bg-slate-800">
                        {currentResult?.sign || 'SEARCHING'}
                      </span>
                      <span className="text-xs text-cyan-300 font-semibold">
                        ({Math.round((currentResult?.confidence || 0) * 100)}%)
                      </span>
                    </div>
                    <p className="text-[11px] text-slate-500 mt-1">
                      Hold the sign steady for 2-3 seconds, then tap "Log Target Trial".
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-2 w-full sm:w-auto">
                  <select
                    value={acceptanceTargetLetter}
                    onChange={(e) => setAcceptanceTargetLetter(e.target.value)}
                    className="bg-slate-950 border border-slate-700 text-cyan-300 px-3 py-2 rounded-xl text-xs font-semibold"
                  >
                    {['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'].map((l) => (
                      <option key={l} value={l}>
                        Sign: {l}
                      </option>
                    ))}
                  </select>

                  <button
                    onClick={logAcceptanceTrial}
                    disabled={!isCameraActive || !currentResult}
                    className="flex-1 sm:flex-none px-5 py-2 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 disabled:opacity-40 text-black font-bold text-xs flex items-center justify-center gap-2 cursor-pointer shadow-lg shadow-cyan-500/20 active:scale-95 transition-all"
                  >
                    <CheckCircle2 className="w-4 h-4 fill-black text-cyan-400" />
                    <span>Log Target Trial</span>
                  </button>
                </div>
              </div>

              {/* Logged Trials Table */}
              {acceptanceLog.length > 0 && (
                <div className="space-y-2 pt-1">
                  <div className="flex items-center justify-between text-xs text-slate-400 px-1">
                    <span>Logged Trials ({acceptanceLog.length})</span>
                    <span className="text-emerald-400 font-bold">
                      Accepted: {acceptanceLog.filter((t) => t.accepted).length} / {acceptanceLog.length} (
                      {Math.round((acceptanceLog.filter((t) => t.accepted).length / acceptanceLog.length) * 100)}%)
                    </span>
                  </div>

                  <div className="max-h-48 overflow-y-auto rounded-xl border border-slate-800 bg-slate-950 text-xs">
                    <table className="w-full text-left">
                      <thead className="bg-slate-900/80 text-[10px] text-slate-400 uppercase tracking-wider sticky top-0">
                        <tr>
                          <th className="p-2">Target</th>
                          <th className="p-2">Prediction</th>
                          <th className="p-2">Confidence</th>
                          <th className="p-2">Quality</th>
                          <th className="p-2">Latency</th>
                          <th className="p-2 text-right">Verdict</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/60 font-mono text-[11px]">
                        {acceptanceLog.map((trial) => (
                          <tr key={trial.id} className="hover:bg-slate-900/40">
                            <td className="p-2 font-bold text-cyan-300">{trial.targetLetter}</td>
                            <td className="p-2 text-white">{trial.predictedLetter}</td>
                            <td className="p-2">{trial.confidence}%</td>
                            <td className="p-2">{trial.trackingQualityScore}/100</td>
                            <td className="p-2">{trial.latencyMs}ms</td>
                            <td className="p-2 text-right">
                              {trial.accepted ? (
                                <span className="inline-flex items-center gap-1 text-emerald-400 font-sans font-bold text-[10px] px-2 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/30">
                                  ACCEPTED
                                </span>
                              ) : (
                                <span className="inline-flex items-center gap-1 text-rose-400 font-sans font-bold text-[10px] px-2 py-0.5 rounded bg-rose-500/10 border border-rose-500/30">
                                  MISMATCH
                                </span>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>

            {/* Calibration & 10s Evaluation Protocol */}
            <div className="space-y-3">
              <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2">
                <div>
                  <h4 className="text-sm font-semibold text-slate-200">10-Second Physical Device Evaluation</h4>
                  <p className="text-xs text-slate-400">Record 10 seconds of holding a sign to test camera latency, accuracy, and jitter.</p>
                </div>

                <div className="flex items-center gap-2">
                  <select
                    value={calibrationLetter}
                    onChange={(e) => setCalibrationLetter(e.target.value)}
                    disabled={isCalibrating}
                    className="bg-slate-950 border border-slate-700 text-cyan-300 px-3 py-1.5 rounded-xl text-xs font-semibold"
                  >
                    {Object.keys(ALPHABET_REFERENCE).map((l) => (
                      <option key={l} value={l}>
                        Sign: {l}
                      </option>
                    ))}
                  </select>

                  <button
                    onClick={() => startCalibrationSession(calibrationLetter)}
                    disabled={isCalibrating || !isCameraActive}
                    className="px-4 py-1.5 rounded-xl bg-amber-500 hover:bg-amber-600 disabled:opacity-40 text-black font-semibold text-xs flex items-center gap-1.5 cursor-pointer shadow-md"
                  >
                    <Play className="w-3.5 h-3.5 fill-black" />
                    <span>{isCalibrating ? 'Recording (10s)...' : 'Start 10s Test'}</span>
                  </button>
                </div>
              </div>

              {/* Calibration Results Cards */}
              {calibrationResults && (
                <div className="grid grid-cols-2 sm:grid-cols-6 gap-2 pt-2">
                  <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                    <span className="text-[10px] text-slate-400">Target</span>
                    <p className="text-base font-bold text-cyan-400">{calibrationResults.targetLetter}</p>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                    <span className="text-[10px] text-slate-400">Accuracy</span>
                    <p className="text-base font-bold text-emerald-400">{calibrationResults.accuracyRate}%</p>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                    <span className="text-[10px] text-slate-400">Continuity</span>
                    <p className="text-base font-bold text-slate-200">{calibrationResults.continuityRate}%</p>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                    <span className="text-[10px] text-slate-400">FPS</span>
                    <p className="text-base font-bold text-slate-200">{calibrationResults.avgFps}</p>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                    <span className="text-[10px] text-slate-400">Latency</span>
                    <p className="text-base font-bold text-slate-200">{calibrationResults.avgLatencyMs}ms</p>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                    <span className="text-[10px] text-slate-400">Jitter</span>
                    <p className="text-base font-bold text-amber-400">{calibrationResults.avgJitter}</p>
                  </div>
                </div>
              )}
            </div>

            {/* 90-Frame Benchmark Suite */}
            <div className="space-y-3 pt-4 border-t border-slate-800">
              <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2">
                <div>
                  <h4 className="text-sm font-semibold text-slate-200">Comparative Tracking Benchmark (90 Frames)</h4>
                  <p className="text-xs text-slate-400">Evaluates raw MediaPipe vs Baseline bd86151 vs New Precision Engine.</p>
                </div>

                <button
                  onClick={runBenchmarkSuite}
                  disabled={isRunningBenchmark}
                  className="px-4 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 text-white font-semibold text-xs flex items-center gap-1.5 cursor-pointer shadow-md"
                >
                  {isRunningBenchmark ? (
                    <>
                      <Activity className="w-3.5 h-3.5 animate-spin" />
                      <span>Running 90-frame test...</span>
                    </>
                  ) : (
                    <>
                      <FlaskConical className="w-3.5 h-3.5" />
                      <span>Run Benchmark</span>
                    </>
                  )}
                </button>
              </div>

              {/* Comparative Table */}
              {benchmarkReport && (
                <div className="overflow-x-auto rounded-2xl border border-slate-800 bg-slate-950 text-xs">
                  <table className="w-full text-left">
                    <thead className="bg-slate-900 text-slate-400 border-b border-slate-800">
                      <tr>
                        <th className="p-2.5">Pipeline</th>
                        <th className="p-2.5">Overall MAE</th>
                        <th className="p-2.5">Fingertip MAE</th>
                        <th className="p-2.5">PCK@10px</th>
                        <th className="p-2.5">Continuity</th>
                        <th className="p-2.5">Latency</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800 text-slate-300">
                      <tr>
                        <td className="p-2.5 font-medium">{benchmarkReport.rawMediaPipe.pipelineName}</td>
                        <td className="p-2.5">{benchmarkReport.rawMediaPipe.overallMAE}px</td>
                        <td className="p-2.5">{benchmarkReport.rawMediaPipe.fingertipMAE}px</td>
                        <td className="p-2.5">{benchmarkReport.rawMediaPipe.overallPCK10}%</td>
                        <td className="p-2.5 text-rose-400">{benchmarkReport.rawMediaPipe.continuityRate}%</td>
                        <td className="p-2.5">{benchmarkReport.rawMediaPipe.avgProcessingTimeMs}ms</td>
                      </tr>
                      <tr>
                        <td className="p-2.5 font-medium text-amber-300">{benchmarkReport.baselineCommit.pipelineName}</td>
                        <td className="p-2.5">{benchmarkReport.baselineCommit.overallMAE}px</td>
                        <td className="p-2.5">{benchmarkReport.baselineCommit.fingertipMAE}px</td>
                        <td className="p-2.5">{benchmarkReport.baselineCommit.overallPCK10}%</td>
                        <td className="p-2.5">{benchmarkReport.baselineCommit.continuityRate}%</td>
                        <td className="p-2.5">{benchmarkReport.baselineCommit.avgProcessingTimeMs}ms</td>
                      </tr>
                      <tr className="bg-emerald-950/20 font-bold text-emerald-300">
                        <td className="p-2.5 text-emerald-400 flex items-center gap-1.5">
                          <Check className="w-4 h-4" />
                          {benchmarkReport.newPrecisionEngine.pipelineName}
                        </td>
                        <td className="p-2.5">{benchmarkReport.newPrecisionEngine.overallMAE}px</td>
                        <td className="p-2.5 text-white">{benchmarkReport.newPrecisionEngine.fingertipMAE}px</td>
                        <td className="p-2.5">{benchmarkReport.newPrecisionEngine.overallPCK10}%</td>
                        <td className="p-2.5 text-emerald-400">{benchmarkReport.newPrecisionEngine.continuityRate}%</td>
                        <td className="p-2.5">{benchmarkReport.newPrecisionEngine.avgProcessingTimeMs}ms</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              )}

              {/* TITAN Independent Research Engine Inspector */}
              <div className="p-4 bg-slate-950/60 rounded-2xl border border-indigo-900/40">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-indigo-400" />
                    <div>
                      <h4 className="text-sm font-semibold text-indigo-200">SIGNBRIDGE TITAN Spatial GNN Engine</h4>
                      <p className="text-xs text-slate-400">Independent backup recognition architecture (2-Layer GCN + Aux42 + Uncertainty)</p>
                    </div>
                  </div>
                  <button
                    onClick={() => setShowTitanMonitor(prev => !prev)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                      showTitanMonitor
                        ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30'
                        : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                    }`}
                  >
                    {showTitanMonitor ? 'Monitor Active' : 'Enable TITAN Live'}
                  </button>
                </div>

                {showTitanMonitor && (
                  <div className="p-3 bg-slate-900/80 rounded-xl border border-indigo-800/30 text-xs">
                    {titanResult ? (
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                        <div className="p-2 bg-slate-950/50 rounded-lg">
                          <span className="text-slate-400 block text-[10px]">PREDICTED SIGN</span>
                          <span className="text-lg font-bold text-indigo-400">{titanResult.sign}</span>
                          <span className="text-[10px] text-slate-500 block">Raw: {titanResult.rawSign}</span>
                        </div>
                        <div className="p-2 bg-slate-950/50 rounded-lg">
                          <span className="text-slate-400 block text-[10px]">CALIBRATED CONF</span>
                          <span className="text-base font-bold text-emerald-400">
                            {(titanResult.confidence * 100).toFixed(1)}%
                          </span>
                          <span className="text-[10px] text-slate-500 block">
                            Raw: {(titanResult.uncalibratedConfidence * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="p-2 bg-slate-950/50 rounded-lg">
                          <span className="text-slate-400 block text-[10px]">PREDICTIVE ENTROPY</span>
                          <span className={`text-base font-bold ${titanResult.entropy > 1.2 ? 'text-amber-400' : 'text-slate-200'}`}>
                            {titanResult.entropy.toFixed(3)}
                          </span>
                          <span className="text-[10px] text-slate-500 block">Energy: {titanResult.energy.toFixed(1)}</span>
                        </div>
                        <div className="p-2 bg-slate-950/50 rounded-lg">
                          <span className="text-slate-400 block text-[10px]">REJECTION GATE</span>
                          <span className={`text-xs font-bold px-1.5 py-0.5 rounded ${
                            titanResult.isRejected ? 'bg-rose-500/20 text-rose-300' : 'bg-emerald-500/20 text-emerald-300'
                          }`}>
                            {titanResult.isRejected ? 'REJECTED' : 'ACCEPTED'}
                          </span>
                          <span className="text-[10px] text-slate-400 block mt-1">
                            {titanResult.latencyMs.toFixed(2)} ms latency
                          </span>
                        </div>
                      </div>
                    ) : (
                      <div className="text-slate-400 text-center py-2">
                        Present a hand to the camera to view live TITAN GCN activations and uncertainty estimates.
                      </div>
                    )}
                  </div>
                )}

                {/* 2c. ASL Visual Reference Integration Diagnostics */}
                <div className="mt-4 pt-4 border-t border-slate-800">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-cyan-500 animate-pulse" />
                      <span className="text-xs font-bold text-slate-200">ASL Visual Reference Diagnostics</span>
                      <span className="text-[10px] bg-cyan-900/40 text-cyan-400 px-2 py-0.5 rounded-full border border-cyan-800/50">
                        Gérard Aflague Collection
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => setShowFullASLChartModal(true)}
                        className="px-2.5 py-1 rounded-lg text-xs font-medium bg-slate-800 text-slate-300 hover:bg-slate-700 transition-all cursor-pointer flex items-center gap-1.5"
                      >
                        <BookOpen className="w-3.5 h-3.5 text-cyan-400" />
                        Full Poster
                      </button>
                      <button
                        onClick={() => setShowASLReferenceDiagnostics(prev => !prev)}
                        className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                          showASLReferenceDiagnostics
                            ? 'bg-cyan-600 text-white shadow-lg shadow-cyan-600/30'
                            : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                        }`}
                      >
                        {showASLReferenceDiagnostics ? 'Diagnostics Active' : 'Enable ASL GNN'}
                      </button>
                    </div>
                  </div>

                  {showASLReferenceDiagnostics && (
                    <div className="p-3 bg-slate-900/80 rounded-xl border border-cyan-800/30 text-xs space-y-3">
                      {/* Letter Selector Bar */}
                      <div>
                        <span className="text-[10px] text-slate-400 font-semibold block mb-1.5">
                          SELECT REFERENCE CANONICAL TARGET LETTER:
                        </span>
                        <div className="flex flex-wrap gap-1">
                          {'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('').map(letter => (
                            <button
                              key={letter}
                              onClick={() => setSelectedASLReferenceLetter(letter)}
                              className={`w-6 h-6 rounded text-[11px] font-bold transition-all cursor-pointer ${
                                selectedASLReferenceLetter === letter
                                  ? 'bg-cyan-500 text-slate-950 shadow-md shadow-cyan-500/40'
                                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                              }`}
                            >
                              {letter}
                            </button>
                          ))}
                        </div>
                      </div>

                      {/* Visual Reference Comparison Grid */}
                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 items-center">
                        {/* Reference Crop Preview */}
                        <div className="p-2 bg-slate-950/60 rounded-lg border border-slate-800 flex items-center gap-3">
                          <img
                            src={`/references/crops/${selectedASLReferenceLetter}.png`}
                            alt={`ASL ${selectedASLReferenceLetter}`}
                            className="w-16 h-16 rounded-md object-contain bg-slate-900 border border-slate-700"
                            onError={(e) => {
                              (e.target as HTMLImageElement).style.display = 'none';
                            }}
                          />
                          <div>
                            <span className="text-[10px] text-slate-400 block font-semibold">CANONICAL POSTER</span>
                            <span className="text-xl font-black text-cyan-400">{selectedASLReferenceLetter}</span>
                            <span className="text-[9px] text-slate-500 block">Single-hand ASL</span>
                          </div>
                        </div>

                        {/* Model Inference Result */}
                        <div className="p-2 bg-slate-950/60 rounded-lg border border-slate-800 col-span-2">
                          {titanASLResult ? (
                            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                              <div>
                                <span className="text-[9px] text-slate-400 block">DETECTED SIGN</span>
                                <span className={`text-base font-bold ${
                                  titanASLResult.sign === selectedASLReferenceLetter ? 'text-emerald-400' : 'text-amber-400'
                                }`}>
                                  {titanASLResult.sign}
                                </span>
                                <span className="text-[9px] text-slate-500 block">
                                  {titanASLResult.sign === selectedASLReferenceLetter ? 'MATCH' : 'MISMATCH'}
                                </span>
                              </div>
                              <div>
                                <span className="text-[9px] text-slate-400 block">CALIBRATED CONF</span>
                                <span className="text-base font-bold text-emerald-400">
                                  {(titanASLResult.confidence * 100).toFixed(1)}%
                                </span>
                                <span className="text-[9px] text-slate-500 block">
                                  Raw: {(titanASLResult.uncalibratedConfidence * 100).toFixed(1)}%
                                </span>
                              </div>
                              <div>
                                <span className="text-[9px] text-slate-400 block">ENTROPY / ENERGY</span>
                                <span className="text-base font-bold text-slate-200">
                                  {titanASLResult.entropy.toFixed(2)}
                                </span>
                                <span className="text-[9px] text-slate-500 block">
                                  E: {titanASLResult.energy.toFixed(1)}
                                </span>
                              </div>
                              <div>
                                <span className="text-[9px] text-slate-400 block">STATUS</span>
                                <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                                  titanASLResult.isRejected ? 'bg-rose-500/20 text-rose-300' : 'bg-emerald-500/20 text-emerald-300'
                                }`}>
                                  {titanASLResult.isRejected ? 'REJECTED' : 'ACCEPTED'}
                                </span>
                                <span className="text-[9px] text-slate-400 block mt-1">
                                  {titanASLResult.latencyMs.toFixed(1)} ms
                                </span>
                              </div>
                            </div>
                          ) : (
                            <div className="text-slate-400 text-center py-2 text-[11px]">
                              Present a hand to test against ASL {selectedASLReferenceLetter} canonical pose.
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* 4. Alphabet Reference Guide Modal */}
        {showGuideModal && (
          <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-md flex items-center justify-center p-4">
            <div className="bg-slate-900 border border-slate-800 rounded-3xl max-w-4xl w-full max-h-[85vh] flex flex-col shadow-2xl overflow-hidden animate-fadeIn">
              {/* Modal Header */}
              <div className="p-5 border-b border-slate-800 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-9 h-9 rounded-xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-cyan-400">
                    <BookOpen className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="text-base font-bold text-white">Indian Sign Language Alphabet Guide</h3>
                    <p className="text-xs text-slate-400">Learn the exact hand shape for each letter</p>
                  </div>
                </div>

                <button
                  onClick={() => setShowGuideModal(false)}
                  className="p-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white transition-all cursor-pointer"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              {/* Filter tabs */}
              <div className="px-5 py-3 bg-slate-950/50 border-b border-slate-800 flex items-center gap-2">
                <button
                  onClick={() => setGuideFilter('all')}
                  className={`px-3 py-1 rounded-lg text-xs font-medium cursor-pointer transition-all ${
                    guideFilter === 'all' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'
                  }`}
                >
                  All (A-Z)
                </button>
                <button
                  onClick={() => setGuideFilter('single')}
                  className={`px-3 py-1 rounded-lg text-xs font-medium cursor-pointer transition-all ${
                    guideFilter === 'single' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Single-Hand Signs
                </button>
                <button
                  onClick={() => setGuideFilter('dual')}
                  className={`px-3 py-1 rounded-lg text-xs font-medium cursor-pointer transition-all ${
                    guideFilter === 'dual' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Two-Hand (ISLRTC)
                </button>
              </div>

              {/* Alphabet Grid */}
              <div className="p-5 overflow-y-auto grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
                {filteredGuide.map(([letter, info]) => (
                  <div
                    key={letter}
                    className="p-3.5 rounded-2xl border border-slate-800 bg-slate-950/60 hover:border-slate-700 transition-all flex flex-col justify-between space-y-2"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-2xl font-black text-cyan-400">{letter}</span>
                      <span className="text-[10px] px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 font-medium border border-slate-700">
                        {info.standard}
                      </span>
                    </div>
                    <p className="text-xs text-slate-300 leading-relaxed">{info.desc}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
        {/* ASL Visual Reference Chart Modal */}
        {showFullASLChartModal && (
          <div className="fixed inset-0 z-50 bg-slate-950/85 backdrop-blur-md flex items-center justify-center p-4">
            <div className="bg-slate-900 border border-slate-800 rounded-3xl max-w-3xl w-full max-h-[90vh] flex flex-col shadow-2xl overflow-hidden animate-fadeIn">
              <div className="p-4 border-b border-slate-800 flex items-center justify-between">
                <div>
                  <h3 className="text-base font-bold text-white">American Sign Language Alphabet</h3>
                  <p className="text-xs text-slate-400">Gérard Aflague Collection • Canonical Morphological Reference</p>
                </div>
                <button
                  onClick={() => setShowFullASLChartModal(false)}
                  className="p-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white transition-all cursor-pointer"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              <div className="p-4 overflow-y-auto flex items-center justify-center bg-slate-950">
                <img
                  src="/references/asl_alphabet_chart.png"
                  alt="American Sign Language Alphabet Chart"
                  className="max-h-[75vh] w-auto rounded-xl object-contain shadow-lg border border-slate-800"
                />
              </div>
            </div>
          </div>
        )}
      </main>

      {/* 5. Minimal Modern Footer */}
      <footer className="mt-auto border-t border-slate-900 bg-slate-950 px-4 py-4 text-center text-xs text-slate-500">
        <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>SIGNBRIDGE • Real-Time Indian Sign Language AI</span>
          <span className="text-slate-600 text-[11px]">Powered by MediaPipe & Neural Biomechanical Classification</span>
        </div>
      </footer>
    </div>
  );
}

export default LiveISLAlphabetRecognizer;
