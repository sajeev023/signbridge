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
  Crosshair,
  Shield,
  Play,
  FlaskConical,
  BarChart3,
  TrendingDown,
  Info
} from 'lucide-react';
import { webMediaPipe } from '../services/webMediaPipeService';
import {
  alphabetInferenceEngine,
  AlphabetConvention,
  InferenceResult,
  LandmarkPoint
} from '../services/alphabetInferenceEngine';
import { PersistentHandTrack } from '../services/persistentHandTracker';
import { TrackingQualityReport, TrackingStatusState } from '../services/trackingQualityEngine';
import { speechService } from '../services/speechService';
import { HAND_CONNECTIONS } from '../services/cameraLandmarkDetector';
import { TrackingBenchmarkLab, ComparativeBenchmarkResult } from '../services/trackingBenchmarkLab';

// Reference guide for supported alphabet signs
const ALPHABET_REFERENCE: Record<string, { desc: string; standard: string }> = {
  A: { desc: '1-Hand: Closed fist with thumb on side. 2-Hand: Dominant index on base thumb tip.', standard: 'Unified' },
  B: { desc: '1-Hand: 4 fingers upright, thumb across palm. 2-Hand: Two touching loops.', standard: 'Unified' },
  C: { desc: 'Curved C-arc formed with thumb and curved fingers.', standard: 'Single-Hand' },
  D: { desc: '1-Hand: Index pointing up, thumb touching other fingers. 2-Hand: Vertical index + semicircle loop.', standard: 'Unified' },
  E: { desc: '1-Hand: Fingers curled tightly, thumb tucked. 2-Hand: Dominant index on base index tip.', standard: 'Unified' },
  F: { desc: '1-Hand: Index-thumb loop, 3 fingers upright. 2-Hand: Crossed two fingers.', standard: 'Unified' },
  G: { desc: '1-Hand: Index horizontal, thumb forward. 2-Hand: Fists touching knuckles.', standard: 'Unified' },
  H: { desc: '1-Hand: Index & middle horizontal together. 2-Hand: Flat dominant palm across base palm.', standard: 'Unified' },
  I: { desc: '1-Hand: Pinky pointing upright. 2-Hand: Dominant index on base middle tip.', standard: 'Unified' },
  J: { desc: 'Dynamic: Dominant index starts at I and carves downward curved hook.', standard: 'Dynamic' },
  K: { desc: '1-Hand: Index upright, middle forward, thumb between. 2-Hand: Hooked index on vertical index.', standard: 'Unified' },
  L: { desc: 'Single hand thumb and index at 90° right angle, remaining fingers folded.', standard: 'Single-Hand' },
  M: { desc: '2-Hand: Dominant 3 fingers placed down on base palm.', standard: 'ISLRTC Dual' },
  N: { desc: '2-Hand: Dominant 2 fingers placed down on base palm.', standard: 'ISLRTC Dual' },
  O: { desc: '1-Hand: All fingers and thumb touching tips (O-circle). 2-Hand: Dominant index on base ring tip.', standard: 'Unified' },
  P: { desc: '2-Hand: Non-dominant index vertical + dominant loop at top.', standard: 'ISLRTC Dual' },
  Q: { desc: '2-Hand: Base thumb-index loop + dominant index at bottom edge.', standard: 'ISLRTC Dual' },
  R: { desc: 'Index and middle fingers crossed over each other.', standard: 'Single-Hand' },
  S: { desc: '2-Hand: Both pinkies hooked and locked together.', standard: 'ISLRTC Dual' },
  T: { desc: '2-Hand: Dominant index touches edge of base index base.', standard: 'ISLRTC Dual' },
  U: { desc: '1-Hand: Index & middle upright together. 2-Hand: Dominant index on base pinky tip.', standard: 'Unified' },
  V: { desc: 'Index and middle fingers extended upright spread in a V-shape.', standard: 'Unified' },
  W: { desc: 'Index, middle, ring fingers extended upright spread in a W-shape.', standard: 'Unified' },
  X: { desc: 'Index finger hooked/bent, other fingers curled.', standard: 'Single-Hand' },
  Y: { desc: '1-Hand: Thumb and pinky wide spread (shaka). 2-Hand: Dominant index in base thumb-index cleft.', standard: 'Unified' },
  Z: { desc: 'Dynamic: Index finger traces 3-stroke zigzag Z trajectory in air or palm.', standard: 'Dynamic' }
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

export function LiveISLAlphabetRecognizer() {
  // Telemetry & Camera State
  const [isCameraActive, setIsCameraActive] = useState<boolean>(false);
  const [isInitializing, setIsInitializing] = useState<boolean>(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [facingMode, setFacingMode] = useState<'user' | 'environment'>('user');
  const [fps, setFps] = useState<number>(0);
  const [latencyMs, setLatencyMs] = useState<number>(0);
  const [trackedHandCount, setTrackedHandCount] = useState<number>(0);

  // Diagnostic HUD & View Options
  const [showDiagnosticHUD, setShowDiagnosticHUD] = useState<boolean>(false);
  const [showReferenceDrawer, setShowReferenceDrawer] = useState<boolean>(false);
  const [showCalibrationDeck, setShowCalibrationDeck] = useState<boolean>(false);
  const [showBenchmarkLab, setShowBenchmarkLab] = useState<boolean>(false);
  const [benchmarkReport, setBenchmarkReport] = useState<ComparativeBenchmarkResult | null>(null);
  const [isRunningBenchmark, setIsRunningBenchmark] = useState<boolean>(false);

  // Scientific Convention Selection
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
    recommendation: 'Position hands in camera viewfinder.',
    perFingerTelemetry: [],
    identitySwitchesCount: 0,
    reacquisitionsCount: 0
  });

  // Neural Recognition & Stability Lock
  const [currentResult, setCurrentResult] = useState<InferenceResult | null>(null);
  const [stableLockedLetter, setStableLockedLetter] = useState<string | null>(null);
  const [stabilityProgress, setStabilityProgress] = useState<number>(0);
  const [statusMessage, setStatusMessage] = useState<string>('SYSTEM STANDBY');

  // Text Composition Terminal State
  const [textBuffer, setTextBuffer] = useState<string>('');
  const [isMuted, setIsMuted] = useState<boolean>(false);
  const [copied, setCopied] = useState<boolean>(false);

  // Real-Time Calibration & Benchmark Test Bench State
  const [isCalibrating, setIsCalibrating] = useState<boolean>(false);
  const [calibrationLetter, setCalibrationLetter] = useState<string>('L');
  const [calibrationResults, setCalibrationResults] = useState<BenchmarkMetrics | null>(null);
  const calibrationFramesRef = useRef<Array<{ predicted: string; latency: number; jitter: number; isTracked: boolean }>>([]);
  const calibrationStartTimeRef = useRef<number>(0);

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
    setStatusMessage('OPTICAL SENSORS INACTIVE');
  }, []);

  const startCamera = async () => {
    setCameraError(null);
    setIsInitializing(true);
    setStatusMessage('INITIALIZING MEDIAPIPE WEB GPU ENGINE...');

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
            setStatusMessage('OPTICAL TRACKING ACQUIRED');
            runVisionPipeline();
          }
        };
      }
    } catch (err: unknown) {
      console.error('Camera startup exception:', err);
      setIsInitializing(false);
      setIsCameraActive(false);

      if (err instanceof DOMException && (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError')) {
        setCameraError('Permission Denied: Camera access was blocked by the browser. Enable permissions in the address bar.');
      } else if (err instanceof DOMException && err.name === 'NotFoundError') {
        setCameraError('Sensor Fault: No camera hardware detected on this device.');
      } else {
        setCameraError(err instanceof Error ? err.message : 'Unknown hardware initialization error.');
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

  // High-precision anatomical skeleton & HUD renderer
  const renderPrecisionViewfinder = (
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    tracks: PersistentHandTrack[],
    quality: TrackingQualityReport,
    isLocked: boolean,
    showDiagnostics: boolean
  ) => {
    ctx.clearRect(0, 0, width, height);

    // 1. Viewfinder Corner HUD (Scientific aesthetic)
    const isDegraded = quality.status === 'TRACKING DEGRADED';
    ctx.strokeStyle = isLocked ? '#10b981' : isDegraded ? '#f59e0b' : 'rgba(56, 189, 248, 0.4)';
    ctx.lineWidth = 1.5;
    const corner = 22;
    const pad = 14;

    // Corner reticles
    ctx.beginPath();
    ctx.moveTo(pad, pad + corner);
    ctx.lineTo(pad, pad);
    ctx.lineTo(pad + corner, pad);
    ctx.moveTo(width - pad - corner, pad);
    ctx.lineTo(width - pad, pad);
    ctx.lineTo(width - pad, pad + corner);
    ctx.moveTo(pad, height - pad - corner);
    ctx.lineTo(pad, height - pad);
    ctx.lineTo(pad + corner, height - pad);
    ctx.moveTo(width - pad - corner, height - pad);
    ctx.lineTo(width - pad, height - pad);
    ctx.lineTo(width - pad, height - pad - corner);
    ctx.stroke();

    if (!tracks || tracks.length === 0) {
      // Crosshair center target
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.2)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(width / 2 - 14, height / 2);
      ctx.lineTo(width / 2 + 14, height / 2);
      ctx.moveTo(width / 2, height / 2 - 14);
      ctx.lineTo(width / 2, height / 2 + 14);
      ctx.stroke();
      return;
    }

    // 2. Render Anatomical Bone Chains & Palm Membranes
    tracks.forEach((track) => {
      const lm = track.smoothedLandmarks;
      const isDominant = track.role === 'Dominant';
      const isCoasting = track.isCoasting;
      const boneStroke = isLocked
        ? 'rgba(16, 185, 129, 0.9)'
        : isCoasting
        ? 'rgba(245, 158, 11, 0.75)'
        : isDominant
        ? 'rgba(56, 189, 248, 0.85)'
        : 'rgba(245, 158, 11, 0.85)';
      const jointFill = isLocked ? '#34d399' : isCoasting ? '#f59e0b' : isDominant ? '#38bdf8' : '#fbbf24';

      // Diagnostic View: Render Raw OBSERVED landmarks as subtle cyan crosshairs
      if (showDiagnostics && track.observedLandmarks && track.observedLandmarks.length === 21 && !isCoasting) {
        ctx.strokeStyle = 'rgba(56, 189, 248, 0.4)';
        ctx.lineWidth = 1;
        track.observedLandmarks.forEach((obs) => {
          const ox = obs.x * width;
          const oy = obs.y * height;
          ctx.beginPath();
          ctx.moveTo(ox - 3, oy);
          ctx.lineTo(ox + 3, oy);
          ctx.moveTo(ox, oy - 3);
          ctx.lineTo(ox, oy + 3);
          ctx.stroke();
        });
      }

      // Palm Membrane Polygon (Wrist -> Thumb CMC -> Index MCP -> Mid MCP -> Ring MCP -> Pinky MCP -> Wrist)
      ctx.beginPath();
      ctx.moveTo(lm[0].x * width, lm[0].y * height);
      ctx.lineTo(lm[1].x * width, lm[1].y * height);
      ctx.lineTo(lm[2].x * width, lm[2].y * height);
      ctx.lineTo(lm[5].x * width, lm[5].y * height);
      ctx.lineTo(lm[9].x * width, lm[9].y * height);
      ctx.lineTo(lm[13].x * width, lm[13].y * height);
      ctx.lineTo(lm[17].x * width, lm[17].y * height);
      ctx.closePath();
      ctx.fillStyle = isDominant ? 'rgba(56, 189, 248, 0.08)' : 'rgba(245, 158, 11, 0.08)';
      ctx.fill();

      // Finger Bone Chains with anatomical taper
      for (const [start, end] of HAND_CONNECTIONS) {
        const p1 = lm[start];
        const p2 = lm[end];
        if (p1 && p2) {
          const isMetacarpal = [0, 1, 5, 9, 13, 17].includes(start) && [1, 2, 5, 9, 13, 17].includes(end);
          ctx.lineWidth = isMetacarpal ? 3.2 : 2.0;
          ctx.strokeStyle = boneStroke;
          if (isCoasting) {
            ctx.setLineDash([3, 3]);
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

      // Joints & Fingertip Markers
      for (let i = 0; i < lm.length; i++) {
        const p = lm[i];
        const isTip = [4, 8, 12, 16, 20].includes(i);
        const radius = isTip ? 4.5 : 2.5;
        const isPredicted = p.provenance === 'PREDICTED' || isCoasting;

        ctx.beginPath();
        ctx.arc(p.x * width, p.y * height, radius, 0, 2 * Math.PI);
        if (isPredicted) {
          ctx.fillStyle = 'rgba(245, 158, 11, 0.3)';
          ctx.strokeStyle = '#f59e0b';
          ctx.lineWidth = 1.5;
          ctx.stroke();
          ctx.fill();
        } else {
          ctx.fillStyle = isTip ? '#ffffff' : jointFill;
          ctx.fill();
          ctx.strokeStyle = '#07090d';
          ctx.lineWidth = 1;
          ctx.stroke();
        }

        // Optional Diagnostic View: Render Landmark IDs (0..20)
        if (showDiagnostics) {
          ctx.fillStyle = isPredicted ? 'rgba(245, 158, 11, 0.9)' : 'rgba(226, 232, 240, 0.85)';
          ctx.font = '500 8px monospace';
          ctx.fillText(`${i}${isPredicted ? '*' : ''}`, p.x * width + 5, p.y * height - 3);
        }
      }

      // Track Identity & Role HUD Tag
      const wrist = lm[0];
      ctx.fillStyle = isCoasting ? '#f59e0b' : isDominant ? '#38bdf8' : '#fbbf24';
      ctx.font = 'bold 9px monospace';
      ctx.fillText(
        `TRK_${track.trackId}:${isCoasting ? 'COASTING' : isDominant ? 'DOMINANT' : 'BASE'} (${track.handedness})`,
        wrist.x * width - 40,
        wrist.y * height + 18
      );
    });

    // Diagnostic Mode Provenance Legend Overlay (Top Right)
    if (showDiagnostics) {
      ctx.fillStyle = 'rgba(10, 13, 20, 0.85)';
      ctx.strokeStyle = '#1e2430';
      ctx.lineWidth = 1;
      ctx.fillRect(width - 150, 12, 138, 54);
      ctx.strokeRect(width - 150, 12, 138, 54);

      ctx.font = 'bold 8px monospace';
      ctx.fillStyle = '#38bdf8';
      ctx.fillText('● FILTERED (1€ Tuned)', width - 142, 24);
      ctx.fillStyle = '#94a3b8';
      ctx.fillText('+ OBSERVED (Detector)', width - 142, 38);
      ctx.fillStyle = '#f59e0b';
      ctx.fillText('○ PREDICTED (Coasting)', width - 142, 52);
    }

    // 3. Two-Hand Inter-Contact Radar Aura (Vowel Touch Point)
    if (tracks.length >= 2) {
      const h1Idx = tracks[0].smoothedLandmarks[8];
      const h2Wrist = tracks[1].smoothedLandmarks[0];
      const h2Thumb = tracks[1].smoothedLandmarks[4];
      const h2Idx = tracks[1].smoothedLandmarks[8];
      const h2Mid = tracks[1].smoothedLandmarks[12];
      const h2Ring = tracks[1].smoothedLandmarks[16];
      const h2Pinky = tracks[1].smoothedLandmarks[20];

      const targets = [h2Thumb, h2Idx, h2Mid, h2Ring, h2Pinky];
      for (const t of targets) {
        const dx = (h1Idx.x - t.x) * width;
        const dy = (h1Idx.y - t.y) * height;
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d < 24) {
          // Touch Contact Lock Aura
          ctx.strokeStyle = '#10b981';
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.arc(t.x * width, t.y * height, 10, 0, 2 * Math.PI);
          ctx.stroke();
          ctx.beginPath();
          ctx.arc(t.x * width, t.y * height, 16, 0, 2 * Math.PI);
          ctx.stroke();
        }
      }
    }
  };

  // Real-Time Vision Pipeline
  const runVisionPipeline = () => {
    if (!videoRef.current || !canvasRef.current || videoRef.current.readyState < 2) {
      animationFrameRef.current = requestAnimationFrame(runVisionPipeline);
      return;
    }

    const t0 = performance.now();
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    // Stage A: Duplicate Frame Elimination (Saves mobile GPU/CPU cycles)
    if (video.currentTime === lastVideoTimeRef.current) {
      animationFrameRef.current = requestAnimationFrame(runVisionPipeline);
      return;
    }
    lastVideoTimeRef.current = video.currentTime;

    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
    }

    // Measure FPS
    frameCounterRef.current++;
    if (frameCounterRef.current % 15 === 0) {
      const dt = t0 - lastFrameTimestampRef.current;
      if (dt > 0) setFps(Math.round((15 / dt) * 1000));
      lastFrameTimestampRef.current = t0;
    }

    // Stage B: MediaPipe HandLandmarker Extraction
    const mpResult = webMediaPipe.detect(video, t0);
    const rawHands = (mpResult?.landmarks as LandmarkPoint[][]) || [];
    setTrackedHandCount(rawHands.length);

    // Stage C, D, E: Persistent Tracking + 1€ Filter + Quality Engine + Classifier
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
        recommendation: 'Position hands in viewfinder.',
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

    const t1 = performance.now();
    const procLatency = Number((t1 - t0).toFixed(1));
    setLatencyMs(procLatency);

    // Stability State Machine & Duplicate Prevention
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
      setStatusMessage(result.qualityReport.recommendation);
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
        setStatusMessage(`LOCKED: "${sign}" — RELEASE TO RETRANSITION`);
        isLockedNow = true;
      } else {
        const progress = Math.min(100, Math.round((candidateStreakRef.current / STABILITY_REQUIRED_FRAMES) * 100));
        setStabilityProgress(progress);
        setStatusMessage(`EVALUATING "${sign}" (${progress}%)`);

        if (candidateStreakRef.current >= STABILITY_REQUIRED_FRAMES) {
          isLockedNow = true;
          setStableLockedLetter(sign);
          commitLetter(sign);
          lastAcceptedLetterRef.current = sign;
          candidateStreakRef.current = 0;
          setStabilityProgress(100);
          setStatusMessage(`ACCEPTED: "${sign}"`);
        }
      }
    }

    // Benchmark & Calibration Session Sampling
    if (isCalibrating) {
      calibrationFramesRef.current.push({
        predicted: sign,
        latency: procLatency,
        jitter: result.qualityReport.jitterVariance,
        isTracked: rawHands.length > 0
      });

      if (t0 - calibrationStartTimeRef.current >= 10000) {
        // Conclude 10-second calibration
        concludeCalibration();
      }
    }

    // Stage F: Precision Viewfinder Canvas Rendering
    if (ctx) {
      renderPrecisionViewfinder(
        ctx,
        canvas.width,
        canvas.height,
        result.tracks,
        result.qualityReport,
        isLockedNow,
        showDiagnosticHUD
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

  // Calibration Protocol Handler
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
      avgFps: Math.round((total / 10)),
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

  const activeSign =
    currentResult?.sign !== 'NEUTRAL' && (currentResult?.confidence ?? 0) >= CONFIDENCE_MINIMUM
      ? currentResult?.sign
      : null;

  return (
    <div className="min-h-screen bg-[#07090d] text-[#e2e8f0] flex flex-col font-mono selection:bg-cyan-500 selection:text-black">
      {/* 1. Header Bar */}
      <header className="border-b border-[#181d26] bg-[#0c0f16]/95 backdrop-blur px-4 py-2.5">
        <div className="max-w-6xl mx-auto flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded border border-cyan-800/80 bg-cyan-950/60 flex items-center justify-center text-cyan-400 font-bold text-sm">
              SB
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-sm tracking-wider text-white">SIGNBRIDGE</span>
                <span className="text-[10px] px-1.5 py-0.5 rounded border border-[#232a37] bg-[#121620] text-slate-400">
                  SYS 2.1 • 1€ FILTER
                </span>
              </div>
              <p className="text-[11px] text-slate-400 font-sans">Multi-Stage Optical Tracking & Neural Alphabet Engine</p>
            </div>
          </div>

          {/* Telemetry Strip & Toggles */}
          <div className="flex items-center gap-2 text-[11px]">
            <button
              onClick={() => setShowDiagnosticHUD(!showDiagnosticHUD)}
              className={`px-2 py-1 rounded border flex items-center gap-1.5 transition-colors cursor-pointer ${
                showDiagnosticHUD
                  ? 'border-cyan-800 bg-cyan-950/80 text-cyan-300 font-bold'
                  : 'border-[#1e2430] bg-[#10141d] text-slate-400 hover:text-slate-200'
              }`}
            >
              <Crosshair className="w-3.5 h-3.5 text-cyan-400" />
              <span>{showDiagnosticHUD ? 'DIAG: ON' : 'DIAG'}</span>
            </button>

            <button
              onClick={() => setShowCalibrationDeck(!showCalibrationDeck)}
              className={`px-2 py-1 rounded border flex items-center gap-1.5 transition-colors cursor-pointer ${
                showCalibrationDeck
                  ? 'border-amber-800 bg-amber-950/80 text-amber-300 font-bold'
                  : 'border-[#1e2430] bg-[#10141d] text-slate-400 hover:text-slate-200'
              }`}
            >
              <Gauge className="w-3.5 h-3.5 text-amber-400" />
              <span>CALIBRATION</span>
            </button>

            <button
              onClick={() => setShowBenchmarkLab(!showBenchmarkLab)}
              className={`px-2 py-1 rounded border flex items-center gap-1.5 transition-colors cursor-pointer ${
                showBenchmarkLab
                  ? 'border-emerald-800 bg-emerald-950/80 text-emerald-300 font-bold'
                  : 'border-[#1e2430] bg-[#10141d] text-slate-400 hover:text-slate-200'
              }`}
            >
              <FlaskConical className="w-3.5 h-3.5 text-emerald-400" />
              <span>BENCHMARK LAB</span>
            </button>

            <button
              onClick={() => {
                const n = !isMuted;
                setIsMuted(n);
                speechService.setMuted(n);
              }}
              className={`px-2 py-1 rounded border flex items-center gap-1.5 transition-colors cursor-pointer ${
                isMuted
                  ? 'border-rose-900/60 bg-rose-950/40 text-rose-300'
                  : 'border-[#1e2430] bg-[#10141d] text-slate-300 hover:bg-[#151b27]'
              }`}
            >
              {isMuted ? <VolumeX className="w-3.5 h-3.5 text-rose-400" /> : <Volume2 className="w-3.5 h-3.5 text-cyan-400" />}
              <span className="hidden sm:inline">{isMuted ? 'MUTED' : 'VOICE'}</span>
            </button>

            <button
              onClick={() => setShowReferenceDrawer(!showReferenceDrawer)}
              className="px-2 py-1 rounded border border-[#1e2430] bg-[#10141d] text-slate-300 hover:bg-[#151b27] flex items-center gap-1.5 transition-colors cursor-pointer"
            >
              <Layers className="w-3.5 h-3.5 text-amber-400" />
              <span>REF</span>
              {showReferenceDrawer ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            </button>
          </div>
        </div>
      </header>

      {/* 2. Main Instrument Layout */}
      <main className="flex-1 max-w-6xl w-full mx-auto p-3 sm:p-5 space-y-3 font-sans">
        {/* Error Alert */}
        {cameraError && (
          <div className="p-3.5 rounded border border-rose-900/80 bg-rose-950/30 text-rose-300 text-xs flex items-start gap-3 font-mono">
            <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
            <div>
              <strong className="text-rose-200">OPTICAL ACQUISITION ERROR:</strong> {cameraError}
            </div>
          </div>
        )}

        {/* Real-World Calibration & Diagnostic Deck (Collapsible) */}
        {showCalibrationDeck && (
          <div className="p-4 rounded border border-[#1f2636] bg-[#0c1017] space-y-3 font-mono text-xs animate-fadeIn">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 border-b border-[#1b212c] pb-2">
              <span className="text-amber-400 font-bold tracking-wider flex items-center gap-2">
                <Gauge className="w-4 h-4" />
                EMPIRICAL CAMERA CALIBRATION & BENCHMARK BENCH
              </span>
              <span className="text-[11px] text-slate-400 font-sans">
                Real-World 10-Second Physical Device Evaluation Protocol
              </span>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <div className="flex items-center gap-1.5">
                <span className="text-slate-400">Target Sign:</span>
                <select
                  value={calibrationLetter}
                  onChange={(e) => setCalibrationLetter(e.target.value)}
                  disabled={isCalibrating}
                  className="bg-[#121620] border border-[#232a37] text-cyan-300 px-2 py-1 rounded text-xs"
                >
                  {Object.keys(ALPHABET_REFERENCE).map((l) => (
                    <option key={l} value={l}>
                      {l}
                    </option>
                  ))}
                </select>
              </div>

              <button
                onClick={() => startCalibrationSession(calibrationLetter)}
                disabled={isCalibrating || !isCameraActive}
                className="px-4 py-1.5 rounded border border-amber-600 bg-amber-600/20 hover:bg-amber-600/30 disabled:opacity-40 text-amber-300 font-bold flex items-center gap-1.5 cursor-pointer"
              >
                <Play className="w-3.5 h-3.5" />
                <span>{isCalibrating ? 'RECORDING (10s)...' : 'START 10s CALIBRATION'}</span>
              </button>
            </div>

            {/* Benchmark Report */}
            {calibrationResults && (
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-2 pt-2 border-t border-[#181e28]">
                <div className="p-2 rounded bg-[#090b10] border border-[#181d26]">
                  <span className="text-[10px] text-slate-500">SIGN</span>
                  <p className="text-base font-bold text-cyan-400">{calibrationResults.targetLetter}</p>
                </div>
                <div className="p-2 rounded bg-[#090b10] border border-[#181d26]">
                  <span className="text-[10px] text-slate-500">ACCURACY</span>
                  <p className="text-base font-bold text-emerald-400">{calibrationResults.accuracyRate}%</p>
                </div>
                <div className="p-2 rounded bg-[#090b10] border border-[#181d26]">
                  <span className="text-[10px] text-slate-500">CONTINUITY</span>
                  <p className="text-base font-bold text-slate-200">{calibrationResults.continuityRate}%</p>
                </div>
                <div className="p-2 rounded bg-[#090b10] border border-[#181d26]">
                  <span className="text-[10px] text-slate-500">CAMERA FPS</span>
                  <p className="text-base font-bold text-slate-200">{calibrationResults.avgFps}</p>
                </div>
                <div className="p-2 rounded bg-[#090b10] border border-[#181d26]">
                  <span className="text-[10px] text-slate-500">LATENCY</span>
                  <p className="text-base font-bold text-slate-200">{calibrationResults.avgLatencyMs} ms</p>
                </div>
                <div className="p-2 rounded bg-[#090b10] border border-[#181d26]">
                  <span className="text-[10px] text-slate-500">JITTER VAR</span>
                  <p className="text-base font-bold text-amber-400">{calibrationResults.avgJitter}</p>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Precision Tracking Benchmark Lab Deck (Collapsible) */}
        {showBenchmarkLab && (
          <div className="p-4 rounded border border-[#1f2636] bg-[#0c1017] space-y-4 font-mono text-xs animate-fadeIn">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 border-b border-[#1b212c] pb-2">
              <span className="text-emerald-400 font-bold tracking-wider flex items-center gap-2">
                <FlaskConical className="w-4 h-4 text-emerald-400" />
                PRECISION TRACKING RESEARCH BENCHMARK LAB
              </span>
              <span className="text-[11px] text-slate-400 font-sans">
                Deterministic Ground-Truth 90-Frame Benchmark (Static Jitter, High-Speed Translation, 3-Frame Occlusion)
              </span>
            </div>

            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="text-[11px] text-slate-400 font-sans max-w-xl">
                Executes an automated comparative test across 90 annotated ground-truth keypoint frames to evaluate
                Mean Absolute Error (MAE), Percentage of Correct Keypoints (PCK@10/20), Fingertip precision, and recovery continuity.
              </div>

              <button
                onClick={runBenchmarkSuite}
                disabled={isRunningBenchmark}
                className="px-4 py-2 rounded border border-emerald-600 bg-emerald-600/20 hover:bg-emerald-600/30 disabled:opacity-40 text-emerald-300 font-bold flex items-center gap-2 cursor-pointer transition-colors"
              >
                {isRunningBenchmark ? (
                  <>
                    <Activity className="w-4 h-4 animate-spin text-emerald-400" />
                    <span>BENCHMARKING 90 FRAMES...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 text-emerald-400" />
                    <span>RUN COMPARATIVE BENCHMARK</span>
                  </>
                )}
              </button>
            </div>

            {/* Benchmark Report Results */}
            {benchmarkReport && (
              <div className="space-y-3 pt-2 border-t border-[#181e28]">
                {/* Improvement Summary Cards */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                  <div className="p-2.5 rounded bg-[#090c12] border border-emerald-900/40">
                    <span className="text-[10px] text-slate-400 flex items-center gap-1">
                      <TrendingDown className="w-3 h-3 text-emerald-400" />
                      FINGERTIP PRECISION GAIN
                    </span>
                    <p className="text-lg font-bold text-emerald-400">
                      +{benchmarkReport.improvementMetrics.fingertipMAEReductionPct}%
                    </p>
                    <span className="text-[9px] text-slate-500">
                      {benchmarkReport.baselineCommit.fingertipMAE}px → {benchmarkReport.newPrecisionEngine.fingertipMAE}px
                    </span>
                  </div>

                  <div className="p-2.5 rounded bg-[#090c12] border border-emerald-900/40">
                    <span className="text-[10px] text-slate-400">PCK@10px ACCURACY GAIN</span>
                    <p className="text-lg font-bold text-cyan-400">
                      +{benchmarkReport.improvementMetrics.pck10GainPct}%
                    </p>
                    <span className="text-[9px] text-slate-500">Keypoints ≤10px from GT</span>
                  </div>

                  <div className="p-2.5 rounded bg-[#090c12] border border-emerald-900/40">
                    <span className="text-[10px] text-slate-400">TRACKING CONTINUITY</span>
                    <p className="text-lg font-bold text-slate-200">
                      {benchmarkReport.newPrecisionEngine.continuityRate}%
                    </p>
                    <span className="text-[9px] text-slate-500">Survives sensor dropouts</span>
                  </div>

                  <div className="p-2.5 rounded bg-[#090c12] border border-emerald-900/40">
                    <span className="text-[10px] text-slate-400">PIPELINE OVERHEAD</span>
                    <p className="text-lg font-bold text-amber-400">
                      {benchmarkReport.newPrecisionEngine.avgProcessingTimeMs} ms
                    </p>
                    <span className="text-[9px] text-slate-500">&lt;0.3ms frame budget</span>
                  </div>
                </div>

                {/* Comparative Table */}
                <div className="overflow-x-auto rounded border border-[#181d26] bg-[#07090e]">
                  <table className="w-full text-left text-[11px] font-mono">
                    <thead className="bg-[#0e121a] text-slate-400 border-b border-[#181d26]">
                      <tr>
                        <th className="p-2">PIPELINE ARCHITECTURE</th>
                        <th className="p-2">OVERALL MAE</th>
                        <th className="p-2">FINGERTIP MAE</th>
                        <th className="p-2">PCK @ 10px</th>
                        <th className="p-2">PCK @ 20px</th>
                        <th className="p-2">CONTINUITY</th>
                        <th className="p-2">OVERHEAD</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#141822]">
                      <tr className="text-slate-400 hover:bg-[#0c1017]">
                        <td className="p-2 font-bold text-slate-300">{benchmarkReport.rawMediaPipe.pipelineName}</td>
                        <td className="p-2">{benchmarkReport.rawMediaPipe.overallMAE} px</td>
                        <td className="p-2">{benchmarkReport.rawMediaPipe.fingertipMAE} px</td>
                        <td className="p-2">{benchmarkReport.rawMediaPipe.overallPCK10}%</td>
                        <td className="p-2">{benchmarkReport.rawMediaPipe.overallPCK20}%</td>
                        <td className="p-2 text-rose-400">{benchmarkReport.rawMediaPipe.continuityRate}%</td>
                        <td className="p-2">{benchmarkReport.rawMediaPipe.avgProcessingTimeMs} ms</td>
                      </tr>
                      <tr className="text-slate-400 hover:bg-[#0c1017]">
                        <td className="p-2 font-bold text-amber-300">{benchmarkReport.baselineCommit.pipelineName}</td>
                        <td className="p-2">{benchmarkReport.baselineCommit.overallMAE} px</td>
                        <td className="p-2">{benchmarkReport.baselineCommit.fingertipMAE} px</td>
                        <td className="p-2">{benchmarkReport.baselineCommit.overallPCK10}%</td>
                        <td className="p-2">{benchmarkReport.baselineCommit.overallPCK20}%</td>
                        <td className="p-2">{benchmarkReport.baselineCommit.continuityRate}%</td>
                        <td className="p-2">{benchmarkReport.baselineCommit.avgProcessingTimeMs} ms</td>
                      </tr>
                      <tr className="bg-emerald-950/20 text-emerald-300 font-bold border-t border-emerald-900/60">
                        <td className="p-2 text-emerald-400 flex items-center gap-1.5">
                          <Check className="w-3.5 h-3.5" />
                          {benchmarkReport.newPrecisionEngine.pipelineName}
                        </td>
                        <td className="p-2">{benchmarkReport.newPrecisionEngine.overallMAE} px</td>
                        <td className="p-2 text-white">{benchmarkReport.newPrecisionEngine.fingertipMAE} px</td>
                        <td className="p-2">{benchmarkReport.newPrecisionEngine.overallPCK10}%</td>
                        <td className="p-2">{benchmarkReport.newPrecisionEngine.overallPCK20}%</td>
                        <td className="p-2 text-emerald-400">{benchmarkReport.newPrecisionEngine.continuityRate}%</td>
                        <td className="p-2">{benchmarkReport.newPrecisionEngine.avgProcessingTimeMs} ms</td>
                      </tr>
                    </tbody>
                  </table>
                </div>

                {/* Per-Finger Breakdown */}
                <div className="p-2.5 rounded border border-[#181d26] bg-[#07090e] space-y-1.5">
                  <span className="text-[10px] text-slate-400 font-bold uppercase">
                    PER-FINGER MEAN ABSOLUTE ERROR (BASELINE bd86151 vs NEW HYBRID PRECISION)
                  </span>
                  <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 text-[10px]">
                    <div className="p-1.5 rounded bg-[#0b0e14] border border-[#161a22]">
                      <span className="text-slate-500">THUMB TIP (J4)</span>
                      <p className="font-bold text-emerald-400">
                        {benchmarkReport.newPrecisionEngine.perFingerMAE.thumb} px
                        <span className="text-slate-500 font-normal ml-1">({benchmarkReport.baselineCommit.perFingerMAE.thumb}px)</span>
                      </p>
                    </div>
                    <div className="p-1.5 rounded bg-[#0b0e14] border border-[#161a22]">
                      <span className="text-slate-500">INDEX TIP (J8)</span>
                      <p className="font-bold text-emerald-400">
                        {benchmarkReport.newPrecisionEngine.perFingerMAE.index} px
                        <span className="text-slate-500 font-normal ml-1">({benchmarkReport.baselineCommit.perFingerMAE.index}px)</span>
                      </p>
                    </div>
                    <div className="p-1.5 rounded bg-[#0b0e14] border border-[#161a22]">
                      <span className="text-slate-500">MIDDLE (J12)</span>
                      <p className="font-bold text-emerald-400">
                        {benchmarkReport.newPrecisionEngine.perFingerMAE.middle} px
                        <span className="text-slate-500 font-normal ml-1">({benchmarkReport.baselineCommit.perFingerMAE.middle}px)</span>
                      </p>
                    </div>
                    <div className="p-1.5 rounded bg-[#0b0e14] border border-[#161a22]">
                      <span className="text-slate-500">RING (J16)</span>
                      <p className="font-bold text-emerald-400">
                        {benchmarkReport.newPrecisionEngine.perFingerMAE.ring} px
                        <span className="text-slate-500 font-normal ml-1">({benchmarkReport.baselineCommit.perFingerMAE.ring}px)</span>
                      </p>
                    </div>
                    <div className="p-1.5 rounded bg-[#0b0e14] border border-[#161a22]">
                      <span className="text-slate-500">PINKY (J20)</span>
                      <p className="font-bold text-emerald-400">
                        {benchmarkReport.newPrecisionEngine.perFingerMAE.pinky} px
                        <span className="text-slate-500 font-normal ml-1">({benchmarkReport.baselineCommit.perFingerMAE.pinky}px)</span>
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Alphabet Specification & Linguistic Accordion */}
        {showReferenceDrawer && (
          <div className="p-4 rounded border border-[#1e2430] bg-[#0c1017] space-y-3 font-mono text-xs animate-fadeIn">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 border-b border-[#1b212c] pb-2">
              <span className="text-amber-400 font-bold tracking-wider flex items-center gap-2">
                <Sliders className="w-4 h-4" />
                CONVENTION DEFINITIONS: ISLRTC 2-HAND VS 1-HAND CHART
              </span>
              <span className="text-[11px] text-slate-400 font-sans">
                Supplied Poster: 1-Hand ASL (Trend Ent. 2001) | National Indian Standard: 2-Hand ISLRTC
              </span>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-2 text-[11px] font-mono">
              {Object.entries(ALPHABET_REFERENCE).map(([lettr, d]) => (
                <div key={lettr} className="p-2 rounded border border-[#181e29] bg-[#080b10] flex flex-col justify-between">
                  <div className="flex items-center justify-between border-b border-[#141822] pb-1">
                    <span className="font-bold text-cyan-400 text-sm">{lettr}</span>
                    <span className="text-[9px] px-1 py-0.2 rounded bg-[#10141d] text-slate-400">{d.standard}</span>
                  </div>
                  <p className="text-slate-400 text-[10px] mt-1.5 leading-snug line-clamp-3">{d.desc}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* 3. Primary Viewport & Telemetry Deck */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-3 items-start">
          {/* Viewfinder Column (7 cols) */}
          <div className="lg:col-span-7 space-y-2">
            <div className="relative aspect-[4/3] sm:aspect-video w-full rounded border border-[#1e2430] bg-[#05070a] overflow-hidden flex items-center justify-center shadow-lg">
              <video
                ref={videoRef}
                playsInline
                muted
                autoPlay
                className={`w-full h-full object-cover transform -scale-x-100 ${
                  isCameraActive ? 'opacity-100' : 'opacity-0'
                }`}
              />

              <canvas
                ref={canvasRef}
                className={`absolute inset-0 w-full h-full object-cover transform -scale-x-100 pointer-events-none ${
                  isCameraActive ? 'opacity-100' : 'opacity-0'
                }`}
              />

              {/* Standby Prompt */}
              {!isCameraActive && (
                <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center space-y-4 bg-[#080b10]">
                  <div className="w-12 h-12 rounded border border-[#232b3b] bg-[#0f131a] flex items-center justify-center text-cyan-400">
                    <Camera className="w-6 h-6" />
                  </div>
                  <div className="space-y-1">
                    <p className="text-sm font-mono font-bold tracking-wider text-slate-200">OPTICAL TRACKING OFFLINE</p>
                    <p className="text-xs text-slate-400 font-sans max-w-sm">
                      Initialize camera sensor to engage persistent 21-joint tracking with adaptive One Euro filtering.
                    </p>
                  </div>
                  <button
                    onClick={startCamera}
                    disabled={isInitializing}
                    className="px-6 py-2.5 rounded border border-cyan-500/80 bg-cyan-600 hover:bg-cyan-500 text-black font-mono font-bold text-xs tracking-wider uppercase transition-colors cursor-pointer flex items-center gap-2"
                  >
                    {isInitializing ? (
                      <>
                        <Activity className="w-4 h-4 animate-spin" />
                        <span>INITIALIZING ENGINE...</span>
                      </>
                    ) : (
                      <>
                        <Camera className="w-4 h-4" />
                        <span>START CAMERA</span>
                      </>
                    )}
                  </button>
                  <p className="text-[10px] text-slate-500 font-mono">
                    MULTI-STAGE: MEDIAPIPE + PERSISTENT TRACKING + 1€ FILTER + NEURAL MLP
                  </p>
                </div>
              )}

              {/* Viewport Top Telemetry HUD */}
              {isCameraActive && (
                <div className="absolute top-2.5 left-2.5 right-2.5 flex items-center justify-between text-[10px] font-mono pointer-events-none select-none">
                  <div className="flex items-center gap-1.5 bg-[#0a0d13]/90 px-2 py-1 rounded border border-[#1b222d] backdrop-blur text-slate-300">
                    <span className="text-emerald-400 font-bold">{fps} FPS</span>
                    <span className="text-slate-600">|</span>
                    <span className="text-cyan-400 font-bold">{latencyMs} ms</span>
                    <span className="text-slate-600">|</span>
                    <span className="text-amber-400">{trackedHandCount} HAND{trackedHandCount === 1 ? '' : 'S'}</span>
                  </div>

                  <div className="pointer-events-auto flex items-center gap-1.5">
                    <button
                      onClick={toggleFacingMode}
                      className="p-1 rounded bg-[#0a0d13]/90 border border-[#1b222d] text-slate-300 hover:text-white transition-colors cursor-pointer"
                      title="Flip Sensor"
                    >
                      <RotateCcw className="w-3.5 h-3.5" />
                    </button>
                    <button
                      onClick={stopCamera}
                      className="px-2.5 py-1 rounded border border-rose-800 bg-rose-600 hover:bg-rose-500 text-white font-mono text-[10px] font-bold tracking-wider cursor-pointer flex items-center gap-1"
                    >
                      <CameraOff className="w-3 h-3" />
                      <span>TERMINATE</span>
                    </button>
                  </div>
                </div>
              )}

              {/* Viewport Bottom Status Bar */}
              {isCameraActive && (
                <div className="absolute bottom-2.5 left-2.5 right-2.5 flex items-center justify-between text-[10px] font-mono pointer-events-none bg-[#0a0d13]/85 px-2.5 py-1 rounded border border-[#19202a]">
                  <div className="flex items-center gap-2 truncate">
                    <span
                      className={`px-1.5 py-0.2 rounded font-bold text-[9px] ${
                        trackingReport.status === 'TRACKING'
                          ? 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                          : trackingReport.status === 'TRACKING DEGRADED'
                          ? 'bg-amber-950 text-amber-400 border border-amber-800'
                          : 'bg-slate-800 text-slate-400 border border-slate-700'
                      }`}
                    >
                      {trackingReport.status}
                    </span>
                    <span className="text-slate-300 truncate">{statusMessage}</span>
                  </div>
                  <span className="text-cyan-400 font-bold">{trackingReport.qualityScore}% Q</span>
                </div>
              )}
            </div>

            {/* Mode & Telemetry Status Bar */}
            <div className="flex flex-wrap items-center justify-between gap-2 p-2 rounded border border-[#171c26] bg-[#0c0f16] text-xs font-mono">
              <span className="text-[11px] text-slate-400 flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-cyan-400" />
                CONVENTION:
              </span>
              <div className="flex items-center gap-1">
                {(['auto', 'islrtc_bimanual', 'single_hand_chart'] as AlphabetConvention[]).map((m) => (
                  <button
                    key={m}
                    onClick={() => setConvention(m)}
                    className={`px-2 py-0.5 rounded text-[10px] tracking-wider uppercase transition-colors cursor-pointer ${
                      convention === m
                        ? 'border border-cyan-800 bg-cyan-950/80 text-cyan-300 font-bold'
                        : 'border border-[#1a202c] bg-[#10141d] text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {m === 'auto' ? 'Auto-Detect' : m === 'islrtc_bimanual' ? 'ISLRTC Dual' : '1-Hand Chart'}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Telemetry & Composition Column (5 cols) */}
          <div className="lg:col-span-5 space-y-3 font-mono">
            {/* 1. Dedicated Tracking Quality Gauge */}
            <div className="p-3.5 rounded border border-[#1e2430] bg-[#0d1017] space-y-2.5 text-xs">
              <div className="flex items-center justify-between border-b border-[#161c26] pb-1.5">
                <span className="text-slate-400 tracking-wider uppercase flex items-center gap-1.5 font-bold">
                  <Gauge className="w-3.5 h-3.5 text-cyan-400" />
                  OPTICAL TRACKING QUALITY
                </span>
                <span className="text-emerald-400 font-bold">{trackingReport.qualityScore}% HEALTH</span>
              </div>

              <div className="grid grid-cols-3 gap-2 text-[10px]">
                <div className="p-1.5 rounded bg-[#07090e] border border-[#181d26]">
                  <span className="text-slate-500">STABILITY</span>
                  <p className="font-bold text-slate-200">{trackingReport.stabilityScore}%</p>
                </div>
                <div className="p-1.5 rounded bg-[#07090e] border border-[#181d26]">
                  <span className="text-slate-500">FRAMING</span>
                  <p className="font-bold text-cyan-400">{trackingReport.framing}</p>
                </div>
                <div className="p-1.5 rounded bg-[#07090e] border border-[#181d26]">
                  <span className="text-slate-500">JITTER VAR</span>
                  <p className="font-bold text-amber-400">{trackingReport.jitterVariance}</p>
                </div>
              </div>

              {/* Extended Diagnostic Telemetry Deck */}
              {showDiagnosticHUD && (
                <div className="space-y-2 pt-2 border-t border-[#161c26] text-[10px]">
                  <div className="grid grid-cols-3 gap-2">
                    <div className="p-1.5 rounded bg-[#07090e] border border-[#181d26]">
                      <span className="text-slate-500">BIOMECH</span>
                      <p className="font-bold text-emerald-400">{trackingReport.anatomicalIntegrityScore}%</p>
                    </div>
                    <div className="p-1.5 rounded bg-[#07090e] border border-[#181d26]">
                      <span className="text-slate-500">ID SWAPS</span>
                      <p className="font-bold text-slate-300">{trackingReport.identitySwitchesCount}</p>
                    </div>
                    <div className="p-1.5 rounded bg-[#07090e] border border-[#181d26]">
                      <span className="text-slate-500">REACQUIRED</span>
                      <p className="font-bold text-cyan-400">{trackingReport.reacquisitionsCount}</p>
                    </div>
                  </div>

                  {/* Per-Finger Telemetry */}
                  {trackingReport.perFingerTelemetry.length > 0 && (
                    <div className="space-y-1 pt-1">
                      <span className="text-[9px] text-slate-500 uppercase font-bold">PER-FINGER STABILITY</span>
                      <div className="space-y-1">
                        {trackingReport.perFingerTelemetry.map((f) => (
                          <div key={f.name} className="flex items-center justify-between gap-2">
                            <span className="text-[9px] text-slate-400 w-12">{f.name}</span>
                            <div className="flex-1 h-1.5 rounded-full bg-[#121620] overflow-hidden">
                              <div
                                className={`h-full transition-all duration-100 ${
                                  f.stabilityScore >= 70
                                    ? 'bg-emerald-400'
                                    : f.stabilityScore >= 40
                                    ? 'bg-cyan-400'
                                    : 'bg-amber-400'
                                }`}
                                style={{ width: `${f.stabilityScore}%` }}
                              />
                            </div>
                            <span className="text-[9px] text-slate-400 w-6 text-right">{f.stabilityScore}%</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* 2. Neural Recognition Readout Panel */}
            <div className="p-3.5 rounded border border-[#1e2430] bg-[#0d1017] space-y-3">
              <div className="flex items-center justify-between border-b border-[#161c26] pb-1.5 text-xs">
                <span className="text-slate-400 tracking-wider uppercase flex items-center gap-1.5 font-bold">
                  <Activity className="w-3.5 h-3.5 text-cyan-400" />
                  CLASSIFIER OUTPUT
                </span>
                <span className="text-[11px] text-slate-500 font-mono">
                  CONF: {Math.round((currentResult?.confidence ?? 0) * 100)}%
                </span>
              </div>

              <div className="flex items-center gap-4">
                <div className="w-20 h-20 rounded border border-[#222b3b] bg-[#07090e] flex items-center justify-center">
                  {activeSign ? (
                    <span className="text-4xl font-black text-cyan-400 tracking-widest animate-scaleUp">
                      {activeSign}
                    </span>
                  ) : (
                    <span className="text-2xl text-slate-700">--</span>
                  )}
                </div>

                <div className="flex-1 space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-bold text-slate-200">
                      {activeSign ? `LETTER: ${activeSign}` : 'NO TARGET GESTURE'}
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-400 font-sans leading-snug">
                    {activeSign && ALPHABET_REFERENCE[activeSign]
                      ? ALPHABET_REFERENCE[activeSign].desc
                      : 'Position hand within camera field to trigger recognition.'}
                  </p>
                </div>
              </div>

              {/* Lock Progress Meter */}
              <div className="space-y-1 pt-1 text-[11px]">
                <div className="flex items-center justify-between text-slate-400">
                  <span>TEMPORAL LOCK METER</span>
                  <span className="text-cyan-400 font-bold">{stabilityProgress}%</span>
                </div>
                <div className="w-full h-1.5 rounded-full bg-[#121620] overflow-hidden">
                  <div
                    className="h-full bg-cyan-400 transition-all duration-75"
                    style={{ width: `${stabilityProgress}%` }}
                  />
                </div>
              </div>

              <div className="flex items-center justify-between pt-1">
                <span className="text-[10px] text-slate-400">
                  {stableLockedLetter ? (
                    <span className="text-emerald-400 flex items-center gap-1">
                      <CheckCircle2 className="w-3 h-3" />
                      ACCEPTED IN TERMINAL
                    </span>
                  ) : (
                    'HOLD STEADY TO AUTO-ACCEPT'
                  )}
                </span>

                <button
                  onClick={handleManualAddLetter}
                  disabled={!activeSign}
                  className="px-2.5 py-1 rounded border border-[#232b3b] bg-[#121722] hover:bg-[#181f2e] disabled:opacity-40 text-cyan-400 text-[10px] font-bold uppercase transition-colors cursor-pointer"
                >
                  + Force Accept
                </button>
              </div>
            </div>

            {/* 3. Terminal Composition Deck */}
            <div className="p-3.5 rounded border border-[#1e2430] bg-[#0d1017] space-y-2.5 font-mono">
              <div className="flex items-center justify-between border-b border-[#161c26] pb-1.5 text-xs">
                <span className="text-slate-400 tracking-wider uppercase flex items-center gap-1.5 font-bold">
                  <Terminal className="w-3.5 h-3.5 text-emerald-400" />
                  OUTPUT TERMINAL
                </span>
                <span className="text-[11px] text-slate-500">{textBuffer.length} CHARS</span>
              </div>

              <div className="min-h-[64px] p-2.5 rounded border border-[#1a202c] bg-[#06080c] text-base tracking-widest text-emerald-300 break-words flex items-center justify-between select-all">
                <span>
                  {textBuffer || <span className="text-slate-600 text-xs tracking-normal">Awaiting input...</span>}
                  <span className="inline-block w-2 h-4 ml-1 bg-cyan-400 animate-pulse align-middle" />
                </span>
              </div>

              <div className="grid grid-cols-4 gap-1.5 text-[11px]">
                <button
                  onClick={handleSpace}
                  className="py-1.5 rounded border border-[#202735] bg-[#11151f] hover:bg-[#171d2b] text-slate-200 font-bold uppercase transition-colors cursor-pointer flex items-center justify-center gap-1"
                >
                  <Space className="w-3 h-3 text-cyan-400" />
                  <span>Space</span>
                </button>

                <button
                  onClick={handleBackspace}
                  className="py-1.5 rounded border border-[#202735] bg-[#11151f] hover:bg-[#171d2b] text-slate-200 font-bold uppercase transition-colors cursor-pointer flex items-center justify-center gap-1"
                >
                  <Delete className="w-3 h-3 text-amber-400" />
                  <span>Del</span>
                </button>

                <button
                  onClick={handleClear}
                  className="py-1.5 rounded border border-[#202735] bg-[#11151f] hover:bg-[#171d2b] text-slate-200 font-bold uppercase transition-colors cursor-pointer flex items-center justify-center gap-1"
                >
                  <RotateCcw className="w-3 h-3 text-rose-400" />
                  <span>Clear</span>
                </button>

                <button
                  onClick={handleSpeakText}
                  disabled={!textBuffer.trim()}
                  className="py-1.5 rounded border border-cyan-800 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-40 text-black font-bold uppercase transition-colors cursor-pointer flex items-center justify-center gap-1"
                >
                  <Volume2 className="w-3 h-3" />
                  <span>Speak</span>
                </button>
              </div>

              {textBuffer && (
                <button
                  onClick={handleCopy}
                  className="w-full py-1.5 rounded border border-[#1b222e] bg-[#0a0d13] hover:bg-[#111620] text-slate-300 text-[11px] font-bold uppercase transition-colors cursor-pointer flex items-center justify-center gap-1.5"
                >
                  {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5 text-slate-400" />}
                  <span>{copied ? 'BUFFER COPIED' : 'COPY BUFFER TO CLIPBOARD'}</span>
                </button>
              )}
            </div>
          </div>
        </div>

        {/* 4. Scientific Footnote */}
        <footer className="pt-2.5 border-t border-[#161c26] text-[10px] font-mono text-slate-500 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-1">
          <span>
            SIGNBRIDGE • MULTI-STAGE OPTICAL TRACKING ARCHITECTURE
          </span>
          <span className="text-slate-600">
            One Euro Adaptive Filter • Persistent Track Association • Invariant Neural Classification
          </span>
        </footer>
      </main>
    </div>
  );
}
