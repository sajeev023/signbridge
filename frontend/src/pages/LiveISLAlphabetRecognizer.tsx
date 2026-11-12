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
  Maximize2
} from 'lucide-react';
import { webMediaPipe } from '../services/webMediaPipeService';
import {
  alphabetInferenceEngine,
  AlphabetConvention,
  InferenceResult,
  LandmarkPoint
} from '../services/alphabetInferenceEngine';
import { speechService } from '../services/speechService';
import { HAND_CONNECTIONS } from '../services/cameraLandmarkDetector';

// ISLRTC & Single-Hand Manual Alphabet Specification
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

export function LiveISLAlphabetRecognizer() {
  // Telemetry & Camera State
  const [isCameraActive, setIsCameraActive] = useState<boolean>(false);
  const [isInitializing, setIsInitializing] = useState<boolean>(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [facingMode, setFacingMode] = useState<'user' | 'environment'>('user');
  const [fps, setFps] = useState<number>(0);
  const [latencyMs, setLatencyMs] = useState<number>(0);
  const [trackedHandCount, setTrackedHandCount] = useState<number>(0);

  // Scientific Convention Selection
  const [convention, setConvention] = useState<AlphabetConvention>('auto');

  // Recognition & Stability State Machine
  const [currentResult, setCurrentResult] = useState<InferenceResult | null>(null);
  const [stableLockedLetter, setStableLockedLetter] = useState<string | null>(null);
  const [stabilityProgress, setStabilityProgress] = useState<number>(0);
  const [statusMessage, setStatusMessage] = useState<string>('SYSTEM STANDBY');

  // Text Composition State
  const [textBuffer, setTextBuffer] = useState<string>('');
  const [isMuted, setIsMuted] = useState<boolean>(false);
  const [copied, setCopied] = useState<boolean>(false);
  const [showReferenceDrawer, setShowReferenceDrawer] = useState<boolean>(false);

  // DOM & Animation Refs
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Consecutive-frame stability counters
  const lastFrameTimestampRef = useRef<number>(performance.now());
  const frameCounterRef = useRef<number>(0);
  const candidateLetterRef = useRef<string | null>(null);
  const candidateStreakRef = useRef<number>(0);
  const lastAcceptedLetterRef = useRef<string | null>(null);
  const neutralFramesRef = useRef<number>(0);

  const STABILITY_REQUIRED_FRAMES = 7; // ~230ms at 30 FPS
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
    setIsCameraActive(false);
    setIsInitializing(false);
    setTrackedHandCount(0);
    setFps(0);
    setLatencyMs(0);
    setCurrentResult(null);
    setStableLockedLetter(null);
    setStabilityProgress(0);
    setStatusMessage('CAMERA TERMINATED');
  }, []);

  const startCamera = async () => {
    setCameraError(null);
    setIsInitializing(true);
    setStatusMessage('CALIBRATING NEURAL SENSORS...');

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

  // Render smoothed precision skeletal overlays
  const renderViewfinderOverlay = (
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    hands: LandmarkPoint[][],
    isLocked: boolean
  ) => {
    ctx.clearRect(0, 0, width, height);

    // Viewfinder corner marks (Precision HUD)
    ctx.strokeStyle = isLocked ? '#10b981' : 'rgba(148, 163, 184, 0.35)';
    ctx.lineWidth = 1.5;
    const corner = 24;
    const pad = 16;

    // Top-Left
    ctx.beginPath();
    ctx.moveTo(pad, pad + corner);
    ctx.lineTo(pad, pad);
    ctx.lineTo(pad + corner, pad);
    ctx.stroke();

    // Top-Right
    ctx.beginPath();
    ctx.moveTo(width - pad - corner, pad);
    ctx.lineTo(width - pad, pad);
    ctx.lineTo(width - pad, pad + corner);
    ctx.stroke();

    // Bottom-Left
    ctx.beginPath();
    ctx.moveTo(pad, height - pad - corner);
    ctx.lineTo(pad, height - pad);
    ctx.lineTo(pad + corner, height - pad);
    ctx.stroke();

    // Bottom-Right
    ctx.beginPath();
    ctx.moveTo(width - pad - corner, height - pad);
    ctx.lineTo(width - pad, height - pad);
    ctx.lineTo(width - pad, height - pad - corner);
    ctx.stroke();

    if (!hands || hands.length === 0) {
      // Crosshair center prompt
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.2)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(width / 2 - 12, height / 2);
      ctx.lineTo(width / 2 + 12, height / 2);
      ctx.moveTo(width / 2, height / 2 - 12);
      ctx.lineTo(width / 2, height / 2 + 12);
      ctx.stroke();
      return;
    }

    // Draw hands with precision anti-jitter strokes
    hands.forEach((hand, handIndex) => {
      const isPrimary = handIndex === 0;
      const boneStroke = isLocked
        ? 'rgba(16, 185, 129, 0.9)'
        : isPrimary
        ? 'rgba(56, 189, 248, 0.85)'
        : 'rgba(245, 158, 11, 0.85)';
      const jointFill = isLocked ? '#34d399' : isPrimary ? '#38bdf8' : '#fbbf24';

      // Bones
      ctx.lineWidth = 2.2;
      ctx.strokeStyle = boneStroke;
      for (const [start, end] of HAND_CONNECTIONS) {
        const p1 = hand[start];
        const p2 = hand[end];
        if (p1 && p2) {
          ctx.beginPath();
          ctx.moveTo(p1.x * width, p1.y * height);
          ctx.lineTo(p2.x * width, p2.y * height);
          ctx.stroke();
        }
      }

      // Joints
      for (let i = 0; i < hand.length; i++) {
        const p = hand[i];
        const isTip = [4, 8, 12, 16, 20].includes(i);
        const radius = isTip ? 4 : 2.5;

        ctx.beginPath();
        ctx.arc(p.x * width, p.y * height, radius, 0, 2 * Math.PI);
        ctx.fillStyle = isTip ? '#ffffff' : jointFill;
        ctx.fill();
        ctx.strokeStyle = '#090b0e';
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // Telemetry label
      const wrist = hand[0];
      ctx.fillStyle = isPrimary ? '#38bdf8' : '#fbbf24';
      ctx.font = '500 10px monospace';
      ctx.fillText(isPrimary ? 'H1:DOMINANT' : 'H2:BASE', wrist.x * width - 28, wrist.y * height + 16);
    });
  };

  // Real-time processing loop
  const runVisionPipeline = () => {
    if (!videoRef.current || !canvasRef.current || videoRef.current.readyState < 2) {
      animationFrameRef.current = requestAnimationFrame(runVisionPipeline);
      return;
    }

    const t0 = performance.now();
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
    }

    // Telemetry rate
    frameCounterRef.current++;
    if (frameCounterRef.current % 15 === 0) {
      const dt = t0 - lastFrameTimestampRef.current;
      if (dt > 0) setFps(Math.round((15 / dt) * 1000));
      lastFrameTimestampRef.current = t0;
    }

    // 1. MediaPipe Inference
    const mpResult = webMediaPipe.detect(video, t0);
    const rawHands = (mpResult?.landmarks as LandmarkPoint[][]) || [];
    setTrackedHandCount(rawHands.length);

    // 2. Neural Gesture Classification
    let result: InferenceResult = {
      sign: 'NEUTRAL',
      confidence: 0,
      isDynamic: false,
      handCount: rawHands.length,
      convention: 'ISLRTC (2-Hand)',
      allProbabilities: {}
    };

    if (rawHands.length > 0) {
      result = alphabetInferenceEngine.predict(rawHands, mpResult?.handedness, convention, t0);
    }
    setCurrentResult(result);

    const t1 = performance.now();
    setLatencyMs(Number((t1 - t0).toFixed(1)));

    // 3. Temporal Stability & Duplicate Prevention Filter
    const sign = result.sign;
    const conf = result.confidence;
    let isLockedNow = false;

    if (sign === 'NEUTRAL' || conf < CONFIDENCE_MINIMUM || rawHands.length === 0) {
      neutralFramesRef.current++;
      if (neutralFramesRef.current >= 5) {
        lastAcceptedLetterRef.current = null;
        setStableLockedLetter(null);
      }
      candidateStreakRef.current = 0;
      setStabilityProgress(0);
      setStatusMessage(rawHands.length === 0 ? 'SEARCHING FIELD OF VIEW' : 'TRANSITION / RESTING STATE');
    } else {
      neutralFramesRef.current = 0;

      if (sign === candidateLetterRef.current) {
        candidateStreakRef.current++;
      } else {
        candidateLetterRef.current = sign;
        candidateStreakRef.current = 1;
      }

      // Check if sign is already accepted and held
      if (sign === lastAcceptedLetterRef.current) {
        setStabilityProgress(100);
        setStatusMessage(`LOCKED: "${sign}" — RELEASE HAND TO RE-ENGAGE`);
        isLockedNow = true;
      } else {
        const progress = Math.min(100, Math.round((candidateStreakRef.current / STABILITY_REQUIRED_FRAMES) * 100));
        setStabilityProgress(progress);
        setStatusMessage(`EVALUATING "${sign}" (${progress}%)`);

        // Lock condition met!
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

    // 4. Render Precision HUD
    if (ctx) {
      renderViewfinderOverlay(
        ctx,
        canvas.width,
        canvas.height,
        result.smoothedLandmarks || rawHands,
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

  const activeSign =
    currentResult?.sign !== 'NEUTRAL' && (currentResult?.confidence ?? 0) >= CONFIDENCE_MINIMUM
      ? currentResult?.sign
      : null;

  return (
    <div className="min-h-screen bg-[#07090d] text-[#e2e8f0] flex flex-col font-mono selection:bg-cyan-500 selection:text-black">
      {/* 1. Scientific Instrument Header */}
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
                  SYS 2.1
                </span>
              </div>
              <p className="text-[11px] text-slate-400 font-sans">Real-Time Optical Sign Alphabet Recognition</p>
            </div>
          </div>

          {/* Telemetry Status Strip */}
          <div className="flex items-center gap-2 text-[11px]">
            <div className="px-2 py-1 rounded border border-[#1e2430] bg-[#10141d] flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${isCameraActive ? 'bg-emerald-400 animate-pulse' : 'bg-slate-600'}`} />
              <span className="text-slate-300">{isCameraActive ? 'ONLINE' : 'STANDBY'}</span>
            </div>

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
              className="px-2.5 py-1 rounded border border-[#1e2430] bg-[#10141d] text-slate-300 hover:bg-[#151b27] flex items-center gap-1.5 transition-colors cursor-pointer"
            >
              <Layers className="w-3.5 h-3.5 text-amber-400" />
              <span>CHART REF</span>
              {showReferenceDrawer ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            </button>
          </div>
        </div>
      </header>

      {/* 2. Main Instrument Layout */}
      <main className="flex-1 max-w-6xl w-full mx-auto p-3 sm:p-5 space-y-4 font-sans">
        {/* Error / Diagnostic Banner */}
        {cameraError && (
          <div className="p-3.5 rounded border border-rose-900/80 bg-rose-950/30 text-rose-300 text-xs flex items-start gap-3 font-mono">
            <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
            <div>
              <strong className="text-rose-200">HARDWARE INITIALIZATION FAULT:</strong> {cameraError}
            </div>
          </div>
        )}

        {/* Alphabet Specification & Linguistic Accordion */}
        {showReferenceDrawer && (
          <div className="p-4 rounded border border-[#1e2430] bg-[#0c1017] space-y-3 font-mono text-xs animate-fadeIn">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 border-b border-[#1b212c] pb-2">
              <span className="text-amber-400 font-bold tracking-wider flex items-center gap-2">
                <Sliders className="w-4 h-4" />
                LINGUISTIC SPECIFICATION: ISLRTC VS ASL CHART
              </span>
              <span className="text-[11px] text-slate-400 font-sans">
                Supplied Reference: 1-Hand ASL (Trend Ent. 2001) | Indian Standard: 2-Hand ISLRTC
              </span>
            </div>
            <p className="text-slate-300 text-[11px] font-sans leading-relaxed">
              Standard <strong>Indian Sign Language (ISL)</strong> fingerspelling certified by the ISLRTC is predominantly
              bimanual (two-handed), with vowels formed by indexing the base hand&apos;s 5 fingertips. However, single-handed
              fingerspelling (matching your supplied reference chart) is widely used in informal and mobile contexts.
              SignBridge supports <strong>both conventions</strong> seamlessly:
            </p>
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
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-start">
          {/* Viewfinder Column (7 cols) */}
          <div className="lg:col-span-7 space-y-2">
            <div className="relative aspect-[4/3] sm:aspect-video w-full rounded border border-[#1e2430] bg-[#05070a] overflow-hidden flex items-center justify-center shadow-lg">
              {/* Camera Video Stream */}
              <video
                ref={videoRef}
                playsInline
                muted
                autoPlay
                className={`w-full h-full object-cover transform -scale-x-100 ${
                  isCameraActive ? 'opacity-100' : 'opacity-0'
                }`}
              />

              {/* High-Precision Viewfinder Overlay Canvas */}
              <canvas
                ref={canvasRef}
                className={`absolute inset-0 w-full h-full object-cover transform -scale-x-100 pointer-events-none ${
                  isCameraActive ? 'opacity-100' : 'opacity-0'
                }`}
              />

              {/* Inactive Standby View */}
              {!isCameraActive && (
                <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center space-y-4 bg-[#080b10]">
                  <div className="w-12 h-12 rounded border border-[#232b3b] bg-[#0f131a] flex items-center justify-center text-cyan-400">
                    <Camera className="w-6 h-6" />
                  </div>
                  <div className="space-y-1">
                    <p className="text-sm font-mono font-bold tracking-wider text-slate-200">OPTICAL SENSOR IDLE</p>
                    <p className="text-xs text-slate-400 font-sans max-w-sm">
                      Initialize front camera for live hand-tracking and real-time neural alphabet recognition.
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
                        <span>INITIALIZING...</span>
                      </>
                    ) : (
                      <>
                        <Camera className="w-4 h-4" />
                        <span>START CAMERA</span>
                      </>
                    )}
                  </button>
                  <p className="text-[10px] text-slate-500 font-mono">
                    LOCAL RUNTIME: 100% IN-BROWSER INFERENCE • ZERO CLOUD TELEMETRY
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
                      title="Flip Sensor (Front/Rear)"
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
                <div className="absolute bottom-2.5 left-2.5 right-2.5 flex items-center justify-between text-[10px] font-mono pointer-events-none bg-[#0a0d13]/80 px-2.5 py-1 rounded border border-[#19202a]">
                  <span className="text-slate-300 truncate">{statusMessage}</span>
                  <span className="text-slate-500 font-bold uppercase">{currentResult?.convention || '2-HAND'}</span>
                </div>
              )}
            </div>

            {/* Convention Mode Selector Bar */}
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
            {/* 1. Recognition Readout Panel */}
            <div className="p-4 rounded border border-[#1e2430] bg-[#0d1017] space-y-3.5">
              <div className="flex items-center justify-between border-b border-[#161c26] pb-2 text-xs">
                <span className="text-slate-400 tracking-wider uppercase flex items-center gap-1.5 font-bold">
                  <Activity className="w-3.5 h-3.5 text-cyan-400" />
                  RECOGNITION TELEMETRY
                </span>
                <span className="text-[11px] text-slate-500 font-mono">
                  CONFIDENCE: {Math.round((currentResult?.confidence ?? 0) * 100)}%
                </span>
              </div>

              {/* Primary Output Display */}
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
                      {activeSign ? `CHARACTER: ${activeSign}` : 'NO TARGET GESTURE'}
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-400 font-sans leading-snug">
                    {activeSign && ALPHABET_REFERENCE[activeSign]
                      ? ALPHABET_REFERENCE[activeSign].desc
                      : 'Position hand within camera field to trigger recognition.'}
                  </p>
                </div>
              </div>

              {/* Stability Gauge */}
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

              {/* Quick Accept / Feedback Bar */}
              <div className="flex items-center justify-between pt-1">
                <span className="text-[10px] text-slate-400">
                  {stableLockedLetter ? (
                    <span className="text-emerald-400 flex items-center gap-1">
                      <CheckCircle2 className="w-3 h-3" />
                      LOCKED IN BUFFER
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

            {/* 2. Text Terminal Composition Deck */}
            <div className="p-4 rounded border border-[#1e2430] bg-[#0d1017] space-y-3 font-mono">
              <div className="flex items-center justify-between border-b border-[#161c26] pb-2 text-xs">
                <span className="text-slate-400 tracking-wider uppercase flex items-center gap-1.5 font-bold">
                  <Layers className="w-3.5 h-3.5 text-emerald-400" />
                  OUTPUT TERMINAL
                </span>
                <span className="text-[11px] text-slate-500">{textBuffer.length} CHARS</span>
              </div>

              {/* Monospace Output Monitor */}
              <div className="min-h-[70px] p-3 rounded border border-[#1a202c] bg-[#06080c] text-base tracking-widest text-emerald-300 break-words flex items-center justify-between select-all">
                <span>
                  {textBuffer || <span className="text-slate-600 text-xs tracking-normal">Awaiting input...</span>}
                  <span className="inline-block w-2 h-4 ml-1 bg-cyan-400 animate-pulse align-middle" />
                </span>
              </div>

              {/* Action Controls */}
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

              {/* Copy Action */}
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
        <footer className="pt-3 border-t border-[#161c26] text-[11px] font-mono text-slate-500 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2">
          <span>
            SIGNBRIDGE CORE • ASSISTIVE COMPUTER VISION & NEURAL KINEMATICS
          </span>
          <span className="text-slate-600">
            ISLRTC Bimanual + Unified Manual Alphabet Engine
          </span>
        </footer>
      </main>
    </div>
  );
}
