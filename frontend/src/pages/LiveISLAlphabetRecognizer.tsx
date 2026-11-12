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
  Sparkles,
  Info,
  Layers,
  ShieldCheck,
  Activity,
  Hand,
  BookOpen,
  ChevronDown,
  ChevronUp
} from 'lucide-react';
import { webMediaPipe } from '../services/webMediaPipeService';
import {
  alphabetInferenceEngine,
  InferenceResult,
  LandmarkPoint
} from '../services/alphabetInferenceEngine';
import { speechService } from '../services/speechService';
import { HAND_CONNECTIONS } from '../services/cameraLandmarkDetector';

// ISLRTC manual alphabet descriptions
const ISL_ALPHABET_GUIDE: Record<string, { desc: string; hands: '1-Hand' | '2-Hand' | 'Dynamic'; example: string }> = {
  A: { desc: 'Dominant index finger touches non-dominant thumb tip', hands: '2-Hand', example: 'Vowel A' },
  B: { desc: 'Both hands form loops (thumb+index) touching together like 8', hands: '2-Hand', example: 'Ball' },
  C: { desc: 'Single hand curved into a C arc shape', hands: '1-Hand', example: 'Cat' },
  D: { desc: 'Non-dominant vertical index + dominant semicircle loop touching top & base', hands: '2-Hand', example: 'Dog' },
  E: { desc: 'Dominant index finger touches non-dominant index finger tip', hands: '2-Hand', example: 'Vowel E' },
  F: { desc: 'Dominant index & middle fingers crossed across non-dominant index & middle', hands: '2-Hand', example: 'Fish' },
  G: { desc: 'Both hands curled into fists touching knuckles together', hands: '2-Hand', example: 'Goat' },
  H: { desc: 'Dominant flat open palm placed horizontally across non-dominant flat palm', hands: '2-Hand', example: 'Home' },
  I: { desc: 'Dominant index finger touches non-dominant middle finger tip', hands: '2-Hand', example: 'Vowel I' },
  J: { desc: 'Dominant index starts at middle tip (I) and carves downward curved J stroke', hands: 'Dynamic', example: 'Jug' },
  K: { desc: 'Dominant index bent like hook touching non-dominant vertical index PIP joint', hands: '2-Hand', example: 'Kite' },
  L: { desc: 'Single hand thumb and index extended at 90° right angle, others closed', hands: '1-Hand', example: 'Lion' },
  M: { desc: 'Dominant 3 fingers (index, middle, ring) placed downwards on non-dominant palm', hands: '2-Hand', example: 'Man' },
  N: { desc: 'Dominant 2 fingers (index, middle) placed downwards on non-dominant palm', hands: '2-Hand', example: 'Nest' },
  O: { desc: 'Dominant index finger touches non-dominant ring finger tip', hands: '2-Hand', example: 'Vowel O' },
  P: { desc: 'Non-dominant index vertical + dominant thumb-index loop touching top tip', hands: '2-Hand', example: 'Pen' },
  Q: { desc: 'Non-dominant thumb-index loop + dominant index touching bottom edge', hands: '2-Hand', example: 'Queen' },
  S: { desc: 'Both hands pinky fingers hooked and locked together', hands: '2-Hand', example: 'Sun' },
  T: { desc: 'Dominant index finger touches the edge of non-dominant index base', hands: '2-Hand', example: 'Tree' },
  U: { desc: 'Dominant index finger touches non-dominant pinky finger tip', hands: '2-Hand', example: 'Vowel U' },
  V: { desc: 'Dominant V-shape (index & middle spread) placed on non-dominant palm', hands: '2-Hand', example: 'Van' },
  W: { desc: 'Both hands fingers interlaced upright / 3 fingers in W on palm', hands: '2-Hand', example: 'Water' },
  X: { desc: 'Dominant index finger crossed over non-dominant index finger forming an X', hands: '2-Hand', example: 'X-ray' },
  Y: { desc: 'Dominant index finger placed in the cleft between non-dominant thumb and index', hands: '2-Hand', example: 'Yak' },
  Z: { desc: 'Dominant index finger traces 3-stroke zigzag Z trajectory across palm or air', hands: 'Dynamic', example: 'Zebra' }
};

export function LiveISLAlphabetRecognizer() {
  // Camera & Stream State
  const [isCameraActive, setIsCameraActive] = useState<boolean>(false);
  const [isInitializing, setIsInitializing] = useState<boolean>(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [facingMode, setFacingMode] = useState<'user' | 'environment'>('user');
  const [fps, setFps] = useState<number>(0);
  const [trackedHandCount, setTrackedHandCount] = useState<number>(0);

  // Model & Recognition State
  const [currentResult, setCurrentResult] = useState<InferenceResult | null>(null);
  const [stabilityProgress, setStabilityProgress] = useState<number>(0);
  const [stabilityStatus, setStabilityStatus] = useState<string>('Ready');
  const [requiresNeutral, setRequiresNeutral] = useState<boolean>(false);

  // English Text Builder State
  const [textBuffer, setTextBuffer] = useState<string>('');
  const [isMuted, setIsMuted] = useState<boolean>(false);
  const [copied, setCopied] = useState<boolean>(false);
  const [showVocabGuide, setShowVocabGuide] = useState<boolean>(false);

  // DOM Refs
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Real-time inference & stability tracking refs
  const lastFrameTimeRef = useRef<number>(performance.now());
  const frameCountRef = useRef<number>(0);
  const lastCandidateRef = useRef<string | null>(null);
  const candidateStreakRef = useRef<number>(0);
  const lastCommittedCharRef = useRef<string | null>(null);
  const neutralFramesRef = useRef<number>(0);

  // Required consecutive frames to commit a character (~300ms at 30fps)
  const STABILITY_THRESHOLD_FRAMES = 8;
  const CONFIDENCE_THRESHOLD = 0.70;

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  // Stop camera tracks
  const stopCamera = useCallback(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsCameraActive(false);
    setIsInitializing(false);
    setTrackedHandCount(0);
    setFps(0);
    setCurrentResult(null);
    setStabilityProgress(0);
    setStabilityStatus('Camera stopped');
  }, []);

  // Start camera stream
  const startCamera = async () => {
    setCameraError(null);
    setIsInitializing(true);
    setStabilityStatus('Initializing camera & neural model...');

    try {
      // 1. Initialize MediaPipe Web Vision HandLandmarker
      const mpReady = await webMediaPipe.init();
      if (!mpReady) {
        throw new Error('Failed to initialize client-side MediaPipe Hand Landmarker. Ensure WebAssembly is supported.');
      }

      // 2. Request camera with mobile front camera preference
      const constraints: MediaStreamConstraints = {
        video: {
          facingMode: facingMode,
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30 }
        },
        audio: false
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          if (videoRef.current) {
            videoRef.current.play().catch((err) => {
              console.warn('Video auto-play interrupted:', err);
            });
            setIsCameraActive(true);
            setIsInitializing(false);
            setStabilityStatus('Tracking active. Show an ISL alphabet sign.');
            runRecognitionLoop();
          }
        };
      }
    } catch (err: unknown) {
      console.error('Camera startup failure:', err);
      setIsInitializing(false);
      setIsCameraActive(false);

      if (err instanceof DOMException && (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError')) {
        setCameraError('Camera access was denied. Please allow camera permissions in your browser address bar.');
      } else if (err instanceof DOMException && err.name === 'NotFoundError') {
        setCameraError('No camera device was detected on your system. Please connect a webcam.');
      } else {
        setCameraError(err instanceof Error ? err.message : 'Failed to start camera. Please verify device permissions.');
      }
    }
  };

  // Flip camera (front/back on mobile)
  const toggleFacingMode = () => {
    const nextMode = facingMode === 'user' ? 'environment' : 'user';
    setFacingMode(nextMode);
    if (isCameraActive) {
      stopCamera();
      setTimeout(() => {
        startCamera();
      }, 200);
    }
  };

  // Draw hand skeleton on transparent overlay canvas
  const renderOverlays = (
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    landmarks: LandmarkPoint[][],
    predictedSign: string,
    isConfirmed: boolean
  ) => {
    ctx.clearRect(0, 0, width, height);

    if (!landmarks || landmarks.length === 0) {
      // Draw subtle placement reticle
      ctx.save();
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.25)';
      ctx.lineWidth = 2;
      ctx.setLineDash([8, 8]);
      const rw = width * 0.7;
      const rh = height * 0.65;
      const rx = (width - rw) / 2;
      const ry = (height - rh) / 2;
      ctx.strokeRect(rx, ry, rw, rh);
      ctx.restore();
      return;
    }

    landmarks.forEach((hand, handIdx) => {
      const isDominant = handIdx === 0;
      const boneColor = isDominant ? 'rgba(6, 182, 212, 0.85)' : 'rgba(16, 185, 129, 0.85)';
      const jointColor = isDominant ? '#22d3ee' : '#34d399';

      // 1. Draw Bones
      ctx.lineWidth = 3;
      ctx.strokeStyle = boneColor;
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

      // 2. Draw Joints & Highlights
      for (let i = 0; i < hand.length; i++) {
        const p = hand[i];
        const isTip = [4, 8, 12, 16, 20].includes(i);
        const radius = isTip ? (isConfirmed ? 6.5 : 5) : 3;

        ctx.beginPath();
        ctx.arc(p.x * width, p.y * height, radius, 0, 2 * Math.PI);
        ctx.fillStyle = isTip ? '#ffffff' : jointColor;
        ctx.fill();
        ctx.strokeStyle = isDominant ? '#0891b2' : '#059669';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      // Hand Tag
      const wrist = hand[0];
      ctx.fillStyle = isDominant ? 'rgba(6, 182, 212, 0.9)' : 'rgba(16, 185, 129, 0.9)';
      ctx.font = 'bold 11px sans-serif';
      ctx.fillText(isDominant ? 'Hand 1 (Dominant)' : 'Hand 2 (Base)', wrist.x * width - 40, wrist.y * height + 20);
    });
  };

  // Main real-time frame processing loop
  const runRecognitionLoop = () => {
    if (!videoRef.current || !canvasRef.current || videoRef.current.readyState < 2) {
      animationFrameRef.current = requestAnimationFrame(runRecognitionLoop);
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const now = performance.now();

    // Match canvas dimensions to video
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
    }

    // Measure FPS every 15 frames
    frameCountRef.current++;
    if (frameCountRef.current % 15 === 0) {
      const elapsed = now - lastFrameTimeRef.current;
      if (elapsed > 0) {
        setFps(Math.round((15 / elapsed) * 1000));
      }
      lastFrameTimeRef.current = now;
    }

    // 1. Run MediaPipe Landmark Detection
    const mpResult = webMediaPipe.detect(video, now);
    const landmarks = (mpResult?.landmarks as LandmarkPoint[][]) || [];
    setTrackedHandCount(landmarks.length);

    // 2. Run Spatio-Kinematic Neural Inference
    let result: InferenceResult = {
      sign: 'NEUTRAL',
      confidence: 0,
      isDynamic: false,
      handCount: landmarks.length,
      allProbabilities: {}
    };

    if (landmarks.length > 0) {
      result = alphabetInferenceEngine.predict(landmarks, mpResult?.handedness, now);
    }
    setCurrentResult(result);

    // 3. Temporal Stability & Duplicate Prevention State Machine
    const sign = result.sign;
    const conf = result.confidence;
    let isConfirmedFrame = false;

    if (sign === 'NEUTRAL' || conf < CONFIDENCE_THRESHOLD || landmarks.length === 0) {
      // User is in resting or transition state
      neutralFramesRef.current++;
      if (neutralFramesRef.current >= 6) {
        // Neutral transition satisfied: allow repeating characters
        lastCommittedCharRef.current = null;
        setRequiresNeutral(false);
      }
      candidateStreakRef.current = 0;
      setStabilityProgress(0);
      setStabilityStatus(landmarks.length === 0 ? 'Searching for hands...' : 'Neutral resting pose');
    } else {
      // Valid alphabet sign candidate
      neutralFramesRef.current = 0;

      if (sign === lastCandidateRef.current) {
        candidateStreakRef.current++;
      } else {
        lastCandidateRef.current = sign;
        candidateStreakRef.current = 1;
      }

      // Check duplicate hold prevention
      if (sign === lastCommittedCharRef.current) {
        setStabilityProgress(100);
        setStabilityStatus(`Holding "${sign}". Relax hands to type next.`);
        setRequiresNeutral(true);
      } else {
        // Holding candidate towards stability threshold
        const progress = Math.min(100, Math.round((candidateStreakRef.current / STABILITY_THRESHOLD_FRAMES) * 100));
        setStabilityProgress(progress);
        setStabilityStatus(`Holding "${sign}" (${progress}%)`);

        // Stability commitment reached!
        if (candidateStreakRef.current >= STABILITY_THRESHOLD_FRAMES) {
          isConfirmedFrame = true;
          commitCharacter(sign);
          lastCommittedCharRef.current = sign;
          candidateStreakRef.current = 0;
          setStabilityProgress(100);
          setStabilityStatus(`Confirmed "${sign}"!`);
          setRequiresNeutral(true);
        }
      }
    }

    // 4. Render Neon Landmark Overlays
    if (ctx) {
      renderOverlays(ctx, canvas.width, canvas.height, landmarks, sign, isConfirmedFrame);
    }

    animationFrameRef.current = requestAnimationFrame(runRecognitionLoop);
  };

  // Commit recognized character to word builder
  const commitCharacter = (char: string) => {
    setTextBuffer((prev) => prev + char);
    if (!isMuted) {
      speechService.speak(char, 'en', 1.05);
    }
  };

  // Text Builder Actions
  const handleSpace = () => {
    setTextBuffer((prev) => (prev.endsWith(' ') ? prev : prev + ' '));
    lastCommittedCharRef.current = null;
  };

  const handleBackspace = () => {
    setTextBuffer((prev) => prev.slice(0, -1));
    lastCommittedCharRef.current = null;
  };

  const handleClear = () => {
    setTextBuffer('');
    lastCommittedCharRef.current = null;
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

  const activeSign = currentResult?.sign !== 'NEUTRAL' && (currentResult?.confidence ?? 0) >= CONFIDENCE_THRESHOLD
    ? currentResult?.sign
    : null;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-cyan-500 selection:text-white">
      {/* 1. Header */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur-md sticky top-0 z-40 px-4 py-3 sm:px-6">
        <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-600 to-emerald-500 flex items-center justify-center shadow-lg shadow-cyan-500/20">
              <Hand className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl sm:text-2xl font-black tracking-tight text-white flex items-center gap-2">
                SIGNBRIDGE
                <span className="text-xs px-2 py-0.5 rounded-full bg-cyan-950 border border-cyan-800 text-cyan-300 font-mono font-medium">
                  ISLRTC v2.1
                </span>
              </h1>
              <p className="text-xs text-slate-400">Indian Sign Language Manual Alphabet Recognition System</p>
            </div>
          </div>

          {/* Quick Header Badges & Mute */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => {
                const next = !isMuted;
                setIsMuted(next);
                speechService.setMuted(next);
              }}
              className={`p-2 rounded-lg border text-xs font-medium flex items-center gap-1.5 transition-all ${
                isMuted
                  ? 'bg-rose-950/40 border-rose-800 text-rose-300'
                  : 'bg-slate-800 border-slate-700 text-slate-200 hover:bg-slate-700'
              }`}
              title={isMuted ? 'Unmute Audio' : 'Mute Audio'}
            >
              {isMuted ? <VolumeX className="w-4 h-4 text-rose-400" /> : <Volume2 className="w-4 h-4 text-cyan-400" />}
              <span className="hidden sm:inline">{isMuted ? 'Muted' : 'Voice Output'}</span>
            </button>

            <button
              onClick={() => setShowVocabGuide(!showVocabGuide)}
              className="px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs font-medium text-slate-200 flex items-center gap-1.5 transition-all"
            >
              <BookOpen className="w-4 h-4 text-emerald-400" />
              <span>ISL Alphabet Guide</span>
              {showVocabGuide ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
            </button>
          </div>
        </div>
      </header>

      {/* 2. Main Single-Page Content */}
      <main className="flex-1 max-w-6xl w-full mx-auto p-4 sm:p-6 space-y-5">
        {/* Diagnostic / Error Alert */}
        {cameraError && (
          <div className="p-4 rounded-xl bg-rose-950/60 border border-rose-800 text-rose-200 flex items-start gap-3 animate-fadeIn">
            <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
            <div className="text-sm">
              <p className="font-semibold text-rose-100">Camera Access Error</p>
              <p className="mt-0.5 text-rose-300">{cameraError}</p>
              <p className="mt-2 text-xs text-rose-400/90">
                Tip: On mobile devices, tap the lock or camera icon in your browser URL bar to grant camera permissions.
              </p>
            </div>
          </div>
        )}

        {/* Alphabet Guide Drawer (Collapsible) */}
        {showVocabGuide && (
          <div className="p-4 sm:p-5 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-xl space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-bold text-slate-200 uppercase tracking-wider flex items-center gap-2">
                <Layers className="w-4 h-4 text-cyan-400" />
                ISLRTC Manual Alphabet Reference (A–Z)
              </h2>
              <span className="text-xs text-slate-400">Official Indian Sign Language Standard</span>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-2 text-xs">
              {Object.entries(ISL_ALPHABET_GUIDE).map(([letter, info]) => (
                <div key={letter} className="p-2.5 rounded-lg bg-slate-950/80 border border-slate-800/80 flex flex-col justify-between">
                  <div className="flex items-center justify-between">
                    <span className="text-lg font-black text-cyan-400 font-mono">{letter}</span>
                    <span className="px-1.5 py-0.5 rounded text-[10px] bg-slate-800 text-slate-300 font-medium">
                      {info.hands}
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-400 mt-1 line-clamp-2" title={info.desc}>
                    {info.desc}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* 3. Primary Workspace Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
          {/* Left Column: Live Camera & Landmark Overlay Viewport (7 cols) */}
          <div className="lg:col-span-7 space-y-3">
            <div className="relative aspect-[4/3] sm:aspect-video w-full rounded-2xl overflow-hidden bg-slate-900 border-2 border-slate-800 shadow-2xl flex items-center justify-center">
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

              {/* Transparent Landmark Canvas Overlay */}
              <canvas
                ref={canvasRef}
                className={`absolute inset-0 w-full h-full object-cover transform -scale-x-100 pointer-events-none ${
                  isCameraActive ? 'opacity-100' : 'opacity-0'
                }`}
              />

              {/* Inactive Camera State with Large Prominent Button */}
              {!isCameraActive && (
                <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center bg-gradient-to-b from-slate-900/90 to-slate-950/95 space-y-4">
                  <div className="w-16 h-16 rounded-2xl bg-cyan-950/60 border border-cyan-700/50 flex items-center justify-center text-cyan-400 shadow-inner">
                    <Camera className="w-8 h-8" />
                  </div>
                  <div className="max-w-sm space-y-1">
                    <h3 className="text-lg font-bold text-white">Live Camera Recognition</h3>
                    <p className="text-xs text-slate-400">
                      Grant camera permission to start real-time MediaPipe hand tracking and AI ISL alphabet recognition.
                    </p>
                  </div>
                  <button
                    onClick={startCamera}
                    disabled={isInitializing}
                    className="px-8 py-3.5 rounded-xl bg-gradient-to-r from-cyan-500 to-emerald-500 hover:from-cyan-400 hover:to-emerald-400 text-slate-950 font-bold text-sm tracking-wide shadow-lg shadow-cyan-500/25 active:scale-95 transition-all flex items-center gap-2 cursor-pointer"
                  >
                    {isInitializing ? (
                      <>
                        <Activity className="w-5 h-5 animate-spin" />
                        <span>INITIALIZING...</span>
                      </>
                    ) : (
                      <>
                        <Camera className="w-5 h-5" />
                        <span>START CAMERA</span>
                      </>
                    )}
                  </button>
                  <p className="text-[11px] text-slate-500 flex items-center gap-1.5">
                    <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
                    100% Private: Video is processed locally in your browser
                  </p>
                </div>
              )}

              {/* Active Camera Floating HUD */}
              {isCameraActive && (
                <div className="absolute top-3 left-3 right-3 flex items-center justify-between pointer-events-none">
                  <div className="flex items-center gap-2">
                    <span className="px-2.5 py-1 rounded-md bg-slate-950/80 backdrop-blur-md border border-slate-700 text-xs font-mono font-medium text-emerald-400 flex items-center gap-1.5 shadow">
                      <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                      LIVE
                    </span>
                    <span className="px-2 py-1 rounded-md bg-slate-950/80 backdrop-blur-md border border-slate-700 text-xs font-mono text-slate-300 shadow">
                      {fps} FPS
                    </span>
                    <span className="px-2 py-1 rounded-md bg-slate-950/80 backdrop-blur-md border border-slate-700 text-xs font-mono text-cyan-300 shadow">
                      {trackedHandCount} {trackedHandCount === 1 ? 'Hand' : 'Hands'}
                    </span>
                  </div>

                  <div className="pointer-events-auto flex items-center gap-2">
                    <button
                      onClick={toggleFacingMode}
                      className="p-1.5 rounded-md bg-slate-950/80 backdrop-blur-md border border-slate-700 text-slate-300 hover:text-white hover:bg-slate-800 text-xs shadow transition-all"
                      title="Flip Camera (Front/Rear)"
                    >
                      <RotateCcw className="w-4 h-4" />
                    </button>
                    <button
                      onClick={stopCamera}
                      className="px-3 py-1 rounded-md bg-rose-600 hover:bg-rose-500 text-white text-xs font-bold shadow-lg shadow-rose-600/30 active:scale-95 transition-all flex items-center gap-1.5"
                    >
                      <CameraOff className="w-3.5 h-3.5" />
                      <span>STOP</span>
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Viewport Guidance Bar */}
            <div className="flex items-center justify-between text-xs px-2 text-slate-400">
              <span className="flex items-center gap-1.5">
                <Activity className="w-3.5 h-3.5 text-cyan-400" />
                <span>Status: <strong className="text-slate-200">{stabilityStatus}</strong></span>
              </span>
              <span className="font-mono text-[11px] text-slate-500">
                Front Facing • MediaPipe 21-pt 3D
              </span>
            </div>
          </div>

          {/* Right Column: AI Recognition Output & English Text Builder (5 cols) */}
          <div className="lg:col-span-5 space-y-4">
            {/* 1. Live Recognition Result Card */}
            <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 shadow-xl space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                  <Sparkles className="w-4 h-4 text-cyan-400" />
                  Live AI Recognition
                </span>
                {activeSign && (
                  <span className="text-xs font-mono px-2 py-0.5 rounded bg-emerald-950 border border-emerald-800 text-emerald-300">
                    {Math.round((currentResult?.confidence ?? 0) * 100)}% Confident
                  </span>
                )}
              </div>

              {/* Big Recognized Letter Display */}
              <div className="flex items-center gap-4">
                <div className="w-24 h-24 rounded-2xl bg-gradient-to-tr from-slate-950 to-slate-900 border-2 border-cyan-500/40 flex items-center justify-center shadow-inner">
                  {activeSign ? (
                    <span className="text-5xl font-black text-cyan-400 font-mono tracking-wider animate-scaleUp">
                      {activeSign}
                    </span>
                  ) : (
                    <span className="text-3xl font-mono text-slate-600">--</span>
                  )}
                </div>

                <div className="flex-1 space-y-1.5">
                  <h4 className="text-base font-bold text-white">
                    {activeSign ? `Letter ${activeSign}` : 'No Sign Detected'}
                  </h4>
                  <p className="text-xs text-slate-400">
                    {activeSign && ISL_ALPHABET_GUIDE[activeSign]
                      ? ISL_ALPHABET_GUIDE[activeSign].desc
                      : 'Show one or both hands in camera view to sign.'}
                  </p>
                  {requiresNeutral && (
                    <p className="text-[11px] text-amber-400 flex items-center gap-1">
                      <span>✓ Character entered. Relax hands for next.</span>
                    </p>
                  )}
                </div>
              </div>

              {/* Stability Progress Bar */}
              <div className="space-y-1.5 pt-1">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-400">Temporal Stability Hold</span>
                  <span className="font-mono text-cyan-400">{stabilityProgress}%</span>
                </div>
                <div className="w-full h-2 rounded-full bg-slate-950 border border-slate-800 overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-cyan-500 to-emerald-400 transition-all duration-100 ease-out"
                    style={{ width: `${stabilityProgress}%` }}
                  />
                </div>
              </div>
            </div>

            {/* 2. English Text Builder Card */}
            <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 shadow-xl space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                  <Layers className="w-4 h-4 text-emerald-400" />
                  English Text Output
                </span>
                <span className="text-xs text-slate-500 font-mono">
                  {textBuffer.length} characters
                </span>
              </div>

              {/* Accumulated Text Box */}
              <div className="min-h-[80px] p-3.5 rounded-xl bg-slate-950 border border-slate-800 text-lg sm:text-xl font-mono tracking-widest text-emerald-300 break-words flex items-center justify-between select-all">
                <span>
                  {textBuffer || <span className="text-slate-600 text-sm tracking-normal">Formed words will appear here...</span>}
                  <span className="inline-block w-2.5 h-5 ml-1 bg-cyan-400 animate-pulse align-middle" />
                </span>
              </div>

              {/* Text Control Actions */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                <button
                  onClick={handleSpace}
                  className="px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 active:scale-95 text-slate-200 text-xs font-semibold flex items-center justify-center gap-1.5 border border-slate-700 transition-all"
                >
                  <Space className="w-4 h-4 text-cyan-400" />
                  <span>Space</span>
                </button>

                <button
                  onClick={handleBackspace}
                  className="px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 active:scale-95 text-slate-200 text-xs font-semibold flex items-center justify-center gap-1.5 border border-slate-700 transition-all"
                >
                  <Delete className="w-4 h-4 text-amber-400" />
                  <span>Backspace</span>
                </button>

                <button
                  onClick={handleClear}
                  className="px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 active:scale-95 text-slate-200 text-xs font-semibold flex items-center justify-center gap-1.5 border border-slate-700 transition-all"
                >
                  <RotateCcw className="w-4 h-4 text-rose-400" />
                  <span>Clear</span>
                </button>

                <button
                  onClick={handleSpeakText}
                  disabled={!textBuffer.trim()}
                  className="px-3 py-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 disabled:opacity-40 disabled:hover:bg-cyan-600 active:scale-95 text-white text-xs font-bold flex items-center justify-center gap-1.5 shadow-md shadow-cyan-600/20 transition-all"
                >
                  <Volume2 className="w-4 h-4" />
                  <span>Speak</span>
                </button>
              </div>

              {/* Copy Helper */}
              {textBuffer && (
                <button
                  onClick={handleCopy}
                  className="w-full py-2 rounded-lg bg-slate-950 hover:bg-slate-800 border border-slate-800 text-xs text-slate-300 font-medium flex items-center justify-center gap-1.5 transition-all"
                >
                  {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5 text-slate-400" />}
                  <span>{copied ? 'Copied to Clipboard!' : 'Copy Formed Text'}</span>
                </button>
              )}
            </div>
          </div>
        </div>

        {/* 4. Scientific Footnote & Disclosures */}
        <footer className="pt-4 border-t border-slate-800/80 space-y-2 text-xs text-slate-400">
          <div className="flex items-start gap-2">
            <Info className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
            <p>
              <strong>Linguistic Note:</strong> Manual alphabet fingerspelling represents spoken English orthography and is distinct from natural continuous Indian Sign Language (ISL) grammar and spatial syntax.
            </p>
          </div>
          <div className="flex items-start gap-2">
            <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
            <p>
              <strong>Privacy Architecture:</strong> All video decoding, MediaPipe 21-point hand landmark extraction, and neural network classification execute 100% locally on your device hardware. Zero camera frames or biometric recordings are transmitted or stored.
            </p>
          </div>
        </footer>
      </main>
    </div>
  );
}
