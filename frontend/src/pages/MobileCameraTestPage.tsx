import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Camera, 
  CameraOff, 
  RefreshCw, 
  FlipHorizontal, 
  Volume2, 
  VolumeX, 
  Sparkles, 
  CheckCircle2, 
  AlertCircle, 
  Hand, 
  Activity, 
  Gauge, 
  Zap, 
  Sliders, 
  ShieldCheck, 
  Copy, 
  Check,
  ChevronRight,
  BatteryCharging
} from 'lucide-react';
import { webMediaPipe } from '../services/webMediaPipeService';
import { coordinateAiEngine, CoordinateRecognitionResult } from '../services/coordinateAiEngine';
import { apiClient } from '../services/apiClient';
import { speechService } from '../services/speechService';
import { LandmarkRenderer } from '../services/cameraLandmarkDetector';
import { SupportedLanguage } from '../types';

export const MobileCameraTestPage: React.FC = () => {
  // Camera & Stream States
  const [isCameraActive, setIsCameraActive] = useState<boolean>(false);
  const [isStartingCamera, setIsStartingCamera] = useState<boolean>(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [facingMode, setFacingMode] = useState<'user' | 'environment'>('user');
  const [detectedHandsCount, setDetectedHandsCount] = useState<number>(0);

  // Recognition Mode: 'alphabet' (CoordinateMLP 94-D) or 'words' (10 ISL Vocabulary signs)
  const [recognitionMode, setRecognitionMode] = useState<'alphabet' | 'words'>('alphabet');
  
  // Real-time Recognition Telemetry
  const [predictedSign, setPredictedSign] = useState<string>('NEUTRAL');
  const [confidence, setConfidence] = useState<number>(0.0);
  const [isAccepted, setIsAccepted] = useState<boolean>(false);
  const [statusMessage, setStatusMessage] = useState<string>('Ready to start mobile camera');
  const [fps, setFps] = useState<number>(0);
  const [latencyMs, setLatencyMs] = useState<number>(0);

  // Thermal & Power Optimization (Mobile Overheat Prevention)
  const [throttleMode, setThrottleMode] = useState<'balanced' | 'high' | 'eco'>('balanced');

  // Audio / Speech State
  const [isMuted, setIsMuted] = useState<boolean>(false);
  const [selectedLanguage, setSelectedLanguage] = useState<SupportedLanguage>('en');

  // Layer C AI Assistance State
  const [aiStatus, setAiStatus] = useState<any>({
    active_provider: 'local_deterministic_engine',
    cloud_configured: false
  });
  const [recognizedTokens, setRecognizedTokens] = useState<string[]>([]);
  const [composedSentence, setComposedSentence] = useState<string | null>(null);
  const [isComposing, setIsComposing] = useState<boolean>(false);
  const [copied, setCopied] = useState<boolean>(false);

  // References
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const lastFrameTimeRef = useRef<number>(performance.now());
  const frameCounterRef = useRef<number>(0);
  const isProcessingBackendRef = useRef<boolean>(false);
  const lastSpokenRef = useRef<{ sign: string; time: number }>({ sign: '', time: 0 });

  // Load AI provider status on mount
  useEffect(() => {
    apiClient.getAIStatus().then((st) => setAiStatus(st));
    webMediaPipe.init().catch(() => {});

    // Orientation change listener for responsive re-alignment
    const handleResize = () => {
      if (canvasRef.current && videoRef.current) {
        canvasRef.current.width = videoRef.current.videoWidth || 640;
        canvasRef.current.height = videoRef.current.videoHeight || 480;
      }
    };
    window.addEventListener('resize', handleResize);
    window.addEventListener('orientationchange', handleResize);

    return () => {
      stopCamera();
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('orientationchange', handleResize);
    };
  }, []);

  // Frame throttle target interval based on mobile power mode
  const getThrottleIntervalMs = useCallback(() => {
    if (throttleMode === 'eco') return 80;       // ~12 FPS: maximum battery saving, zero overheat
    if (throttleMode === 'balanced') return 50;  // ~20 FPS: optimal mobile balance
    return 33;                                  // ~30 FPS: high performance
  }, [throttleMode]);

  // Start Camera with permission and constraints
  const startCamera = async (mode: 'user' | 'environment' = facingMode) => {
    setCameraError(null);
    setIsStartingCamera(true);
    setStatusMessage('Requesting camera access...');

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Camera media devices API not available. Ensure you are on HTTPS or localhost.');
      }

      // Stop any existing stream
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }

      await webMediaPipe.init();

      // Mobile camera constraints: ideal facingMode with fallback
      const constraints: MediaStreamConstraints = {
        video: {
          facingMode: { ideal: mode },
          width: { ideal: 640 },
          height: { ideal: 480 }
        },
        audio: false
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.setAttribute('playsinline', 'true');
        videoRef.current.setAttribute('webkit-playsinline', 'true');
        videoRef.current.muted = true;
        await videoRef.current.play();
      }

      setIsCameraActive(true);
      setIsStartingCamera(false);
      setStatusMessage('Camera active • Show hand sign');
      startInferenceLoop();
    } catch (err: any) {
      setIsCameraActive(false);
      setIsStartingCamera(false);
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setCameraError('Camera permission was denied. Please tap the lock/settings icon in your browser address bar and allow camera access.');
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        setCameraError('No camera found on this device.');
      } else {
        setCameraError(`Camera Error: ${err.message || 'Unable to access camera'}`);
      }
      setStatusMessage('Camera error occurred');
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
    setIsCameraActive(false);
    setDetectedHandsCount(0);
    setStatusMessage('Camera stopped');
  };

  const toggleFacingMode = () => {
    const nextMode = facingMode === 'user' ? 'environment' : 'user';
    setFacingMode(nextMode);
    if (isCameraActive) {
      startCamera(nextMode);
    }
  };

  // Real-Time Inference Loop
  const startInferenceLoop = () => {
    let lastInferenceTime = 0;
    let fpsFrames = 0;
    let fpsStartTime = performance.now();

    const loop = (currentTime: number) => {
      const video = videoRef.current;
      const canvas = canvasRef.current;

      if (video && canvas && video.readyState >= 2) {
        const ctx = canvas.getContext('2d');
        if (ctx) {
          canvas.width = video.videoWidth || 640;
          canvas.height = video.videoHeight || 480;

          // Draw camera frame (mirror if front camera)
          ctx.save();
          if (facingMode === 'user') {
            ctx.translate(canvas.width, 0);
            ctx.scale(-1, 1);
          }
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          ctx.restore();

          // Calculate FPS
          fpsFrames++;
          const fpsElapsed = currentTime - fpsStartTime;
          if (fpsElapsed >= 1000) {
            setFps(Math.round((fpsFrames * 1000) / fpsElapsed));
            fpsFrames = 0;
            fpsStartTime = currentTime;
          }

          // Throttle detection to prevent mobile CPU overload
          const intervalMs = getThrottleIntervalMs();
          if (currentTime - lastInferenceTime >= intervalMs) {
            lastInferenceTime = currentTime;
            frameCounterRef.current++;

            const t0 = performance.now();
            const detection = webMediaPipe.detect(video, currentTime);

            if (detection && detection.landmarks && detection.landmarks.length > 0) {
              const handsCount = detection.landmarks.length;
              setDetectedHandsCount(handsCount);

              let leftHand: any[] | null = null;
              let rightHand: any[] | null = null;

              for (let i = 0; i < handsCount; i++) {
                const lm = detection.landmarks[i];
                const handedness = detection.handednesses?.[i]?.[0]?.categoryName || 'Right';
                if (handedness === 'Left') leftHand = lm;
                else rightHand = lm;
              }

              // Draw neon landmark overlays
              LandmarkRenderer.drawLandmarks(ctx, canvas.width, canvas.height, {
                leftHand: leftHand as any,
                rightHand: rightHand as any,
                pose: null
              });

              // BRANCH A: Alphabet Recognition (94-D CoordinateMLP on primary hand)
              if (recognitionMode === 'alphabet') {
                const primaryHand = rightHand || leftHand || detection.landmarks[0];
                const result: CoordinateRecognitionResult = coordinateAiEngine.recognizeHand(primaryHand, currentTime);
                const elapsed = performance.now() - t0;
                setLatencyMs(Math.round(elapsed * 10) / 10);

                if (result.isAccepted) {
                  setPredictedSign(result.predictedLetter);
                  setConfidence(result.confidence);
                  setIsAccepted(true);
                  setStatusMessage(`Recognized Letter: ${result.predictedLetter}`);

                  // Buffer recognized token
                  commitToken(result.predictedLetter);
                } else {
                  setPredictedSign(result.bestCandidate || 'UNCERTAIN');
                  setConfidence(result.confidence);
                  setIsAccepted(false);
                  setStatusMessage(result.confidence < 0.70 ? 'Low confidence rejection (< 70%)' : 'Stabilizing gesture...');
                }
              }

              // BRANCH B: Word Vocabulary Recognition (10 ISL Signs via Backend BiLSTM)
              else {
                if (!isProcessingBackendRef.current) {
                  isProcessingBackendRef.current = true;
                  apiClient.predictFrame({
                    frame_id: frameCounterRef.current,
                    timestamp_ms: currentTime,
                    has_hands: true,
                    hand_count: handsCount,
                    left_hand: leftHand,
                    right_hand: rightHand,
                    mode: 'words'
                  }).then((res) => {
                    const elapsed = performance.now() - t0;
                    setLatencyMs(Math.round(elapsed * 10) / 10);
                    if (res.status === 'recognized') {
                      const signName = res.sign_id || res.predicted_sign || 'UNKNOWN';
                      setPredictedSign(signName.toUpperCase());
                      setConfidence(res.confidence || 0.95);
                      setIsAccepted(true);
                      setStatusMessage(`Recognized ISL Sign: ${signName}`);
                      commitToken(signName.toUpperCase());
                    } else if (res.status === 'buffering') {
                      setIsAccepted(false);
                      setPredictedSign('BUFFERING');
                      setStatusMessage(res.message || 'Buffering motion sequence...');
                    } else {
                      setIsAccepted(false);
                      setPredictedSign('UNCERTAIN');
                      setConfidence(res.confidence || 0.0);
                      setStatusMessage('Uncertain motion • Please hold sign');
                    }
                  }).catch(() => {
                    // Fallback
                  }).finally(() => {
                    isProcessingBackendRef.current = false;
                  });
                }
              }
            } else {
              // No hand detected
              setDetectedHandsCount(0);
              setIsAccepted(false);
              if (frameCounterRef.current % 10 === 0) {
                setPredictedSign('NO HAND');
                setConfidence(0.0);
                setStatusMessage('Place your hand in front of the camera');
              }
            }
          }
        }
      }
      animFrameRef.current = requestAnimationFrame(loop);
    };

    animFrameRef.current = requestAnimationFrame(loop);
  };

  // Commit recognized token with debounce and speech
  const commitToken = (token: string) => {
    const now = Date.now();
    if (token === lastSpokenRef.current.sign && now - lastSpokenRef.current.time < 2000) {
      return; // Debounce repeated tokens
    }

    lastSpokenRef.current = { sign: token, time: now };

    setRecognizedTokens((prev) => {
      const next = [...prev, token].slice(-8); // keep last 8 tokens
      return next;
    });

    if (!isMuted) {
      speechService.speak(token, selectedLanguage);
    }
  };

  // Layer C: AI Sentence Composition
  const handleComposeSentence = async () => {
    if (recognizedTokens.length === 0) return;
    setIsComposing(true);
    try {
      const res = await apiClient.composeSentence(recognizedTokens, selectedLanguage);
      setComposedSentence(res.composed_text);
      if (!isMuted && res.composed_text) {
        speechService.speak(res.composed_text, selectedLanguage);
      }
    } catch {
      // ignore
    } finally {
      setIsComposing(false);
    }
  };

  return (
    <div className="min-h-screen bg-navy-950 text-white flex flex-col p-3 pb-24 sm:p-6 max-w-lg mx-auto">
      {/* Top Mobile Bar */}
      <div className="flex items-center justify-between py-2 border-b border-navy-800 mb-3">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-xl bg-brand-600 flex items-center justify-center shadow-glow">
            <Camera className="w-4 h-4 text-white" />
          </div>
          <div>
            <h1 className="text-sm font-bold text-white tracking-tight">SIGNBRIDGE Mobile</h1>
            <p className="text-[10px] text-slate-400 font-mono">Real-Camera Field Acceptance</p>
          </div>
        </div>

        {/* Top Status Badges */}
        <div className="flex items-center gap-1.5">
          <div className={`px-2 py-0.5 rounded-full text-[10px] font-mono border flex items-center gap-1 ${
            isCameraActive ? 'bg-emerald-950/80 text-emerald-400 border-emerald-800' : 'bg-navy-900 text-slate-400 border-navy-800'
          }`}>
            <span className={`w-1.5 h-1.5 rounded-full ${isCameraActive ? 'bg-emerald-400 animate-pulse' : 'bg-slate-500'}`} />
            <span>{isCameraActive ? 'LIVE' : 'IDLE'}</span>
          </div>

          <button
            onClick={() => setIsMuted(!isMuted)}
            className={`p-1.5 rounded-lg border ${
              isMuted ? 'bg-rose-950 text-rose-300 border-rose-800' : 'bg-navy-900 text-slate-300 border-navy-800'
            }`}
          >
            {isMuted ? <VolumeX className="w-3.5 h-3.5" /> : <Volume2 className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>

      {/* Camera Viewport with Touch Controls */}
      <div className="relative aspect-[4/3] bg-black rounded-3xl overflow-hidden border border-navy-800 shadow-2xl flex items-center justify-center">
        <video 
          ref={videoRef} 
          playsInline 
          webkit-playsinline="true"
          muted 
          className="hidden" 
        />
        <canvas 
          ref={canvasRef} 
          className="w-full h-full object-cover" 
        />

        {/* Live HUD Overlay */}
        {isCameraActive && (
          <>
            <div className="absolute top-3 left-3 right-3 flex items-center justify-between pointer-events-none">
              <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-navy-950/80 backdrop-blur-md border border-navy-800 text-[10px] font-mono text-sky-400">
                <Hand className="w-3 h-3" />
                <span>{detectedHandsCount > 0 ? `${detectedHandsCount} Hand${detectedHandsCount > 1 ? 's' : ''}` : 'Searching'}</span>
              </div>

              <div className="flex items-center gap-2">
                <div className="flex items-center gap-1 px-2.5 py-1 rounded-full bg-navy-950/80 backdrop-blur-md border border-navy-800 text-[10px] font-mono text-slate-300">
                  <Activity className="w-3 h-3 text-brand-400" />
                  <span>{fps} FPS</span>
                </div>
                <div className="flex items-center gap-1 px-2.5 py-1 rounded-full bg-navy-950/80 backdrop-blur-md border border-navy-800 text-[10px] font-mono text-amber-300">
                  <Gauge className="w-3 h-3" />
                  <span>{latencyMs}ms</span>
                </div>
              </div>
            </div>

            {/* Camera Switch Floating Button */}
            <button
              onClick={toggleFacingMode}
              className="absolute bottom-3 right-3 p-3 rounded-full bg-navy-950/90 hover:bg-navy-900 border border-navy-700 text-white shadow-lg active:scale-90 transition-transform"
              title="Switch Front/Rear Camera"
            >
              <FlipHorizontal className="w-5 h-5 text-brand-400" />
            </button>
          </>
        )}

        {/* Camera Inactive State */}
        {!isCameraActive && (
          <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center bg-gradient-to-b from-navy-950/95 to-navy-900/95 space-y-3">
            <div className="w-14 h-14 rounded-2xl bg-brand-600/20 border border-brand-500/40 flex items-center justify-center text-brand-400 shadow-glow">
              <Camera className="w-7 h-7" />
            </div>
            <div>
              <h2 className="text-base font-bold text-white">Mobile Camera Verification</h2>
              <p className="text-xs text-slate-400 max-w-xs mt-1">
                Point your phone's camera at your hand to test live sign recognition in real time.
              </p>
            </div>

            <button
              onClick={() => startCamera(facingMode)}
              disabled={isStartingCamera}
              className="px-6 py-3 rounded-2xl bg-brand-600 hover:bg-brand-500 active:scale-95 text-white font-bold text-sm shadow-glow flex items-center gap-2 transition-all"
            >
              {isStartingCamera ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Requesting Access...</span>
                </>
              ) : (
                <>
                  <Camera className="w-4 h-4" />
                  <span>Enable Mobile Camera</span>
                </>
              )}
            </button>
          </div>
        )}
      </div>

      {/* Error Alert Banner */}
      {cameraError && (
        <div className="mt-3 p-3 bg-rose-950/80 border border-rose-800 rounded-2xl text-xs text-rose-200 flex items-start gap-2 shadow-lg">
          <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
          <div className="flex-1">
            <strong className="block font-bold">Camera Permission Required</strong>
            <span>{cameraError}</span>
          </div>
        </div>
      )}

      {/* Real-World Acceptance Telemetry Card */}
      <div className="mt-3 bg-navy-900 border border-navy-800 rounded-2xl p-4 shadow-subtle space-y-3">
        <div className="flex items-center justify-between border-b border-navy-800 pb-2">
          <span className="text-[11px] font-mono uppercase text-slate-400 font-bold flex items-center gap-1.5">
            <ShieldCheck className="w-3.5 h-3.5 text-brand-400" />
            <span>Acceptance Telemetry</span>
          </span>
          <span className="text-[10px] font-mono text-slate-500">
            Model: {recognitionMode === 'alphabet' ? 'CoordinateMLP 94-D' : 'BiLSTM 153-D'}
          </span>
        </div>

        {/* Large Recognition Display */}
        <div className="flex items-center justify-between bg-navy-950 p-3 rounded-xl border border-navy-850">
          <div>
            <span className="text-[10px] font-mono text-slate-500 uppercase block">Current Prediction</span>
            <div className="flex items-center gap-2">
              <span className="text-2xl font-black text-white tracking-wider font-mono">
                {predictedSign}
              </span>
              <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold uppercase ${
                isAccepted ? 'bg-emerald-950 text-emerald-400 border border-emerald-800' : 'bg-amber-950 text-amber-400 border border-amber-800'
              }`}>
                {isAccepted ? 'ACCEPTED' : 'REJECTED'}
              </span>
            </div>
          </div>

          <div className="text-right">
            <span className="text-[10px] font-mono text-slate-500 uppercase block">Confidence</span>
            <span className={`text-xl font-bold font-mono ${
              confidence >= 0.70 ? 'text-emerald-400' : 'text-slate-400'
            }`}>
              {Math.round(confidence * 100)}%
            </span>
          </div>
        </div>

        {/* Status Line */}
        <p className="text-xs text-slate-300 font-mono flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-brand-400" />
          <span>{statusMessage}</span>
        </p>

        {/* Mode & Power Throttling Toggles */}
        <div className="grid grid-cols-2 gap-2 pt-2 border-t border-navy-850 text-xs">
          <div>
            <span className="text-[10px] text-slate-500 font-mono block mb-1">Recognition Pipeline</span>
            <div className="flex rounded-lg bg-navy-950 p-0.5 border border-navy-800">
              <button
                onClick={() => setRecognitionMode('alphabet')}
                className={`flex-1 py-1 rounded text-[11px] font-bold transition-all ${
                  recognitionMode === 'alphabet' ? 'bg-brand-600 text-white shadow' : 'text-slate-400 hover:text-white'
                }`}
              >
                Alphabet
              </button>
              <button
                onClick={() => setRecognitionMode('words')}
                className={`flex-1 py-1 rounded text-[11px] font-bold transition-all ${
                  recognitionMode === 'words' ? 'bg-brand-600 text-white shadow' : 'text-slate-400 hover:text-white'
                }`}
              >
                10 Words
              </button>
            </div>
          </div>

          <div>
            <span className="text-[10px] text-slate-500 font-mono block mb-1">Power / Anti-Heat</span>
            <div className="flex rounded-lg bg-navy-950 p-0.5 border border-navy-800">
              {(['eco', 'balanced', 'high'] as const).map((m) => (
                <button
                  key={m}
                  onClick={() => setThrottleMode(m)}
                  className={`flex-1 py-1 rounded text-[10px] uppercase font-bold transition-all ${
                    throttleMode === m ? 'bg-amber-600 text-white shadow' : 'text-slate-400 hover:text-white'
                  }`}
                >
                  {m}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Layer C: AI Sentence Composition Mobile Card */}
      <div className="mt-3 bg-navy-900 border border-brand-800/60 rounded-2xl p-4 shadow-subtle space-y-2.5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-brand-400" />
            <span className="text-xs font-bold text-white uppercase tracking-wider font-mono">
              AI Smart Composer
            </span>
          </div>

          <span className="text-[9px] font-mono px-2 py-0.5 rounded-full bg-navy-950 border border-navy-800 text-slate-400">
            {aiStatus.active_provider}
          </span>
        </div>

        {/* Tokens Stream */}
        <div className="flex items-center gap-1 flex-wrap min-h-[32px] p-2 bg-navy-950 rounded-xl border border-navy-850">
          {recognizedTokens.length === 0 ? (
            <span className="text-[11px] text-slate-500 italic">Sign gestures to collect tokens...</span>
          ) : (
            recognizedTokens.map((tok, idx) => (
              <span key={idx} className="px-2 py-0.5 rounded bg-brand-950 text-brand-300 border border-brand-800 font-mono text-xs font-bold">
                {tok}
              </span>
            ))
          )}
        </div>

        {/* Compose Button */}
        <button
          onClick={handleComposeSentence}
          disabled={recognizedTokens.length === 0 || isComposing}
          className="w-full py-2 rounded-xl bg-brand-600 hover:bg-brand-500 disabled:opacity-40 text-white font-bold text-xs shadow-glow flex items-center justify-center gap-1.5 transition-all"
        >
          {isComposing ? (
            <>
              <RefreshCw className="w-3.5 h-3.5 animate-spin" />
              <span>Composing Sentence...</span>
            </>
          ) : (
            <>
              <Sparkles className="w-3.5 h-3.5" />
              <span>Synthesize Sentence (Layer C)</span>
            </>
          )}
        </button>

        {/* Composed Output */}
        {composedSentence && (
          <div className="p-3 bg-navy-950 border border-brand-800 rounded-xl space-y-1.5">
            <p className="text-xs font-medium text-white leading-relaxed">
              "{composedSentence}"
            </p>
            <div className="flex items-center justify-between pt-1 border-t border-navy-850">
              <span className="text-[9px] font-mono text-brand-400">Localized Speech</span>
              <div className="flex items-center gap-1">
                <button
                  onClick={() => speechService.speak(composedSentence, selectedLanguage)}
                  className="p-1 rounded text-slate-300 hover:text-white"
                  title="Speak"
                >
                  <Volume2 className="w-3.5 h-3.5 text-brand-400" />
                </button>
                <button
                  onClick={() => {
                    navigator.clipboard?.writeText(composedSentence);
                    setCopied(true);
                    setTimeout(() => setCopied(false), 2000);
                  }}
                  className="p-1 rounded text-slate-300 hover:text-white"
                  title="Copy"
                >
                  {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
