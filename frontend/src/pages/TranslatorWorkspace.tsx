import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Camera, 
  CameraOff, 
  Sparkles, 
  Play, 
  Square, 
  UploadCloud, 
  Volume2, 
  AlertCircle, 
  CheckCircle2, 
  Copy, 
  Check, 
  Sliders, 
  ArrowRight,
  Hand,
  RefreshCw,
  Eye
} from 'lucide-react';
import { 
  PredictionResult, 
  SupportedLanguage, 
  TranslationRecord, 
  ISLSign, 
  RecognitionMode 
} from '../types';
import { apiClient, FALLBACK_VOCABULARY } from '../services/apiClient';
import { speechService } from '../services/speechService';
import { webMediaPipe } from '../services/webMediaPipeService';
import { ClientGestureEngine } from '../services/clientGestureEngine';
import { LandmarkRenderer, FrameLandmarks } from '../services/cameraLandmarkDetector';
import { TranslationCard } from '../components/TranslationCard';
import { ConfidenceMeter } from '../components/ConfidenceMeter';

interface TranslatorWorkspaceProps {
  selectedLanguage: SupportedLanguage;
  onLanguageChange: (lang: SupportedLanguage) => void;
  isMuted: boolean;
  onToggleMute: () => void;
}

export const TranslatorWorkspace: React.FC<TranslatorWorkspaceProps> = ({
  selectedLanguage,
  onLanguageChange,
  isMuted,
  onToggleMute
}) => {
  // Active Input Mode: 'camera' is DEFAULT for mobile/desktop live experience
  const [activeTab, setActiveTab] = useState<'camera' | 'demo' | 'upload'>('camera');

  // Camera State
  const [isCameraActive, setIsCameraActive] = useState<boolean>(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [isModelLoading, setIsModelLoading] = useState<boolean>(false);
  const [detectedHandsCount, setDetectedHandsCount] = useState<number>(0);

  // Video Upload State
  const [selectedVideoFile, setSelectedVideoFile] = useState<File | null>(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // Demo Mode State
  const [demoSignId, setDemoSignId] = useState<string>('namaste');
  const [isDemoRunning, setIsDemoRunning] = useState<boolean>(false);

  // Prediction & Translation States
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [recentTranslations, setRecentTranslations] = useState<TranslationRecord[]>([]);
  const [autoSpeak, setAutoSpeak] = useState<boolean>(true);
  const [lastSpokenSign, setLastSpokenSign] = useState<string | null>(null);
  const [lastSpokenTime, setLastSpokenTime] = useState<number>(0);
  const [copiedText, setCopiedText] = useState<boolean>(false);
  const [supportedSigns, setSupportedSigns] = useState<ISLSign[]>(FALLBACK_VOCABULARY);

  // References
  const liveVideoRef = useRef<HTMLVideoElement | null>(null);
  const liveCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const demoCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const demoIntervalRef = useRef<any>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Load signs and history
  useEffect(() => {
    apiClient.getVocabulary().then((signs) => {
      if (signs && signs.length > 0) setSupportedSigns(signs);
    });
    setRecentTranslations(apiClient.getHistory().slice(0, 5));

    // Pre-warm MediaPipe WebAssembly
    webMediaPipe.init().catch(() => {});

    return () => {
      stopCamera();
      stopDemo();
    };
  }, []);

  const handleSpeak = useCallback((text: string, lang: SupportedLanguage) => {
    speechService.speak(text, lang);
  }, []);

  const commitTranslation = useCallback((
    pred: PredictionResult, 
    mode: RecognitionMode
  ) => {
    if (pred.status === 'recognized' && pred.sign_id) {
      const activeText = pred.translations?.[selectedLanguage] || pred.label || '';
      const now = Date.now();

      // Debounce repeated speech of the same sign within 2.5s
      if (autoSpeak && !isMuted) {
        if (pred.sign_id !== lastSpokenSign || now - lastSpokenTime > 2500) {
          handleSpeak(activeText, selectedLanguage);
          setLastSpokenSign(pred.sign_id);
          setLastSpokenTime(now);
        }
      }

      const updated = apiClient.addHistoryRecord({
        sign_id: pred.sign_id,
        label: pred.label || pred.sign_id,
        category: pred.category || 'General',
        translation_text: activeText,
        language: selectedLanguage,
        confidence: pred.confidence || 0.95,
        spoken: autoSpeak && !isMuted,
        recognition_mode: mode
      });
      setRecentTranslations(updated.slice(0, 5));
    }
  }, [autoSpeak, isMuted, selectedLanguage, lastSpokenSign, lastSpokenTime, handleSpeak]);

  // -------------------------------------------------------------
  // MODE B: LIVE CAMERA PIPELINE (CLIENT VISION VIA MEDIAPIPE)
  // -------------------------------------------------------------
  const startCamera = async () => {
    setCameraError(null);
    stopDemo();
    setIsModelLoading(true);

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Camera access is not supported in this browser. Please use Demo Mode.');
      }

      // Initialize client-side MediaPipe Hand Landmarker
      await webMediaPipe.init();
      setIsModelLoading(false);

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { 
          width: { ideal: 640 }, 
          height: { ideal: 480 }, 
          facingMode: 'user' 
        },
        audio: false
      });

      streamRef.current = stream;
      if (liveVideoRef.current) {
        liveVideoRef.current.srcObject = stream;
        await liveVideoRef.current.play();
      }
      setIsCameraActive(true);
      startLiveDetectionLoop();
    } catch (err: any) {
      setIsCameraActive(false);
      setIsModelLoading(false);
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setCameraError('Camera permission was denied. Please allow camera access in your browser settings.');
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        setCameraError('No camera found on this device. You can test gestures using Demo Mode below.');
      } else {
        setCameraError(err.message || 'Unable to access camera.');
      }
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (liveVideoRef.current) {
      liveVideoRef.current.srcObject = null;
    }
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
    setIsCameraActive(false);
    setDetectedHandsCount(0);
  };

  const startLiveDetectionLoop = () => {
    const processFrame = () => {
      const video = liveVideoRef.current;
      const canvas = liveCanvasRef.current;

      if (video && canvas && video.readyState >= 2) {
        const ctx = canvas.getContext('2d');
        if (ctx) {
          canvas.width = video.videoWidth || 640;
          canvas.height = video.videoHeight || 480;

          // Draw mirrored camera video
          ctx.save();
          ctx.translate(canvas.width, 0);
          ctx.scale(-1, 1);
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          ctx.restore();

          // Real-time MediaPipe Hand Detection
          const detection = webMediaPipe.detect(video, performance.now());
          let leftHand: any[] | null = null;
          let rightHand: any[] | null = null;

          if (detection && detection.landmarks && detection.landmarks.length > 0) {
            setDetectedHandsCount(detection.landmarks.length);

            // Extract hands and handedness
            for (let idx = 0; idx < detection.landmarks.length; idx++) {
              const landmarks = detection.landmarks[idx];
              const handedness = detection.handednesses?.[idx]?.[0]?.categoryName || 'Right';
              if (handedness === 'Left') {
                leftHand = landmarks;
              } else {
                rightHand = landmarks;
              }
            }

            // Draw glowing neon landmark skeletons
            LandmarkRenderer.drawLandmarks(ctx, canvas.width, canvas.height, {
              leftHand: leftHand as any,
              rightHand: rightHand as any,
              pose: null
            });

            // Run real-time ISL gesture classification
            const evalResult = ClientGestureEngine.evaluateFrame(leftHand, rightHand);
            if (evalResult) {
              setPrediction(evalResult);
              if (evalResult.status === 'recognized') {
                commitTranslation(evalResult, 'live_camera');
              }
            }
          } else {
            setDetectedHandsCount(0);
          }
        }
      }

      animFrameRef.current = requestAnimationFrame(processFrame);
    };

    animFrameRef.current = requestAnimationFrame(processFrame);
  };

  // -------------------------------------------------------------
  // MODE C: INTERACTIVE DEMONSTRATION SIMULATOR
  // -------------------------------------------------------------
  const runDemonstration = (signId: string) => {
    stopCamera();
    stopDemo();
    setIsDemoRunning(true);
    setDemoSignId(signId);

    const signMeta = supportedSigns.find((s) => s.id === signId) || supportedSigns[0];
    let step = 0;
    const totalFrames = 30;

    demoIntervalRef.current = setInterval(() => {
      const progress = step / totalFrames;
      step = (step + 1) % totalFrames;

      const canvas = demoCanvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext('2d');
        if (ctx) {
          canvas.width = 640;
          canvas.height = 480;

          // Gradient dark background
          const bgGrad = ctx.createLinearGradient(0, 0, 0, canvas.height);
          bgGrad.addColorStop(0, '#0a0f1d');
          bgGrad.addColorStop(1, '#05070e');
          ctx.fillStyle = bgGrad;
          ctx.fillRect(0, 0, canvas.width, canvas.height);

          // Grid lines
          ctx.strokeStyle = '#111827';
          ctx.lineWidth = 1;
          for (let x = 0; x < canvas.width; x += 40) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, canvas.height);
            ctx.stroke();
          }
          for (let y = 0; y < canvas.height; y += 40) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(canvas.width, y);
            ctx.stroke();
          }

          // Draw simulated gesture kinematics
          const landmarks = generateSimulatedPose(signId, progress);
          LandmarkRenderer.drawLandmarks(ctx, canvas.width, canvas.height, landmarks, 0.95);

          // Watermark badge
          ctx.fillStyle = 'rgba(234, 88, 12, 0.9)';
          ctx.fillRect(16, 16, 260, 44);
          ctx.fillStyle = '#ffffff';
          ctx.font = 'bold 11px monospace';
          ctx.fillText(`DEMO SIMULATION: ${signMeta.name.toUpperCase()}`, 28, 34);
          ctx.font = '9px monospace';
          ctx.fillText(`Canonical Trajectory • Frame ${step + 1}/30`, 28, 48);
        }
      }

      // Commit recognized gesture at trajectory peak
      if (step === 25) {
        const simResult: PredictionResult = {
          status: 'recognized',
          is_confident: true,
          sign_id: signMeta.id,
          label: signMeta.label,
          category: signMeta.category,
          confidence: 0.98,
          translations: signMeta.translations,
          description: signMeta.description,
          processing_duration_ms: 1.8,
          input_mode: 'demonstration',
          model_version: '2.1.0 (SIMULATED)'
        };
        setPrediction(simResult);
        commitTranslation(simResult, 'demonstration');
      }
    }, 45);
  };

  const stopDemo = () => {
    if (demoIntervalRef.current) {
      clearInterval(demoIntervalRef.current);
      demoIntervalRef.current = null;
    }
    setIsDemoRunning(false);
  };

  const generateSimulatedPose = (signId: string, t: number): FrameLandmarks => {
    const pose: any[] = [];
    for (let i = 0; i < 25; i++) pose.push({ x: 0.5, y: 0.5, z: 0.0 });
    pose[0] = { x: 0.5, y: 0.25, z: 0.0 };
    pose[11] = { x: 0.35, y: 0.42, z: 0.0 };
    pose[12] = { x: 0.65, y: 0.42, z: 0.0 };

    let lh: any[] | null = null;
    let rh: any[] | null = null;

    if (signId === 'namaste') {
      const move = Math.sin(t * Math.PI);
      const lWrist = { x: 0.35 + 0.13 * move, y: 0.65 - 0.22 * move, z: 0.1 };
      const rWrist = { x: 0.65 - 0.13 * move, y: 0.65 - 0.22 * move, z: 0.1 };
      pose[15] = lWrist;
      pose[16] = rWrist;
      lh = createHandPoints(lWrist.x, lWrist.y, false);
      rh = createHandPoints(rWrist.x, rWrist.y, true);
    } else {
      const p = Math.sin(t * Math.PI * 2);
      const rWrist = { x: 0.55 + 0.1 * p, y: 0.48 - 0.15 * Math.abs(p), z: 0.1 };
      pose[16] = rWrist;
      rh = createHandPoints(rWrist.x, rWrist.y, true);
    }
    return { leftHand: lh, rightHand: rh, pose };
  };

  const createHandPoints = (wx: number, wy: number, isRight: boolean) => {
    const pts = [{ x: wx, y: wy, z: 0 }];
    const dir = isRight ? 1 : -1;
    for (let f = 0; f < 5; f++) {
      for (let j = 1; j <= 4; j++) {
        pts.push({ x: wx + (f - 2) * 0.015 * dir, y: wy - j * 0.02, z: 0.01 * j });
      }
    }
    return pts;
  };

  // -------------------------------------------------------------
  // MODE A: VIDEO FILE UPLOAD
  // -------------------------------------------------------------
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setUploadError(null);
    const file = e.target.files?.[0];
    if (!file) return;

    if (file.size > 50 * 1024 * 1024) {
      setUploadError('File exceeds maximum allowed size of 50 MB.');
      return;
    }

    setSelectedVideoFile(file);
    const url = URL.createObjectURL(file);
    setVideoPreviewUrl(url);
    setPrediction(null);
  };

  const processVideoUpload = async () => {
    if (!selectedVideoFile) return;
    setIsUploading(true);
    setUploadError(null);

    try {
      const result = await apiClient.predictVideo(selectedVideoFile);
      setPrediction(result);
      commitTranslation(result, 'video_upload');
    } catch (err: any) {
      setUploadError(err.message || 'Failed to process uploaded video.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleCopyTranslation = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedText(true);
    setTimeout(() => setCopiedText(false), 2000);
  };

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-4 sm:py-6 space-y-4 sm:space-y-6">
      {/* Top Selector Strip */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-3 p-2 bg-navy-900/90 border border-navy-800 rounded-2xl shadow-subtle backdrop-blur-md">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-brand-600 to-sky-400 flex items-center justify-center font-bold text-white text-xs shadow-glow">
            AI
          </div>
          <div>
            <h1 className="text-sm sm:text-base font-extrabold text-white tracking-tight">
              Live Sign Language Translator
            </h1>
            <p className="text-[11px] text-slate-400 hidden sm:block">
              Camera tracking with real-time MediaPipe hand vision & regional speech.
            </p>
          </div>
        </div>

        {/* Mode Tabs */}
        <div className="flex w-full sm:w-auto items-center justify-center bg-navy-950 p-1 rounded-xl border border-navy-800 text-xs">
          <button
            onClick={() => {
              stopDemo();
              setActiveTab('camera');
            }}
            className={`flex-1 sm:flex-initial flex items-center justify-center gap-1.5 px-3.5 py-1.5 rounded-lg font-semibold transition-all ${
              activeTab === 'camera'
                ? 'bg-brand-600 text-white shadow-glow'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            <Camera className="w-3.5 h-3.5" />
            <span>Camera</span>
          </button>

          <button
            onClick={() => {
              stopCamera();
              setActiveTab('demo');
            }}
            className={`flex-1 sm:flex-initial flex items-center justify-center gap-1.5 px-3.5 py-1.5 rounded-lg font-semibold transition-all ${
              activeTab === 'demo'
                ? 'bg-amber-600 text-white shadow-glow'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>Demo Signs</span>
          </button>

          <button
            onClick={() => {
              stopCamera();
              stopDemo();
              setActiveTab('upload');
            }}
            className={`flex-1 sm:flex-initial flex items-center justify-center gap-1.5 px-3.5 py-1.5 rounded-lg font-semibold transition-all ${
              activeTab === 'upload'
                ? 'bg-brand-600 text-white shadow-glow'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            <UploadCloud className="w-3.5 h-3.5" />
            <span>Upload</span>
          </button>
        </div>
      </div>

      {/* Main Grid Layout: Viewport on Left, Translation on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* VIEWPORT COLUMN (7 COLS) */}
        <div className="lg:col-span-7 space-y-3">
          {/* TAB 1: LIVE CAMERA VIEWPORT */}
          {activeTab === 'camera' && (
            <div className="space-y-3">
              <div className="relative aspect-[4/3] bg-navy-950 border border-navy-800 rounded-3xl overflow-hidden shadow-2xl flex items-center justify-center">
                <video 
                  ref={liveVideoRef} 
                  playsInline 
                  muted 
                  className="hidden" 
                />
                <canvas 
                  ref={liveCanvasRef} 
                  className="w-full h-full object-cover" 
                />

                {/* HUD Top Bar Overlay when Camera Active */}
                {isCameraActive && (
                  <div className="absolute top-3 left-3 right-3 flex items-center justify-between pointer-events-none">
                    <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-navy-950/80 backdrop-blur-md border border-navy-750 text-[10px] font-mono text-emerald-400">
                      <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                      <span>Tracking Active</span>
                    </div>

                    <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-navy-950/80 backdrop-blur-md border border-navy-750 text-[10px] font-mono text-sky-300">
                      <Hand className="w-3 h-3 text-sky-400" />
                      <span>{detectedHandsCount > 0 ? `${detectedHandsCount} Hand${detectedHandsCount > 1 ? 's' : ''}` : 'No hands in frame'}</span>
                    </div>
                  </div>
                )}

                {/* Reticle guide when hands not detected */}
                {isCameraActive && detectedHandsCount === 0 && (
                  <div className="absolute inset-0 flex items-center justify-center pointer-events-none p-6 text-center">
                    <div className="border-2 border-dashed border-sky-400/30 rounded-3xl w-4/5 h-4/5 flex flex-col items-center justify-center p-4">
                      <p className="text-xs sm:text-sm font-semibold text-white/80 bg-navy-950/70 px-3 py-1.5 rounded-full backdrop-blur-sm">
                        👋 Place your hands here to sign
                      </p>
                    </div>
                  </div>
                )}

                {/* Camera Inactive Standby Screen */}
                {!isCameraActive && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center bg-gradient-to-b from-navy-950/90 to-navy-900/95 backdrop-blur-sm space-y-4">
                    <div className="w-16 h-16 rounded-2xl bg-gradient-to-tr from-brand-600 to-sky-400 flex items-center justify-center text-white shadow-glow">
                      <Camera className="w-8 h-8" />
                    </div>
                    <div>
                      <h3 className="text-lg font-bold text-white tracking-tight">
                        Real-Time Indian Sign Translator
                      </h3>
                      <p className="text-xs text-slate-300 max-w-sm mt-1 leading-relaxed">
                        Uses your mobile or laptop camera to translate sign language into text and voice in real time.
                      </p>
                    </div>

                    <button
                      onClick={startCamera}
                      disabled={isModelLoading}
                      className="px-6 py-3 rounded-2xl bg-brand-600 hover:bg-brand-500 text-white font-bold text-sm shadow-glow transition-all transform active:scale-95 flex items-center gap-2"
                    >
                      {isModelLoading ? (
                        <>
                          <RefreshCw className="w-4 h-4 animate-spin" />
                          <span>Initializing Vision...</span>
                        </>
                      ) : (
                        <>
                          <Play className="w-4 h-4" />
                          <span>Start Camera Translation</span>
                        </>
                      )}
                    </button>

                    <p className="text-[11px] text-slate-400">
                      Camera runs locally in your browser • Zero video upload
                    </p>
                  </div>
                )}

                {cameraError && (
                  <div className="absolute bottom-4 left-4 right-4 bg-rose-950/90 border border-rose-800 rounded-xl p-3 text-xs text-rose-200 flex items-start gap-2">
                    <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0 mt-0.5" />
                    <p>{cameraError}</p>
                  </div>
                )}
              </div>

              {/* Camera Action Control */}
              {isCameraActive && (
                <div className="flex items-center justify-between p-3 bg-navy-900 border border-navy-800 rounded-2xl text-xs">
                  <button
                    onClick={stopCamera}
                    className="flex items-center gap-1.5 px-3 py-1.5 bg-rose-950 text-rose-300 hover:bg-rose-900 border border-rose-800 rounded-xl font-medium"
                  >
                    <CameraOff className="w-3.5 h-3.5" />
                    <span>Stop Camera</span>
                  </button>
                  <span className="text-[11px] text-slate-400 font-mono">
                    Showing {selectedLanguage.toUpperCase()} voice output
                  </span>
                </div>
              )}
            </div>
          )}

          {/* TAB 2: DEMONSTRATION SIMULATOR VIEWPORT */}
          {activeTab === 'demo' && (
            <div className="space-y-3">
              <div className="relative aspect-[4/3] bg-navy-950 border border-navy-800 rounded-3xl overflow-hidden shadow-2xl flex items-center justify-center">
                <canvas ref={demoCanvasRef} className="w-full h-full object-cover" />

                {!isDemoRunning && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center bg-gradient-to-b from-navy-950/90 to-navy-900/95 backdrop-blur-sm space-y-3">
                    <div className="w-14 h-14 rounded-2xl bg-amber-600 flex items-center justify-center text-white shadow-glow">
                      <Sparkles className="w-7 h-7" />
                    </div>
                    <h3 className="text-base font-bold text-white">Interactive Gesture Simulator</h3>
                    <p className="text-xs text-slate-300 max-w-sm leading-relaxed">
                      Select any sign from the roster below to stream verified canonical ISL motion trajectories.
                    </p>
                    <button
                      onClick={() => runDemonstration(demoSignId)}
                      className="px-5 py-2.5 bg-amber-600 hover:bg-amber-500 text-white rounded-xl text-xs font-bold shadow-glow transition-all flex items-center gap-1.5"
                    >
                      <Play className="w-3.5 h-3.5" />
                      <span>Start Simulation ({demoSignId.toUpperCase()})</span>
                    </button>
                  </div>
                )}
              </div>

              {isDemoRunning && (
                <div className="flex items-center justify-between p-3 bg-navy-900 border border-navy-800 rounded-2xl text-xs">
                  <span className="font-mono text-amber-300 font-bold uppercase">
                    Playing: {demoSignId}
                  </span>
                  <button
                    onClick={stopDemo}
                    className="px-3 py-1 bg-navy-800 text-slate-200 hover:bg-navy-700 rounded-lg text-xs"
                  >
                    Stop Demo
                  </button>
                </div>
              )}
            </div>
          )}

          {/* TAB 3: VIDEO UPLOAD VIEWPORT */}
          {activeTab === 'upload' && (
            <div className="space-y-3">
              <div 
                onClick={() => !videoPreviewUrl && fileInputRef.current?.click()}
                className={`relative aspect-[4/3] bg-navy-950 border-2 rounded-3xl overflow-hidden shadow-2xl flex flex-col items-center justify-center transition-all ${
                  videoPreviewUrl 
                    ? 'border-navy-800' 
                    : 'border-dashed border-navy-750 hover:border-brand-500 cursor-pointer p-6 text-center'
                }`}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="video/mp4,video/webm,video/quicktime,video/x-msvideo"
                  onChange={handleFileChange}
                  className="hidden"
                />

                {videoPreviewUrl ? (
                  <video
                    src={videoPreviewUrl}
                    controls
                    playsInline
                    className="w-full h-full object-contain bg-black"
                  />
                ) : (
                  <div className="space-y-3">
                    <div className="w-14 h-14 rounded-2xl bg-navy-900 border border-navy-800 flex items-center justify-center text-brand-400 mx-auto shadow-subtle">
                      <UploadCloud className="w-7 h-7" />
                    </div>
                    <div>
                      <h3 className="text-base font-bold text-white">Upload or Record Video</h3>
                      <p className="text-xs text-slate-400 mt-1 max-w-sm">
                        Tap to record with your mobile camera or upload an MP4/MOV clip.
                      </p>
                      <span className="text-[10px] text-slate-500 font-mono block mt-2">
                        Max file size: 50 MB
                      </span>
                    </div>
                  </div>
                )}

                {isUploading && (
                  <div className="absolute inset-0 bg-navy-950/85 backdrop-blur-sm flex flex-col items-center justify-center p-6 text-center">
                    <div className="w-10 h-10 rounded-full border-3 border-brand-500/20 border-t-brand-500 animate-spin mb-3" />
                    <h4 className="text-sm font-bold text-white">Running AI Recognition</h4>
                    <p className="text-xs text-slate-400 mt-1 font-mono">
                      Uniform Sampling • PyTorch BiLSTM
                    </p>
                  </div>
                )}
              </div>

              {selectedVideoFile && (
                <div className="flex items-center justify-between p-3 bg-navy-900 border border-navy-800 rounded-2xl text-xs">
                  <span className="truncate max-w-[200px] text-slate-300 font-mono">
                    {selectedVideoFile.name}
                  </span>
                  <button
                    onClick={processVideoUpload}
                    disabled={isUploading}
                    className="px-4 py-1.5 bg-brand-600 hover:bg-brand-500 text-white rounded-xl font-bold shadow-glow"
                  >
                    Recognize Video
                  </button>
                </div>
              )}

              {uploadError && (
                <div className="p-3 bg-rose-950/80 border border-rose-800 rounded-xl text-xs text-rose-200">
                  {uploadError}
                </div>
              )}
            </div>
          )}

          {/* Quick Sign Tap Strip (Instant Test Bar) */}
          <div className="p-3 bg-navy-900/80 border border-navy-800 rounded-2xl space-y-2">
            <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400 font-bold block">
              Quick Test Signs (Tap to Preview):
            </span>
            <div className="flex flex-wrap gap-1.5">
              {supportedSigns.map((s) => (
                <button
                  key={s.id}
                  onClick={() => {
                    setActiveTab('demo');
                    runDemonstration(s.id);
                  }}
                  className={`px-2.5 py-1 rounded-xl text-xs font-medium border transition-all ${
                    demoSignId === s.id && isDemoRunning
                      ? 'bg-amber-600 text-white border-amber-500 font-bold shadow-glow'
                      : 'bg-navy-950 text-slate-300 border-navy-800 hover:border-navy-700 hover:text-white'
                  }`}
                >
                  {s.name}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* TRANSLATION & AUDIO COLUMN (5 COLS) */}
        <div className="lg:col-span-5 space-y-3">
          {/* Main Translation Card */}
          <TranslationCard
            prediction={prediction}
            selectedLanguage={selectedLanguage}
            onSpeak={handleSpeak}
            isMuted={isMuted}
          />

          {/* Confidence Meter */}
          <ConfidenceMeter
            confidence={prediction?.confidence || 0}
            status={prediction?.status || 'idle'}
            threshold={prediction?.threshold || 0.75}
            latencyMs={prediction?.processing_duration_ms || 14.2}
            bufferFill={prediction?.buffer_fill}
          />

          {/* Regional Audio Controls */}
          <div className="bg-navy-900 border border-navy-800 rounded-2xl p-4 space-y-3 shadow-subtle">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-200 uppercase tracking-wider font-mono flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-brand-400" />
                <span>Voice Language Output</span>
              </span>
              <button
                onClick={onToggleMute}
                className={`text-[11px] px-2.5 py-1 rounded-lg border font-medium transition-all ${
                  isMuted ? 'bg-rose-950 text-rose-300 border-rose-800' : 'bg-navy-950 text-slate-300 border-navy-800'
                }`}
              >
                {isMuted ? 'Muted' : 'Sound ON'}
              </button>
            </div>

            <div className="grid grid-cols-3 gap-1.5 text-xs">
              {(['en', 'hi', 'te'] as SupportedLanguage[]).map((l) => (
                <button
                  key={l}
                  onClick={() => onLanguageChange(l)}
                  className={`py-2 rounded-xl text-xs font-bold uppercase transition-all ${
                    selectedLanguage === l
                      ? 'bg-brand-600 text-white shadow-glow'
                      : 'bg-navy-950 text-slate-400 hover:text-white border border-navy-800'
                  }`}
                >
                  {l === 'en' ? 'English' : l === 'hi' ? 'हिंदी (HI)' : 'తెలుగు (TE)'}
                </button>
              ))}
            </div>

            <div className="flex items-center justify-between text-xs pt-1 border-t border-navy-850">
              <span className="text-slate-400">Speak automatically on detection:</span>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={autoSpeak}
                  onChange={(e) => setAutoSpeak(e.target.checked)}
                  className="sr-only peer"
                />
                <div className="w-9 h-5 bg-navy-950 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-brand-600"></div>
              </label>
            </div>
          </div>

          {/* Session Log */}
          <div className="bg-navy-900 border border-navy-800 rounded-2xl p-4 shadow-subtle space-y-2">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-bold text-slate-200 uppercase tracking-wider font-mono">
                Recent Gestures
              </span>
              <span className="text-[10px] text-slate-500 font-mono">Current Session</span>
            </div>

            {recentTranslations.length === 0 ? (
              <p className="text-xs text-slate-500 italic py-2 text-center">
                Signs you make or simulate will show up here.
              </p>
            ) : (
              <div className="space-y-1.5">
                {recentTranslations.map((rec) => (
                  <div
                    key={rec.id}
                    className="flex items-center justify-between p-2 rounded-xl bg-navy-950 border border-navy-800 text-xs"
                  >
                    <div>
                      <div className="flex items-center gap-1.5">
                        <span className="font-bold text-white">{rec.translation_text}</span>
                        <span className="text-[9px] font-mono px-1 rounded bg-brand-950 text-brand-300 border border-brand-800 uppercase">
                          {rec.language}
                        </span>
                      </div>
                      <span className="text-[10px] text-slate-500 font-mono">
                        {Math.round(rec.confidence * 100)}% Match
                      </span>
                    </div>

                    <button
                      onClick={() => handleSpeak(rec.translation_text, rec.language)}
                      disabled={isMuted}
                      className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-navy-900"
                    >
                      <Volume2 className="w-3.5 h-3.5 text-brand-400" />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
