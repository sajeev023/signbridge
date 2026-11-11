import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  UploadCloud, 
  Camera, 
  CameraOff, 
  Sparkles, 
  Play, 
  Square, 
  RefreshCw, 
  Volume2, 
  AlertCircle, 
  CheckCircle2, 
  FileVideo, 
  Sliders, 
  Copy, 
  Check, 
  Tag, 
  Clock, 
  Layers,
  ArrowRight,
  ShieldAlert
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
  // Active Input Mode: 'upload' | 'camera' | 'demo'
  const [activeTab, setActiveTab] = useState<'upload' | 'camera' | 'demo'>('upload');

  // Video Upload State
  const [selectedVideoFile, setSelectedVideoFile] = useState<File | null>(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // Camera State
  const [isCameraActive, setIsCameraActive] = useState<boolean>(false);
  const [hasCameraDevice, setHasCameraDevice] = useState<boolean | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);

  // Demo Mode State
  const [demoSignId, setDemoSignId] = useState<string>('namaste');
  const [isDemoRunning, setIsDemoRunning] = useState<boolean>(false);

  // Shared Prediction & Translation States
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [recentTranslations, setRecentTranslations] = useState<TranslationRecord[]>([]);
  const [autoSpeak, setAutoSpeak] = useState<boolean>(true);
  const [copiedText, setCopiedText] = useState<boolean>(false);
  const [supportedSigns, setSupportedSigns] = useState<ISLSign[]>(FALLBACK_VOCABULARY);

  // References
  const videoElemRef = useRef<HTMLVideoElement | null>(null);
  const liveVideoRef = useRef<HTMLVideoElement | null>(null);
  const liveCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const demoCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const demoIntervalRef = useRef<any>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Initial vocabulary and hardware detection
  useEffect(() => {
    apiClient.getVocabulary().then((signs) => {
      if (signs && signs.length > 0) setSupportedSigns(signs);
    });
    setRecentTranslations(apiClient.getHistory().slice(0, 5));

    // Detect webcam availability
    if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
      navigator.mediaDevices.enumerateDevices().then((devices) => {
        const videoInputs = devices.filter((d) => d.kind === 'videoinput');
        setHasCameraDevice(videoInputs.length > 0);
      }).catch(() => {
        setHasCameraDevice(false);
      });
    } else {
      setHasCameraDevice(false);
    }
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
      if (autoSpeak && !isMuted) {
        handleSpeak(activeText, selectedLanguage);
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
  }, [autoSpeak, isMuted, selectedLanguage, handleSpeak]);

  // -------------------------------------------------------------
  // MODE A: VIDEO UPLOAD PIPELINE
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

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setUploadError(null);
    const file = e.dataTransfer.files?.[0];
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

  // -------------------------------------------------------------
  // MODE B: LIVE CAMERA PIPELINE
  // -------------------------------------------------------------
  const startCamera = async () => {
    setCameraError(null);

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Camera access API is not available in this browser environment.');
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        audio: false
      });

      streamRef.current = stream;
      if (liveVideoRef.current) {
        liveVideoRef.current.srcObject = stream;
        liveVideoRef.current.play();
      }
      setIsCameraActive(true);
      startCameraRenderLoop();
    } catch (err: any) {
      setIsCameraActive(false);
      if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        setCameraError('No physical webcam was detected on this device. Please switch to Video Upload or Demo Mode.');
      } else if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setCameraError('Camera access was denied by your browser settings. Please permit camera access.');
      } else {
        setCameraError(`Camera error: ${err.message || 'Unable to start camera'}.`);
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
  };

  const startCameraRenderLoop = () => {
    const render = () => {
      const video = liveVideoRef.current;
      const canvas = liveCanvasRef.current;
      if (video && canvas && video.readyState >= 2) {
        const ctx = canvas.getContext('2d');
        if (ctx) {
          canvas.width = video.videoWidth || 640;
          canvas.height = video.videoHeight || 480;

          ctx.save();
          ctx.translate(canvas.width, 0);
          ctx.scale(-1, 1);
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          ctx.restore();
        }
      }
      animFrameRef.current = requestAnimationFrame(render);
    };
    animFrameRef.current = requestAnimationFrame(render);
  };

  // -------------------------------------------------------------
  // MODE C: DEMONSTRATION MODE (SIMULATED / LABELED)
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

          // Clear background
          ctx.fillStyle = '#080c16';
          ctx.fillRect(0, 0, canvas.width, canvas.height);

          // Grid
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

          // Draw simulated trajectory
          const landmarks = generateSimulatedPose(signId, progress);
          LandmarkRenderer.drawLandmarks(ctx, canvas.width, canvas.height, landmarks, 0.95);

          // DEMO MODE WATERMARK
          ctx.fillStyle = 'rgba(234, 88, 12, 0.9)';
          ctx.fillRect(16, 16, 260, 48);
          ctx.fillStyle = '#ffffff';
          ctx.font = 'bold 11px monospace';
          ctx.fillText(`DEMONSTRATION MODE: ${signId.toUpperCase()}`, 28, 36);
          ctx.font = '9px monospace';
          ctx.fillText(`Simulated Trajectory • Frame ${step + 1}/30`, 28, 50);
        }
      }

      // If at end of gesture stroke, commit simulated result
      if (step === 28) {
        const simResult: PredictionResult = {
          status: 'recognized',
          is_confident: true,
          sign_id: signMeta.id,
          label: signMeta.label,
          category: signMeta.category,
          confidence: 0.985,
          translations: signMeta.translations,
          description: signMeta.description,
          processing_duration_ms: 2.1,
          attention_weights: Array.from({ length: 30 }, (_, i) => (i === 15 ? 0.35 : 0.02)),
          input_mode: 'demonstration',
          model_version: '2.0.0 (SIMULATED)'
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

  useEffect(() => {
    return () => {
      stopCamera();
      stopDemo();
    };
  }, []);

  const handleCopyTranslation = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedText(true);
    setTimeout(() => setCopiedText(false), 2000);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
      {/* Workspace Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-navy-850">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
              AI Translator Workspace
            </h1>
            <span className="px-2 py-0.5 text-xs font-mono font-medium rounded-full bg-brand-950 text-brand-300 border border-brand-800">
              V2.0
            </span>
          </div>
          <p className="text-xs sm:text-sm text-slate-400 mt-1">
            Triple-mode Indian Sign Language translation pipeline: Video Upload, Live Camera, and Demonstration.
          </p>
        </div>

        {/* Input Mode Tabs */}
        <div className="flex items-center bg-navy-900 border border-navy-800 rounded-xl p-1 text-xs">
          <button
            onClick={() => {
              stopCamera();
              stopDemo();
              setActiveTab('upload');
            }}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-medium transition-all ${
              activeTab === 'upload'
                ? 'bg-brand-600 text-white font-semibold shadow-sm'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            <UploadCloud className="w-3.5 h-3.5" />
            <span>Upload Video</span>
          </button>

          <button
            onClick={() => {
              stopDemo();
              setActiveTab('camera');
            }}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-medium transition-all ${
              activeTab === 'camera'
                ? 'bg-brand-600 text-white font-semibold shadow-sm'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            <Camera className="w-3.5 h-3.5" />
            <span>Live Camera</span>
          </button>

          <button
            onClick={() => {
              stopCamera();
              setActiveTab('demo');
            }}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-medium transition-all ${
              activeTab === 'demo'
                ? 'bg-amber-600 text-white font-semibold shadow-sm'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            <Sparkles className="w-3.5 h-3.5 text-amber-300" />
            <span>Demo Mode</span>
          </button>
        </div>
      </div>

      {/* Main Two-Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* LEFT COLUMN: Input Mode Viewport (7 Cols) */}
        <div className="lg:col-span-7 space-y-4">
          {/* TAB 1: VIDEO UPLOAD (Mode A) */}
          {activeTab === 'upload' && (
            <div className="space-y-4">
              <div 
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleDrop}
                onClick={() => !videoPreviewUrl && fileInputRef.current?.click()}
                className={`relative aspect-[4/3] bg-navy-950 border-2 rounded-2xl overflow-hidden shadow-glow flex flex-col items-center justify-center transition-all ${
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
                    ref={videoElemRef}
                    src={videoPreviewUrl}
                    controls
                    playsInline
                    className="w-full h-full object-contain bg-black"
                  />
                ) : (
                  <div className="space-y-3">
                    <div className="w-16 h-16 rounded-2xl bg-navy-900 border border-navy-800 flex items-center justify-center text-brand-400 mx-auto shadow-subtle">
                      <UploadCloud className="w-8 h-8" />
                    </div>
                    <div>
                      <h3 className="text-base font-bold text-white">Upload ISL Video File</h3>
                      <p className="text-xs text-slate-400 mt-1 max-w-sm">
                        Drag and drop an MP4, WebM, or MOV video clip, or click to browse.
                      </p>
                      <span className="text-[10px] text-slate-500 font-mono block mt-2">
                        Max file size: 50 MB • Recommended: 1–5 seconds per sign
                      </span>
                    </div>
                  </div>
                )}

                {/* Processing Overlay */}
                {isUploading && (
                  <div className="absolute inset-0 bg-navy-950/85 backdrop-blur-sm flex flex-col items-center justify-center p-6 text-center">
                    <div className="w-12 h-12 rounded-full border-3 border-brand-500/20 border-t-brand-500 animate-spin mb-3" />
                    <h4 className="text-sm font-bold text-white">Analyzing Video Kinematics</h4>
                    <p className="text-xs text-slate-400 mt-1 font-mono">
                      Decoding frames • MediaPipe Landmarks • PyTorch BiLSTM
                    </p>
                  </div>
                )}
              </div>

              {/* Upload Controls Bar */}
              <div className="flex flex-wrap items-center justify-between gap-3 bg-navy-900 border border-navy-800 rounded-xl p-3 text-xs">
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="px-3 py-1.5 bg-navy-800 hover:bg-navy-750 text-slate-200 rounded-lg font-medium transition-all"
                  >
                    {selectedVideoFile ? 'Choose Another Video' : 'Select File'}
                  </button>

                  {selectedVideoFile && (
                    <button
                      onClick={processVideoUpload}
                      disabled={isUploading}
                      className="flex items-center gap-1.5 px-4 py-1.5 bg-brand-600 hover:bg-brand-500 text-white rounded-lg font-semibold shadow-glow transition-all disabled:opacity-50"
                    >
                      <Play className="w-3.5 h-3.5" />
                      <span>Run AI Recognition</span>
                    </button>
                  )}
                </div>

                <div className="text-[11px] font-mono text-slate-400">
                  {selectedVideoFile ? (
                    <span className="text-slate-300 truncate max-w-xs block">
                      {selectedVideoFile.name} ({(selectedVideoFile.size / (1024 * 1024)).toFixed(1)} MB)
                    </span>
                  ) : (
                    <span>No webcam needed • Upload prerecorded signs</span>
                  )}
                </div>
              </div>

              {uploadError && (
                <div className="p-3 bg-rose-950/80 border border-rose-800 rounded-xl text-xs text-rose-200 flex items-start gap-2">
                  <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0 mt-0.5" />
                  <p>{uploadError}</p>
                </div>
              )}
            </div>
          )}

          {/* TAB 2: LIVE CAMERA (Mode B) */}
          {activeTab === 'camera' && (
            <div className="space-y-4">
              <div className="relative aspect-[4/3] bg-navy-950 border border-navy-800 rounded-2xl overflow-hidden shadow-glow flex items-center justify-center">
                <video ref={liveVideoRef} playsInline muted className="hidden" />
                <canvas ref={liveCanvasRef} className="w-full h-full object-cover" />

                {!isCameraActive && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center bg-navy-950/95 backdrop-blur-sm">
                    {hasCameraDevice === false ? (
                      <div className="space-y-3 max-w-sm">
                        <div className="w-14 h-14 rounded-2xl bg-amber-950/60 border border-amber-800/80 flex items-center justify-center text-amber-400 mx-auto shadow-subtle">
                          <CameraOff className="w-7 h-7" />
                        </div>
                        <h3 className="text-base font-bold text-white">No Physical Webcam Detected</h3>
                        <p className="text-xs text-slate-400 leading-relaxed">
                          Your current desktop workstation does not have an attached camera.
                          You can comfortably test the complete AI translation pipeline using Video Upload or Demonstration mode.
                        </p>
                        <div className="pt-2 flex flex-wrap gap-2 justify-center">
                          <button
                            onClick={() => setActiveTab('upload')}
                            className="px-4 py-2 bg-brand-600 hover:bg-brand-500 text-white rounded-xl text-xs font-semibold shadow-glow transition-all"
                          >
                            Switch to Video Upload
                          </button>
                          <button
                            onClick={() => setActiveTab('demo')}
                            className="px-4 py-2 bg-navy-850 hover:bg-navy-800 text-slate-200 rounded-xl text-xs font-medium border border-navy-750 transition-all"
                          >
                            Try Demo Mode
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div className="space-y-3">
                        <div className="w-14 h-14 rounded-2xl bg-navy-900 border border-navy-800 flex items-center justify-center text-slate-400 mx-auto shadow-subtle">
                          <Camera className="w-7 h-7" />
                        </div>
                        <h3 className="text-base font-bold text-white">Live Camera Standby</h3>
                        <p className="text-xs text-slate-400 max-w-sm">
                          Click below to start your webcam and enable real-time landmark tracking.
                        </p>
                        <button
                          onClick={startCamera}
                          className="px-5 py-2 bg-brand-600 hover:bg-brand-500 text-white rounded-xl text-xs font-semibold shadow-glow transition-all inline-flex items-center gap-1.5"
                        >
                          <Play className="w-3.5 h-3.5" />
                          <span>Start Webcam</span>
                        </button>
                      </div>
                    )}
                  </div>
                )}

                {cameraError && (
                  <div className="absolute bottom-4 left-4 right-4 bg-rose-950/90 border border-rose-800 rounded-xl p-3 text-xs text-rose-200 flex items-start gap-2">
                    <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-semibold">Camera Access Notice</p>
                      <p className="text-[11px] text-rose-300 mt-0.5">{cameraError}</p>
                    </div>
                  </div>
                )}
              </div>

              {/* Camera Controls */}
              {isCameraActive && (
                <div className="flex items-center justify-between bg-navy-900 border border-navy-800 rounded-xl p-3 text-xs">
                  <button
                    onClick={stopCamera}
                    className="flex items-center gap-1.5 px-3 py-1.5 bg-rose-950 hover:bg-rose-900 text-rose-300 border border-rose-800 rounded-lg font-medium"
                  >
                    <CameraOff className="w-3.5 h-3.5" />
                    <span>Stop Camera</span>
                  </button>
                  <span className="text-[11px] font-mono text-emerald-400 flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                    <span>Camera Live (60 FPS)</span>
                  </span>
                </div>
              )}
            </div>
          )}

          {/* TAB 3: DEMONSTRATION MODE (Mode C) */}
          {activeTab === 'demo' && (
            <div className="space-y-4">
              <div className="relative aspect-[4/3] bg-navy-950 border border-navy-800 rounded-2xl overflow-hidden shadow-glow flex items-center justify-center">
                <canvas ref={demoCanvasRef} className="w-full h-full object-cover" />

                {!isDemoRunning && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center bg-navy-950/95 backdrop-blur-sm">
                    <div className="w-14 h-14 rounded-2xl bg-amber-950/60 border border-amber-800/80 flex items-center justify-center text-amber-400 mx-auto shadow-subtle mb-3">
                      <Sparkles className="w-7 h-7" />
                    </div>
                    <div className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-amber-950 text-amber-300 border border-amber-800 text-[10px] font-mono font-bold mb-2 uppercase">
                      Exhibition Demonstration Mode
                    </div>
                    <h3 className="text-base font-bold text-white">Simulated Kinematic Demonstration</h3>
                    <p className="text-xs text-slate-400 max-w-sm mt-1 leading-relaxed">
                      Select any sign from the roster below to stream verified canonical ISL motion trajectories.
                      Ideal for demonstration booths when a live webcam or signer is unavailable.
                    </p>
                    <button
                      onClick={() => runDemonstration(demoSignId)}
                      className="mt-4 px-5 py-2 bg-amber-600 hover:bg-amber-500 text-white rounded-xl text-xs font-semibold shadow-glow transition-all inline-flex items-center gap-1.5"
                    >
                      <Play className="w-3.5 h-3.5" />
                      <span>Start Simulation ({demoSignId})</span>
                    </button>
                  </div>
                )}
              </div>

              {/* Demo Sign Selector */}
              <div className="bg-navy-900 border border-navy-800 rounded-xl p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider font-mono flex items-center gap-1.5">
                    <Sparkles className="w-3.5 h-3.5 text-amber-400" />
                    <span>Select Demonstration Sign</span>
                  </span>
                  {isDemoRunning && (
                    <button
                      onClick={stopDemo}
                      className="px-2 py-0.5 bg-slate-800 text-slate-300 text-[10px] rounded hover:bg-slate-700"
                    >
                      Halt Demo
                    </button>
                  )}
                </div>

                <div className="flex flex-wrap gap-1.5">
                  {supportedSigns.map((s) => (
                    <button
                      key={s.id}
                      onClick={() => runDemonstration(s.id)}
                      className={`px-2.5 py-1 text-xs rounded-lg border transition-all ${
                        demoSignId === s.id && isDemoRunning
                          ? 'bg-amber-600 text-white border-amber-500 font-semibold shadow-glow'
                          : 'bg-navy-950 text-slate-300 border-navy-800 hover:border-navy-700 hover:text-white'
                      }`}
                    >
                      {s.name}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* RIGHT COLUMN: Translation Card, Confidence, Audio & Recent Logs (5 Cols) */}
        <div className="lg:col-span-5 space-y-4">
          {/* Mode Pill Indicator */}
          <div className="flex items-center justify-between px-3.5 py-2 rounded-xl bg-navy-900 border border-navy-800 text-xs">
            <span className="text-slate-400">Current Input Mode:</span>
            <span className={`px-2 py-0.5 rounded-full text-[10px] font-mono font-bold uppercase ${
              activeTab === 'upload' 
                ? 'bg-brand-950 text-brand-300 border border-brand-800'
                : activeTab === 'camera'
                ? 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                : 'bg-amber-950 text-amber-300 border border-amber-800'
            }`}>
              {activeTab === 'upload' ? 'Mode A: Video Upload' : activeTab === 'camera' ? 'Mode B: Live Camera' : 'Mode C: Demonstration'}
            </span>
          </div>

          {/* Primary Translation Output Card */}
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
            latencyMs={prediction?.processing_duration_ms || prediction?.latency_ms}
            bufferFill={prediction?.buffer_fill}
          />

          {/* Video Metadata Inspector (If Video Upload) */}
          {prediction?.video_metadata && (
            <div className="bg-navy-900 border border-navy-800 rounded-xl p-3.5 text-xs font-mono space-y-1">
              <span className="text-[10px] uppercase text-slate-400 font-bold block mb-1">
                Decoded Video Metadata:
              </span>
              <div className="grid grid-cols-2 gap-2 text-[11px] text-slate-300">
                <div>Duration: <strong className="text-white">{prediction.video_metadata.duration_seconds}s</strong></div>
                <div>Frames: <strong className="text-white">{prediction.video_metadata.total_frames}</strong></div>
                <div>FPS: <strong className="text-white">{prediction.video_metadata.fps}</strong></div>
                <div>Sampled: <strong className="text-sky-400">{prediction.video_metadata.sampled_frames} frames</strong></div>
              </div>
            </div>
          )}

          {/* Voice Output Controls */}
          <div className="bg-navy-900 border border-navy-800 rounded-xl p-4 shadow-subtle space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider font-mono flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-brand-400" />
                <span>Regional Voice Output</span>
              </span>
              <button
                onClick={onToggleMute}
                className={`text-[11px] px-2 py-0.5 rounded border transition-all ${
                  isMuted ? 'bg-rose-950 text-rose-300 border-rose-800' : 'bg-navy-950 text-slate-300 border-navy-800'
                }`}
              >
                {isMuted ? 'Audio Muted' : 'Sound Active'}
              </button>
            </div>

            {/* Language Selector */}
            <div className="flex items-center justify-between text-xs pt-1">
              <span className="text-slate-400">Spoken Language:</span>
              <div className="flex gap-1">
                {(['en', 'hi', 'te'] as SupportedLanguage[]).map((l) => (
                  <button
                    key={l}
                    onClick={() => onLanguageChange(l)}
                    className={`px-2.5 py-1 rounded text-xs font-medium uppercase transition-all ${
                      selectedLanguage === l
                        ? 'bg-brand-600 text-white font-semibold'
                        : 'bg-navy-950 text-slate-400 hover:text-white border border-navy-800'
                    }`}
                  >
                    {l === 'en' ? 'English' : l === 'hi' ? 'Hindi' : 'Telugu'}
                  </button>
                ))}
              </div>
            </div>

            {/* Auto-Speak Toggle */}
            <div className="flex items-center justify-between text-xs pt-1 border-t border-navy-850">
              <span className="text-slate-400">Auto-speak upon detection:</span>
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

          {/* Recent Translations Feed */}
          <div className="bg-navy-900 border border-navy-800 rounded-xl p-4 shadow-subtle">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider font-mono">
                Recent Translations
              </span>
              <span className="text-[10px] text-slate-500 font-mono">Session Log</span>
            </div>

            {recentTranslations.length === 0 ? (
              <p className="text-xs text-slate-500 italic py-3 text-center">
                No gestures recorded yet in this session.
              </p>
            ) : (
              <div className="space-y-2">
                {recentTranslations.map((rec) => (
                  <div
                    key={rec.id}
                    className="flex items-center justify-between p-2.5 rounded-lg bg-navy-950 border border-navy-800 text-xs"
                  >
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="font-bold text-white">{rec.translation_text}</span>
                        <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-navy-900 text-slate-400 border border-navy-800 uppercase">
                          {rec.language}
                        </span>
                        <span className={`text-[9px] font-mono px-1.5 py-0.2 rounded uppercase ${
                          rec.recognition_mode === 'video_upload' 
                            ? 'text-brand-300 bg-brand-950 border border-brand-800' 
                            : rec.recognition_mode === 'live_camera'
                            ? 'text-emerald-300 bg-emerald-950 border border-emerald-800'
                            : 'text-amber-300 bg-amber-950 border border-amber-800'
                        }`}>
                          {rec.recognition_mode === 'video_upload' ? 'Upload' : rec.recognition_mode === 'live_camera' ? 'Camera' : 'Demo'}
                        </span>
                      </div>
                      <span className="text-[10px] text-slate-500 font-mono">
                        {Math.round(rec.confidence * 100)}% match • {new Date(rec.timestamp).toLocaleTimeString()}
                      </span>
                    </div>

                    <div className="flex items-center gap-1">
                      <button
                        onClick={() => handleSpeak(rec.translation_text, rec.language)}
                        disabled={isMuted}
                        className="p-1.5 rounded text-slate-400 hover:text-white hover:bg-navy-900 transition-all"
                        title="Replay Voice Audio"
                      >
                        <Volume2 className="w-3.5 h-3.5" />
                      </button>
                      <button
                        onClick={() => handleCopyTranslation(rec.translation_text)}
                        className="p-1.5 rounded text-slate-400 hover:text-white hover:bg-navy-900 transition-all"
                        title="Copy Text"
                      >
                        <Copy className="w-3.5 h-3.5" />
                      </button>
                    </div>
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
