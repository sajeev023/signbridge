import React, { useState, useEffect, useRef } from 'react';
import {
  ShieldCheck,
  Camera,
  Layers,
  Database,
  Download,
  Trash2,
  AlertTriangle,
  CheckCircle,
  HelpCircle,
  X,
  Play,
  RotateCcw,
  Sliders,
  Sparkles,
  Info,
  Activity,
  Cpu
} from 'lucide-react';
import { LandmarkPoint, alphabetInferenceEngine } from '../services/alphabetInferenceEngine';
import letterASpecialistModel from '../services/letter_a_specialist_model.json';

export interface ZeroLeakSample {
  id: string;
  timestamp: string;
  signerId: string;
  cohort: 'train' | 'val' | 'test';
  letter: string;
  isDynamic: boolean;
  metadata: {
    distance: string;
    orientation: string;
    lighting: string;
    handedness: string;
    resolution: string;
    fps: number;
    trackingQualityScore: number;
  };
  landmarks?: LandmarkPoint[];
  trajectory?: { x: number; y: number; t: number }[];
  features42?: number[];
  resampledTrajectory32?: number[];
}

interface ZeroLeakDataCollectorProps {
  isOpen: boolean;
  onClose: () => void;
  currentLandmarks: LandmarkPoint[] | null;
  trackingQualityReport?: any;
  isCameraActive: boolean;
  currentFps: number;
  videoElement?: HTMLVideoElement | null;
}

const ASL_LETTERS = [
  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
  'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
  'U', 'V', 'W', 'X', 'Y', 'Z'
];

export const ZeroLeakDataCollector: React.FC<ZeroLeakDataCollectorProps> = ({
  isOpen,
  onClose,
  currentLandmarks,
  trackingQualityReport,
  isCameraActive,
  currentFps,
  videoElement
}) => {
  // Protocol Consent & Privacy
  const [hasConsented, setHasConsented] = useState<boolean>(false);
  const [showProtocolHelp, setShowProtocolHelp] = useState<boolean>(false);

  // Signer & Session Information
  const [signerId, setSignerId] = useState<string>('Signer_P21');
  const [cohort, setCohort] = useState<'train' | 'val' | 'test'>('test');
  const [sessionId] = useState<string>(() => `SES_${Date.now().toString(36).toUpperCase()}`);

  // Environmental Metadata
  const [distance, setDistance] = useState<string>('Medium (0.5m-1.0m)');
  const [orientation, setOrientation] = useState<string>('Frontal (0°)');
  const [lighting, setLighting] = useState<string>('Fluorescent Office');
  const [handedness, setHandedness] = useState<string>('Right');

  // Sign Selection
  const [selectedLetter, setSelectedLetter] = useState<string>('A');

  // Storage
  const [recordedSamples, setRecordedSamples] = useState<ZeroLeakSample[]>([]);
  const [lastActionMsg, setLastActionMsg] = useState<string | null>(null);

  // Dynamic Recording State
  const [isRecordingDynamic, setIsRecordingDynamic] = useState<boolean>(false);
  const [dynamicProgress, setDynamicProgress] = useState<number>(0);
  const dynamicBufferRef = useRef<{ x: number; y: number; t: number }[]>([]);

  // Automatically adjust default cohort split based on signer index (P01-P15: train, P16-P20: val, P21-P25: test)
  useEffect(() => {
    const match = signerId.match(/P(\d+)/i);
    if (match) {
      const idx = parseInt(match[1], 10);
      if (idx <= 15) setCohort('train');
      else if (idx <= 20) setCohort('val');
      else setCohort('test');
    }
  }, [signerId]);

  if (!isOpen) return null;

  const isDynamicSign = selectedLetter === 'J' || selectedLetter === 'Z';

  // Compute canonical 42 continuous biomechanical invariant features
  const extractBiomechanics42 = (pts: LandmarkPoint[]): number[] => {
    if (!pts || pts.length < 21) return new Array(42).fill(0);
    return alphabetInferenceEngine.extractSingleHandFeatures(pts);
  };

  // Lightweight forward pass for Letter A Specialist Discriminator (calibrated MLP)
  const runLetterASpecialistMLP = (x: number[]): { probA: number; probOther: number; latencyMs: number } => {
    const t0 = performance.now();
    const w: any = (letterASpecialistModel as any).weights;
    const temp: number = (letterASpecialistModel as any).temperature || 1.2;

    // Layer 1: Linear 42 -> 32 + BatchNorm1d + ReLU
    const z1 = new Float32Array(32);
    for (let i = 0; i < 32; i++) {
      let sum = w.b1[i];
      const row = w.w1[i];
      for (let j = 0; j < 42; j++) {
        sum += row[j] * x[j];
      }
      const norm = (sum - w.bn1_mean[i]) / Math.sqrt(w.bn1_var[i] + 1e-5);
      const bn = norm * w.bn1_weight[i] + w.bn1_bias[i];
      z1[i] = bn > 0 ? bn : 0;
    }

    // Layer 2: Linear 32 -> 16 + ReLU
    const z2 = new Float32Array(16);
    for (let i = 0; i < 16; i++) {
      let sum = w.b2[i];
      const row = w.w2[i];
      for (let j = 0; j < 32; j++) {
        sum += row[j] * z1[j];
      }
      z2[i] = sum > 0 ? sum : 0;
    }

    // Layer 3: Linear 16 -> 2 + Temperature scaling
    let l0 = w.b3[0];
    let l1 = w.b3[1];
    for (let j = 0; j < 16; j++) {
      l0 += w.w3[0][j] * z2[j];
      l1 += w.w3[1][j] * z2[j];
    }
    l0 /= temp;
    l1 /= temp;

    const maxL = Math.max(l0, l1);
    const exp0 = Math.exp(l0 - maxL);
    const exp1 = Math.exp(l1 - maxL);
    const sumExp = exp0 + exp1;

    const latencyMs = performance.now() - t0;
    return {
      probOther: exp0 / sumExp,
      probA: exp1 / sumExp,
      latencyMs
    };
  };

  // Comprehensive Real-Time Evaluation Diagnostics for Letter A
  const evaluateLetterADiagnostics = (pts: LandmarkPoint[] | null) => {
    if (!pts || pts.length < 21) {
      return {
        landmarkCount: 0,
        featureCount: 0,
        qualityScore: 0,
        isSuitable: false,
        fistCurled: false,
        thumbElevated: false,
        thumbLateral: false,
        thumbNotCrossing: false,
        thumbNotTucked: false,
        probA: 0,
        probOther: 1.0,
        predictedLabel: 'NO_HAND',
        feedback: 'No hand detected. Position hand in camera view.',
        latencyMs: 0
      };
    }

    const feats = extractBiomechanics42(pts);
    const mlp = runLetterASpecialistMLP(feats);

    const extIdx = feats[1];
    const extMid = feats[2];
    const extRing = feats[3];
    const extPinky = feats[4];
    const cosSpread = feats[15];
    const distThumbMid = feats[20];
    const thumbX = feats[24];
    const thumbY = feats[25];

    const fistCurled = extIdx < 0.35 && extMid < 0.35 && extRing < 0.35 && extPinky < 0.35;
    const thumbElevated = thumbY >= 0.55;
    const thumbLateral = thumbX >= 0.12;
    const thumbNotCrossing = cosSpread < 0.15;
    const thumbNotTucked = distThumbMid >= 0.38;

    const isSuitable = fistCurled && thumbElevated && thumbLateral && thumbNotCrossing && thumbNotTucked;

    let predictedLabel = 'REJECT_LOW_CONFIDENCE';
    let feedback = 'Pose matches canonical ASL Letter A criteria. Ready to capture.';

    if (mlp.probA >= 0.50 && isSuitable) {
      predictedLabel = 'A (CONFIRMED)';
    } else if (mlp.probA >= 0.50 && !isSuitable) {
      predictedLabel = 'REJECT_HARD_NEGATIVE';
      if (!thumbNotCrossing) feedback = 'Thumb crosses knuckles (confusable with Letter S).';
      else if (!thumbElevated) feedback = 'Thumb folded horizontally (confusable with Letter E).';
      else if (!thumbNotTucked) feedback = 'Thumb tucked under fingers (confusable with M/N/T).';
      else feedback = 'Biomechanical gate failure.';
    } else if (!fistCurled) {
      predictedLabel = 'REJECT_LOOSE_FIST';
      feedback = 'Curl index, middle, ring & pinky fingers tightly into palm.';
    } else if (!thumbElevated) {
      predictedLabel = 'REJECT_THUMB_LOW';
      feedback = 'Raise thumb tip upright along radial edge of index finger (too low, resembles E).';
    } else if (!thumbLateral) {
      predictedLabel = 'REJECT_THUMB_CROSS';
      feedback = 'Keep thumb on lateral radial margin (crossing palm, resembles S).';
    } else if (!thumbNotTucked) {
      predictedLabel = 'REJECT_THUMB_TUCK';
      feedback = 'Do not tuck thumb under curled fingers (resembles M/N/T).';
    } else {
      predictedLabel = 'REJECT_NON_A';
      feedback = 'Handshape does not match Letter A profile.';
    }

    return {
      landmarkCount: pts.length,
      featureCount: feats.length,
      qualityScore: trackingQualityReport?.qualityScore ?? 100,
      isSuitable,
      fistCurled,
      thumbElevated,
      thumbLateral,
      thumbNotCrossing,
      thumbNotTucked,
      probA: mlp.probA,
      probOther: mlp.probOther,
      predictedLabel,
      feedback,
      latencyMs: mlp.latencyMs
    };
  };

  // Static capture
  const handleCaptureStatic = () => {
    if (!hasConsented) {
      alert('Ethical protocol compliance requires consenting to non-PII recording before capture.');
      return;
    }
    if (!currentLandmarks || currentLandmarks.length < 21) {
      setLastActionMsg('⚠️ No hand detected. Align hand within camera frame.');
      return;
    }

    const resolution = videoElement
      ? `${videoElement.videoWidth}x${videoElement.videoHeight}`
      : '640x480';

    const qualityScore = trackingQualityReport?.qualityScore ?? 100;
    const feats42 = extractBiomechanics42(currentLandmarks);

    const sample: ZeroLeakSample = {
      id: `SMP_${selectedLetter}_${Date.now()}`,
      timestamp: new Date().toISOString(),
      signerId,
      cohort,
      letter: selectedLetter,
      isDynamic: false,
      metadata: {
        distance,
        orientation,
        lighting,
        handedness,
        resolution,
        fps: currentFps,
        trackingQualityScore: qualityScore
      },
      landmarks: JSON.parse(JSON.stringify(currentLandmarks)),
      features42: feats42
    };

    setRecordedSamples(prev => [sample, ...prev]);
    setLastActionMsg(`✅ Captured static '${selectedLetter}' for ${signerId} (${cohort})`);
  };

  // Dynamic trajectory capture (16-frame buffer)
  const handleStartDynamicRecording = () => {
    if (!hasConsented) {
      alert('Ethical protocol compliance requires consenting to non-PII recording before capture.');
      return;
    }
    if (isRecordingDynamic) return;

    setIsRecordingDynamic(true);
    setDynamicProgress(0);
    dynamicBufferRef.current = [];

    const activeTipIndex = selectedLetter === 'J' ? 20 : 8; // Pinky tip for J, Index tip for Z
    const durationMs = 1200;
    const intervalMs = 40;
    let elapsed = 0;

    const interval = setInterval(() => {
      elapsed += intervalMs;
      const progress = Math.min(100, Math.round((elapsed / durationMs) * 100));
      setDynamicProgress(progress);

      if (currentLandmarks && currentLandmarks.length > activeTipIndex) {
        const pt = currentLandmarks[activeTipIndex];
        dynamicBufferRef.current.push({ x: pt.x, y: pt.y, t: Date.now() });
      }

      if (elapsed >= durationMs) {
        clearInterval(interval);
        setIsRecordingDynamic(false);
        saveDynamicEpisode(dynamicBufferRef.current);
      }
    }, intervalMs);
  };

  const saveDynamicEpisode = (rawPts: { x: number; y: number; t: number }[]) => {
    if (rawPts.length < 8) {
      setLastActionMsg(`⚠️ Insufficient trajectory frames (${rawPts.length}/16). Repeat gesture.`);
      return;
    }

    // Resample to 16 normalized points (32 features)
    const resampled32: number[] = [];
    const p0 = rawPts[0];
    const wrist = currentLandmarks ? currentLandmarks[0] : { x: 0.5, y: 0.8 };
    const midMcp = currentLandmarks ? currentLandmarks[9] : { x: 0.5, y: 0.6 };
    const palmScale = Math.hypot(midMcp.x - wrist.x, midMcp.y - wrist.y) || 0.15;

    for (let i = 0; i < 16; i++) {
      const idx = Math.min(rawPts.length - 1, Math.round((i / 15) * (rawPts.length - 1)));
      const p = rawPts[idx];
      resampled32.push((p.x - p0.x) / palmScale);
      resampled32.push((p.y - p0.y) / palmScale);
    }

    const resolution = videoElement
      ? `${videoElement.videoWidth}x${videoElement.videoHeight}`
      : '640x480';

    const sample: ZeroLeakSample = {
      id: `DYN_${selectedLetter}_${Date.now()}`,
      timestamp: new Date().toISOString(),
      signerId,
      cohort,
      letter: selectedLetter,
      isDynamic: true,
      metadata: {
        distance,
        orientation,
        lighting,
        handedness,
        resolution,
        fps: currentFps,
        trackingQualityScore: trackingQualityReport?.qualityScore ?? 100
      },
      trajectory: rawPts,
      resampledTrajectory32: resampled32
    };

    setRecordedSamples(prev => [sample, ...prev]);
    setLastActionMsg(`✅ Captured dynamic '${selectedLetter}' trajectory (${rawPts.length} frames resampled to 16)`);
  };

  // Delete sample
  const handleDeleteSample = (id: string) => {
    setRecordedSamples(prev => prev.filter(s => s.id !== id));
    setLastActionMsg('🗑️ Sample removed from session.');
  };

  // Export JSON
  const handleExportJson = () => {
    if (recordedSamples.length === 0) {
      alert('No samples recorded in this session to export.');
      return;
    }

    const exportManifest = {
      exportVersion: '1.0.0-zero-leak',
      sessionMetadata: {
        sessionId,
        exportTimestamp: new Date().toISOString(),
        totalSamples: recordedSamples.length,
        signerCohortDistribution: {
          train: recordedSamples.filter(s => s.cohort === 'train').length,
          val: recordedSamples.filter(s => s.cohort === 'val').length,
          test: recordedSamples.filter(s => s.cohort === 'test').length
        },
        ethicalCompliance: {
          piiCollected: false,
          consentObtained: hasConsented,
          rightToEraseAvailable: true
        }
      },
      records: recordedSamples
    };

    const blob = new Blob([JSON.stringify(exportManifest, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ZERO_LEAK_DATASET_${signerId}_${sessionId}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    setLastActionMsg('💾 Dataset exported successfully as JSON.');
  };

  // Letters recorded for current signer
  const capturedLettersForSigner = new Set(
    recordedSamples.filter(s => s.signerId === signerId).map(s => s.letter)
  );

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-3 bg-black/80 backdrop-blur-md overflow-y-auto">
      <div className="relative w-full max-w-5xl bg-slate-900 border border-amber-500/40 rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[92vh]">
        {/* Header */}
        <div className="p-4 bg-gradient-to-r from-amber-950/60 via-slate-900 to-indigo-950/60 border-b border-amber-500/30 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-amber-500/20 border border-amber-400/40 flex items-center justify-center text-amber-300">
              <Database className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-base font-bold text-slate-100">SIGNBRIDGE TITAN: Zero-Leak Data Collector Studio</h2>
                <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-amber-500/20 border border-amber-500/30 text-amber-300">
                  Frozen Validation Infrastructure
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Ethical, non-PII dataset acquisition tool for independent human generalization benchmarking.
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowProtocolHelp(!showProtocolHelp)}
              className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
              title="View Collection Protocol Guide"
            >
              <HelpCircle className="w-4 h-4" />
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-slate-200 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Protocol Guide Banner */}
        {showProtocolHelp && (
          <div className="p-4 bg-amber-950/40 border-b border-amber-500/30 text-xs text-amber-200/90 flex flex-col gap-2">
            <div className="flex items-center gap-2 font-semibold text-amber-300">
              <Info className="w-4 h-4" />
              <span>Scientific Zero-Leak Acquisition Standard</span>
            </div>
            <p>
              To maintain scientific integrity, <strong>Signer_P21 through Signer_P25</strong> are designated as permanently held-out test participants.
              Record all 26 letters (A–Z) per participant with variations in lighting, distance, and orientation. Never use test samples for training or tuning.
            </p>
          </div>
        )}

        <div className="flex-1 p-4 overflow-y-auto space-y-4 text-xs">
          {/* 1. Consent and Ethics Confirmation */}
          <div className={`p-3 rounded-xl border transition-all ${
            hasConsented
              ? 'bg-emerald-950/20 border-emerald-500/40 text-emerald-300'
              : 'bg-rose-950/20 border-rose-500/40 text-rose-300'
          }`}>
            <label className="flex items-start gap-2.5 cursor-pointer">
              <input
                type="checkbox"
                checked={hasConsented}
                onChange={e => setHasConsented(e.target.checked)}
                className="mt-0.5 rounded border-slate-700 text-emerald-500 focus:ring-emerald-400 cursor-pointer"
              />
              <div>
                <span className="font-semibold text-sm">
                  {hasConsented ? 'Consent Verified: Ethical Protocol Compliance Active' : 'Consent Required Before Data Acquisition'}
                </span>
                <p className="text-[11px] text-slate-400 mt-0.5">
                  I agree to record anonymized hand kinematic coordinates for open scientific validation. No facial photographs, video streams, or PII are retained.
                </p>
              </div>
            </label>
          </div>

          {/* 2. Participant & Cohort Setup */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 p-3 bg-slate-800/40 rounded-xl border border-slate-700/60">
            <div>
              <label className="text-slate-400 block mb-1">Signer Pseudonym ID</label>
              <select
                value={signerId}
                onChange={e => setSignerId(e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-slate-200 focus:border-amber-400 outline-none"
              >
                <optgroup label="Final Human Evaluation Cohort (Signer_P26–P35)">
                  {Array.from({ length: 10 }, (_, i) => {
                    const id = `Signer_P${String(i + 26).padStart(2, '0')}`;
                    return <option key={id} value={id}>{id} (Held-Out Final Eval)</option>;
                  })}
                </optgroup>
                <optgroup label="Held-Out Test Cohort (Zero-Leak)">
                  <option value="Signer_P21">Signer_P21 (Held-out Test)</option>
                  <option value="Signer_P22">Signer_P22 (Held-out Test)</option>
                  <option value="Signer_P23">Signer_P23 (Held-out Test)</option>
                  <option value="Signer_P24">Signer_P24 (Held-out Test)</option>
                  <option value="Signer_P25">Signer_P25 (Held-out Test)</option>
                </optgroup>
                <optgroup label="Validation Cohort">
                  <option value="Signer_P16">Signer_P16 (Validation)</option>
                  <option value="Signer_P17">Signer_P17 (Validation)</option>
                  <option value="Signer_P18">Signer_P18 (Validation)</option>
                  <option value="Signer_P19">Signer_P19 (Validation)</option>
                  <option value="Signer_P20">Signer_P20 (Validation)</option>
                </optgroup>
                <optgroup label="Training Cohort">
                  {Array.from({ length: 15 }, (_, i) => {
                    const id = `Signer_P${String(i + 1).padStart(2, '0')}`;
                    return <option key={id} value={id}>{id} (Train)</option>;
                  })}
                </optgroup>
              </select>
            </div>

            <div>
              <label className="text-slate-400 block mb-1">Cohort Partition</label>
              <span className={`inline-block w-full text-center py-2 px-3 rounded-lg font-bold uppercase tracking-wider text-xs border ${
                cohort === 'test'
                  ? 'bg-rose-950/40 border-rose-500/50 text-rose-300'
                  : cohort === 'val'
                  ? 'bg-indigo-950/40 border-indigo-500/50 text-indigo-300'
                  : 'bg-emerald-950/40 border-emerald-500/50 text-emerald-300'
              }`}>
                {cohort === 'test' ? '🔒 Frozen Held-Out Test' : cohort === 'val' ? '🔍 Validation' : '📚 Training'}
              </span>
            </div>

            <div>
              <label className="text-slate-400 block mb-1">Camera Distance</label>
              <select
                value={distance}
                onChange={e => setDistance(e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-slate-200 outline-none"
              >
                <option value="Near (<0.5m)">Near (&lt;0.5m)</option>
                <option value="Medium (0.5m-1.0m)">Medium (0.5m–1.0m)</option>
                <option value="Far (>1.0m)">Far (&gt;1.0m)</option>
              </select>
            </div>

            <div>
              <label className="text-slate-400 block mb-1">Hand Orientation</label>
              <select
                value={orientation}
                onChange={e => setOrientation(e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-slate-200 outline-none"
              >
                <option value="Frontal (0°)">Frontal (0°)</option>
                <option value="Lateral 30°">Lateral 30°</option>
                <option value="Lateral 45°">Lateral 45°</option>
                <option value="Lateral 60°">Lateral 60°</option>
                <option value="Angled/Tilted">Angled / Tilted</option>
              </select>
            </div>
          </div>

          {/* 3. Secondary Environmental Metadata */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 p-3 bg-slate-800/20 rounded-xl border border-slate-700/40">
            <div>
              <label className="text-slate-400 block mb-1">Lighting Condition</label>
              <select
                value={lighting}
                onChange={e => setLighting(e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-slate-200 outline-none"
              >
                <option value="Natural Daylight">Natural Daylight</option>
                <option value="Fluorescent Office">Fluorescent Office</option>
                <option value="Dim Ambient">Dim Ambient</option>
                <option value="Strong Backlit">Strong Backlit</option>
              </select>
            </div>

            <div>
              <label className="text-slate-400 block mb-1">Dominant Hand</label>
              <select
                value={handedness}
                onChange={e => setHandedness(e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-slate-200 outline-none"
              >
                <option value="Right">Right Hand</option>
                <option value="Left">Left Hand</option>
              </select>
            </div>

            <div className="flex flex-col justify-end">
              <div className="p-2 rounded-lg bg-slate-900/80 border border-slate-700/80 text-[11px] text-slate-400 flex items-center justify-between">
                <span>Live Feed:</span>
                <span className={isCameraActive ? 'text-emerald-400 font-medium' : 'text-rose-400'}>
                  {isCameraActive ? `Active (${currentFps} FPS)` : 'Standby'}
                </span>
                <span>Tracked Hand:</span>
                <span className={currentLandmarks ? 'text-cyan-400 font-medium' : 'text-slate-500'}>
                  {currentLandmarks ? 'Detected (21 pts)' : 'None'}
                </span>
              </div>
            </div>
          </div>

          {/* 4. Letter Selector Matrix */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="font-semibold text-slate-300">
                Select Target ASL Sign (A–Z) — {capturedLettersForSigner.size}/26 Recorded for {signerId}
              </span>
              <span className="text-[11px] text-slate-400">
                {isDynamicSign ? 'Dynamic Trajectory Gesture (J/Z)' : 'Static Handshape'}
              </span>
            </div>

            {/* Quick-Select Benchmark Bar for Letter A & Hard Negatives */}
            <div className="flex items-center gap-2 p-2 mb-2.5 bg-slate-800/60 rounded-xl border border-amber-500/30 overflow-x-auto">
              <span className="font-semibold text-amber-300 text-[11px] whitespace-nowrap flex items-center gap-1">
                <Sparkles className="w-3.5 h-3.5 text-amber-400" />
                Phase 1 Quick-Select:
              </span>
              {[
                { letter: 'A', role: 'Target' },
                { letter: 'E', role: 'Hard Neg' },
                { letter: 'S', role: 'Hard Neg' },
                { letter: 'M', role: 'Hard Neg' },
                { letter: 'N', role: 'Hard Neg' },
                { letter: 'T', role: 'Hard Neg' }
              ].map(item => (
                <button
                  key={item.letter}
                  onClick={() => setSelectedLetter(item.letter)}
                  className={`px-2.5 py-1 rounded-lg text-[11px] font-bold transition-all border whitespace-nowrap flex items-center gap-1 ${
                    selectedLetter === item.letter
                      ? 'bg-amber-500 text-slate-950 border-amber-400 shadow-md ring-1 ring-amber-400/50'
                      : 'bg-slate-900/80 text-slate-300 border-slate-700 hover:border-amber-400/50 hover:bg-slate-800'
                  }`}
                >
                  <span className="font-mono">{item.letter}</span>
                  <span className="text-[10px] font-normal opacity-80">({item.role})</span>
                </button>
              ))}
            </div>

            <div className="grid grid-cols-7 sm:grid-cols-9 md:grid-cols-13 gap-1.5">
              {ASL_LETTERS.map(letter => {
                const isCaptured = capturedLettersForSigner.has(letter);
                const isSelected = selectedLetter === letter;
                const isDyn = letter === 'J' || letter === 'Z';

                return (
                  <button
                    key={letter}
                    onClick={() => setSelectedLetter(letter)}
                    className={`relative p-2 rounded-xl text-center font-bold text-sm transition-all border ${
                      isSelected
                        ? 'border-amber-400 bg-amber-500/20 text-amber-200 ring-2 ring-amber-400/30'
                        : isCaptured
                        ? 'border-emerald-500/40 bg-emerald-950/20 text-emerald-300 hover:bg-emerald-900/30'
                        : 'border-slate-700/80 bg-slate-800/60 text-slate-400 hover:bg-slate-700/60 hover:text-slate-200'
                    }`}
                  >
                    {letter}
                    {isDyn && (
                      <span className="absolute -top-1 -right-1 w-2 h-2 rounded-full bg-cyan-400 ring-2 ring-slate-900" title="Dynamic Gesture" />
                    )}
                    {isCaptured && (
                      <span className="absolute -bottom-1 -right-1 w-2 h-2 rounded-full bg-emerald-400 ring-2 ring-slate-900" title="Captured" />
                    )}
                  </button>
                );
              })}
            </div>
          </div>

          {/* 4.5 Real-Time Evaluation Diagnostics HUD */}
          {(() => {
            const diag = evaluateLetterADiagnostics(currentLandmarks);
            return (
              <div className="p-3.5 bg-slate-900/90 rounded-xl border border-amber-500/40 space-y-3">
                <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-3">
                  <div className="flex items-center gap-3">
                    <div className="relative w-14 h-14 rounded-lg overflow-hidden border border-amber-400/50 bg-slate-950 flex-shrink-0">
                      <img
                        src="/references/crops/A.png"
                        alt="ASL Letter A Reference"
                        className="w-full h-full object-contain"
                      />
                      <span className="absolute bottom-0 right-0 bg-amber-500 text-slate-950 text-[9px] font-bold px-1 rounded-tl">
                        Aa
                      </span>
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="font-bold text-sm text-slate-100">Live Evaluation Diagnostics HUD</span>
                        <span className={`px-2 py-0.5 rounded text-[10px] font-semibold border ${
                          diag.predictedLabel === 'A (CONFIRMED)'
                            ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30'
                            : diag.predictedLabel.startsWith('REJECT_HARD_NEGATIVE')
                            ? 'bg-rose-500/20 text-rose-300 border-rose-500/30'
                            : 'bg-amber-500/20 text-amber-300 border-amber-500/30'
                        }`}>
                          {diag.predictedLabel}
                        </span>
                      </div>
                      <p className="text-xs text-slate-300 mt-0.5">
                        {diag.feedback}
                      </p>
                    </div>
                  </div>

                  {/* Telemetry Chips */}
                  <div className="flex flex-wrap items-center gap-2 text-[10px]">
                    <span className="px-2 py-1 rounded bg-slate-800 text-cyan-300 border border-slate-700 flex items-center gap-1 font-mono">
                      <Activity className="w-3 h-3" />
                      Landmarks: {diag.landmarkCount}/21
                    </span>
                    <span className="px-2 py-1 rounded bg-slate-800 text-indigo-300 border border-slate-700 flex items-center gap-1 font-mono">
                      <Cpu className="w-3 h-3" />
                      Features: {diag.featureCount}/42
                    </span>
                    <span className="px-2 py-1 rounded bg-slate-800 text-amber-300 border border-slate-700 font-mono">
                      Latency: {diag.latencyMs.toFixed(2)}ms
                    </span>
                    <span className="px-2 py-1 rounded bg-slate-800 text-slate-300 border border-slate-700 font-mono">
                      {currentFps} FPS
                    </span>
                  </div>
                </div>

                {/* Real-time probability bar */}
                <div className="bg-slate-950/70 p-2.5 rounded-lg border border-slate-800 space-y-1.5">
                  <div className="flex items-center justify-between text-[11px]">
                    <span className="text-slate-400">Letter A Specialist Probability P(A | x):</span>
                    <span className="font-mono font-bold text-amber-300">
                      {(diag.probA * 100).toFixed(1)}% (Non-A: {(diag.probOther * 100).toFixed(1)}%)
                    </span>
                  </div>
                  <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden flex">
                    <div
                      className="bg-gradient-to-r from-amber-500 to-emerald-400 h-full transition-all duration-150"
                      style={{ width: `${Math.min(100, Math.max(0, diag.probA * 100))}%` }}
                    />
                  </div>
                </div>

                {/* Kinematic Gate Checklist */}
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-2 text-[11px] bg-slate-950/60 p-2.5 rounded-lg border border-slate-800">
                  <div className={`flex items-center gap-1.5 ${diag.fistCurled ? 'text-emerald-400' : 'text-slate-500'}`}>
                    <span>{diag.fistCurled ? '✓' : '✗'}</span>
                    <span>4 Fingers Curled</span>
                  </div>
                  <div className={`flex items-center gap-1.5 ${diag.thumbElevated ? 'text-emerald-400' : 'text-slate-500'}`}>
                    <span>{diag.thumbElevated ? '✓' : '✗'}</span>
                    <span>Thumb Upright</span>
                  </div>
                  <div className={`flex items-center gap-1.5 ${diag.thumbLateral ? 'text-emerald-400' : 'text-slate-500'}`}>
                    <span>{diag.thumbLateral ? '✓' : '✗'}</span>
                    <span>Thumb Lateral</span>
                  </div>
                  <div className={`flex items-center gap-1.5 ${diag.thumbNotCrossing ? 'text-emerald-400' : 'text-slate-500'}`}>
                    <span>{diag.thumbNotCrossing ? '✓' : '✗'}</span>
                    <span>No Cross (≠S)</span>
                  </div>
                  <div className={`flex items-center gap-1.5 ${diag.thumbNotTucked ? 'text-emerald-400' : 'text-slate-500'}`}>
                    <span>{diag.thumbNotTucked ? '✓' : '✗'}</span>
                    <span>No Tuck (≠M/T)</span>
                  </div>
                </div>
              </div>
            );
          })()}

          {/* 5. Capture Controls */}
          <div className="p-4 bg-slate-800/50 rounded-xl border border-slate-700 flex flex-col sm:flex-row items-center justify-between gap-4">
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xl font-bold text-amber-300">Target Sign: {selectedLetter}</span>
                <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-slate-700 text-slate-300">
                  {isDynamicSign ? 'Dynamic Trajectory (16 Frames)' : 'Static 21-Landmark + 42 Features'}
                </span>
              </div>
              <p className="text-slate-400 text-[11px] mt-0.5">
                {isDynamicSign
                  ? selectedLetter === 'J'
                    ? 'Start with extended pinky and trace a downward "J" hook in 3D air.'
                    : 'Point index finger and trace a 3-stroke horizontal "Z" zigzag.'
                  : 'Hold stationary pose facing camera. Ensure palm and fingers are clearly visible.'}
              </p>
            </div>

            <div className="flex items-center gap-3">
              {isDynamicSign ? (
                <button
                  onClick={handleStartDynamicRecording}
                  disabled={!hasConsented || isRecordingDynamic || !isCameraActive}
                  className={`px-5 py-2.5 rounded-xl font-bold flex items-center gap-2 shadow-lg transition-all ${
                    isRecordingDynamic
                      ? 'bg-rose-600 text-white animate-pulse'
                      : 'bg-gradient-to-r from-cyan-600 to-indigo-600 hover:from-cyan-500 hover:to-indigo-500 text-white'
                  } disabled:opacity-50 disabled:cursor-not-allowed`}
                >
                  <Play className="w-4 h-4" />
                  <span>{isRecordingDynamic ? `Recording (${dynamicProgress}%)` : `Record ${selectedLetter} (1.2s)`}</span>
                </button>
              ) : (
                <button
                  onClick={handleCaptureStatic}
                  disabled={!hasConsented || !isCameraActive || !currentLandmarks}
                  className="px-5 py-2.5 rounded-xl font-bold bg-gradient-to-r from-amber-600 to-emerald-600 hover:from-amber-500 hover:to-emerald-500 text-white flex items-center gap-2 shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
                >
                  <Camera className="w-4 h-4" />
                  <span>Capture Static '{selectedLetter}'</span>
                </button>
              )}
            </div>
          </div>

          {/* Action Status Feedback */}
          {lastActionMsg && (
            <div className="p-2.5 rounded-lg bg-slate-800/80 border border-slate-700 text-slate-200 text-center font-medium animate-fadeIn">
              {lastActionMsg}
            </div>
          )}

          {/* 6. Captured Samples Manifest Table */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-semibold text-slate-300">
                Session Sample Buffer ({recordedSamples.length} total captures)
              </span>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => {
                    if (confirm('Erase all recorded samples in this session? This action complies with participant right-to-erase.')) {
                      setRecordedSamples([]);
                      setLastActionMsg('🧹 Session buffer cleared.');
                    }
                  }}
                  disabled={recordedSamples.length === 0}
                  className="px-2.5 py-1 rounded-lg border border-rose-900/60 bg-rose-950/30 hover:bg-rose-900/40 text-rose-300 text-[11px] flex items-center gap-1 disabled:opacity-40 cursor-pointer"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  <span>Clear Buffer</span>
                </button>
                <button
                  onClick={handleExportJson}
                  disabled={recordedSamples.length === 0}
                  className="px-3 py-1 rounded-lg border border-emerald-500/50 bg-emerald-950/40 hover:bg-emerald-900/50 text-emerald-200 text-[11px] font-semibold flex items-center gap-1.5 disabled:opacity-40 cursor-pointer shadow"
                >
                  <Download className="w-3.5 h-3.5 text-emerald-400" />
                  <span>Export Zero-Leak Dataset (JSON)</span>
                </button>
              </div>
            </div>

            <div className="max-h-48 overflow-y-auto border border-slate-800 rounded-xl bg-slate-950/40">
              {recordedSamples.length === 0 ? (
                <div className="p-6 text-center text-slate-500">
                  No samples collected yet. Select a letter, verify consent, and capture.
                </div>
              ) : (
                <table className="w-full text-left border-collapse text-[11px]">
                  <thead>
                    <tr className="border-b border-slate-800 bg-slate-900/60 text-slate-400">
                      <th className="p-2">Sample ID</th>
                      <th className="p-2">Signer</th>
                      <th className="p-2">Cohort</th>
                      <th className="p-2">Letter</th>
                      <th className="p-2">Type</th>
                      <th className="p-2">Distance / Angle</th>
                      <th className="p-2">Lighting</th>
                      <th className="p-2 text-right">Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recordedSamples.map(s => (
                      <tr key={s.id} className="border-b border-slate-900/60 hover:bg-slate-800/30">
                        <td className="p-2 font-mono text-slate-300">{s.id}</td>
                        <td className="p-2 text-slate-300">{s.signerId}</td>
                        <td className="p-2">
                          <span className={`px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase ${
                            s.cohort === 'test'
                              ? 'bg-rose-950/60 text-rose-300 border border-rose-800/40'
                              : s.cohort === 'val'
                              ? 'bg-indigo-950/60 text-indigo-300 border border-indigo-800/40'
                              : 'bg-emerald-950/60 text-emerald-300 border border-emerald-800/40'
                          }`}>
                            {s.cohort}
                          </span>
                        </td>
                        <td className="p-2 font-bold text-amber-300">{s.letter}</td>
                        <td className="p-2 text-slate-400">{s.isDynamic ? 'Dynamic (16-pt)' : 'Static (21-pt)'}</td>
                        <td className="p-2 text-slate-400">{s.metadata.distance} / {s.metadata.orientation}</td>
                        <td className="p-2 text-slate-400">{s.metadata.lighting}</td>
                        <td className="p-2 text-right">
                          <button
                            onClick={() => handleDeleteSample(s.id)}
                            className="p-1 rounded hover:bg-rose-950/60 text-rose-400 hover:text-rose-200 transition-colors"
                            title="Delete sample"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-3 bg-slate-950 border-t border-slate-800 flex items-center justify-between text-[11px] text-slate-400">
          <div className="flex items-center gap-3">
            <span>Session: <strong className="text-slate-300">{sessionId}</strong></span>
            <span>Active Signer: <strong className="text-amber-300">{signerId}</strong> ({cohort})</span>
          </div>
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium transition-colors"
          >
            Close Studio
          </button>
        </div>
      </div>
    </div>
  );
};
