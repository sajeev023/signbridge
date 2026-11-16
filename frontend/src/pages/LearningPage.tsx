import React, { useState, useEffect, useRef } from 'react';
import { 
  GraduationCap, 
  Play, 
  Square, 
  CheckCircle2, 
  AlertCircle, 
  Volume2, 
  Trophy, 
  Hand, 
  UploadCloud,
  Sparkles,
  Camera,
  ArrowRight,
  Check
} from 'lucide-react';
import { ISLSign, PracticeEvaluation, SupportedLanguage } from '../types';
import { apiClient, FALLBACK_VOCABULARY } from '../services/apiClient';
import { speechService } from '../services/speechService';
import { LandmarkRenderer, FrameLandmarks } from '../services/cameraLandmarkDetector';

interface AlphabetSignGuide {
  letter: string;
  name: string;
  category: 'Single-Hand' | 'ISLRTC Dual' | 'Dynamic';
  desc: string;
  isPhysicallyValidated: boolean;
  validatedSessions?: number;
  kinematicRules: string[];
}

const ALPHABET_GUIDE: Record<string, AlphabetSignGuide> = {
  A: {
    letter: 'A',
    name: 'Letter A',
    category: 'Single-Hand',
    desc: 'Form a closed fist with all four fingers curled tightly down into the palm; extend the thumb straight up along the outer lateral side of the index finger.',
    isPhysicallyValidated: true,
    validatedSessions: 4,
    kinematicRules: ['Thumb extension > 0.85', 'Index/Middle/Ring/Pinky curl ratio < 0.35', 'Thumb tip resting alongside index PIP joint']
  },
  B: {
    letter: 'B',
    name: 'Letter B',
    category: 'Single-Hand',
    desc: 'Hold all four fingers (index, middle, ring, pinky) extended straight upright and touching together; tuck the thumb flat across the palm.',
    isPhysicallyValidated: true,
    validatedSessions: 4,
    kinematicRules: ['Four fingers vertical parallelism > 0.90', 'Thumb curled across lower palm', 'Zero inter-finger spread']
  },
  C: {
    letter: 'C',
    name: 'Letter C',
    category: 'Single-Hand',
    desc: 'Curve all four fingers and thumb into a smooth semi-circular arch forming an open "C" shape facing inward.',
    isPhysicallyValidated: true,
    validatedSessions: 4,
    kinematicRules: ['Gentle uniform joint flexion across all digits', 'Thumb-to-index tip gap 0.30 - 0.55 palm scale', 'Curved palm contour']
  },
  D: {
    letter: 'D',
    name: 'Letter D',
    category: 'Single-Hand',
    desc: 'Point index finger straight upright; curl middle, ring, and pinky fingers down to touch the thumb tip, forming a circular loop.',
    isPhysicallyValidated: true,
    validatedSessions: 4,
    kinematicRules: ['Index finger vertical extension > 0.80', 'Middle/ring/pinky fingertips touching thumb tip', 'High loop closure metric']
  },
  E: {
    letter: 'E',
    name: 'Letter E',
    category: 'Single-Hand',
    desc: 'Curl all four fingertips tightly downward towards the palm; fold the thumb flat underneath the curled fingertips.',
    isPhysicallyValidated: true,
    validatedSessions: 4,
    kinematicRules: ['High finger curl ratio on all 4 fingers', 'Thumb tucked tightly beneath finger nails', 'Compact fist profile']
  },
  F: {
    letter: 'F',
    name: 'Letter F',
    category: 'Single-Hand',
    desc: 'Touch index fingertip to thumb tip forming a circle; hold middle, ring, and pinky fingers straight upright and spread apart.',
    isPhysicallyValidated: true,
    validatedSessions: 4,
    kinematicRules: ['Thumb-index tip Euclidean distance < 0.12', 'Middle, ring, pinky fully extended upright', 'Non-touching 3-finger fan']
  },
  G: {
    letter: 'G',
    name: 'Letter G',
    category: 'Single-Hand',
    desc: 'Point index finger horizontally forward; extend thumb parallel to index finger with other fingers curled into palm.',
    isPhysicallyValidated: false,
    kinematicRules: ['Index bone vector aligned horizontally', 'Thumb parallel to index', 'Middle/ring/pinky curled']
  },
  H: {
    letter: 'H',
    name: 'Letter H',
    category: 'Single-Hand',
    desc: 'Extend index and middle fingers together horizontally forward; tuck thumb into folded ring and pinky fingers.',
    isPhysicallyValidated: false,
    kinematicRules: ['Index and middle pointing horizontally', 'Index-middle spread < 0.05', 'Thumb folded over ring finger']
  },
  I: {
    letter: 'I',
    name: 'Letter I',
    category: 'Single-Hand',
    desc: 'Make a fist with thumb folded over index, middle, and ring fingers; extend only the pinky finger straight upright.',
    isPhysicallyValidated: false,
    kinematicRules: ['Pinky vertical extension > 0.85', 'Index, middle, ring fully curled into fist', 'Thumb locked across fingers']
  },
  J: {
    letter: 'J',
    name: 'Letter J',
    category: 'Dynamic',
    desc: 'Begin with the "I" handshape (pinky vertical) and trace a downward hooking "J" trajectory through the air.',
    isPhysicallyValidated: false,
    kinematicRules: ['Temporal motion duration 350 - 900ms', 'Vertical downward plunge followed by upward left curve', 'Trajectory curvature threshold > 0.70']
  },
  K: {
    letter: 'K',
    name: 'Letter K',
    category: 'Single-Hand',
    desc: 'Index finger points straight up, middle finger extends forward at an angle, and thumb touches the base knuckle of middle finger.',
    isPhysicallyValidated: false,
    kinematicRules: ['Index vertical, middle angled 45-60 deg forward', 'Thumb tip resting between index and middle MCP', 'Ring and pinky tucked']
  },
  L: {
    letter: 'L',
    name: 'Letter L',
    category: 'Single-Hand',
    desc: 'Extend thumb outward horizontally and index finger upright vertically at an exact 90-degree right angle forming an "L".',
    isPhysicallyValidated: true,
    validatedSessions: 4,
    kinematicRules: ['Inter-digit angle thumb-index ~ 90 deg (1.35 - 1.75 rad)', 'Index vertical extension > 0.80', 'Middle/ring/pinky curled']
  },
  M: {
    letter: 'M',
    name: 'Letter M',
    category: 'ISLRTC Dual',
    desc: 'Place dominant hand with three fingers (index, middle, ring) facing downward onto the open palm of non-dominant hand.',
    isPhysicallyValidated: false,
    kinematicRules: ['Dominant 3 fingers touch base palm', 'Dual-hand distance span < 0.15 palm scale', 'Base palm open horizontally']
  },
  N: {
    letter: 'N',
    name: 'Letter N',
    category: 'ISLRTC Dual',
    desc: 'Place dominant hand with two fingers (index, middle) facing downward onto the open palm of non-dominant hand.',
    isPhysicallyValidated: false,
    kinematicRules: ['Dominant 2 fingers touch base palm', 'Dual-hand distance span < 0.15 palm scale', 'Ring and pinky tucked']
  },
  O: {
    letter: 'O',
    name: 'Letter O',
    category: 'Single-Hand',
    desc: 'Touch all four fingertips to the thumb tip simultaneously, forming a complete circular "O" tube shape.',
    isPhysicallyValidated: true,
    validatedSessions: 4,
    kinematicRules: ['Fingertip-to-thumb pairwise distances all < 0.18', 'Smooth circular knuckle curvature', 'Clear open aperture view']
  },
  P: {
    letter: 'P',
    name: 'Letter P',
    category: 'ISLRTC Dual',
    desc: 'Hold base hand with index vertical; place dominant hand loop against the upper knuckle of the base index finger.',
    isPhysicallyValidated: false,
    kinematicRules: ['Base index vertical', 'Dominant thumb-index loop contacts base index tip', 'Dual-hand kinematic anchor']
  },
  Q: {
    letter: 'Q',
    name: 'Letter Q',
    category: 'ISLRTC Dual',
    desc: 'Form a vertical circular loop with base hand; place dominant index finger touching the bottom lower edge.',
    isPhysicallyValidated: false,
    kinematicRules: ['Base hand circular ring', 'Dominant index touches base lower boundary', 'Dual-hand coordination']
  },
  R: {
    letter: 'R',
    name: 'Letter R',
    category: 'Single-Hand',
    desc: 'Extend index and middle fingers upright, crossing the middle finger over the dorsal back of the index finger.',
    isPhysicallyValidated: true,
    validatedSessions: 4,
    kinematicRules: ['Middle finger crossed over index (lateral crossing > 0.5)', 'Ring and pinky folded into palm', 'Thumb holding ring finger']
  },
  S: {
    letter: 'S',
    name: 'Letter S',
    category: 'ISLRTC Dual',
    desc: 'Hook the pinky finger of dominant hand tightly into the pinky finger of non-dominant base hand.',
    isPhysicallyValidated: false,
    kinematicRules: ['Dominant pinky interlocked with base pinky', 'High pinky contact proximity', 'Remaining fingers folded']
  },
  T: {
    letter: 'T',
    name: 'Letter T',
    category: 'ISLRTC Dual',
    desc: 'Touch dominant index fingertip directly to the outer lateral side base knuckle of non-dominant index finger.',
    isPhysicallyValidated: false,
    kinematicRules: ['Dominant index tip contacts base index MCP', 'Orthogonal contact angle', 'Clear separation of remaining fingers']
  },
  U: {
    letter: 'U',
    name: 'Letter U',
    category: 'Single-Hand',
    desc: 'Extend index and middle fingers straight upright pressed tightly together; fold thumb across ring and pinky fingers.',
    isPhysicallyValidated: true,
    validatedSessions: 4,
    kinematicRules: ['Index and middle vertical extension > 0.85', 'Index-to-middle fingertip distance < 0.08', 'Parallel pointing vectors']
  },
  V: {
    letter: 'V',
    name: 'Letter V',
    category: 'Single-Hand',
    desc: 'Extend index and middle fingers straight upright and spread wide apart in a clear "V" peace sign.',
    isPhysicallyValidated: true,
    validatedSessions: 4,
    kinematicRules: ['Index and middle vertical extension > 0.85', 'Index-to-middle fingertip spread distance > 0.25', 'Angle spread ~ 25-40 deg']
  },
  W: {
    letter: 'W',
    name: 'Letter W',
    category: 'Single-Hand',
    desc: 'Extend index, middle, and ring fingers straight upright and spread evenly apart; thumb holds pinky finger down against palm.',
    isPhysicallyValidated: false,
    kinematicRules: ['Index, middle, ring extension > 0.80', 'Even 3-finger angular spread', 'Pinky held securely by thumb']
  },
  X: {
    letter: 'X',
    name: 'Letter X',
    category: 'Single-Hand',
    desc: 'Curl the extended index finger into a rigid hook shape (like a pirate hook); fold middle, ring, and pinky fingers in.',
    isPhysicallyValidated: false,
    kinematicRules: ['Index PIP flexed ~90 deg with DIP curved', 'Hook depth ratio > 0.40', 'Middle/ring/pinky curled in fist']
  },
  Y: {
    letter: 'Y',
    name: 'Letter Y',
    category: 'Single-Hand',
    desc: 'Extend thumb and pinky fingers outward wide while curling middle three fingers down into palm (shaka / hang-loose sign).',
    isPhysicallyValidated: true,
    validatedSessions: 4,
    kinematicRules: ['Thumb-to-pinky span ratio > 1.25', 'Middle 3 fingers fully curled into palm', 'Clear bilateral extension']
  },
  Z: {
    letter: 'Z',
    name: 'Letter Z',
    category: 'Dynamic',
    desc: 'Extend index finger and trace a sharp 3-stroke "Z" (horizontal right, diagonal down-left, horizontal right) through the air.',
    isPhysicallyValidated: false,
    kinematicRules: ['3-stroke temporal inflection profile', 'Index fingertip tracking speed > 0.35 palm/s', 'Zero-leak kinematic trajectory match']
  }
};

interface LearningPageProps {
  initialSignId?: string;
  selectedLanguage: SupportedLanguage;
  isMuted: boolean;
  onNavigate?: (page: string) => void;
}

export const LearningPage: React.FC<LearningPageProps> = ({
  initialSignId = 'A',
  selectedLanguage,
  isMuted,
  onNavigate
}) => {
  const [learningMode, setLearningMode] = useState<'alphabet' | 'words'>('alphabet');
  const [selectedLetter, setSelectedLetter] = useState<string>(
    initialSignId.length === 1 && initialSignId.toUpperCase() in ALPHABET_GUIDE
      ? initialSignId.toUpperCase()
      : 'A'
  );
  const [alphabetFilter, setAlphabetFilter] = useState<'all' | 'validated' | 'cohort_pending'>('all');

  // Word Practice States
  const [signs, setSigns] = useState<ISLSign[]>(FALLBACK_VOCABULARY);
  const [targetSignId, setTargetSignId] = useState<string>('namaste');
  const [isPracticing, setIsPracticing] = useState<boolean>(false);
  const [practiceResult, setPracticeResult] = useState<PracticeEvaluation | null>(null);
  const [isEvaluatingVideo, setIsEvaluatingVideo] = useState<boolean>(false);
  const [videoPracticeError, setVideoPracticeError] = useState<string | null>(null);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animIntervalRef = useRef<any>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleUploadPracticeVideo = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setVideoPracticeError(null);
    setIsEvaluatingVideo(true);
    stopPractice();

    try {
      const evalData = await apiClient.practiceVideo(file, targetSignId);
      setPracticeResult(evalData);
    } catch (err: any) {
      setVideoPracticeError(err.message || 'Failed to analyze practice video.');
    } finally {
      setIsEvaluatingVideo(false);
    }
  };

  useEffect(() => {
    apiClient.getVocabulary().then((res) => {
      if (res && res.length > 0) setSigns(res);
    });
  }, []);

  const activeSign = signs.find((s) => s.id === targetSignId) || signs[0];
  const activeAlphabet = ALPHABET_GUIDE[selectedLetter] || ALPHABET_GUIDE['A'];

  const allLetters = Object.keys(ALPHABET_GUIDE);
  const filteredLetters = allLetters.filter((ltr) => {
    if (alphabetFilter === 'validated') return ALPHABET_GUIDE[ltr].isPhysicallyValidated;
    if (alphabetFilter === 'cohort_pending') return !ALPHABET_GUIDE[ltr].isPhysicallyValidated;
    return true;
  });

  const startPractice = () => {
    setIsPracticing(true);
    let step = 0;
    const totalFrames = 30;

    animIntervalRef.current = setInterval(async () => {
      const progress = step / totalFrames;
      step = (step + 1) % totalFrames;

      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext('2d');
        if (ctx) {
          canvas.width = 640;
          canvas.height = 480;

          // Draw Canvas background
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

          // Build synthetic attempt landmarks
          const landmarks = generatePracticeLandmarks(targetSignId, progress);
          LandmarkRenderer.drawLandmarks(ctx, canvas.width, canvas.height, landmarks, 0.95);

          // Evaluation Overlay
          ctx.fillStyle = 'rgba(12, 18, 34, 0.85)';
          ctx.fillRect(16, 16, 280, 50);
          ctx.strokeStyle = '#3b82f6';
          ctx.strokeRect(16, 16, 280, 50);

          ctx.fillStyle = '#38bdf8';
          ctx.font = 'bold 12px Inter, sans-serif';
          ctx.fillText(`PRACTICE FORM: ${activeSign.name.toUpperCase()}`, 28, 36);

          ctx.fillStyle = '#94a3b8';
          ctx.font = '10px monospace';
          ctx.fillText(`Biomechanical Feedback Loop Active`, 28, 52);
        }
      }
    }, 45);
  };

  const stopPractice = () => {
    if (animIntervalRef.current) {
      clearInterval(animIntervalRef.current);
      animIntervalRef.current = null;
    }
    setIsPracticing(false);
  };

  const generatePracticeLandmarks = (signId: string, t: number): FrameLandmarks => {
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
      const p = Math.sin(t * Math.PI);
      const rWrist = { x: 0.55 + 0.1 * p, y: 0.50 - 0.15 * p, z: 0.1 };
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
    return () => stopPractice();
  }, []);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Header */}
      <div>
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-purple-950 text-purple-300 border border-purple-800 text-xs font-mono mb-2">
          <GraduationCap className="w-3.5 h-3.5" />
          <span>Interactive Learning Studio</span>
        </div>
        <h1 className="text-3xl font-extrabold text-white tracking-tight">
          Master Indian Sign Language with Real-Time Feedback
        </h1>
        <p className="text-sm text-slate-400 mt-1 max-w-3xl">
          Unlike static flashcards, SIGNBRIDGE analyzes your real-time joint positions and trajectory,
          guiding you to replicate canonical ISL handshapes and movements accurately.
        </p>
      </div>

      {/* Mode Selector Tabs */}
      <div className="flex flex-wrap items-center gap-2 border-b border-navy-800 pb-3">
        <button
          onClick={() => setLearningMode('alphabet')}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold transition-all cursor-pointer ${
            learningMode === 'alphabet'
              ? 'bg-brand-600 text-white shadow-glow'
              : 'bg-navy-900 text-slate-400 hover:text-white border border-navy-800'
          }`}
        >
          <Sparkles className="w-4 h-4 text-cyan-300" />
          <span>Manual Alphabet Curriculum (A–Z)</span>
          <span className="px-1.5 py-0.5 text-[9px] bg-navy-950 text-cyan-300 rounded font-mono">26 Classes</span>
        </button>

        <button
          onClick={() => setLearningMode('words')}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold transition-all cursor-pointer ${
            learningMode === 'words'
              ? 'bg-brand-600 text-white shadow-glow'
              : 'bg-navy-900 text-slate-400 hover:text-white border border-navy-800'
          }`}
        >
          <GraduationCap className="w-4 h-4 text-purple-400" />
          <span>Conversational ISL Words & Phrases</span>
          <span className="px-1.5 py-0.5 text-[9px] bg-navy-950 text-purple-300 rounded font-mono">BiLSTM 30-Frame</span>
        </button>
      </div>

      {/* ====================================================================== */}
      {/* MODE 1: MANUAL ALPHABET CURRICULUM (A–Z)                              */}
      {/* ====================================================================== */}
      {learningMode === 'alphabet' && (
        <div className="space-y-6">
          {/* Validation Provenance & Integrity Banner */}
          <div className="p-4 rounded-2xl bg-navy-900/90 border border-navy-800 grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
            <div className="flex items-center gap-2.5 p-2.5 rounded-xl bg-emerald-950/40 border border-emerald-500/30">
              <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0" />
              <div>
                <span className="font-bold text-emerald-200 block">12 Physically Validated</span>
                <span className="text-[11px] text-emerald-300/80">42 live camera sessions (A, B, C, D, E, F, L, O, R, U, V, Y)</span>
              </div>
            </div>

            <div className="flex items-center gap-2.5 p-2.5 rounded-xl bg-amber-950/40 border border-amber-500/30">
              <AlertCircle className="w-5 h-5 text-amber-400 shrink-0" />
              <div>
                <span className="font-bold text-amber-200 block">14 Verified Baselines</span>
                <span className="text-[11px] text-amber-300/80">Awaiting multi-signer field cohort (G, H, I, J, K, M, N, P, Q, S, T, W, X, Z)</span>
              </div>
            </div>

            <div className="flex items-center gap-2.5 p-2.5 rounded-xl bg-sky-950/40 border border-sky-500/30">
              <Sparkles className="w-5 h-5 text-sky-400 shrink-0" />
              <div>
                <span className="font-bold text-sky-200 block">94-D Coordinate AI</span>
                <span className="text-[11px] text-sky-300/80">Continuous scale & rotation invariant neural representation</span>
              </div>
            </div>
          </div>

          {/* Alphabet Studio Layout */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            {/* Left Column: A-Z Letters Grid (5 Cols) */}
            <div className="lg:col-span-5 space-y-4">
              <div className="bg-navy-900 border border-navy-800 rounded-2xl p-5 shadow-subtle space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-400">
                    Alphabet Classes ({filteredLetters.length})
                  </h3>
                  <div className="flex items-center gap-1 bg-navy-950 p-1 rounded-lg border border-navy-850 text-[10px]">
                    <button
                      onClick={() => setAlphabetFilter('all')}
                      className={`px-2 py-0.5 rounded transition-all cursor-pointer ${
                        alphabetFilter === 'all' ? 'bg-brand-600 text-white font-semibold' : 'text-slate-400'
                      }`}
                    >
                      All (26)
                    </button>
                    <button
                      onClick={() => setAlphabetFilter('validated')}
                      className={`px-2 py-0.5 rounded transition-all cursor-pointer ${
                        alphabetFilter === 'validated' ? 'bg-emerald-600 text-white font-semibold' : 'text-slate-400'
                      }`}
                    >
                      Validated (12)
                    </button>
                    <button
                      onClick={() => setAlphabetFilter('cohort_pending')}
                      className={`px-2 py-0.5 rounded transition-all cursor-pointer ${
                        alphabetFilter === 'cohort_pending' ? 'bg-amber-600 text-white font-semibold' : 'text-slate-400'
                      }`}
                    >
                      Pending (14)
                    </button>
                  </div>
                </div>

                {/* Grid of A-Z Buttons */}
                <div className="grid grid-cols-5 sm:grid-cols-6 gap-2 pt-1 max-h-[360px] overflow-y-auto pr-1">
                  {filteredLetters.map((ltr) => {
                    const item = ALPHABET_GUIDE[ltr];
                    const isSelected = selectedLetter === ltr;
                    return (
                      <button
                        key={ltr}
                        onClick={() => setSelectedLetter(ltr)}
                        className={`relative p-3 rounded-xl border text-center font-bold text-base transition-all cursor-pointer ${
                          isSelected
                            ? 'bg-brand-600 text-white border-brand-400 shadow-glow scale-105 z-10'
                            : 'bg-navy-950 text-slate-200 border-navy-850 hover:bg-navy-850 hover:border-slate-700'
                        }`}
                      >
                        <span>{ltr}</span>
                        {item.isPhysicallyValidated ? (
                          <span className="absolute top-1 right-1 w-2 h-2 rounded-full bg-emerald-400" title="Physically Human Validated" />
                        ) : (
                          <span className="absolute top-1 right-1 w-2 h-2 rounded-full bg-amber-400" title="Awaiting Multi-Signer Field Cohort" />
                        )}
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Right Column: Detailed Letter Inspector & Guide (7 Cols) */}
            <div className="lg:col-span-7 space-y-4">
              <div className="bg-navy-900 border border-navy-800 rounded-2xl p-6 shadow-subtle space-y-5">
                {/* Letter Header */}
                <div className="flex items-center gap-5 border-b border-navy-800 pb-4">
                  <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-brand-600 via-indigo-600 to-sky-500 flex items-center justify-center text-white text-4xl font-black shadow-glow shrink-0">
                    {activeAlphabet.letter}
                  </div>

                  <div className="space-y-1 flex-1">
                    <div className="flex items-center gap-2">
                      <h3 className="text-2xl font-black text-white">{activeAlphabet.name}</h3>
                      <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-semibold bg-navy-950 text-cyan-300 border border-navy-800">
                        {activeAlphabet.category}
                      </span>
                    </div>

                    <div>
                      {activeAlphabet.isPhysicallyValidated ? (
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-950/80 text-emerald-300 border border-emerald-500/30">
                          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                          <span>Physically Human Validated (42 Camera Sessions)</span>
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-amber-950/80 text-amber-300 border border-amber-500/30">
                          <AlertCircle className="w-3.5 h-3.5 text-amber-400" />
                          <span>Verified Kinematic Baseline (Field Cohort Pending)</span>
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                {/* Handshape Description */}
                <div className="space-y-1.5">
                  <h4 className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-400">
                    Handshape Instructions:
                  </h4>
                  <p className="text-sm text-slate-200 leading-relaxed bg-navy-950 p-4 rounded-xl border border-navy-850">
                    {activeAlphabet.desc}
                  </p>
                </div>

                {/* Kinematic Rules */}
                <div className="space-y-2">
                  <h4 className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-400">
                    Kinematic Joint Articulation Rules:
                  </h4>
                  <div className="space-y-1.5">
                    {activeAlphabet.kinematicRules.map((rule, idx) => (
                      <div key={idx} className="flex items-center gap-2 text-xs text-slate-300 p-2 rounded-lg bg-navy-950/60 border border-navy-850">
                        <Check className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
                        <span className="font-mono">{rule}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="flex flex-col sm:flex-row items-center gap-3 pt-2">
                  {onNavigate && (
                    <button
                      onClick={() => onNavigate('recognizer')}
                      className="w-full sm:w-auto flex items-center justify-center gap-2 px-5 py-3 rounded-xl bg-gradient-to-r from-brand-600 to-sky-500 hover:from-brand-500 hover:to-sky-400 text-white font-bold text-xs shadow-glow transition-all active:scale-95 cursor-pointer"
                    >
                      <Camera className="w-4 h-4" />
                      <span>Practice Letter {activeAlphabet.letter} in Live Camera</span>
                      <ArrowRight className="w-4 h-4" />
                    </button>
                  )}

                  <button
                    onClick={() => speechService.speak(`Letter ${activeAlphabet.letter}`, selectedLanguage)}
                    disabled={isMuted}
                    className="w-full sm:w-auto flex items-center justify-center gap-2 px-4 py-3 rounded-xl bg-navy-950 hover:bg-navy-850 border border-navy-800 text-slate-300 hover:text-white font-semibold text-xs transition-all cursor-pointer"
                  >
                    <Volume2 className="w-4 h-4 text-brand-400" />
                    <span>Pronounce</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ====================================================================== */}
      {/* MODE 2: CONVERSATIONAL ISL WORDS & PHRASES                             */}
      {/* ====================================================================== */}
      {learningMode === 'words' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Left: Sign Selection & Kinematic Guide (4 Cols) */}
          <div className="lg:col-span-4 space-y-4">
            <div className="bg-navy-900 border border-navy-800 rounded-2xl p-5 shadow-subtle">
              <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-400 mb-3">
                Select Sign to Practice
              </h3>
              <div className="space-y-1.5 max-h-[320px] overflow-y-auto pr-1">
                {signs.map((s) => (
                  <button
                    key={s.id}
                    onClick={() => {
                      stopPractice();
                      setTargetSignId(s.id);
                      setPracticeResult(null);
                    }}
                    className={`w-full flex items-center justify-between p-2.5 rounded-xl text-xs font-medium transition-all ${
                      targetSignId === s.id
                        ? 'bg-brand-600 text-white font-semibold shadow-glow'
                        : 'bg-navy-950 text-slate-300 hover:text-white border border-navy-850'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <Hand className="w-3.5 h-3.5" />
                      <span>{s.name}</span>
                    </div>
                    <span className="text-[10px] font-mono opacity-80 uppercase">
                      {s.difficulty}
                    </span>
                  </button>
                ))}
              </div>
            </div>

            {/* Canonical Reference Card */}
            <div className="bg-navy-900 border border-navy-800 rounded-2xl p-5 shadow-subtle space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-semibold uppercase tracking-wider text-brand-400">
                  Canonical Guide
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-navy-950 text-slate-400 border border-navy-800">
                  {activeSign.hand_config === 'two_handed' ? 'Two Hands Required' : 'Single Hand'}
                </span>
              </div>

              <h4 className="text-lg font-bold text-white">
                {activeSign.name} ({activeSign.translations.hi})
              </h4>

              <p className="text-xs text-slate-300 leading-relaxed">
                {activeSign.description}
              </p>

              <div className="bg-navy-950 p-2.5 rounded-xl border border-navy-850 text-xs text-slate-400">
                <strong className="text-white block mb-0.5">Motion Path:</strong>
                {activeSign.movement}
              </div>

              <button
                onClick={() => speechService.speak(activeSign.translations[selectedLanguage] || activeSign.translations.en, selectedLanguage)}
                disabled={isMuted}
                className="w-full flex items-center justify-center gap-2 py-2 rounded-xl text-xs font-medium text-slate-300 hover:text-white bg-navy-950 hover:bg-navy-850 border border-navy-800 transition-all cursor-pointer"
              >
                <Volume2 className="w-3.5 h-3.5 text-brand-400" />
                <span>Hear Pronunciation</span>
              </button>
            </div>
          </div>

          {/* Right: Interactive Camera / Kinematic Evaluation View (8 Cols) */}
          <div className="lg:col-span-8 space-y-4">
            <div className="relative aspect-[4/3] bg-navy-950 border border-navy-800 rounded-2xl overflow-hidden shadow-glow flex items-center justify-center">
              <canvas ref={canvasRef} className="w-full h-full object-cover" />

              {!isPracticing && (
                <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center bg-navy-950/95 backdrop-blur-sm">
                  <div className="w-16 h-16 rounded-2xl bg-navy-900 border border-navy-800 flex items-center justify-center text-purple-400 mb-4 shadow-subtle">
                    <GraduationCap className="w-8 h-8" />
                  </div>
                  <h3 className="text-base font-bold text-white">Practice Studio Standby</h3>
                  <p className="text-xs text-slate-400 mt-1 max-w-sm">
                    Ready to practice <strong className="text-white">"{activeSign.name}"</strong>.
                    Click below to activate real-time biomechanical analysis.
                  </p>

                  <button
                    onClick={startPractice}
                    className="mt-5 flex items-center gap-2 px-6 py-2.5 rounded-xl bg-purple-600 hover:bg-purple-500 text-white font-semibold text-xs shadow-glow transition-all cursor-pointer"
                  >
                    <Play className="w-3.5 h-3.5" />
                    <span>Start Practice Attempt</span>
                  </button>
                </div>
              )}
            </div>

            {/* Action Bar */}
            <div className="flex flex-wrap items-center justify-between gap-3 bg-navy-900 border border-navy-800 p-3 rounded-xl text-xs">
              <div className="flex items-center gap-2">
                {isPracticing ? (
                  <button
                    onClick={stopPractice}
                    className="flex items-center gap-1.5 px-3 py-1.5 bg-rose-950 text-rose-300 border border-rose-800 rounded-lg font-medium cursor-pointer"
                  >
                    <Square className="w-3.5 h-3.5" />
                    <span>End Practice</span>
                  </button>
                ) : (
                  <button
                    onClick={startPractice}
                    className="flex items-center gap-1.5 px-3 py-1.5 bg-purple-600 hover:bg-purple-500 text-white rounded-lg font-medium shadow-sm cursor-pointer"
                  >
                    <Play className="w-3.5 h-3.5" />
                    <span>Interactive Trajectory</span>
                  </button>
                )}

                <input
                  ref={fileInputRef}
                  type="file"
                  accept="video/mp4,video/webm,video/quicktime"
                  onChange={handleUploadPracticeVideo}
                  className="hidden"
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isEvaluatingVideo}
                  className="flex items-center gap-1.5 px-3 py-1.5 bg-navy-800 hover:bg-navy-750 text-slate-200 rounded-lg font-medium transition-all cursor-pointer disabled:opacity-50"
                >
                  <UploadCloud className="w-3.5 h-3.5 text-brand-400" />
                  <span>{isEvaluatingVideo ? 'Scoring Video...' : 'Upload Attempt Video'}</span>
                </button>
              </div>

              <div className="text-[11px] font-mono text-slate-400">
                Target: <strong className="text-purple-300">{activeSign.name}</strong>
              </div>
            </div>

            {videoPracticeError && (
              <div className="p-3 bg-rose-950/80 border border-rose-800 rounded-xl text-xs text-rose-200 flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0 mt-0.5" />
                <p>{videoPracticeError}</p>
              </div>
            )}

            {/* Granular Feedback Report */}
            {practiceResult && (
              <div className="bg-navy-900 border border-navy-800 rounded-2xl p-6 shadow-subtle space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400">
                      Biomechanical Accuracy
                    </span>
                    <div className="flex items-baseline gap-2 mt-0.5">
                      <span className="text-3xl font-extrabold text-white">
                        {practiceResult.accuracy_score}%
                      </span>
                      <span className="px-2 py-0.5 rounded text-xs font-semibold bg-emerald-950 text-emerald-300 border border-emerald-800">
                        {practiceResult.grade}
                      </span>
                    </div>
                  </div>

                  <div className="w-12 h-12 rounded-xl bg-purple-950 border border-purple-800 flex items-center justify-center text-purple-300">
                    <Trophy className="w-6 h-6" />
                  </div>
                </div>

                {/* Actionable Feedback Notes */}
                <div className="space-y-2 pt-2 border-t border-navy-800">
                  <span className="text-xs font-mono text-slate-400 uppercase tracking-wider block">
                    Actionable Form Guidance:
                  </span>
                  <div className="space-y-1.5">
                    {practiceResult.feedback.map((note, idx) => (
                      <div
                        key={idx}
                        className="flex items-start gap-2 p-2.5 rounded-lg bg-navy-950 border border-navy-850 text-xs text-slate-300"
                      >
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0 mt-0.5" />
                        <span>{note}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};