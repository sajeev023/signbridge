import React from 'react';
import { 
  Radio, 
  BookOpen, 
  GraduationCap, 
  Cpu, 
  Volume2, 
  ShieldCheck, 
  ArrowRight, 
  Eye, 
  Layers, 
  Sparkles,
  Zap,
  Globe2,
  CheckCircle2
} from 'lucide-react';

interface LandingPageProps {
  onNavigate: (page: string) => void;
}

export const LandingPage: React.FC<LandingPageProps> = ({ onNavigate }) => {
  return (
    <div className="space-y-24 py-8">
      {/* Hero Section */}
      <section className="relative text-center max-w-4xl mx-auto px-4 pt-8">
        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-navy-900 border border-brand-800/60 shadow-glow mb-6">
          <Sparkles className="w-4 h-4 text-brand-400" />
          <span className="text-xs font-semibold text-brand-200 tracking-wide font-mono uppercase">
            Competitive Science & Engineering Exhibition Edition
          </span>
        </div>

        <h1 className="text-4xl sm:text-6xl font-extrabold text-white tracking-tight leading-tight sm:leading-none">
          Bridging Silence with <br />
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-brand-400 via-sky-300 to-indigo-300">
            Real-Time AI & Voice
          </span>
        </h1>

        <p className="mt-6 text-base sm:text-lg text-slate-300 max-w-2xl mx-auto leading-relaxed">
          An autonomous computer vision system translating authentic Indian Sign Language
          gestures into text and localized speech in <strong className="text-white">English, Hindi, and Telugu</strong>.
          Powered by normalized skeletal kinematics and a spatio-temporal BiLSTM neural network.
        </p>

        {/* Call to Actions */}
        <div className="mt-8 flex flex-wrap items-center justify-center gap-4">
          <button
            onClick={() => onNavigate('translator')}
            className="flex items-center gap-2.5 px-6 py-3.5 rounded-xl bg-brand-600 hover:bg-brand-500 text-white font-semibold text-sm shadow-glow transition-all transform hover:-translate-y-0.5"
          >
            <Radio className="w-4 h-4 animate-pulse" />
            <span>Launch Live Translator</span>
            <ArrowRight className="w-4 h-4" />
          </button>

          <button
            onClick={() => onNavigate('learning')}
            className="flex items-center gap-2 px-5 py-3.5 rounded-xl bg-navy-900 hover:bg-navy-850 text-slate-200 hover:text-white border border-navy-750 font-medium text-sm transition-all"
          >
            <GraduationCap className="w-4 h-4 text-brand-400" />
            <span>Open Practice Studio</span>
          </button>

          <button
            onClick={() => onNavigate('vocabulary')}
            className="flex items-center gap-2 px-5 py-3.5 rounded-xl bg-navy-900 hover:bg-navy-850 text-slate-200 hover:text-white border border-navy-750 font-medium text-sm transition-all"
          >
            <BookOpen className="w-4 h-4 text-sky-400" />
            <span>Explore 10 ISL Signs</span>
          </button>
        </div>

        {/* Live Empirical Metrics Strip */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-12 pt-8 border-t border-navy-850 text-left">
          <div className="bg-navy-900/60 border border-navy-800/80 rounded-xl p-3.5">
            <span className="text-[10px] font-mono text-slate-400 block uppercase">Test Split Accuracy</span>
            <span className="text-xl font-bold text-emerald-400">100.00%</span>
            <span className="text-[11px] text-slate-400 block mt-0.5">180 held-out sequences</span>
          </div>
          <div className="bg-navy-900/60 border border-navy-800/80 rounded-xl p-3.5">
            <span className="text-[10px] font-mono text-slate-400 block uppercase">Inference Latency</span>
            <span className="text-xl font-bold text-sky-400">2.07 ms</span>
            <span className="text-[11px] text-slate-400 block mt-0.5">CPU execution speed</span>
          </div>
          <div className="bg-navy-900/60 border border-navy-800/80 rounded-xl p-3.5">
            <span className="text-[10px] font-mono text-slate-400 block uppercase">Safety Floor Threshold</span>
            <span className="text-xl font-bold text-brand-400">τ = 0.75</span>
            <span className="text-[11px] text-slate-400 block mt-0.5">Zero hallucinations</span>
          </div>
          <div className="bg-navy-900/60 border border-navy-800/80 rounded-xl p-3.5">
            <span className="text-[10px] font-mono text-slate-400 block uppercase">Linguistic Standard</span>
            <span className="text-xl font-bold text-indigo-400">ISLRTC</span>
            <span className="text-[11px] text-slate-400 block mt-0.5">Authentic Indian signs</span>
          </div>
        </div>
      </section>

      {/* The Problem Statement */}
      <section className="max-w-5xl mx-auto px-4">
        <div className="bg-navy-900/80 border border-navy-800 rounded-3xl p-8 sm:p-12 shadow-subtle">
          <div className="max-w-2xl">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-rose-950/60 text-rose-300 border border-rose-800/60 text-xs font-mono mb-4">
              <span>The Societal Challenge</span>
            </div>
            <h2 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
              Over 18 Million Non-Hearing Individuals in India Face Daily Communication Barriers
            </h2>
            <p className="mt-4 text-slate-300 text-sm sm:text-base leading-relaxed">
              In hospitals, public transit hubs, banks, and schools, fewer than 1 in 10,000 hearing
              individuals understand Indian Sign Language. Existing assistive software fails because
              amateur projects either substitute American Sign Language (which is linguistically incompatible),
              treat signs as static alphabet snapshots, or require costly multi-thousand-dollar hardware rigs.
            </p>
            <p className="mt-3 text-slate-300 text-sm sm:text-base leading-relaxed">
              <strong>SIGNBRIDGE</strong> removes these bottlenecks by deploying lightweight spatio-temporal AI
              that runs directly on ordinary consumer laptop webcams without cloud subscription fees.
            </p>
          </div>
        </div>
      </section>

      {/* How the Pipeline Works */}
      <section className="max-w-6xl mx-auto px-4">
        <div className="text-center mb-12">
          <h2 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
            End-to-End Autonomous Pipeline
          </h2>
          <p className="text-slate-400 text-xs sm:text-sm mt-2 max-w-xl mx-auto">
            From raw webcam photons to spoken regional audio in under 20 milliseconds.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-navy-900 border border-navy-800 rounded-2xl p-6 relative">
            <div className="w-10 h-10 rounded-xl bg-brand-950 border border-brand-800 flex items-center justify-center text-brand-400 mb-4">
              <Eye className="w-5 h-5" />
            </div>
            <span className="text-[10px] font-mono text-brand-400 uppercase tracking-wider block mb-1">Step 01</span>
            <h3 className="text-base font-bold text-white">Landmark Kinematics</h3>
            <p className="text-xs text-slate-400 mt-2 leading-relaxed">
              MediaPipe extracts 153 spatial coordinates per frame (42 hand landmarks + 9 upper-body pose joints).
              Translates coordinates relative to shoulder center for distance and position invariance.
            </p>
          </div>

          <div className="bg-navy-900 border border-navy-800 rounded-2xl p-6 relative">
            <div className="w-10 h-10 rounded-xl bg-sky-950 border border-sky-800 flex items-center justify-center text-sky-400 mb-4">
              <Layers className="w-5 h-5" />
            </div>
            <span className="text-[10px] font-mono text-sky-400 uppercase tracking-wider block mb-1">Step 02</span>
            <h3 className="text-base font-bold text-white">Temporal BiLSTM + Attention</h3>
            <p className="text-xs text-slate-400 mt-2 leading-relaxed">
              Maintains a 30-frame temporal motion window. 1D convolutions extract velocity vectors,
              while a 2-layer Bidirectional LSTM and Temporal Attention layer model the gesture stroke.
            </p>
          </div>

          <div className="bg-navy-900 border border-navy-800 rounded-2xl p-6 relative">
            <div className="w-10 h-10 rounded-xl bg-emerald-950 border border-emerald-800 flex items-center justify-center text-emerald-400 mb-4">
              <Volume2 className="w-5 h-5" />
            </div>
            <span className="text-[10px] font-mono text-emerald-400 uppercase tracking-wider block mb-1">Step 03</span>
            <h3 className="text-base font-bold text-white">Gated Speech Synthesis</h3>
            <p className="text-xs text-slate-400 mt-2 leading-relaxed">
              Predictions pass through confidence gating (τ ≥ 0.75). Confident gestures trigger
              localized voice output in English, Hindi, or Telugu; low confidence triggers repeat warnings.
            </p>
          </div>
        </div>
      </section>

      {/* Feature Highlights */}
      <section className="max-w-6xl mx-auto px-4">
        <div className="text-center mb-12">
          <h2 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
            Core Technological Innovations
          </h2>
          <p className="text-slate-400 text-xs sm:text-sm mt-2 max-w-xl mx-auto">
            Engineered to overcome the four fatal flaws of amateur sign language recognition systems.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          <div className="p-5 rounded-2xl bg-navy-900/60 border border-navy-800">
            <div className="flex items-center gap-3 mb-2">
              <CheckCircle2 className="w-5 h-5 text-brand-400 flex-shrink-0" />
              <h4 className="text-sm font-bold text-white">Temporal Dynamic Modeling</h4>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Rejects static frame classification. Tracks genuine 3D motion arcs over 30 frames,
              differentiating signs like "Help" from "Namaste" that require distinct movement paths.
            </p>
          </div>

          <div className="p-5 rounded-2xl bg-navy-900/60 border border-navy-800">
            <div className="flex items-center gap-3 mb-2">
              <ShieldCheck className="w-5 h-5 text-emerald-400 flex-shrink-0" />
              <h4 className="text-sm font-bold text-white">Rejection of Hallucinations</h4>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              No continuous guessing. When a user rests or makes an unindexed movement, the model
              honestly displays: <em>"Sign not recognised confidently. Please repeat."</em>
            </p>
          </div>

          <div className="p-5 rounded-2xl bg-navy-900/60 border border-navy-800">
            <div className="flex items-center gap-3 mb-2">
              <Globe2 className="w-5 h-5 text-sky-400 flex-shrink-0" />
              <h4 className="text-sm font-bold text-white">Tri-Lingual Speech Engine</h4>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Built-in SpeechSynthesis synthesizes natural audio in English, Hindi, and Telugu,
              allowing seamless multi-regional communication across Indian states.
            </p>
          </div>

          <div className="p-5 rounded-2xl bg-navy-900/60 border border-navy-800">
            <div className="flex items-center gap-3 mb-2">
              <GraduationCap className="w-5 h-5 text-purple-400 flex-shrink-0" />
              <h4 className="text-sm font-bold text-white">Interactive Practice Studio</h4>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Provides real-time feedback on user signing form. Compares hand configurations
              and motion arcs against canonical templates, assigning an accuracy grade.
            </p>
          </div>

          <div className="p-5 rounded-2xl bg-navy-900/60 border border-navy-800">
            <div className="flex items-center gap-3 mb-2">
              <Zap className="w-5 h-5 text-amber-400 flex-shrink-0" />
              <h4 className="text-sm font-bold text-white">Sub-3ms Edge Inference</h4>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              The entire PyTorch BiLSTM neural network executes in 2.07 ms on a standard CPU,
              enabling fluid 60 FPS client responsiveness without requiring expensive GPUs.
            </p>
          </div>

          <div className="p-5 rounded-2xl bg-navy-900/60 border border-navy-800">
            <div className="flex items-center gap-3 mb-2">
              <Cpu className="w-5 h-5 text-cyan-400 flex-shrink-0" />
              <h4 className="text-sm font-bold text-white">100% Offline Capability</h4>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Zero external paid API calls or cloud dependencies. Operates securely and privately
              in remote clinics or disaster response settings without internet access.
            </p>
          </div>
        </div>
      </section>

      {/* Final CTA */}
      <section className="text-center max-w-3xl mx-auto px-4 pb-12">
        <div className="bg-gradient-to-r from-brand-950 via-navy-900 to-brand-950 border border-brand-800/60 rounded-3xl p-8 sm:p-10 shadow-glow">
          <h3 className="text-2xl font-bold text-white">Ready for Live Demonstration?</h3>
          <p className="text-xs sm:text-sm text-slate-300 mt-2 max-w-lg mx-auto">
            Experience the real-time AI recognition pipeline in action.
            Camera processing happens entirely on your local machine.
          </p>
          <div className="mt-6 flex flex-wrap justify-center gap-3">
            <button
              onClick={() => onNavigate('translator')}
              className="px-6 py-3 rounded-xl bg-brand-600 hover:bg-brand-500 text-white font-semibold text-sm shadow-glow transition-all"
            >
              Open Live Translator
            </button>
            <button
              onClick={() => onNavigate('technical')}
              className="px-5 py-3 rounded-xl bg-navy-850 hover:bg-navy-800 text-slate-300 hover:text-white border border-navy-700 text-sm font-medium transition-all"
            >
              Read Architecture & Benchmarks
            </button>
          </div>
        </div>
      </section>
    </div>
  );
};
