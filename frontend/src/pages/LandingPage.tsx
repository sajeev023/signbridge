import React from 'react';
import { 
  Camera, 
  BookOpen, 
  GraduationCap, 
  Volume2, 
  ShieldCheck, 
  ArrowRight, 
  Sparkles,
  Zap,
  Globe2,
  CheckCircle2,
  Layers,
  Hand
} from 'lucide-react';
import { FALLBACK_VOCABULARY } from '../services/apiClient';

interface LandingPageProps {
  onNavigate: (page: string) => void;
}

export const LandingPage: React.FC<LandingPageProps> = ({ onNavigate }) => {
  return (
    <div className="space-y-12 sm:space-y-20 py-4 sm:py-8 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
      {/* Hero Section */}
      <section className="text-center max-w-3xl mx-auto pt-2 sm:pt-6 space-y-4 sm:space-y-6">
        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-navy-900 border border-brand-800/80 shadow-glow">
          <Sparkles className="w-3.5 h-3.5 text-brand-400 animate-pulse" />
          <span className="text-xs font-bold text-brand-200 tracking-wide font-mono uppercase">
            Indian Sign Language & Manual Alphabet AI Platform
          </span>
        </div>

        <h1 className="text-3xl sm:text-5xl lg:text-6xl font-black text-white tracking-tight leading-tight sm:leading-none">
          Translate Sign Language to <br className="hidden sm:block" />
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-brand-400 via-sky-300 to-indigo-300">
            Voice & Text in Real Time
          </span>
        </h1>

        <p className="text-sm sm:text-base text-slate-300 max-w-xl mx-auto leading-relaxed">
          Break communication barriers with real-time computer vision AI.
          Translates 26-letter manual alphabet signs and 10 core ISL gestures into spoken <strong className="text-white">English, Hindi, and Telugu</strong> directly in your browser.
        </p>

        {/* Primary Action Buttons */}
        <div className="pt-2 flex flex-col sm:flex-row items-center justify-center gap-3">
          <button
            onClick={() => onNavigate('recognizer')}
            className="w-full sm:w-auto flex items-center justify-center gap-2 px-6 py-3.5 rounded-2xl bg-gradient-to-r from-brand-600 via-indigo-600 to-sky-500 hover:from-brand-500 hover:to-sky-400 text-white font-bold text-sm shadow-glow transition-all transform active:scale-95"
          >
            <Sparkles className="w-4 h-4 text-cyan-300" />
            <span>Launch A-Z Alphabet AI</span>
            <ArrowRight className="w-4 h-4" />
          </button>

          <button
            onClick={() => onNavigate('translator')}
            className="w-full sm:w-auto flex items-center justify-center gap-2 px-5 py-3.5 rounded-2xl bg-navy-900 hover:bg-navy-850 text-slate-200 hover:text-white border border-navy-750 font-semibold text-sm transition-all"
          >
            <Camera className="w-4 h-4 text-sky-400" />
            <span>Word Translator (10 ISL Signs)</span>
          </button>

          <button
            onClick={() => onNavigate('learning')}
            className="w-full sm:w-auto flex items-center justify-center gap-2 px-5 py-3.5 rounded-2xl bg-navy-900 hover:bg-navy-850 text-slate-200 hover:text-white border border-navy-750 font-semibold text-sm transition-all"
          >
            <GraduationCap className="w-4 h-4 text-brand-400" />
            <span>Practice Studio</span>
          </button>
        </div>

        {/* KPI Strip */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 pt-4 text-left">
          <div className="bg-navy-900/80 border border-navy-800 rounded-2xl p-3">
            <span className="text-[10px] font-mono text-slate-400 uppercase block">Vision Latency</span>
            <span className="text-lg font-bold text-sky-400">1.78 ms</span>
            <span className="text-[10px] text-slate-400 block">Sub-2ms CPU inference</span>
          </div>
          <div className="bg-navy-900/80 border border-navy-800 rounded-2xl p-3">
            <span className="text-[10px] font-mono text-slate-400 uppercase block">Spoken Languages</span>
            <span className="text-lg font-bold text-emerald-400">3 Regional</span>
            <span className="text-[10px] text-slate-400 block">English, Hindi, Telugu</span>
          </div>
          <div className="bg-navy-900/80 border border-navy-800 rounded-2xl p-3">
            <span className="text-[10px] font-mono text-slate-400 uppercase block">Safety Threshold</span>
            <span className="text-lg font-bold text-brand-400">τ = 0.75</span>
            <span className="text-[10px] text-slate-400 block">Calibrated confidence</span>
          </div>
          <div className="bg-navy-900/80 border border-navy-800 rounded-2xl p-3">
            <span className="text-[10px] font-mono text-slate-400 uppercase block">Linguistic Scope</span>
            <span className="text-lg font-bold text-indigo-400">26 A-Z + 10 ISL</span>
            <span className="text-[10px] text-slate-400 block">Single-hand & ISLRTC signs</span>
          </div>
        </div>
      </section>

      {/* 3 Modes Overview */}
      <section className="space-y-6">
        <div className="text-center space-y-1">
          <h2 className="text-xl sm:text-2xl font-black text-white tracking-tight">
            Designed for Every Device & Context
          </h2>
          <p className="text-xs sm:text-sm text-slate-400">
            Works on phones, laptops, and exhibition booths with zero install.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div 
            onClick={() => onNavigate('recognizer')}
            className="p-5 rounded-2xl bg-navy-900/90 border border-navy-800 hover:border-cyan-500/50 transition-all cursor-pointer group shadow-subtle flex flex-col justify-between"
          >
            <div>
              <div className="w-10 h-10 rounded-xl bg-cyan-600/20 border border-cyan-500/40 flex items-center justify-center text-cyan-400 mb-3 group-hover:scale-105 transition-transform">
                <Sparkles className="w-5 h-5" />
              </div>
              <h3 className="text-base font-bold text-white group-hover:text-cyan-300 transition-colors">
                A–Z Alphabet AI
              </h3>
              <p className="text-xs text-slate-400 mt-1.5 leading-relaxed">
                94-dimensional Coordinate AI neural engine recognizing all 26 manual alphabet handshapes (Single-Hand ASL / International Standard) and dynamic J/Z gestures at sub-2ms latency, with experimental ISLRTC bimanual mode.
              </p>
            </div>
            <span className="inline-flex items-center gap-1 text-[11px] font-bold text-cyan-400 mt-4">
              <span>Launch Alphabet AI</span>
              <ArrowRight className="w-3 h-3" />
            </span>
          </div>

          <div 
            onClick={() => onNavigate('translator')}
            className="p-5 rounded-2xl bg-navy-900/90 border border-navy-800 hover:border-brand-500/50 transition-all cursor-pointer group shadow-subtle flex flex-col justify-between"
          >
            <div>
              <div className="w-10 h-10 rounded-xl bg-brand-600/20 border border-brand-500/40 flex items-center justify-center text-brand-400 mb-3 group-hover:scale-105 transition-transform">
                <Camera className="w-5 h-5" />
              </div>
              <h3 className="text-base font-bold text-white group-hover:text-brand-300 transition-colors">
                Core ISL Word Translator
              </h3>
              <p className="text-xs text-slate-400 mt-1.5 leading-relaxed">
                Live camera tracking for 10 canonical Indian Sign Language vocabulary signs with instant tri-lingual speech synthesis. (Note: Continuous open-domain sentence translation is experimental).
              </p>
            </div>
            <span className="inline-flex items-center gap-1 text-[11px] font-bold text-brand-400 mt-4">
              <span>Start Camera (10 Signs)</span>
              <ArrowRight className="w-3 h-3" />
            </span>
          </div>

          <div 
            onClick={() => onNavigate('learning')}
            className="p-5 rounded-2xl bg-navy-900/90 border border-navy-800 hover:border-purple-500/50 transition-all cursor-pointer group shadow-subtle flex flex-col justify-between"
          >
            <div>
              <div className="w-10 h-10 rounded-xl bg-purple-600/20 border border-purple-500/40 flex items-center justify-center text-purple-400 mb-3 group-hover:scale-105 transition-transform">
                <GraduationCap className="w-5 h-5" />
              </div>
              <h3 className="text-base font-bold text-white group-hover:text-purple-300 transition-colors">
                Practice Studio
              </h3>
              <p className="text-xs text-slate-400 mt-1.5 leading-relaxed">
                Learn manual alphabet and word signs with real-time biomechanical feedback and verified joint angle guides.
              </p>
            </div>
            <span className="inline-flex items-center gap-1 text-[11px] font-bold text-purple-400 mt-4">
              <span>Open Studio</span>
              <ArrowRight className="w-3 h-3" />
            </span>
          </div>

          <div 
            onClick={() => onNavigate('vocabulary')}
            className="p-5 rounded-2xl bg-navy-900/90 border border-navy-800 hover:border-emerald-500/50 transition-all cursor-pointer group shadow-subtle flex flex-col justify-between"
          >
            <div>
              <div className="w-10 h-10 rounded-xl bg-emerald-600/20 border border-emerald-500/40 flex items-center justify-center text-emerald-400 mb-3 group-hover:scale-105 transition-transform">
                <Sparkles className="w-5 h-5" />
              </div>
              <h3 className="text-base font-bold text-white group-hover:text-emerald-300 transition-colors">
                ISL Vocabulary
              </h3>
              <p className="text-xs text-slate-400 mt-1.5 leading-relaxed">
                Explore standardized signs cataloged in strict alignment with ISLRTC curriculum specifications.
              </p>
            </div>
            <span className="inline-flex items-center gap-1 text-[11px] font-bold text-emerald-400 mt-4">
              <span>Browse Vocabulary</span>
              <ArrowRight className="w-3 h-3" />
            </span>
          </div>
        </div>
      </section>

      {/* Vocabulary Showcase (10 ISL Signs Grid) */}
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl sm:text-2xl font-black text-white tracking-tight">
              Supported ISL Vocabulary Catalog
            </h2>
            <p className="text-xs text-slate-400">
              10 official gestures curated in alignment with ISLRTC standards.
            </p>
          </div>
          <button
            onClick={() => onNavigate('vocabulary')}
            className="text-xs text-brand-400 hover:text-brand-300 font-bold flex items-center gap-1"
          >
            <span>View All</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2.5">
          {FALLBACK_VOCABULARY.map((s) => (
            <div
              key={s.id}
              onClick={() => onNavigate('translator')}
              className="p-3.5 rounded-2xl bg-navy-900/80 border border-navy-800 hover:border-navy-700 transition-all cursor-pointer group space-y-1.5"
            >
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-navy-950 text-slate-400 border border-navy-850 uppercase">
                  {s.category}
                </span>
                <span className="text-[10px] text-brand-400 font-bold">
                  {s.difficulty}
                </span>
              </div>
              <h4 className="text-sm font-bold text-white group-hover:text-brand-300 transition-colors">
                {s.name}
              </h4>
              <p className="text-[11px] text-slate-400 truncate">
                {s.translations.hi}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* Linguistic Standards & Architecture Disclosure */}
      <section className="p-5 sm:p-6 rounded-3xl bg-navy-900/60 border border-slate-800 text-xs space-y-3">
        <div className="flex items-center gap-2 text-slate-200 font-bold text-sm">
          <BookOpen className="w-4 h-4 text-brand-400" />
          <span>Linguistic Standards & Architecture Demarcation</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-slate-400 leading-relaxed">
          <div className="p-3.5 rounded-xl bg-navy-950/80 border border-slate-800/80 space-y-1.5">
            <span className="font-semibold text-cyan-300 block">Manual Alphabet (A–Z)</span>
            <p>
              The primary Coordinate AI engine uses a 94-dimensional normalized biomechanical feature extractor trained on the <strong>Single-Hand Manual Alphabet (ASL / International Single-Hand Standard)</strong>. For dual-hand bimanual letters (such as M, N, P, Q, S, T), the system provides an experimental ISLRTC Bimanual mode.
            </p>
          </div>
          <div className="p-3.5 rounded-xl bg-navy-950/80 border border-slate-800/80 space-y-1.5">
            <span className="font-semibold text-indigo-300 block">Word & Gesture Translator</span>
            <p>
              The word recognition pipeline is powered by a PyTorch BiLSTM trained on <strong>10 canonical ISL vocabulary signs</strong> (<em>namaste, thank_you, help, yes, no, water, food, emergency, doctor, please</em>) aligned with ISLRTC curriculum standards. Open-domain continuous conversational sentence recognition is an active research baseline.
            </p>
          </div>
        </div>
      </section>

      {/* Bottom CTA Card */}
      <section className="p-6 sm:p-8 rounded-3xl bg-gradient-to-r from-brand-950 via-navy-900 to-indigo-950 border border-brand-800/80 text-center space-y-3 shadow-glow">
        <h3 className="text-xl sm:text-2xl font-black text-white tracking-tight">
          Ready to Test Sign Language AI?
        </h3>
        <p className="text-xs sm:text-sm text-slate-300 max-w-md mx-auto">
          Start your camera and make authentic gestures. The system translates and speaks instantly.
        </p>
        <button
          onClick={() => onNavigate('translator')}
          className="px-6 py-3 rounded-2xl bg-brand-600 hover:bg-brand-500 text-white font-bold text-xs sm:text-sm shadow-glow transition-all"
        >
          Launch Live Translator
        </button>
      </section>
    </div>
  );
};
