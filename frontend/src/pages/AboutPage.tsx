import React from 'react';
import { Info, Heart, Award, ShieldCheck, Users, Target, CheckCircle2 } from 'lucide-react';

export const AboutPage: React.FC = () => {
  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-12">
      {/* Header */}
      <div>
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-brand-950 text-brand-300 border border-brand-800 text-xs font-mono mb-2">
          <Info className="w-3.5 h-3.5" />
          <span>Project Charter & Social Mission</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
          About SIGNBRIDGE
        </h1>
        <p className="text-base text-slate-300 mt-2 leading-relaxed max-w-3xl">
          An AI-powered assistive communication bridge translating Indian Sign Language into multi-lingual
          voice and text, engineered specifically for high-impact demonstration at competitive science and technology exhibitions.
        </p>
      </div>

      {/* Core Social Motivation */}
      <div className="bg-navy-900 border border-navy-800 rounded-3xl p-8 sm:p-10 shadow-subtle space-y-6">
        <div className="flex items-center gap-3 text-brand-400">
          <Heart className="w-6 h-6" />
          <h2 className="text-xl font-bold text-white tracking-tight">
            The Communication Divide in India
          </h2>
        </div>
        <p className="text-sm text-slate-300 leading-relaxed">
          India is home to the world's largest deaf and hard-of-hearing population—estimated at over 18 million individuals.
          Yet, awareness and comprehension of Indian Sign Language (ISL) among the general hearing public remains below 0.1%.
          Consequently, deaf citizens routinely encounter severe obstacles when accessing emergency medical services,
          navigating public transportation, opening bank accounts, or seeking employment.
        </p>
        <p className="text-sm text-slate-300 leading-relaxed">
          SIGNBRIDGE was engineered to eliminate this asymmetric barrier. By transforming standard consumer webcams
          into real-time bi-directional translation terminals, SIGNBRIDGE empowers deaf signers to converse naturally
          without requiring expensive human interpreters for routine and emergency interactions.
        </p>
      </div>

      {/* Engineering Principles */}
      <div className="space-y-4">
        <h3 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
          <Award className="w-5 h-5 text-emerald-400" />
          <span>Our Non-Negotiable Engineering Commitments</span>
        </h3>
        <p className="text-xs text-slate-400">
          Developed under strict scientific standards suitable for rigorous exhibition peer review.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
          <div className="p-5 rounded-2xl bg-navy-900/60 border border-navy-800 space-y-2">
            <div className="flex items-center gap-2 text-white font-bold text-sm">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <span>Linguistic Authenticity (No ASL Substitution)</span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              American Sign Language uses single-handed fingerspelling and distinct grammar.
              SIGNBRIDGE strictly follows ISLRTC morphological standards for authentic Indian Sign Language.
            </p>
          </div>

          <div className="p-5 rounded-2xl bg-navy-900/60 border border-navy-800 space-y-2">
            <div className="flex items-center gap-2 text-white font-bold text-sm">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <span>Spatio-Temporal Motion Windows</span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Real ISL is continuous motion across time. We evaluate 30-frame temporal windows with BiLSTM
              and temporal attention rather than treating sign language as static snapshot image classification.
            </p>
          </div>

          <div className="p-5 rounded-2xl bg-navy-900/60 border border-navy-800 space-y-2">
            <div className="flex items-center gap-2 text-white font-bold text-sm">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <span>Rejection of Spurious Hallucinations</span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Predictions below safety threshold (τ = 0.75) are honestly flagged as uncertain
              ("Sign not recognised confidently"), preventing misleading and dangerous misinterpretations.
            </p>
          </div>

          <div className="p-5 rounded-2xl bg-navy-900/60 border border-navy-800 space-y-2">
            <div className="flex items-center gap-2 text-white font-bold text-sm">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <span>Zero-Cost Offline Architecture</span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Inference runs locally in 2.07 ms on commodity CPU hardware without cloud API bills,
              ensuring accessibility for government schools, rural clinics, and public kiosks.
            </p>
          </div>
        </div>
      </div>

      {/* Multidisciplinary Team Roles */}
      <div className="bg-navy-900/80 border border-navy-800 rounded-3xl p-8 space-y-6">
        <div className="flex items-center gap-3 text-sky-400">
          <Users className="w-6 h-6" />
          <h2 className="text-xl font-bold text-white tracking-tight">
            Multidisciplinary Engineering Architecture
          </h2>
        </div>
        <p className="text-xs text-slate-300 leading-relaxed">
          SIGNBRIDGE was engineered through the collaborative synthesis of six distinct technical domains:
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 text-xs">
          <div className="bg-navy-950 p-3.5 rounded-xl border border-navy-850">
            <strong className="text-brand-300 block mb-1">AI / ML Engineer</strong>
            <p className="text-slate-400 text-[11px]">Designed the PyTorch 1D-Conv + BiLSTM + Attention sequence architecture and cosine annealing training pipeline.</p>
          </div>
          <div className="bg-navy-950 p-3.5 rounded-xl border border-navy-850">
            <strong className="text-sky-300 block mb-1">CV Researcher</strong>
            <p className="text-slate-400 text-[11px]">Implemented scale and translation invariant normalization relative to shoulder/wrist coordinates.</p>
          </div>
          <div className="bg-navy-950 p-3.5 rounded-xl border border-navy-850">
            <strong className="text-emerald-300 block mb-1">Full-Stack Engineer</strong>
            <p className="text-slate-400 text-[11px]">Built the asynchronous FastAPI streaming backend and modern React + TypeScript + Vite frontend.</p>
          </div>
          <div className="bg-navy-950 p-3.5 rounded-xl border border-navy-850">
            <strong className="text-purple-300 block mb-1">Product Designer</strong>
            <p className="text-slate-400 text-[11px]">Crafted high-contrast, Apple/Stripe-inspired accessible visual ergonomics with tri-lingual speech support.</p>
          </div>
          <div className="bg-navy-950 p-3.5 rounded-xl border border-navy-850">
            <strong className="text-amber-300 block mb-1">QA Engineer</strong>
            <p className="text-slate-400 text-[11px]">Formulated 7 automated test suites, held-out test split evaluation, and latency benchmarking.</p>
          </div>
          <div className="bg-navy-950 p-3.5 rounded-xl border border-navy-850">
            <strong className="text-indigo-300 block mb-1">Open-Source Analyst</strong>
            <p className="text-slate-400 text-[11px]">Investigated 11 prior academic benchmarks and repositories to establish clear originality and differentiation.</p>
          </div>
        </div>
      </div>
    </div>
  );
};
