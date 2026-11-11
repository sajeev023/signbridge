import React from 'react';
import { ShieldCheck, Lock, EyeOff, Server, HardDrive, CheckCircle2 } from 'lucide-react';

export const PrivacyPage: React.FC = () => {
  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-10">
      <div>
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-950 text-emerald-300 border border-emerald-800 text-xs font-mono mb-3">
          <ShieldCheck className="w-3.5 h-3.5" />
          <span>Privacy & Data Sovereignty Charter</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
          Privacy Policy & Security Architecture
        </h1>
        <p className="text-sm text-slate-300 mt-2 leading-relaxed">
          SIGNBRIDGE is designed on strict privacy-first principles. We believe assistive technology
          must protect user autonomy, dignity, and personal communication privacy.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="p-6 rounded-2xl bg-navy-900 border border-navy-800 space-y-3 shadow-subtle">
          <div className="w-10 h-10 rounded-xl bg-navy-950 border border-navy-800 flex items-center justify-center text-emerald-400">
            <EyeOff className="w-5 h-5" />
          </div>
          <h3 className="text-base font-bold text-white">No Cloud Video Retention</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Live webcam frames are processed strictly in-memory (RAM) and immediately discarded.
            No camera feed, video stream, or biometric facial capture is ever recorded, stored on disk,
            or transmitted to third-party services.
          </p>
        </div>

        <div className="p-6 rounded-2xl bg-navy-900 border border-navy-800 space-y-3 shadow-subtle">
          <div className="w-10 h-10 rounded-xl bg-navy-950 border border-navy-800 flex items-center justify-center text-brand-400">
            <HardDrive className="w-5 h-5" />
          </div>
          <h3 className="text-base font-bold text-white">Automatic Video Purging</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            When you upload a video in Video Upload Mode, it is temporarily stored in an isolated system directory
            solely for the duration of landmark decoding (sub-500ms), and is immediately deleted in a mandatory
            server-side cleanup routine upon completion.
          </p>
        </div>

        <div className="p-6 rounded-2xl bg-navy-900 border border-navy-800 space-y-3 shadow-subtle">
          <div className="w-10 h-10 rounded-xl bg-navy-950 border border-navy-800 flex items-center justify-center text-sky-400">
            <Lock className="w-5 h-5" />
          </div>
          <h3 className="text-base font-bold text-white">Local-Only Session Storage</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Your translation history is maintained exclusively within your browser's private local storage.
            You have full sovereignty to export your history as a JSON file or purge it completely at any time with one click.
          </p>
        </div>

        <div className="p-6 rounded-2xl bg-navy-900 border border-navy-800 space-y-3 shadow-subtle">
          <div className="w-10 h-10 rounded-xl bg-navy-950 border border-navy-800 flex items-center justify-center text-purple-400">
            <Server className="w-5 h-5" />
          </div>
          <h3 className="text-base font-bold text-white">Edge-First Inference</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Core AI recognition executes on local CPU hardware using our self-contained PyTorch BiLSTM neural model.
            We do not rely on paid external cloud vision APIs, preventing external telemetry profiling.
          </p>
        </div>
      </div>

      <div className="bg-navy-900/60 border border-navy-800 rounded-2xl p-6 space-y-3">
        <h4 className="text-sm font-bold text-white flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          <span>User Consent & Data Rights Summary</span>
        </h4>
        <ul className="space-y-1.5 text-xs text-slate-400 list-disc list-inside">
          <li>Camera permissions can be revoked at any time via your browser settings.</li>
          <li>Video uploads are strictly user-initiated; the system never accesses local storage automatically.</li>
          <li>Audio synthesis is generated on your device using native browser SpeechSynthesis engines.</li>
          <li>We collect zero telemetry, tracking cookies, or advertising identifiers.</li>
        </ul>
      </div>
    </div>
  );
};
