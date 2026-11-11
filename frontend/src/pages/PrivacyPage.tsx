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
          <h3 className="text-base font-bold text-white">Live Camera In-Memory Processing</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            In Live Camera mode, video frames are processed strictly in-memory (RAM) by the browser
            and immediately overwritten. No camera feed, video stream, or biometric facial capture is ever
            persisted to disk or forwarded to third-party tracking services.
          </p>
        </div>

        <div className="p-6 rounded-2xl bg-navy-900 border border-navy-800 space-y-3 shadow-subtle">
          <div className="w-10 h-10 rounded-xl bg-navy-950 border border-navy-800 flex items-center justify-center text-brand-400">
            <HardDrive className="w-5 h-5" />
          </div>
          <h3 className="text-base font-bold text-white">Transient Server-Side Video Ingestion</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            In Video Upload mode, uploaded clips are ingested by the backend server into an isolated temporary
            system folder exclusively for the duration of frame decoding (sub-500ms). The video file is
            automatically and irrevocably purged from disk upon completion or error.
          </p>
        </div>

        <div className="p-6 rounded-2xl bg-navy-900 border border-navy-800 space-y-3 shadow-subtle">
          <div className="w-10 h-10 rounded-xl bg-navy-950 border border-navy-800 flex items-center justify-center text-sky-400">
            <Lock className="w-5 h-5" />
          </div>
          <h3 className="text-base font-bold text-white">Client-Controlled Local History</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Your translation session history is maintained exclusively within your browser's private local storage.
            No server database records user translations. You can export your session as JSON or purge it
            with one click at any time.
          </p>
        </div>

        <div className="p-6 rounded-2xl bg-navy-900 border border-navy-800 space-y-3 shadow-subtle">
          <div className="w-10 h-10 rounded-xl bg-navy-950 border border-navy-800 flex items-center justify-center text-purple-400">
            <Server className="w-5 h-5" />
          </div>
          <h3 className="text-base font-bold text-white">Self-Contained Model Execution</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            AI recognition runs on self-hosted PyTorch CPU weights and local MediaPipe models.
            The system never invokes proprietary external cloud vision APIs, preventing commercial profiling
            or data harvesting of assistive conversations.
          </p>
        </div>
      </div>

      <div className="bg-navy-900/60 border border-navy-800 rounded-2xl p-6 space-y-3">
        <h4 className="text-sm font-bold text-white flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          <span>User Consent & Operational Transparency</span>
        </h4>
        <ul className="space-y-1.5 text-xs text-slate-400 list-disc list-inside">
          <li><strong>Live Camera:</strong> Client-side frame tracking in RAM; zero server upload of video streams.</li>
          <li><strong>Video Upload:</strong> Ephemeral server-side processing with immediate automated file deletion.</li>
          <li><strong>Demonstration Mode:</strong> Simulated trajectory streaming running entirely on the client.</li>
          <li><strong>Audio:</strong> Synthesized locally via browser SpeechSynthesis API.</li>
          <li><strong>Analytics:</strong> Zero third-party telemetry, trackers, or commercial cookies.</li>
        </ul>
      </div>
    </div>
  );
};
