import React from 'react';
import { Volume2, AlertCircle, CheckCircle2, Tag, Activity } from 'lucide-react';
import { PredictionResult, SupportedLanguage } from '../types';

interface TranslationCardProps {
  prediction: PredictionResult | null;
  selectedLanguage: SupportedLanguage;
  onSpeak: (text: string, lang: SupportedLanguage) => void;
  isMuted: boolean;
}

export const TranslationCard: React.FC<TranslationCardProps> = ({
  prediction,
  selectedLanguage,
  onSpeak,
  isMuted
}) => {
  if (!prediction) {
    return (
      <div className="bg-navy-900 border border-navy-800 rounded-2xl p-6 text-center text-slate-400 min-h-[220px] flex flex-col items-center justify-center">
        <Activity className="w-8 h-8 text-slate-500 mb-2 animate-pulse" />
        <p className="text-sm font-medium">Ready to translate</p>
        <p className="text-xs text-slate-500 mt-1 max-w-xs">
          Perform any supported Indian Sign Language gesture in view of the camera to begin.
        </p>
      </div>
    );
  }

  // Handle Uncertain Fallback
  if (prediction.status === 'uncertain') {
    return (
      <div className="bg-amber-950/20 border border-amber-800/40 rounded-2xl p-6 transition-all min-h-[220px] flex flex-col justify-between">
        <div>
          <div className="flex items-center gap-2 text-amber-400 mb-2">
            <AlertCircle className="w-5 h-5 flex-shrink-0" />
            <span className="text-xs font-semibold uppercase tracking-wider font-mono">
              Uncertain Prediction Filtered
            </span>
          </div>
          <h3 className="text-xl font-bold text-slate-200 mt-1">
            "Sign not recognised confidently. Please repeat."
          </h3>
          <p className="text-xs text-slate-400 mt-2 leading-relaxed">
            Observed candidate had confidence of {Math.round((prediction.confidence || 0) * 100)}%,
            which falls below the safety floor of {Math.round((prediction.threshold || 0.75) * 100)}%.
            SIGNBRIDGE rejects hallucinations rather than guessing.
          </p>
        </div>
        <div className="mt-4 pt-3 border-t border-amber-800/30 text-[11px] font-mono text-amber-300/80">
          Tip: Ensure clear lighting and execute the complete stroke arc.
        </div>
      </div>
    );
  }

  // Handle Buffering
  if (prediction.status === 'buffering') {
    return (
      <div className="bg-navy-900 border border-navy-800 rounded-2xl p-6 min-h-[220px] flex flex-col justify-center items-center text-center">
        <div className="w-10 h-10 rounded-full border-2 border-brand-500/20 border-t-brand-500 animate-spin mb-3" />
        <p className="text-sm font-medium text-slate-200">Processing Temporal Motion Window</p>
        <p className="text-xs text-slate-400 mt-1 font-mono">
          {prediction.buffer_count || 0} / 30 frames collected
        </p>
      </div>
    );
  }

  // Handle Confident Recognition
  const activeTranslation = prediction.translations?.[selectedLanguage] || prediction.label || '';

  return (
    <div className="bg-gradient-to-b from-navy-900 to-navy-850 border border-navy-750 rounded-2xl p-6 shadow-glow transition-all min-h-[220px] flex flex-col justify-between">
      <div>
        {/* Header tags: Category & Status */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-brand-950 text-brand-300 border border-brand-800">
              <Tag className="w-3 h-3" />
              {prediction.category || 'General'}
            </span>
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-mono font-medium bg-emerald-950 text-emerald-300 border border-emerald-800">
              <CheckCircle2 className="w-3 h-3" />
              {Math.round((prediction.confidence || 0) * 100)}% Match
            </span>
          </div>

          {/* Voice Replay button */}
          <button
            onClick={() => onSpeak(activeTranslation, selectedLanguage)}
            disabled={isMuted}
            className={`p-2 rounded-xl transition-all flex items-center gap-1.5 text-xs font-semibold ${
              isMuted
                ? 'bg-slate-800/40 text-slate-500 cursor-not-allowed'
                : 'bg-brand-600 hover:bg-brand-500 text-white shadow-glow'
            }`}
            title="Replay Voice Audio"
          >
            <Volume2 className="w-4 h-4" />
            <span className="hidden sm:inline">Play Voice</span>
          </button>
        </div>

        {/* Primary Translation Output */}
        <div className="mt-2">
          <div className="text-xs uppercase font-mono text-slate-400 tracking-wider">
            Detected ISL Translation ({selectedLanguage.toUpperCase()}):
          </div>
          <h2 className="text-2xl sm:text-3xl font-extrabold text-white mt-1 tracking-tight leading-tight">
            {activeTranslation}
          </h2>
        </div>

        {/* Secondary multi-lingual translations */}
        <div className="grid grid-cols-3 gap-2 mt-4 pt-3 border-t border-navy-800 text-xs">
          <div className="bg-navy-950/70 p-2 rounded-lg border border-navy-800/70">
            <span className="text-[10px] text-slate-500 font-mono block">English</span>
            <span className="font-medium text-slate-200 line-clamp-1">
              {prediction.translations?.en || '-'}
            </span>
          </div>
          <div className="bg-navy-950/70 p-2 rounded-lg border border-navy-800/70">
            <span className="text-[10px] text-slate-500 font-mono block">Hindi (हिंदी)</span>
            <span className="font-medium text-slate-200 line-clamp-1">
              {prediction.translations?.hi || '-'}
            </span>
          </div>
          <div className="bg-navy-950/70 p-2 rounded-lg border border-navy-800/70">
            <span className="text-[10px] text-slate-500 font-mono block">Telugu (తెలుగు)</span>
            <span className="font-medium text-slate-200 line-clamp-1">
              {prediction.translations?.te || '-'}
            </span>
          </div>
        </div>
      </div>

      {/* Temporal Attention Heatmap Sparkline */}
      {prediction.attention_weights && prediction.attention_weights.length > 0 && (
        <div className="mt-4 pt-3 border-t border-navy-800">
          <div className="flex items-center justify-between text-[10px] font-mono text-slate-400 mb-1.5">
            <span>Temporal Attention Dynamics (30 Frames)</span>
            <span>Climax Peak Weight</span>
          </div>
          <div className="h-4 flex items-end gap-[2px] bg-navy-950/80 p-0.5 rounded border border-navy-800">
            {prediction.attention_weights.map((w, idx) => {
              const maxW = Math.max(...(prediction.attention_weights || [1]));
              const heightPct = Math.max(15, Math.round((w / (maxW || 1)) * 100));
              const isPeak = w === maxW;
              return (
                <div
                  key={idx}
                  className={`flex-1 rounded-t-[1px] transition-all ${
                    isPeak ? 'bg-sky-400' : 'bg-brand-600/70'
                  }`}
                  style={{ height: `${heightPct}%` }}
                  title={`Frame ${idx + 1}: weight ${(w * 100).toFixed(1)}%`}
                />
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};
