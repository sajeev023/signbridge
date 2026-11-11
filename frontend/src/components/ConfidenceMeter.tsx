import React from 'react';
import { ShieldCheck, AlertTriangle, Clock, Layers } from 'lucide-react';
import { PredictionStatus } from '../types';

interface ConfidenceMeterProps {
  confidence: number;
  status: PredictionStatus;
  threshold?: number;
  latencyMs?: number;
  bufferFill?: number;
}

export const ConfidenceMeter: React.FC<ConfidenceMeterProps> = ({
  confidence,
  status,
  threshold = 0.75,
  latencyMs,
  bufferFill
}) => {
  const percentage = Math.round(confidence * 100);
  const thresholdPct = Math.round(threshold * 100);

  // Status-driven colors
  let barColor = 'bg-brand-500';
  let badgeColor = 'bg-brand-950 text-brand-300 border-brand-800';
  let statusText = 'AWAITING GESTURE';

  if (status === 'buffering') {
    barColor = 'bg-sky-500';
    badgeColor = 'bg-sky-950 text-sky-300 border-sky-800';
    statusText = `BUFFERING (${Math.round((bufferFill || 0) * 100)}%)`;
  } else if (status === 'recognized') {
    barColor = 'bg-emerald-500';
    badgeColor = 'bg-emerald-950 text-emerald-300 border-emerald-800';
    statusText = 'CONFIDENT RECOGNITION';
  } else if (status === 'uncertain') {
    barColor = 'bg-amber-500';
    badgeColor = 'bg-amber-950 text-amber-300 border-amber-800';
    statusText = 'BELOW THRESHOLD (FALLBACK)';
  } else if (status === 'evaluating') {
    barColor = 'bg-indigo-500';
    badgeColor = 'bg-indigo-950 text-indigo-300 border-indigo-800';
    statusText = 'TEMPORAL STABILIZING';
  }

  return (
    <div className="bg-navy-900 border border-navy-800 rounded-xl p-4 shadow-subtle">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-slate-400" />
          <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
            Neural Confidence Score
          </span>
        </div>
        <div className="flex items-center gap-3 font-mono text-xs">
          {latencyMs !== undefined && (
            <span className="flex items-center gap-1 text-slate-400">
              <Clock className="w-3 h-3 text-brand-400" />
              {latencyMs.toFixed(1)} ms
            </span>
          )}
          <span className="text-sm font-bold text-white">
            {status === 'buffering' ? `${Math.round((bufferFill || 0) * 100)}%` : `${percentage}%`}
          </span>
        </div>
      </div>

      {/* Progress Bar with Threshold Marker */}
      <div className="relative w-full h-3 bg-navy-950 rounded-full overflow-hidden border border-navy-800">
        <div
          className={`h-full transition-all duration-200 rounded-full ${barColor}`}
          style={{ width: `${status === 'buffering' ? (bufferFill || 0) * 100 : percentage}%` }}
        />
        {/* Threshold line */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-rose-400/80 z-10"
          style={{ left: `${thresholdPct}%` }}
          title={`Rejection Threshold: ${thresholdPct}%`}
        />
      </div>

      {/* Meter Legend & Status Badge */}
      <div className="flex items-center justify-between mt-2.5 text-[11px]">
        <div className="flex items-center gap-1.5 text-slate-400 font-mono">
          <span className="w-1.5 h-1.5 rounded-full bg-rose-400" />
          <span>Floor: {thresholdPct}%</span>
        </div>
        <span className={`px-2 py-0.5 text-[10px] font-mono font-medium rounded-full border ${badgeColor}`}>
          {statusText}
        </span>
      </div>
    </div>
  );
};
