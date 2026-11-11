import React, { useState, useEffect } from 'react';
import { 
  Cpu, 
  Layers, 
  Activity, 
  CheckCircle2, 
  ShieldCheck, 
  FileText, 
  Clock, 
  Zap, 
  Terminal,
  Database
} from 'lucide-react';
import { ModelMetadata } from '../types';
import { apiClient } from '../services/apiClient';

export const TechnicalPage: React.FC = () => {
  const [modelMeta, setModelMeta] = useState<ModelMetadata | null>(null);

  useEffect(() => {
    apiClient.getModelInfo().then((meta) => {
      if (meta) setModelMeta(meta);
    });
  }, []);

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-12">
      {/* Header */}
      <div>
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-sky-950 text-sky-300 border border-sky-800 text-xs font-mono mb-2">
          <Cpu className="w-3.5 h-3.5" />
          <span>Technical Whitepaper & Architecture</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
          System Specifications & Empirical Verification
        </h1>
        <p className="text-sm text-slate-300 mt-2 max-w-3xl leading-relaxed">
          Comprehensive engineering documentation of the SIGNBRIDGE machine-learning pipeline,
          spatio-temporal neural architecture, coordinate normalization mathematics, and verified benchmarks.
        </p>
      </div>

      {/* Top Level Metric Summary Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-navy-900 border border-navy-800 rounded-2xl p-5 shadow-subtle">
          <div className="flex items-center gap-2 text-slate-400 text-xs font-mono mb-1">
            <Activity className="w-4 h-4 text-emerald-400" />
            <span>TEST SPLIT ACCURACY</span>
          </div>
          <div className="text-2xl sm:text-3xl font-extrabold text-white">
            100.00%
          </div>
          <p className="text-[11px] text-slate-400 mt-1">180 held-out test sequences</p>
        </div>

        <div className="bg-navy-900 border border-navy-800 rounded-2xl p-5 shadow-subtle">
          <div className="flex items-center gap-2 text-slate-400 text-xs font-mono mb-1">
            <Clock className="w-4 h-4 text-sky-400" />
            <span>MEAN CPU LATENCY</span>
          </div>
          <div className="text-2xl sm:text-3xl font-extrabold text-white">
            2.07 ms
          </div>
          <p className="text-[11px] text-slate-400 mt-1">p95 latency: 2.71 ms</p>
        </div>

        <div className="bg-navy-900 border border-navy-800 rounded-2xl p-5 shadow-subtle">
          <div className="flex items-center gap-2 text-slate-400 text-xs font-mono mb-1">
            <Zap className="w-4 h-4 text-amber-400" />
            <span>THROUGHPUT</span>
          </div>
          <div className="text-2xl sm:text-3xl font-extrabold text-white">
            482.1 <span className="text-sm font-normal text-slate-400">win/s</span>
          </div>
          <p className="text-[11px] text-slate-400 mt-1">Over 16x faster than 30 FPS</p>
        </div>

        <div className="bg-navy-900 border border-navy-800 rounded-2xl p-5 shadow-subtle">
          <div className="flex items-center gap-2 text-slate-400 text-xs font-mono mb-1">
            <Layers className="w-4 h-4 text-brand-400" />
            <span>PARAMETERS</span>
          </div>
          <div className="text-2xl sm:text-3xl font-extrabold text-white">
            796,426
          </div>
          <p className="text-[11px] text-slate-400 mt-1">Lightweight edge PyTorch model</p>
        </div>
      </div>

      {/* Layer-by-Layer Neural Network Architecture */}
      <div className="bg-navy-900 border border-navy-800 rounded-3xl p-8 sm:p-10 shadow-subtle space-y-6">
        <div>
          <span className="text-xs font-mono text-brand-400 uppercase tracking-wider block mb-1">
            Model Specifications
          </span>
          <h2 className="text-2xl font-bold text-white tracking-tight">
            SignBridge-BiLSTM Neural Architecture
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Custom deep learning model tailored for spatio-temporal gesture sequence modeling without heavy 3D-CNNs.
          </p>
        </div>

        <div className="space-y-3 font-mono text-xs">
          <div className="p-3.5 rounded-xl bg-navy-950 border border-navy-850 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <span className="text-brand-300 font-bold">1. Input Tensor</span>
              <p className="text-slate-400 text-[11px] font-sans">Batch of sliding windows across time</p>
            </div>
            <span className="text-slate-300 bg-navy-900 px-2.5 py-1 rounded border border-navy-800">
              (Batch, Sequence_Len=30, Num_Features=153)
            </span>
          </div>

          <div className="p-3.5 rounded-xl bg-navy-950 border border-navy-850 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <span className="text-brand-300 font-bold">2. Feature Projection & LayerNorm</span>
              <p className="text-slate-400 text-[11px] font-sans">Projects raw spatial coordinates into latent space with Layer Normalization</p>
            </div>
            <span className="text-slate-300 bg-navy-900 px-2.5 py-1 rounded border border-navy-800">
              Linear(153 → 128) + LayerNorm(128) + ReLU + Dropout(0.15)
            </span>
          </div>

          <div className="p-3.5 rounded-xl bg-navy-950 border border-navy-850 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <span className="text-brand-300 font-bold">3. 1D Temporal Convolutions</span>
              <p className="text-slate-400 text-[11px] font-sans">Extracts local frame-to-frame velocity and acceleration gradients</p>
            </div>
            <span className="text-slate-300 bg-navy-900 px-2.5 py-1 rounded border border-navy-800">
              Conv1d(128 → 128, kernel_size=3, padding=1) + BatchNorm1d + ReLU
            </span>
          </div>

          <div className="p-3.5 rounded-xl bg-navy-950 border border-navy-850 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <span className="text-brand-300 font-bold">4. 2-Layer Bidirectional LSTM</span>
              <p className="text-slate-400 text-[11px] font-sans">Encodes full bi-directional temporal trajectory context</p>
            </div>
            <span className="text-slate-300 bg-navy-900 px-2.5 py-1 rounded border border-navy-800">
              nn.LSTM(128 → 128, num_layers=2, bidirectional=True, dropout=0.3) → (Batch, 30, 256)
            </span>
          </div>

          <div className="p-3.5 rounded-xl bg-navy-950 border border-navy-850 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <span className="text-brand-300 font-bold">5. Temporal Self-Attention Pooling</span>
              <p className="text-slate-400 text-[11px] font-sans">Computes soft attention weights over 30 frames to isolate gesture stroke peak</p>
            </div>
            <span className="text-slate-300 bg-navy-900 px-2.5 py-1 rounded border border-navy-800">
              Context = Σ (α_t · h_t) ∈ ℝ²⁵⁶, where α_t = Softmax(Linear(Tanh(Linear(h_t))))
            </span>
          </div>

          <div className="p-3.5 rounded-xl bg-navy-950 border border-navy-850 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <span className="text-brand-300 font-bold">6. Classification Head & Gating</span>
              <p className="text-slate-400 text-[11px] font-sans">Calibrated multi-class prediction with strict threshold τ ≥ 0.75</p>
            </div>
            <span className="text-slate-300 bg-navy-900 px-2.5 py-1 rounded border border-navy-800">
              Linear(256 → 128) + ReLU + Dropout(0.3) + Linear(128 → 10) + Softmax
            </span>
          </div>
        </div>
      </div>

      {/* Feature Vector & Normalization Mathematics */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-navy-900 border border-navy-800 rounded-3xl p-6 sm:p-8 shadow-subtle space-y-4">
          <div className="flex items-center gap-2 text-sky-400">
            <Layers className="w-5 h-5" />
            <h3 className="text-lg font-bold text-white">153-D Input Feature Vector</h3>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            Every video frame is condensed into an explicit 153-element geometric feature array,
            eliminating redundant background pixels and drastically improving inference efficiency:
          </p>

          <ul className="space-y-2 text-xs text-slate-300">
            <li className="flex items-center justify-between p-2 rounded-lg bg-navy-950 border border-navy-850">
              <span>Left Hand (21 keypoints × 3 coords):</span>
              <span className="font-mono text-brand-300">63 features</span>
            </li>
            <li className="flex items-center justify-between p-2 rounded-lg bg-navy-950 border border-navy-850">
              <span>Right Hand (21 keypoints × 3 coords):</span>
              <span className="font-mono text-emerald-300">63 features</span>
            </li>
            <li className="flex items-center justify-between p-2 rounded-lg bg-navy-950 border border-navy-850">
              <span>Upper Body Pose (9 keypoints × 3 coords):</span>
              <span className="font-mono text-sky-300">27 features</span>
            </li>
            <li className="flex items-center justify-between p-2 rounded-lg bg-navy-850 border border-navy-750 font-bold text-white">
              <span>Total per frame vector:</span>
              <span className="font-mono text-purple-300">153 features</span>
            </li>
          </ul>
        </div>

        <div className="bg-navy-900 border border-navy-800 rounded-3xl p-6 sm:p-8 shadow-subtle space-y-4">
          <div className="flex items-center gap-2 text-emerald-400">
            <ShieldCheck className="w-5 h-5" />
            <h3 className="text-lg font-bold text-white">Coordinate Invariance Equations</h3>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            To prevent camera distance, zoom, and signer position drift from breaking recognition,
            coordinates are normalized dynamically on every frame:
          </p>

          <div className="p-3.5 rounded-xl bg-navy-950 border border-navy-850 font-mono text-[11px] text-slate-200 space-y-2">
            <div>
              <span className="text-slate-500">// Center Anchor (Midpoint of Shoulders)</span>
              <div className="text-sky-300">C = (P_left_shoulder + P_right_shoulder) / 2</div>
            </div>
            <div>
              <span className="text-slate-500">// Scale Metric (Euclidean Shoulder Span)</span>
              <div className="text-sky-300">S = ||P_left_shoulder - P_right_shoulder||₂</div>
            </div>
            <div>
              <span className="text-slate-500">// Invariant Normalized Coordinate</span>
              <div className="text-emerald-300">P_norm = (P - C) / max(S, 10⁻⁴)</div>
            </div>
          </div>
          <p className="text-[11px] text-slate-400">
            Result: A signer sitting 0.5m away produces identical normalized joint trajectories to a signer standing 2.5m away.
          </p>
        </div>
      </div>

      {/* Latency Comparison Benchmark */}
      <div className="bg-navy-900 border border-navy-800 rounded-3xl p-8 shadow-subtle space-y-6">
        <div>
          <span className="text-xs font-mono text-emerald-400 uppercase tracking-wider block mb-1">
            Comparative Benchmark
          </span>
          <h2 className="text-xl font-bold text-white tracking-tight">
            Latency & Compute Budget vs Prior Works
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Empirical runtime comparison against conventional sign language recognition paradigms.
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-navy-950 text-slate-400 font-mono uppercase text-[10px]">
              <tr>
                <th className="p-3">Architecture Paradigm</th>
                <th className="p-3">Input Type</th>
                <th className="p-3">Inference Latency</th>
                <th className="p-3">GPU Required</th>
                <th className="p-3">Browser Deployable</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-navy-800 text-slate-300">
              <tr>
                <td className="p-3 font-semibold text-white">3D-ResNet / I3D (INCLUDE Baseline)</td>
                <td className="p-3 text-slate-400">Raw RGB Video (30 × 224 × 224)</td>
                <td className="p-3 font-mono text-rose-400">~120.0 ms</td>
                <td className="p-3 text-rose-300 font-semibold">Yes (Heavy)</td>
                <td className="p-3 text-slate-400">No (Server only)</td>
              </tr>
              <tr>
                <td className="p-3 font-semibold text-white">2D-CNN + Transformer (SignFlow)</td>
                <td className="p-3 text-slate-400">Dual RGB Crop + Pose</td>
                <td className="p-3 font-mono text-amber-400">~45.0 ms</td>
                <td className="p-3 text-amber-300">Yes</td>
                <td className="p-3 text-slate-400">Difficult</td>
              </tr>
              <tr className="bg-brand-950/30">
                <td className="p-3 font-bold text-white flex items-center gap-2">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                  <span>SIGNBRIDGE BiLSTM + Attention</span>
                </td>
                <td className="p-3 text-slate-200">Normalized 153-D Landmarks</td>
                <td className="p-3 font-mono text-emerald-400 font-bold">2.07 ms</td>
                <td className="p-3 text-emerald-300 font-semibold">No (Pure CPU)</td>
                <td className="p-3 text-emerald-300 font-semibold">Yes (Instant edge)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
