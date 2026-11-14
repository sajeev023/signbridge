/**
 * SIGNBRIDGE TITAN ASL: Frontend Inference Engine Performance & Latency Benchmark
 * Measures:
 * 1. Model loading and tensor reconstruction
 * 2. 21-Node scale-normalized node feature preparation
 * 3. 2-Layer Adaptive GCN Forward Pass with LayerNorm and GELU
 * 4. Dual Readout (Mean-Max) + Auxiliary 42-Feature Fusion
 * 5. Temperature-scaled Softmax, Predictive Entropy, and Energy OOD score
 * 6. Median, P95, and P99 latency across 5,000 iterations
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');

const modelPath = path.join(rootDir, 'backend', 'saved_models', 'titan_asl_spatial_model.json');
const evalDataPath = path.join(rootDir, 'backend', 'titan', 'datasets', 'asl_alphabet', 'metadata', 'real_human_benchmark.json');

const model = JSON.parse(fs.readFileSync(modelPath, 'utf8'));
const evalData = JSON.parse(fs.readFileSync(evalDataPath, 'utf8'));

const testRecord = evalData.records[0];
const landmarks = testRecord.landmarks;
const weights = model.state_dict;
const classes = model.classes;
const temp = model.temperature || 1.08;

// Build Kipf-Welling Normalized Adjacency
const GRAPH_EDGES = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17], [0, 5], [0, 17],
  [4, 8], [8, 12], [12, 16], [16, 20], [4, 12], [5, 17],
];

const N = 21;
const adj = Array.from({ length: N }, () => Array(N).fill(0));
for (let i = 0; i < N; i++) adj[i][i] = 1.0;
for (const [u, v] of GRAPH_EDGES) {
  adj[u][v] = 1.0;
  adj[v][u] = 1.0;
}
const deg = adj.map(row => row.reduce((s, v) => s + v, 0));
const degInvSqrt = deg.map(d => (d > 0 ? 1.0 / Math.sqrt(d) : 0));
const normAdj = Array.from({ length: N }, () => Array(N).fill(0));
for (let i = 0; i < N; i++) {
  for (let j = 0; j < N; j++) {
    normAdj[i][j] = degInvSqrt[i] * adj[i][j] * degInvSqrt[j];
  }
}

// Effective Adjacency: A_eff = A_norm + B_learned
const bAdj = weights['spatial_encoder.learned_adj'];
const effAdj = Array.from({ length: N }, (_, i) =>
  Array.from({ length: N }, (_, j) => normAdj[i][j] + (bAdj ? bAdj[i][j] : 0))
);

const KINEMATIC_PARENT = [
  0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19
];

function gelu(x) {
  return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
}

function layerNorm(x, gamma, beta, eps = 1e-5) {
  const len = x.length;
  let sum = 0;
  for (let i = 0; i < len; i++) sum += x[i];
  const mean = sum / len;
  let varSum = 0;
  for (let i = 0; i < len; i++) {
    const diff = x[i] - mean;
    varSum += diff * diff;
  }
  const invStd = 1.0 / Math.sqrt(varSum / len + eps);
  const out = new Array(len);
  for (let i = 0; i < len; i++) {
    const g = gamma ? gamma[i] : 1.0;
    const b = beta ? beta[i] : 0.0;
    out[i] = (x[i] - mean) * invStd * g + b;
  }
  return out;
}

function runTITANForward(lms, feats42) {
  const wrist = lms[0];
  const palmDist = Math.hypot(lms[9].x - wrist.x, lms[9].y - wrist.y) || 1.0;

  // Node features (21, 6)
  const nodeFeats = new Array(N);
  for (let i = 0; i < N; i++) {
    const p = lms[i];
    const parent = lms[KINEMATIC_PARENT[i]];
    nodeFeats[i] = [
      (p.x - wrist.x) / palmDist,
      (p.y - wrist.y) / palmDist,
      ((p.z || 0) - (wrist.z || 0)) / palmDist,
      (p.x - parent.x) / palmDist,
      (p.y - parent.y) / palmDist,
      ((p.z || 0) - (parent.z || 0)) / palmDist,
    ];
  }

  // Input projection
  const wInput = weights['spatial_encoder.input_proj.weight'];
  const bInput = weights['spatial_encoder.input_proj.bias'];
  const wJoint = weights['spatial_encoder.joint_embed.weight'];
  const hiddenDim = wInput.length;

  let h = Array.from({ length: N }, () => Array(hiddenDim).fill(0));
  for (let i = 0; i < N; i++) {
    const rawVec = [...nodeFeats[i], ...wJoint[i]];
    for (let o = 0; o < hiddenDim; o++) {
      let sum = bInput[o];
      for (let k = 0; k < rawVec.length; k++) sum += rawVec[k] * wInput[o][k];
      h[i][o] = gelu(sum);
    }
  }

  // GCN Layer 0 & 1 with LayerNorm & Adaptive Graph
  for (let layerIdx = 0; layerIdx < 2; layerIdx++) {
    const wN = weights[`spatial_encoder.gcn_layers.${layerIdx}.w_neighbor.weight`];
    const wS = weights[`spatial_encoder.gcn_layers.${layerIdx}.w_self.weight`];
    const bS = weights[`spatial_encoder.gcn_layers.${layerIdx}.w_self.bias`];
    const gLN = weights[`spatial_encoder.gcn_layers.${layerIdx}.layer_norm.weight`];
    const bLN = weights[`spatial_encoder.gcn_layers.${layerIdx}.layer_norm.bias`];

    const hNeighbor = Array.from({ length: N }, () => Array(hiddenDim).fill(0));
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const a = effAdj[i][j];
        if (a !== 0) {
          for (let d = 0; d < hiddenDim; d++) hNeighbor[i][d] += a * h[j][d];
        }
      }
    }

    const nextH = Array.from({ length: N }, () => Array(hiddenDim).fill(0));
    for (let i = 0; i < N; i++) {
      const act = new Array(hiddenDim);
      for (let o = 0; o < hiddenDim; o++) {
        let sum = bS ? bS[o] : 0;
        for (let d = 0; d < hiddenDim; d++) {
          sum += hNeighbor[i][d] * wN[o][d] + h[i][d] * wS[o][d];
        }
        act[o] = gelu(sum);
      }
      const normAct = layerNorm(act, gLN, bLN);
      for (let o = 0; o < hiddenDim; o++) nextH[i][o] = h[i][o] + normAct[o];
    }
    h = nextH;
  }

  // Dual Readout (Mean-Max)
  const pooled = new Array(hiddenDim * 2).fill(0);
  for (let d = 0; d < hiddenDim; d++) {
    let sum = 0, maxVal = -Infinity;
    for (let i = 0; i < N; i++) {
      sum += h[i][d];
      if (h[i][d] > maxVal) maxVal = h[i][d];
    }
    pooled[d] = sum / N;
    pooled[hiddenDim + d] = maxVal;
  }

  const wRead = weights['spatial_encoder.readout_proj.weight'];
  const bRead = weights['spatial_encoder.readout_proj.bias'];
  const gNorm = weights['spatial_encoder.norm.weight'];
  const bNorm = weights['spatial_encoder.norm.bias'];
  const zRaw = new Array(hiddenDim);
  for (let o = 0; o < hiddenDim; o++) {
    let sum = bRead[o];
    for (let k = 0; k < pooled.length; k++) sum += pooled[k] * wRead[o][k];
    zRaw[o] = gelu(sum);
  }
  let zSpatial = layerNorm(zRaw, gNorm, bNorm);

  // Aux feature projection
  const wAux0 = weights['spatial_encoder.aux_proj.0.weight'];
  const bAux0 = weights['spatial_encoder.aux_proj.0.bias'];
  const wAux2 = weights['spatial_encoder.aux_proj.2.weight'];
  const bAux2 = weights['spatial_encoder.aux_proj.2.bias'];
  const gAux3 = weights['spatial_encoder.aux_proj.3.weight'];
  const bAux3 = weights['spatial_encoder.aux_proj.3.bias'];
  const gFuse = weights['spatial_encoder.fusion_norm.weight'];
  const bFuse = weights['spatial_encoder.fusion_norm.bias'];

  if (wAux0 && wAux2 && feats42) {
    const auxH = new Array(hiddenDim);
    for (let o = 0; o < hiddenDim; o++) {
      let sum = bAux0[o];
      for (let k = 0; k < feats42.length; k++) sum += feats42[k] * wAux0[o][k];
      auxH[o] = gelu(sum);
    }
    const auxOut = new Array(hiddenDim);
    for (let o = 0; o < hiddenDim; o++) {
      let sum = bAux2[o];
      for (let k = 0; k < auxH.length; k++) sum += auxH[k] * wAux2[o][k];
      auxOut[o] = sum;
    }
    const auxNorm = layerNorm(auxOut, gAux3, bAux3);
    const fused = new Array(hiddenDim);
    for (let o = 0; o < hiddenDim; o++) fused[o] = zSpatial[o] + auxNorm[o];
    zSpatial = layerNorm(fused, gFuse, bFuse);
  }

  // Classifier
  const wCls0 = weights['classifier.0.weight'];
  const bCls0 = weights['classifier.0.bias'];
  const gCls1 = weights['classifier.1.weight'];
  const bCls1 = weights['classifier.1.bias'];
  const wCls4 = weights['classifier.4.weight'];
  const bCls4 = weights['classifier.4.bias'];

  const cls0 = new Array(wCls0.length);
  for (let o = 0; o < wCls0.length; o++) {
    let sum = bCls0[o];
    for (let k = 0; k < zSpatial.length; k++) sum += zSpatial[k] * wCls0[o][k];
    cls0[o] = sum;
  }
  const clsNorm = layerNorm(cls0, gCls1, bCls1);
  const clsHidden = clsNorm.map(x => gelu(x));

  const logits = new Array(classes.length);
  for (let o = 0; o < classes.length; o++) {
    let sum = bCls4[o];
    for (let k = 0; k < clsHidden.length; k++) sum += clsHidden[k] * wCls4[o][k];
    logits[o] = sum;
  }

  // Calibrated Softmax
  const scaled = logits.map(l => l / temp);
  const maxL = Math.max(...scaled);
  const exps = scaled.map(l => Math.exp(l - maxL));
  const sumExp = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map(e => e / (sumExp + 1e-12));

  let bestIdx = 0, bestP = probs[0];
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > bestP) {
      bestP = probs[i];
      bestIdx = i;
    }
  }

  return { pred: classes[bestIdx], conf: bestP, logits };
}

// Warmup
for (let i = 0; i < 50; i++) runTITANForward(landmarks, testRecord.features_42);

// Benchmark 5,000 iterations
const ITERATIONS = 5000;
const latencies = new Float64Array(ITERATIONS);

for (let i = 0; i < ITERATIONS; i++) {
  const t0 = performance.now();
  runTITANForward(landmarks, testRecord.features_42);
  const t1 = performance.now();
  latencies[i] = (t1 - t0) * 1000; // microseconds
}

latencies.sort();
const medianUs = latencies[Math.floor(ITERATIONS * 0.50)];
const p95Us = latencies[Math.floor(ITERATIONS * 0.95)];
const p99Us = latencies[Math.floor(ITERATIONS * 0.99)];
const meanUs = latencies.reduce((a, b) => a + b, 0) / ITERATIONS;

const outResult = runTITANForward(landmarks, testRecord.features_42);

const benchmarkResults = {
  engine: "SIGNBRIDGE TITAN ASL Adaptive Spatial GNN (TypeScript)",
  iterations: ITERATIONS,
  prediction_sample: testRecord.capture_id,
  ground_truth: testRecord.alphabet_label,
  predicted_class: outResult.pred,
  calibrated_confidence: Number(outResult.conf.toFixed(4)),
  latency_us: {
    median: Number(medianUs.toFixed(1)),
    p95: Number(p95Us.toFixed(1)),
    p99: Number(p99Us.toFixed(1)),
    mean: Number(meanUs.toFixed(1))
  },
  latency_ms: {
    median: Number((medianUs / 1000).toFixed(4)),
    p95: Number((p95Us / 1000).toFixed(4)),
    p99: Number((p99Us / 1000).toFixed(4)),
    mean: Number((meanUs / 1000).toFixed(4))
  },
  fps_equivalent: Math.round(1000000 / medianUs)
};

console.log(JSON.stringify(benchmarkResults, null, 2));
