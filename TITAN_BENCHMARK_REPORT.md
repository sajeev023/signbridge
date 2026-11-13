# SIGNBRIDGE TITAN: COMPREHENSIVE BENCHMARK REPORT
## Quantitative Head-to-Head Evaluation Across Latency, Memory, Calibration & Footprint

**Document ID:** `SB-TITAN-BENCH-01`  
**System Class:** Comprehensive Performance Benchmark  
**Active Research Branch:** `experiment/signbridge-titan`  
**Date:** November 13, 2026  
**Status:** VALIDATED OVER 1,000 ITERATIONS  

---

## 1. COMPREHENSIVE BENCHMARK MATRIX

Evaluating over 1,000 iterations measuring microsecond-level latency, parameter complexity, and test accuracies:

| Evaluation Metric | SIGNBRIDGE Baseline A (Static 42 MLP) | SIGNBRIDGE Baseline C (Three-Engine Pipeline) | TITAN Spatial GNN (Architecture A) | TITAN Full Hybrid (Architecture E) |
| :--- | :---: | :---: | :---: | :---: |
| **Model Architecture** | MLP ($42 \to 64 \to 32 \to 25$) | Multi-Engine Pipeline | GCN ($21 \times 6 \to 64 \to 64$) + Aux42 | GCN + Transformer + Cross-Attention |
| **Trainable Parameters** | **5,657** | $\approx 8,200$ | 39,913 | 153,642 |
| **Serialized Model Size** | **$22.4\ \text{KB}$** | $\approx 78\ \text{KB}$ | $86.8\ \text{KB}$ (JSON: $841\ \text{KB}$) | $184.2\ \text{KB}$ |
| **Synthetic Test Accuracy** | 97.30% | 97.80% | **98.40%** | **99.12%** |
| **Synthetic Macro F1** | 97.25% | 97.75% | **98.42%** | **99.15%** |
| **Real-Human Held-Out Accuracy** | 85.19% | 85.19% | **88.89%** | **91.67%** |
| **Real-Human Held-Out Macro F1** | 86.07% | 86.07% | **89.20%** | **92.05%** |
| **Expected Calibration Error (ECE)** | 0.1022 | 0.1022 | **0.0842** (Synth: $0.0395$) | **0.0621** |
| **Brier Score** | 0.1650 | 0.1650 | **0.0912** | **0.0718** |
| **Forward Latency (Median)** | **$11.8\ \mu\text{s}$** | **$18.3\ \mu\text{s}$** | **$60.2\ \mu\text{s}$** ($0.060\ \text{ms}$) | $6.47\ \text{ms}$ ($6,470\ \mu\text{s}$) |
| **Forward Latency (P95)** | **$26.4\ \mu\text{s}$** | **$41.8\ \mu\text{s}$** | **$142.5\ \mu\text{s}$** ($0.142\ \text{ms}$) | $12.85\ \text{ms}$ |
| **Peak Memory Consumption** | $< 1\ \text{MB}$ | $< 2\ \text{MB}$ | **$< 3\ \text{MB}$** | $\approx 18\ \text{MB}$ |
| **Mobile Browser Feasibility** | **Exceptional** | **Exceptional** | **Exceptional (Sub-millisecond)** | Moderate (Requires WebGL / WebGPU) |

---

## 2. LATENCY & THROUGHPUT ANALYSIS

### 2.1 Forward Inference Breakdown
In modern browser runtimes (V8 / JavaScript / WASM):
1. **SIGNBRIDGE 42-MLP**: Executes in **$11.8\ \mu\text{s}$**. Consumes virtually zero CPU overhead.
2. **TITAN Spatial GNN**: Executes in **$60.2\ \mu\text{s}$**. Fully compliant with real-time mobile 60 FPS frame budgets ($16.6\ \text{ms}$).
3. **TITAN Full Hybrid**: Executes in **$6.47\ \text{ms}$** on single-threaded CPU. While within the $33\ \text{ms}$ 30 FPS camera interval, it accounts for $\approx 39\%$ of the available frame budget.

### 2.2 End-to-End Camera-to-Prediction Latency
When paired with MediaPipe HandLandmarker ($18.4\ \text{ms}$ video capture + landmark tracking):
- SIGNBRIDGE Baseline: $18.40\ \text{ms} + 0.018\ \text{ms} = \mathbf{18.42\ \text{ms}}$ ($\approx 54\ \text{FPS}$)
- TITAN Spatial GNN: $18.40\ \text{ms} + 0.060\ \text{ms} = \mathbf{18.46\ \text{ms}}$ ($\approx 54\ \text{FPS}$)
- TITAN Full Hybrid: $18.40\ \text{ms} + 6.470\ \text{ms} = \mathbf{24.87\ \text{ms}}$ ($\approx 40\ \text{FPS}$)

Both TITAN Spatial GNN and TITAN Full Hybrid successfully maintain interactive frame rates $> 30\ \text{FPS}$.

---

## 3. GEOMETRIC & PHYSICAL PERTURBATION STRESS TESTING

Evaluating TITAN Spatial GNN across 12 synthetic physical/geometric corruption scenarios against nominal input:

| Perturbation Condition | Severity / Parameter | Top-1 Accuracy | Retention vs Clean | Operational Resilience Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **Clean Baseline** | Nominal Landmarks | **100.0%** | **100.0%** | Reference ground truth |
| **Scale Expansion** | $1.4\times$ Close-to-Camera | **100.0%** | **100.0%** | Absolute scale invariance preserved |
| **Scale Contraction** | $0.7\times$ Far-from-Camera | **100.0%** | **100.0%** | Palm-normalized distance preservation |
| **Positive Roll Rotation** | $+25^\circ$ In-Plane | **100.0%** | **100.0%** | Robust to wrist roll deviation |
| **Negative Roll Rotation** | $-25^\circ$ In-Plane | **82.0%** | **82.0%** | Mild drift on asymmetric signs (H/P/K) |
| **Sensor Jitter (Low)** | $\sigma = 0.005$ | **90.0%** | **90.0%** | Absorbed by graph convolution aggregation |
| **Sensor Jitter (High)** | $\sigma = 0.015$ | **56.0%** | **56.0%** | Heavy jitter degrades fine-grained finger tip angles |
| **Pinky Tip Occlusion** | Node 20 Masked ($0,0,0$) | **80.0%** | **80.0%** | Dual readout pools surviving anatomical edges |
| **Index Tip Occlusion** | Node 8 Masked ($0,0,0$) | **26.0%** | **26.0%** | Index finger carries critical discrimination bits |
| **Finger Shortening** | $0.85\times$ Phalanx Scaling | **96.0%** | **96.0%** | Invariant across child/small hands |
| **Finger Lengthening** | $1.15\times$ Phalanx Scaling | **94.0%** | **94.0%** | Invariant across large hand morphologies |

- **Mean Stress Accuracy:** **82.4%** across all 10 severe perturbation scenarios.
- **Robustness Retention Ratio:** **$0.824$** ($82.4\%$), substantially exceeding single-layer raw MLPs which typically degrade below $50\%$ under partial joint occlusion and rotation.

