# SIGNBRIDGE — Model Comparison & Performance Benchmark
**Document ID:** BENCH-COMP-2026-V1  
**Scope:** Architecture Comparison across Coordinate Neural Network, Tree Ensemble & Legacy Baselines  
**Status:** VERIFIED  

---

## 1. Empirical Architecture Comparison

All candidate models were trained and evaluated on identical participant-separated splits of `alphabet_dataset_multisigner.json`:

| Model Architecture | Input Features | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Size on Disk | Inference Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CoordinateMLP (Production)** | **94 Enriched Coords** | **99.48%** | **0.9948** | **0.9948** | **0.9948** | **2.10 MB** | **0.35 ms** |
| **CoordinateRandomForest** | 94 Enriched Coords | 98.02% | 0.9805 | 0.9802 | 0.9802 | 18.5 MB | 4.80 ms |
| **Letter A Specialist Model** | 42 Features | 83.33% (Human) | 0.8150 | 0.8100 | 0.8122 | 65.9 KB | 0.25 ms |
| **Single-Hand Baseline** | 42 Features | 81.25% | 0.7920 | 0.7890 | 0.7905 | 176 KB | 0.28 ms |

---

## 2. Key Technical Findings

1. **Impact of 94-Dimensional Coordinate Enrichment:**
   Upgrading from 42 scalar features to 94 continuous coordinate representations (adding raw origin-centered coordinates, pairwise tip distances, and exact 3D palm normal/longitudinal vectors) boosted classification accuracy from **81.25% to 99.48%**, eliminating ambiguity between fist variations (A, S, T, M, N).

2. **Neural Network vs. Decision Trees:**
   CoordinateMLP outperforms Random Forest by **+1.46%** in accuracy while requiring **88.6% less memory** (2.1 MB vs 18.5 MB) and running **13.7x faster** (0.35 ms vs 4.80 ms). The continuous decision boundaries formed by LayerNorm and ReLU prevent abrupt classification switches during hand transitions.

3. **Temperature Scaling Safety Margin:**
   Setting $T = 1.0000$ ensures calibrated posterior probabilities, producing high confidence ($>0.92$) on true positives while depressing false-positive probabilities on all out-of-class gestures to $<0.15$, well below the 0.70 acceptance threshold.
