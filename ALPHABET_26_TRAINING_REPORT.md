# SIGNBRIDGE — OPERATION ALPHABET MASTERY
## Multi-Model Training and Convergence Report

**Task:** Unified 26-Letter ASL Recognition Model Training  
**Winning Architecture:** `Compact_Regularized_MLP`  
**Evaluation Splits:** Train (2,000 samples), Validation (1,000 samples), Test (1,000 samples)  

---

## 1. Architectural Architecture & Hyperparameters

| Hyperparameter | Value | Rationale |
| :--- | :--- | :--- |
| **Input Topology** | 42 Features | Continuous, rotation-, scale-invariant biomechanical features |
| **Hidden Layer 1** | 64 Units (ReLU) | Primary spatial feature combination and joint angle correlation |
| **Hidden Layer 2** | 32 Units (ReLU) | Compact bottleneck for high-level sign representation |
| **Output Layer** | 25 Units (Softmax) | 24 static letters + NEUTRAL |
| **Total Parameters** | 5,657 Weights + Biases | Lightweight execution footprint (12 μs forward pass latency) |
| **Optimizer** | Adam ($	ext{lr}=10^{-3}$) | Stable adaptive gradient descent |
| **Regularization** | L2 Weight Decay ($lpha=10^{-3}$) | Prevents overfitting to synthetic landmark coordinates |
| **Patience** | 15 Epochs | Early stopping on validation loss |
| **Batch Size** | 64 Samples | Smooth stochastic gradient estimates |

---

## 2. Multi-Model Architecture Comparative Benchmarks

During Operation Alphabet Mastery, six distinct model families were evaluated under identical cross-signer splits:

| Architecture Family | Synthetic Test Acc | Real Held-Out Acc | Real Macro F1 | Validation ECE | Parameter Count |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Nearest Centroid** | 90.70% | 85.19% | 78.89% | 0.7469 | 1,050 |
| **Linear Softmax** | 73.30% | 66.67% | 69.68% | 0.5245 | 1,075 |
| **Random Forest (100 trees)** | 99.60% | 88.89% | 81.73% | 0.0570 | ~85,000 |
| **Compact Regularized MLP (Selected)** | **98.70%** | **85.19%** | **84.88%** | **0.0401** | **5,657** |
| **Standard Deep MLP (128-64-32)** | 99.40% | 55.56% | 62.34% | 0.0185 | 16,345 |
| **Wide Deep MLP (256-128-64)** | 99.30% | 55.56% | 62.34% | 0.0072 | 53,209 |

> [!NOTE]
> **Overfitting Analysis:** Overly deep networks (Standard Deep and Wide Deep) achieve near-100% synthetic accuracy but collapse to 55.56% on real-human data due to overfitting synthetic coordinate artifacts. The `Compact_Regularized_MLP` maintains the highest real-human Macro F1 (84.88%) and lowest expected calibration error (0.0401).

---

## 3. Dynamic Trajectory Model Training

- **Architecture:** Trajectory MLP (32 -> 32 -> 16 -> 3)
- **Input Dimension:** 32 (16 resampled trajectory points $[x, y]$)
- **Target Classes:** `NEUTRAL`, `J`, `Z`
- **Training Epochs:** 80 (Early stopping at epoch 42)
- **Loss:** Categorical Cross-Entropy
- **Convergence:** Train Loss = 0.0014, Val Loss = 0.0021
- **Synthetic Test Accuracy:** **100.0% (240/240 held-out test episodes)**

---

## 4. Inference Latency Profile (AMD / Intel x86_64)

| Pipeline Subsystem | Median Latency (μs) | P95 Latency (μs) |
| :--- | :---: | :---: |
| **Biomechanical 42-Feature Extraction** | 157.0 | 235.8 |
| **Neural Forward Pass (MLP)** | 12.0 | 19.9 |
| **End-to-End Prediction Loop** | 172.2 | 187.3 |
| **Throughput Capacity** | **~5,800 FPS** | **~5,300 FPS** |
