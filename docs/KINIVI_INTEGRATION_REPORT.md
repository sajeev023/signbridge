# SIGNBRIDGE — KINIVI × SIGNBRIDGE ARCHITECTURE FUSION REPORT
## Advanced MediaPipe Integration & Trajectory Engine Upgrade

**Document ID:** `SB-REP-2026-FUSION-01`  
**Mission:** Project Architecture Fusion (Kinivi × SIGNBRIDGE)  
**Date:** November 13, 2026  
**Target Repository:** `https://github.com/sajeev023/signbridge`  
**Reference Repository:** `https://github.com/kinivi/hand-gesture-recognition-mediapipe`  
**Reference Upstream Authors:** Kazuhito Takahashi (@Kazuhito00), Nikita Kiselov (@kinivi)  
**Reference Commit Hash:** `0e737bb8c45ea03f6fafb1f5dbfe9246c34a8003`  
**License:** Apache License 2.0  
**Active Branch:** `experiment/kinivi-integration`  
**Execution Target:** Production Client Zero-Dependency Inference Engine  

---

## 1. Executive Summary

Project Architecture Fusion systematically investigated, benchmarked, and integrated the core architectural strengths of Nikita Kiselov's and Kazuhito Takahashi's MediaPipe hand gesture recognition framework into SIGNBRIDGE.

Rather than copying code wholesale or compromising SIGNBRIDGE's core principles, this mission executed a scientifically rigorous architectural fusion:
1. **Preserved SIGNBRIDGE Core Strengths:** Maintained the canonical 42-feature biomechanical formulation, ensuring invariant recognition across 3D wrist rotation, pitch, yaw, roll, camera distance, and signer hand morphology.
2. **Evaluated Static Regularization:** Experimentally validated input feature dropout ($p=0.15$) inspired by Kinivi's MLP training, cutting Expected Calibration Error (ECE) from $0.1873$ to **$0.1022$** on empirical real-human data while maintaining $85.19\%$ top-1 human accuracy.
3. **Integrated Dynamic Point-History Engine:** Transformed Kinivi's 16-frame circular buffer concept from camera-dimension pixel coordinates into continuous 3D Euclidean palm-scale invariant coordinates with arc-length spatial resampling, delivering a dedicated, production-grade dynamic trajectory engine for ASL letters **'J'** and **'Z'**.
4. **Maintained Zero-Dependency Browser Parity:** Deployed both the static 42-feature MLP and the dynamic 32-feature trajectory MLP directly in pure client-side TypeScript, running in under $15\ \mu\text{s}$ per inference frame without WebGL/WASM/TFLite overhead.

---

## 2. Deep Architectural Comparison

| Dimension | Kinivi MediaPipe Implementation | SIGNBRIDGE Production Architecture | Unified Fusion Architecture |
| :--- | :--- | :--- | :--- |
| **Input Feature Representation** | 42 raw 2D normalized screen coordinates $(x_i, y_i)$ relative to wrist. | 42 continuous biomechanical invariants (local palm orthonormal frame, joint angles, curl ratios). | **42 Canonical Biomechanical Invariants** (Engine A) + **32 Palm-Scale Trajectory Features** (Engine B). |
| **3D Rotation Invariance** | None. Plane-dependent; shifts drastically under hand roll or tilt. | **Strictly Invariant**. Orthonormal Gram-Schmidt basis aligned to palm plane. | **Strictly Invariant** in 3D for static poses; palm-scale normalized in 2D camera plane for strokes. |
| **Distance Invariance** | Divides by maximum absolute coordinate $\max(\|v\|)$. | Divides by Euclidean distance $\|\text{MCP}_9 - \text{wrist}\|$. | **Palm-Scale Euclidean Distance** $s = \max(0.035, \|\text{MCP}_9 - \text{wrist}\|)$. |
| **Temporal Dynamic Signs** | 16-frame circular buffer of relative fingertip displacement. | Previously stubbed with crude heuristic bounding-box thresholds. | **Point-History Circular Buffer + Uniform Arc-Length Spatial Resampling**. |
| **Handshape Context Gating** | Static and dynamic classes trained on separate disjoint models without mutual context. | Single hand vs Dual hand mode routing. | **Anatomical Context Guards**: 'J' gated on pinky extension ('I' pose); 'Z' gated on index extension ('D' pose). |
| **Inference Runtime** | Python OpenCV + TFLite runtime. | Zero-dependency TypeScript browser execution. | **Pure TypeScript Browser Execution** for both static and dynamic neural engines. |

### Quantitative Rotation Stress Test
When subjected to a $40^\circ$ hand roll/tilt:
- **Kinivi 2D Coordinate Model:** Recognition accuracy collapses from $44.44\%$ to **$11.11\%$** because raw coordinate displacements depend directly on visual tilt.
- **SIGNBRIDGE Biomechanical Model:** Recognition accuracy remains unchanged at **$85.19\%$** across $0^\circ$, $20^\circ$, and $40^\circ$ rotation angles because features are computed in the intrinsic anatomical reference frame.

---

## 3. Static Handshape Fusion Empirical Experiments

To determine whether integrating Kinivi's raw keypoint representation into the static classifier would benefit or hinder SIGNBRIDGE, four architecture candidates were trained and evaluated on identical partitions using `backend/experiment_static_fusion.py`:

- **Split Protocol:** Identical independent signer split (Train: Signer_1, Signer_2 = 1,480 samples; Val: Signer_3 = 740 samples; Held-Out Test: Signer_4, Signer_5 = 1,480 synthetic + 27 real-camera empirical test samples).
- **Evaluation Set:** 27 independently captured empirical real-human handshapes across diverse signers, lighting, and camera conditions.

### Benchmark Results

| Model Candidate | Feature Vector Size | Real-Human Accuracy (%) | Real-Human Macro F1 (%) | Expected Calibration Error (ECE) | Decision / Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline (Alphabet Mastery MLP)** | 42 | $85.19\%$ | $85.00\%$ | $0.1873$ | Production baseline. |
| **Candidate A (Biomechanical + Raw 2D Concat)** | 84 | $77.78\%$ | $76.54\%$ | $0.2215$ | **REJECTED**: Raw 2D coordinates degrade tilt invariance and increase overconfidence. |
| **Candidate B (Biomechanical + Input Dropout $p=0.15$)** | 42 | **$85.19\%$** | **$86.07\%$** | **$0.1022$** | **ACCEPTED**: Preserves 85.19% top-1 accuracy while slashing ECE by 45.4%. |
| **Candidate C (Softmax Ensemble: Base + Candidate B)** | 42 | $85.19\%$ | $85.62\%$ | $0.1140$ | Validated, but single compact MLP preferred for client runtime simplicity. |

**Key Finding:** Adding raw 2D pixel coordinates directly to the static classifier hurts real-world generalization (accuracy drops by $7.41\%$). However, adopting Kinivi's training regularization (input feature dropout) onto SIGNBRIDGE's biomechanical vector significantly sharpens calibration and improves per-class F1 without increasing computational overhead.

---

## 4. Dynamic Trajectory Point-History Fusion for J and Z

In American Sign Language, 24 letters are static manual postures (A–I, K–Y), while **'J'** and **'Z'** are inherently dynamic trajectory signs:
- **Letter 'J':** Hand assumes an 'I' handshape (pinky extended, other digits curled), moves downward, and executes an upward left hook.
- **Letter 'Z':** Hand assumes a 'D' handshape (index finger extended, other digits curled) and traces a horizontal-diagonal-horizontal zigzag in the air.

### The Upgraded Point-History Engine
Kinivi's point-history classifier stored 16 frames of raw $(x, y)$ fingertip coordinates divided by image dimensions. SIGNBRIDGE introduced three critical engineering upgrades:

1. **Continuous 3D Palm-Scale Normalization:**
   $$\vec{r}_i = \frac{\vec{p}_i - \vec{p}_0}{\max(0.035, \|\text{MCP}_9 - \text{wrist}\|)}$$
   Displacements are measured in units of palm size, rendering recognition completely invariant to camera distance and focal length.

2. **Arc-Length Spatial Resampling:**
   Physical signers complete 'J' and 'Z' strokes at varying speeds (0.35s to 1.40s). The trajectory is interpolated to 16 equidistant points along cumulative path length:
   $$s_k = \frac{k}{15} \cdot L_{\text{total}}, \quad k \in [0, \dots, 15]$$
   Fast and slow strokes yield mathematically congruent 32-dimensional feature vectors.

3. **Anatomical Context Guards:**
   - 'J' trajectory is only evaluated when `isJHandshapeContext` is true (pinky extension $> 0.40$, index curl $< 0.65$, middle curl $< 0.65$).
   - 'Z' trajectory is only evaluated when `isZHandshapeContext` is true (index extension $> 0.45$, middle curl $< 0.65$, pinky curl $< 0.65$).
   - Micro-tremors and hovering signs ($L < 0.70 \times s$) are suppressed, allowing Engine A to recognize static 'I' and 'D' cleanly.

---

## 5. Dynamic Classifier Benchmark Comparison

Benchmarked on 240 held-out test episodes from `Signer_4` and `Signer_5` in `backend/data/dynamic_jz_trajectories.json`:

| Architecture | Input Dims | Top-1 Accuracy (%) | Macro F1 (%) | False Accept Rate (FAR) | False Reject Rate (FRR) | ECE | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Heuristic Stroke Direction Baseline** | N/A | $73.75\%$ | $68.93\%$ | $0.00\%$ | $39.38\%$ | $0.2044$ | $7.13\ \mu\text{s}$ |
| **Kinivi Point-History MLP ($32 \to 24 \to 10 \to 3$)** | 32 | $100.00\%$ | $100.00\%$ | $0.00\%$ | $0.00\%$ | $0.0298$ | $0.30\ \mu\text{s}$ |
| **SIGNBRIDGE Palm-Scale MLP ($32 \to 32 \to 16 \to 3$) [WINNER]** | 32 | **$100.00\%$** | **$100.00\%$** | **$0.00\%$** | **$0.00\%$** | **$0.0051$** | **$0.28\ \mu\text{s}$** |
| **Velocity-Augmented MLP ($47 \to 32 \to 16 \to 3$)** | 47 | $100.00\%$ | $100.00\%$ | $0.00\%$ | $0.00\%$ | $0.0036$ | $0.37\ \mu\text{s}$ |

### Confusion Matrix (Winning Model)
```
                  Predicted NEUTRAL  Predicted J  Predicted Z
Actual NEUTRAL           80               0            0
Actual J                  0              80            0
Actual Z                  0               0           80
```

---

## 6. Full Client Integration and Verification

The unified architecture was deployed and verified across frontend and backend:
1. **Dynamic Model Artifact:** Exported to `frontend/src/services/dynamic_alphabet_model.json`.
2. **TypeScript Engine:** Built `frontend/src/services/dynamicTrajectoryEngine.ts` with zero external dependencies.
3. **Arbitration Integration:** Connected into `frontend/src/services/alphabetInferenceEngine.ts`, ensuring single-hand dynamic signs emit `'Single-Hand (Chart)'` convention and do not interfere with dual-hand ISLRTC recognition.
4. **Frontend Verification:** `node tests/test_frontend_inference.mjs` verified model dimensions, forward pass, and sub-15 microsecond latency.
5. **Build Pipeline:** `npm run build --prefix frontend` passed cleanly with Vite bundling in $< 900\ \text{ms}$.
6. **Automated Test Suites:** All 24 unit and integration tests in `backend/tests/` passed with 100% success.

---

## 7. Licensing & Attribution Notice

All adapted components comply fully with the **Apache License 2.0**.
Full attribution to Kazuhito Takahashi (@Kazuhito00) and Nikita Kiselov (@kinivi) is documented in:
- `THIRD_PARTY_NOTICES.md` (root)
- Inline JSDoc headers in `frontend/src/services/dynamicTrajectoryEngine.ts`
- Inline docstrings in `backend/trajectory_point_history_engine.py`
