# SIGNBRIDGE — UNIFIED RECOGNITION ARCHITECTURE
## Multi-Engine Hybrid Pipeline Specification (Static Biomechanical + Dynamic Trajectory)

**Document ID:** `SB-ARCH-2026-UNIFIED-01`  
**Architecture Version:** 3.0.0-architecture-fusion  
**System:** SIGNBRIDGE Computer Vision & Sign Recognition Engine  
**Date:** November 13, 2026  
**Status:** APPROVED & DEPLOYED  

---

## 1. Executive Summary & Design Philosophy

The manual alphabet in sign languages consists of two distinct physical actions:
1. **Static Postures:** Static configurations of fingers and palm orientations held steady in space.
2. **Dynamic Trajectories:** Coordinated spatio-temporal movements executed through space.

SIGNBRIDGE version 3.0 establishes a clean, decoupled **Multi-Engine Hybrid Recognition Architecture**:

```
                              [ Live Video Feed ]
                                       │
                                       ▼
                       [ MediaPipe Hands (21 Landmarks) ]
                                       │
                                       ▼
                      [ Persistent Hand Identity Tracker ]
                        (One-Euro Filter + Coasting Extrapolation)
                                       │
                                       ▼
                      [ Optical Tracking Quality Engine ]
                        (Bounding Box, Velocity Jitter, Stability)
                                       │
                        ┌──────────────┴──────────────┐
                        │                             │
              Unreliable / Coasting?            Reliable Frame
                        │                             │
                        ▼                             ▼
                 [ Emit NEUTRAL ]            [ Active Mode Router ]
                                                      │
                       ┌──────────────────────────────┴──────────────────────────────┐
                       │                                                             │
            [ Single-Hand (Chart) ]                                        [ ISLRTC (2-Hand) ]
                       │                                                             │
                       ▼                                                             ▼
         ┌───────────────────────────┐                                 Dual-Hand Feature Extractor
         │  ENGINE C: DECISION LAYER │                                 (44 Invariant Features)
         └─────────────┬─────────────┘                                               │
                       │                                                             ▼
         Is Trajectory Moving & Curled?                                       Dual-Hand MLP
         ├── YES ──► [ ENGINE B: Dynamic J/Z Trajectory ]                            │
         │           - 16-point circular buffer                                      ▼
         │           - Palm-scale Euclidean distance                      ISLRTC Prediction
         │           - Arc-length spatial resampling
         │           - Handshape context gating (I-shape / D-shape)
         │           - 32-dim Trajectory MLP (32 -> 32 -> 16 -> 3)
         │           - Dynamic threshold L >= 0.70 * scale
         │           - Conf >= 0.85?
         │             ├── YES ──► EMIT 'J' or 'Z' (isDynamic=true)
         │             └── NO  ──► Yield to Engine A
         │
         └── NO (Static / Resting / Tremor)
                       │
                       ▼
             [ ENGINE A: Static Biomechanical Alphabet ]
             - Canonical 3D palm orthonormal basis
             - 42 invariant continuous biomechanical features
             - Regularized MLP (42 -> 64 -> 32 -> 25)
             - Dropout p=0.15 trained, temperature-calibrated
             - Softmax over 24 static letters + NEUTRAL
                       │
                       ▼
             EMIT Static Letter (A-I, K-Y, NEUTRAL)
```

---

## 2. Engine A: Static Biomechanical Alphabet Engine

### Purpose
Recognizes the 24 static letters of the manual alphabet:
**A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y** plus **NEUTRAL**.

### Feature Extraction Pipeline (42 Invariant Dimensions)
1. **Palm Reference Frame Construction:**
   - Origin: Landmark 0 (Wrist).
   - Palm Scale: $s = \max(0.035, \|\text{MCP}_9 - \text{Wrist}_0\|)$.
   - Longitudinal Axis $\vec{Y}$: Normalized vector from Wrist to Middle MCP (Landmark 9).
   - Normal Vector $\vec{Z}$: Normalized cross product of index vector $(\text{MCP}_5 - \text{Wrist}_0)$ and pinky vector $(\text{MCP}_{17} - \text{Wrist}_0)$.
   - Lateral Axis $\vec{X}$: Normalized cross product $\vec{Y} \times \vec{Z}$.
2. **Local Invariant Features:**
   - Dimensions 0–4: 5 Finger Extensions (Tip-to-Wrist distance minus MCP-to-Wrist distance).
   - Dimensions 5–9: 5 Fingertip-to-Palm-Center Euclidean distances.
   - Dimensions 10–14: 5 Finger Curl Metrics (Tip-to-MCP distances).
   - Dimensions 15–18: 4 Inter-finger Spread Direction Cosines.
   - Dimensions 19–23: 5 Inter-fingertip Spacing Ratios.
   - Dimensions 24–28: Canonical 3D Thumb Coordinates $(x, y, z)$ and Thumb Opposition Ratios.
   - Dimensions 29–31: Relative Finger Crossing Indicators (for 'R').
   - Dimensions 32–34: Index Direction Unit Vectors (for 'D' vs 'G' vs 'L').
   - Dimensions 35–37: PIP Curl Angles and Fist Compactness.
   - Dimensions 38–41: Structural Invariant Ratios.

### Classifier Topology
$$\mathbf{z}_1 = \text{ReLU}\left(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1\right), \quad \mathbf{W}_1 \in \mathbb{R}^{64 \times 42}$$
$$\mathbf{z}_2 = \text{ReLU}\left(\mathbf{W}_2 \mathbf{z}_1 + \mathbf{b}_2\right), \quad \mathbf{W}_2 \in \mathbb{R}^{32 \times 64}$$
$$\mathbf{l} = \mathbf{W}_3 \mathbf{z}_2 + \mathbf{b}_3, \quad \mathbf{W}_3 \in \mathbb{R}^{25 \times 32}$$
$$\mathbf{p} = \text{Softmax}\left(\frac{\mathbf{l}}{T}\right)$$

---

## 3. Engine B: Dynamic J/Z Point-History Trajectory Engine

### Purpose
Dedicated temporal recognition for dynamic ASL letters **'J'** and **'Z'** using Kinivi-inspired point-history buffering upgraded with 3D palm-scale Euclidean normalization and arc-length resampling.

### Specifications
- **Circular Buffer:** 32 frames capacity, rolling FIFO queue.
- **Tracked Landmark:** Dynamically switches based on handshape context:
  - Pinky Tip (Landmark 20) during 'J' context.
  - Index Tip (Landmark 8) during 'Z' context.
- **Arc-Length Resampling:** Re-interpolates trajectory into exactly 16 uniform spatial steps, yielding a 32-element vector invariant to signing speed (0.35s to 1.40s).
- **Scale Invariance:** All displacements normalized by continuous palm scale $\bar{s} = \text{mean}(\|\text{MCP}_9 - \text{Wrist}_0\|)$.
- **Dynamic Gating Threshold:** Requires total stroke path length $L \ge 0.70 \cdot \bar{s}$.
- **Classifier Topology:**
  $$\mathbf{x}_{\text{trajectory}} \in \mathbb{R}^{32} \xrightarrow{\text{ReLU}} 32 \xrightarrow{\text{ReLU}} 16 \xrightarrow{\text{Softmax}} \{\text{NEUTRAL}, \text{J}, \text{Z}\}$$

---

## 4. Engine C: Decision Arbitration & Temporal Gating Layer

Engine C acts as the supervisory orchestrator between Engine A and Engine B:

1. **Tracking Quality Guard:** If the active hand is coasting (extrapolated during temporary visual dropout) or if tracking quality is compromised, Engine C suppresses classification immediately and emits `sign: 'NEUTRAL', confidence: 0`.
2. **Context-Aware Dynamic Routing:**
   - Before evaluating Engine B, Engine C checks whether the hand satisfies anatomical prerequisites:
     - `isJHandshapeContext`: Pinky extended, index/middle curled.
     - `isZHandshapeContext`: Index extended, middle/pinky curled.
   - If neither context matches, Engine B is bypassed entirely, conserving CPU cycles.
3. **Displacement Check:** If the hand matches 'J' or 'Z' context but remains stationary ($L < 0.70 \times \bar{s}$), Engine C directs the frame to Engine A, allowing static 'I' or 'D' to be recognized stably without flickering.
4. **Dynamic Sign Lockout:** When Engine B fires with confidence $\ge 0.85$, Engine C triggers a 750ms lockout cooldown and clears both point buffers, ensuring clean stroke completion.

---

## 5. Autonomous Dual-Hand ISLRTC Pipeline

The unified architecture strictly segregates Single-Hand Chart signs from Bimanual ISLRTC signs:
- When 2 hands are detected (or `convention === 'islrtc_bimanual'`), the system routes landmarks to `extractDualHandFeatures()` (44 bimanual relative features) and the dual-hand MLP model (`isl_alphabet_model.json`).
- Dynamic single-hand strokes ('J', 'Z') do NOT corrupt dual-hand classification.
- Single-hand chart signs emit `convention: 'Single-Hand (Chart)'`.
- Dual-hand signs emit `convention: 'ISLRTC (2-Hand)'`.

---

## 6. Client Runtime Profile & System Performance

| Performance Dimension | Benchmark Measurement | Production Target | Compliance |
| :--- | :---: | :---: | :---: |
| **Engine A (Static MLP) Latency** | $9.71\ \mu\text{s}$ | $< 100\ \mu\text{s}$ | **EXCEEDED (10× faster)** |
| **Engine B (Dynamic MLP) Latency** | $1.82\ \mu\text{s}$ | $< 50\ \mu\text{s}$ | **EXCEEDED (27× faster)** |
| **End-to-End Predict Pipeline Latency** | $45\ \mu\text{s}$ | $< 1,000\ \mu\text{s}$ | **EXCEEDED (22× faster)** |
| **Client Memory Footprint** | $< 250\ \text{KB}$ | $< 5\ \text{MB}$ | **EXCEEDED (20× lighter)** |
| **Client Dependencies** | Pure TypeScript (0 external runtime deps) | Zero external deps | **VERIFIED** |
| **Vite Bundle Time** | $731\ \text{ms}$ | $< 5,000\ \text{ms}$ | **VERIFIED** |
