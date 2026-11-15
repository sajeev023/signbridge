# SIGNBRIDGE INTEGRATION ARCHITECTURE
**System:** SIGNBRIDGE Recognition Engine  
**Branch:** `experiment/open-source-integration`  
**Engineer:** Byte (Senior Systems & ML Architect)  
**Date:** 2026-11-15 03:48:08 UTC  

---

## 1. System Architecture Overview

The upgraded SIGNBRIDGE recognition engine combines four decoupled stages:
1. **3D Orthonormal Canonical Landmark Extraction (SIGNBRIDGE Native)**
2. **Dual-Path Static & Specialist Inference (SIGNBRIDGE Multi-Signer MLP + Frozen Specialist)**
3. **Trajectory-Resampled Point History Engine for J & Z (Adapted from Kinivi with 3D Palm Invariance)**
4. **Temporal Hysteresis Mode Filter (Adapted from Nochnack with 800ms Debounce and Flush)**

```
                      +-----------------------------+
                      | MediaPipe Hands (21 3D LMs) |
                      +--------------+--------------+
                                     |
                                     v
                  +-----------------------------------+
                  | Canonical 3D Orthonormal Basis    |
                  | Translation, Gram-Schmidt, Scale  |
                  +------------------+----------------+
                                     |
                +--------------------+--------------------+
                |                                         |
     (Static Features: 42D)                   (Index Tip Trajectory: 3D)
                |                                         |
                v                                         v
+-------------------------------+       +-----------------------------------+
| 24-Letter Static MLP Ensemble |       | Kinivi-Adapted Point History      |
| + Letter A Specialist (Frozen)|       | 16-Frame Buffer + Arc Resampling  |
+---------------+---------------+       +-----------------+-----------------+
                |                                         |
                | (Static Letter + Conf)                  | (Dynamic J / Z + Conf)
                +--------------------+--------------------+
                                     |
                                     v
                  +-----------------------------------+
                  | Dynamic-over-Static Priority Gate |
                  | (Routes J/Z when trajectory fires)|
                  +------------------+----------------+
                                     |
                                     v
                  +-----------------------------------+
                  | Nochnack Temporal Hysteresis      |
                  | K-Frame Mode Agreement + Debounce |
                  | + Neutral State Reset             |
                  +------------------+----------------+
                                     |
                                     v
                        Verified Letter Output Stream
```

---

## 2. Decoupled Pipeline Modalities

To prevent cross-talk and performance degradation, SIGNBRIDGE enforces strict modality isolation:
1. **Static Alphabet Recognition (A–I, K–Y):** Evaluated strictly on single-frame 42D canonical features. Never delayed by sequence windows.
2. **Dynamic Alphabet Recognition (J, Z):** Evaluated strictly on the index fingertip circular trajectory buffer when kinematic motion exceeds the activation threshold.
3. **ISLRTC Whole-Body Two-Handed Engine:** Retained as a separate, uncoupled module for continuous Indian Sign Language phrases.
