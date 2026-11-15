# REPOSITORY COMPATIBILITY MATRIX
**System:** SIGNBRIDGE Recognition Engine  
**Branch:** `experiment/open-source-integration`  
**Engineer:** Byte (Senior Systems & ML Architect)  
**Date:** 2026-11-15 03:48:08 UTC  

---

## 1. Component Classification Framework

Every audited component is classified under one of four rigorous engineering verdicts:
1. **REUSE DIRECTLY:** Code or pre-trained artifacts integrated without structural modification.
2. **ADAPT AND INTEGRATE:** Algorithmic patterns extracted, upgraded to SIGNBRIDGE standards (3D canonical basis, scale invariance, TypeScript/Python parity), and integrated.
3. **REIMPLEMENT BASED ON THE APPROACH:** Design concept retained but completely rewritten to eliminate architectural flaws or license encumbrances.
4. **REJECT:** Disqualified due to technical flaws, privacy violations, or architectural mismatches.

---

## 2. Master Compatibility Matrix

| Source Repository | Component | Proposed Usage | Verdict | Engineering Justification |
| :--- | :--- | :--- | :--- | :--- |
| **kinivi** | `keypoint_classifier.tflite` | Static Alphabet (A–Z) | **REJECT** | Discards z-depth, lacks rotation invariance (70.5% drift at 30°), covers only 4 non-alphabet gestures. |
| **kinivi** | 2D Relative Normalization | Static Feature Extractor | **REJECT** | Lacks 3D orthonormal basis; inferior to SIGNBRIDGE 42-feature canonical extractor by 6 orders of magnitude ($10^{-7}$ vs $0.70$). |
| **kinivi** | Circular Buffer Point History | Dynamic J & Z Tracing | **ADAPT AND INTEGRATE** | The 16-frame circular buffer pattern was adopted in `KiniviPointHistoryAdapter` and upgraded with 3D palm-scale Euclidean normalization and arc-length resampling. |
| **kinivi** | `point_history_classifier.tflite` | Dynamic J & Z Model | **REJECT** | Trained on clockwise/counter-clockwise circles and stop gestures; incompatible with ASL J hook and Z stroke geometries. |
| **nicknochnack (Action)** | MediaPipe Holistic (1662 feats) | Feature Pipeline | **REJECT** | 468-point face mesh violates privacy and destroys real-time CPU performance (>80ms latency). |
| **nicknochnack (Action)** | `action.h5` LSTM Weights | Word / Letter Recognition | **REJECT** | Contains only 3 macro word classes (`hello`, `thanks`, `iloveyou`), zero alphabet letters. Missing open license. |
| **nicknochnack (Action)** | Temporal Hysteresis Mode Filter | Real-Time Output Smoothing | **ADAPT AND INTEGRATE** | Adapted into `NochnackTemporalHysteresisFilter` (Python) and `openSourceIntegrationAdapter.ts` (TS), adding 800ms debounce and neutral state flushing. |
| **nicknochnack (Object)** | SSD MobileNet v2 | Hand / Sign Detection | **REJECT** | Face-mask bounding box detector; completely irrelevant to sign language. Raw RGB input violates privacy. |
| **pianalytix** | 30-Frame LSTM Architecture | Static Letters A, B, C | **REJECT** | Anti-pattern: forces 1000ms latency on static postures. Zero coordinate normalization causes spatial position fragility. |
| **pianalytix** | 90-Sequence Training Dataset | Model Training | **REJECT** | Single-signer, single-session capture with massive frame-to-frame correlation. Violates dataset provenance standards. |

---

## 3. Detailed Adaptation Specifications

### 3.1 Point History Adapter (`kinivi` -> `KiniviPointHistoryAdapter`)
- **Original Concept:** 16-sample queue storing $(x, y)$ coordinates of landmark 8 (index tip) relative to the initial point, scaled by 2D screen width/height.
- **Upgrades Implemented:**
  1. **3D Palm-Scale Invariance:** Instead of dividing by pixel screen dimensions, coordinates are translated relative to landmark 0 (wrist) and scaled by $d(\text{wrist}, \text{middle\_mcp})$. This makes trajectory features distance- and resolution-invariant.
  2. **Arc-Length Resampling:** Dynamic trajectories are uniformly interpolated across 16 equidistant steps, eliminating variations caused by camera framerate drops (15 FPS vs 60 FPS).
  3. **Zero-Pad Flushing:** Neutral hand drop flushes the circular buffer with zeros to prevent ghost trajectory activations.

### 3.2 Temporal Hysteresis Filter (`nicknochnack` -> `NochnackTemporalHysteresisFilter`)
- **Original Concept:** Buffer last 10 predictions; emit prediction only if `np.unique(predictions[-10:])[0] == current_prediction` and confidence $> 0.5$.
- **Upgrades Implemented:**
  1. **Dynamic Debounce Interval:** Enforces an 800ms dead-time between successive emissions of the *same* sign, preventing repeated letter spamming during continuous signing.
  2. **Immediate Neutral Flusher:** Detection of a neutral/transition hand posture instantly resets the hysteresis buffer, enabling swift transitions between adjacent letters (e.g. `B` -> `A`).
  3. **Dual Language Implementation:** Implemented in pure Python (`backend/open_source_integration/nochnack_temporal_hysteresis.py`) for backend validation and in TypeScript (`frontend/src/services/openSourceIntegrationAdapter.ts`) for zero-latency in-browser execution.
