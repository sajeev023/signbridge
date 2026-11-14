# SIGNBRIDGE — OPERATION ALPHABET MASTERY
## Dynamic Alphabet Letters J and Z Validation Report

**Scope:** Temporal point-history trajectory architecture, handshape context gating, speed/scale invariance, and empirical fidelity for ASL dynamic letters 'J' and 'Z'.

---

## 1. Kinematic & Anatomical Specification

### Letter J Kinematic Specification
- **Initial Handshape Context:** ASL letter 'I' handshape:
  - Pinky finger extended upright ($ext_{pinky} > 0.40$).
  - Index and middle fingers flexed tightly into palm ($curl_{idx} < 0.65, curl_{mid} < 0.65$).
  - Thumb folded across curled digits.
- **Trajectory Geometry:** Pinky fingertip traces a downward vertical stroke ($>0.08$ normalized displacement), followed by an inward/upward medial hook arc.
- **Temporal Duration:** 350 ms to 950 ms.

### Letter Z Kinematic Specification
- **Initial Handshape Context:** ASL letter 'D' / '1' handshape:
  - Index finger extended upright ($ext_{idx} > 0.45$).
  - Middle and pinky fingers flexed into palm ($curl_{mid} < 0.65, curl_{pinky} < 0.65$).
  - Thumb opposed or folded.
- **Trajectory Geometry:** Index fingertip traces a 3-stroke zigzag in the sagittal-coronal plane:
  - Stroke 1: Horizontal shift from left to right.
  - Stroke 2: Diagonal downward stroke from right to left.
  - Stroke 3: Horizontal shift from left to right.
- **Temporal Duration:** 450 ms to 1,200 ms.

---

## 2. Dynamic Trajectory Point-History Architecture

```
[MediaPipe 21 Landmarks]
           │
           ▼
[Handshape Context Gate] ─── Fails Context ───► [Reject Dynamic Path]
           │ Passes Context
           ▼
[Circular Point-History Buffer] (Length = 16 frames, Δt <= 300ms)
           │
           ▼
[Palm-Scale Normalization] (d_euclidean / scale_palm)
           │
           ▼
[Arc-Length Spatial Resampling] (Uniform 16 points -> 32 features [x, y])
           │
           ▼
[Path Length Threshold Check] (Path >= 0.70x palmScale)
           │ Passes Displacement
           ▼
[Trajectory MLP Classifier] (32 -> 32 -> 16 -> 3)
           │
           ▼
[Output: J / Z / NEUTRAL] (Confidence >= 0.85)
```

---

## 3. Empirical Stress Scenarios & Robustness Testing

16 automated test scenarios in `backend/tests/test_dynamic_trajectory_suite.py` and `test_dynamic_stress_scenarios.py` confirm:

| Scenario / Stress Test | Input Condition | Expected Behavior | Measured Result |
| :--- | :--- | :--- | :---: |
| **Variable Speed J** | Drawing speed 0.5x, 1.0x, 2.0x, 2.5x | Arc-length resampling normalizes temporal variation | **PASSED (100%)** |
| **Variable Speed Z** | Drawing speed 0.5x, 1.0x, 2.0x, 2.5x | Arc-length resampling normalizes temporal variation | **PASSED (100%)** |
| **Static Letter I Isolation** | Stationary 'I' hand held for 3 seconds | Path length < 0.20; dynamic engine suppressed | **PASSED (100%)** |
| **Static Letter D Isolation** | Stationary 'D' hand held for 3 seconds | Path length < 0.20; dynamic engine suppressed | **PASSED (100%)** |
| **Incomplete Strokes** | Aborted J hook or 1-stroke partial Z | Path length < 0.70 or rejected by MLP | **PASSED (100%)** |
| **Inverted Trajectories** | Backward J (upward) or inverted Z | Neural classifier outputs NEUTRAL | **PASSED (100%)** |
| **Micro-Tremor Suppression** | Physiological hand tremor ($\sigma=0.015$) | Cumulative displacement remains sub-threshold | **PASSED (100%)** |
| **Sensor Dropout Recovery** | Missing tracking frames for 250 ms | OneEuroFilter coasts; buffer maintains continuity | **PASSED (100%)** |
| **Hand Loss / Reacquisition**| Tracking lost for > 300 ms | Buffer resets instantly; prevents stale cross-hand stroke | **PASSED (100%)** |
| **Camera Distance Invariance**| Palm scale scaled 0.5x to 2.0x | Euclidean normalization eliminates camera depth scale | **PASSED (100%)** |

---

## 4. Dataset Provenance Disclosure

> [!WARNING]
> **Data Provenance Reality:**  
> All 600 dynamic episodes in `backend/data/dynamic_jz_trajectories.json` are **synthetic kinematic trajectories** generated via biomechanical simulation. There are currently **ZERO genuine physical video recordings** of human signers performing J and Z in the repository. Full certification for J and Z remains blocked pending field collection.
