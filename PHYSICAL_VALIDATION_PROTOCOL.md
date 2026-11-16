# SIGNBRIDGE PHYSICAL VALIDATION PROTOCOL
**System:** SIGNBRIDGE ASL Recognition Pipeline  
**Document Code:** `SOP-VALIDATION-CV-002`  
**Engineer:** Senior QA & Computer Vision Validation Engineer  
**Date:** 2026-11-15 04:05:42 UTC  
**Status:** ACTIVE TESTING SPECIFICATION (READY FOR HUMAN PARTICIPANTS)  

---

## 1. Ethical Governance & Participant Protections

To guarantee zero data leakage and protect participant privacy, the physical camera validation protocol strictly prohibits storing raw sensor data.

```
                    +------------------------------------+
                    |   Live USB / Web Camera Feed       |
                    +-----------------+------------------+
                                      |
                                      v
                    +------------------------------------+
                    | MediaPipe Hands (Client-Side Memory|
                    | Frame Processed & Immediately Lost |
                    +-----------------+------------------+
                                      |
                                      v
     [ STRICT PRIVACY GATE: NO RGB, NO FACE MESH, NO AUDIO SAVED ]
                                      |
                                      v
                    +------------------------------------+
                    | 21 3D Landmark Floats (x, y, z)    |
                    | + 16-Point Fingertip Trajectory    |
                    +-----------------+------------------+
                                      |
                                      v
                    +------------------------------------+
                    | Encrypted Session JSON Manifest    |
                    | Pseudonymized (e.g., P_SIGNER_01)  |
                    +------------------------------------+
```

### 1.1 Mandatory Informed Consent Gate
Before any session begins, the participant must review and sign the informed consent agreement. The automated runner [session_protocol.py](file:///c:/Users/LFDC/Downloads/signbridge-main/backend/validation/session_protocol.py) will immediately terminate with an exception if `informed_consent_granted=False`.

### 1.2 Participant Pseudonymization
All sessions must use standardized pseudonym format:
- Training Cohort: `P_TRAIN_01` through `P_TRAIN_05`
- Held-Out Evaluation Cohort: `P_TEST_01` through `P_TEST_03`
No real names, emails, IP addresses, or demographic identifiers may be recorded in session manifests.

### 1.3 Cohort Separation Rule
Participants assigned to the `TRAIN_COHORT` cannot be used for `HELDOUT_EVAL_COHORT` benchmarks. Multi-signer generalization must be proven strictly on un-encountered hands.

---

## 2. Test Environment & Parameter Grid

Each physical participant must be recorded across a structured 36-trial parameter grid:

| Parameter Dimension | Allowed Levels | Validation Goal |
| :--- | :--- | :--- |
| **Camera Distance** | `near_35cm` (30–40 cm) | Evaluates large palm-scale invariance |
| | `medium_55cm` (50–65 cm) | Standard nominal operating range |
| | `far_85cm` (80–95 cm) | Evaluates low-resolution landmark tracking |
| **Camera Orientation**| `frontal` ($0^\circ$) | Baseline optical axis |
| | `15deg_pitch` (Tilted up/down)| Evaluates vertical perspective foreshortening |
| | `15deg_roll` (Hand tilted $\pm 15^\circ$)| Evaluates orthonormal rotational invariance |
| | `15deg_yaw` (Rotated side-to-side)| Evaluates out-of-plane finger depth occlusion |
| **Lighting Tier** | `normal_indoor` (300–500 lux)| Standard office / home lighting |
| | `dim_warm` (80–150 lux) | Low light with shadow casting |
| | `bright_overhead` (> 800 lux)| High contrast / skin reflectance wash |

---

## 3. Standard Operating Procedure (Step-by-Step)

### Step 1: Pre-Flight Camera Calibration
1. Launch ZeroLeak Camera Studio.
2. Verify frame rate is $\ge 25	ext{ FPS}$.
3. Verify hand landmark visibility ratio is $\ge 0.90$.
4. Ensure camera lens is clean and focal point is locked on the hand.

### Step 2: Static Sign Acquisition (Letters A–I, K–Y)
1. Operator announces target letter (e.g., "Letter G").
2. Participant starts from a relaxed resting hand position (`NEUTRAL`).
3. Participant forms the requested handshape and holds it steadily for 3.0 seconds in front of the camera.
4. System records 21 landmarks, predicted sign, confidence score, and status (`EMITTED` / `HOLDING`).
5. Participant returns hand to resting position (`NEUTRAL`) for 1.0 second.

### Step 3: Dynamic Sign Acquisition (Letters J and Z)
1. Operator announces dynamic letter ("Letter J" or "Letter Z").
2. Participant assumes initial handshape context:
   - For `J`: Pinky extended, other fingers curled.
   - For `Z`: Index extended, other fingers curled.
3. Participant traces the stroke in air:
   - For `J`: Downward stroke with an upward hook to the left.
   - For `Z`: Horizontal stroke, diagonal down-left, horizontal right.
4. System records 16-point circular buffer trajectory and validates kinematic bounds.
5. Participant returns to `NEUTRAL`.

### Step 4: Session Persistence
At the completion of all 26 classes, execute `session.save_session()`. The manifest is automatically verified and saved to `backend/data/physical_validation_sessions/<pseudonym>_<timestamp>.json`.
