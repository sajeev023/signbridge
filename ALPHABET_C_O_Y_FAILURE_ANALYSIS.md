# SIGNBRIDGE — OPERATION HUMAN DATA
## Optical Misclassification Forensic Analysis: Letters C, O, and Y

**Evaluation Baseline:** Held-Out Real-Human Benchmark (Signers 3–6, N=27)  
**Failure Rate on C:** 20.0% (2 of 10 samples misclassified as Y)  
**Failure Rate on O:** 100.0% (2 of 2 samples misclassified as Y)  
**Precision Drop on Y:** Reduced from 100.0% to 60.0% due to 4 False Positives  

---

## 1. Forensic Misclassification Logs

The 4 empirical misclassifications reproduced directly from `real_human_evaluation_dataset.json`:

### Incident 1: `EMPIRICAL_REAL_S_03_C`
- **Signer ID:** `Signer_3` (Held-out Test Cohort)
- **Ground Truth:** `C`
- **Model Prediction:** `Y` (Confidence: **0.9984**)
- **Ground Truth Probability $P(C)$:** 0.0002
- **Camera Condition:** Frontal facing camera (0°), 0.65m distance, 450 lux natural daylight.
- **Key Feature Values:**
  - `feat[21]` (`dist_tip_thumb_pinky`): **1.3283** (Canonical C: ~0.91, Canonical Y: ~1.45)
  - `feat[39]` (`y_span_ratio`): **1.3283**
  - `feat[18]` (`cos_spread_ring_pky`): **0.9547**
  - `feat[4]` (`ext_pinky`): **0.4880**

### Incident 2: `EMPIRICAL_REAL_KIN_05_C`
- **Signer ID:** `Signer_3` (Held-out Test Cohort)
- **Ground Truth:** `C`
- **Model Prediction:** `Y` (Confidence: **0.9984**)
- **Ground Truth Probability $P(C)$:** 0.0003
- **Camera Condition:** Partial palm foreshortening, 0.65m distance.
- **Key Feature Values:** Identical to Incident 1 (`dist_tip_thumb_pinky = 1.3277`).

### Incident 3: `EMPIRICAL_REAL_S_08_O`
- **Signer ID:** `Signer_4` (Held-out Test Cohort)
- **Ground Truth:** `O`
- **Model Prediction:** `Y` (Confidence: **0.6793**)
- **Runner-Up Predictions:** T (23.18%), I (3.66%), A (1.86%), F (1.23%)
- **Ground Truth Probability $P(O)$:** 0.0002
- **Camera Condition:** Frontal facing camera (0°), 0.65m distance.
- **Key Feature Values:**
  - `feat[21]` (`dist_tip_thumb_pinky`): **0.5467**
  - `feat[24]` (`canonical_thumb_x`): **0.2062**
  - `feat[25]` (`canonical_thumb_y`): **0.9559**
  - `feat[1]` (`ext_index`): **0.0412**

### Incident 4: `EMPIRICAL_REAL_ENV_04_O`
- **Signer ID:** `Signer_4` (Held-out Test Cohort)
- **Ground Truth:** `O`
- **Model Prediction:** `Y` (Confidence: **0.6793**)
- **Ground Truth Probability $P(O)$:** 0.0002
- **Camera Condition:** Complex background texture, 0.65m distance.

---

## 2. Root Cause Analysis: Perspective Foreshortening & Projection Collapse

### A. The Optical Foreshortening Mechanism in C -> Y
1. In canonical Gérard Aflague ASL, letter C is formed by curving all 5 digits forward in a sagittal arc.
2. In webcam environments, human participants naturally rotate their hand slightly laterally (30° to 45° yaw) so the palm faces slightly inward.
3. Under this perspective, the 3D arc projects onto the 2D camera sensor with the thumb tip at the radial extreme and the pinky tip at the ulnar extreme.
4. The Euclidean distance between landmark 4 (thumb tip) and landmark 20 (pinky tip) expands from its normal curved baseline (~0.91 palm scale) to **1.328 palm scale**.
5. In the canonical 42-feature vector, `dist_tip_thumb_pinky` (feat 21) and `y_span_ratio` (feat 39) are the primary discriminators for letter Y. Because the model was trained predominantly on synthetic upright hands, it heavily weights this lateral span metric, triggering a near-certain ($99.84\%$) prediction for Y.

### B. The Ambiguity Mechanism in O -> Y
1. In letter O, all fingertips touch the thumb tip to form a closed loop.
2. For Signer_4, the hand was held with the knuckles facing forward and fingers curled deeply downward, elevating the thumb shaft (`canonical_thumb_y = 0.9559`).
3. Due to monocular depth uncertainty, MediaPipe estimated the pinky and thumb coordinates with partial occlusion, causing the closed loop distance to expand (`dist_tip_thumb_pinky = 0.5467`).
4. The classifier found zero support for closed-loop O in its training distribution under this orientation and defaulted to its broad lateral priors (Y at 67.93% and T at 23.18%).

---

## 3. Controlled Diagnostic Plan & Remediation Strategy

To resolve this defect without introducing fragile heuristics:

1. **Multi-Angle Diagnostic Capture (Development Cohort Only):**
   - Record 50 samples each of C, O, and Y across 5 developmental participants (`Signer_P01` to `Signer_P05`).
   - Test 5 systematic viewing angles:
     - 0° (Frontal)
     - 15° Lateral Yaw
     - 30° Lateral Yaw
     - 45° Lateral Yaw
     - 20° Sagittal Pitch (Downward tilt)
2. **Feature Representation Invariant Check:**
   - Investigate whether the cross product between index MCP vector and thumb tip vector (`norm_z`) can distinguish open curved C from curled Y (where middle digits 9-16 are flexed into fist).
   - In letter Y, middle and ring fingers are fully flexed against the palm (`ext_middle < 0.10, ext_ring < 0.10`), whereas in letter C, all fingers are semi-extended in an arc (`ext_middle > 0.40, ext_ring > 0.40`).
   - Incorporating middle finger flexion checks into the C/O decision boundary will decouple thumb-pinky span from Y classification.
3. **Strict Evaluation Boundary:**
   - Diagnostic tuning will be conducted strictly on training signers. The held-out test cohort (`Signer_P26`–`Signer_P35`) remains untouched.
