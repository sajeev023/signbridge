# SIGNBRIDGE — ASL LETTER A SPECIALIST
## Physical Human Validation Protocol & Empirical Study Plan

**Branch**: `experiment/letter-a-mastery`  
**Protocol Version**: 1.0.0-FROZEN  
**Target Subject**: ASL Manual Alphabet — Isolated Letter A Recognition  
**Status**: APPROVED FOR PHYSICAL DATA COLLECTION  

---

### 1. Objective & Scope

Establish the empirical generalization, discrimination resilience, and calibration of the frozen Letter A Specialist architecture across $\ge 10$ unobserved human participants under live camera capture.

This protocol prohibits post-hoc threshold adjustment, model retraining, or selective sample omission.

---

### 2. Participant Cohort & Non-PII Safeguards

#### Participant Allocation
- **Cohort Range**: 10 distinct human participants assigned pseudonymous identifiers `Signer_P26` through `Signer_P35`.
- **Demographic & Physiological Diversity**: Variation in hand morphology (palm width, finger length, aspect ratio), skin tone, and joint flexibility.
- **Handedness**: Both dominant and non-dominant hands tested ($70\%$ right-hand, $30\%$ left-hand).

#### Privacy & Zero-Leak Safeguards
- **Zero Raw Video Retention**: Camera frames are processed strictly in volatile browser memory. No RGB frames, images, audio, or face landmarks are written to disk or transmitted over network.
- **Landmark-Only Serialization**: Captured data records contain only 21 3D coordinate triplets $(x, y, z)$, 42 extracted biomechanical features, timestamp, signer ID, and ground truth label.
- **Local Storage**: All exports remain client-side in browser local storage or downloadable JSON format.

---

### 3. Sampling Design & Quotas

Each participant executes a structured session generating a minimum of 70 validated samples:

| Target Class | Sample Quota per Signer | Gesture Specifications | Permitted Perturbations |
| :--- | :--- | :--- | :--- |
| **Letter A (Positive)** | **$\ge 30$** | Fist curled, thumb upright and flush against radial margin of index finger | Camera yaw $\pm 15^\circ$, pitch $\pm 10^\circ$, distance $0.4\text{m}$–$1.0\text{m}$ |
| **Letter E (Hard Negative)** | $\ge 8$ | 4 fingers folded down tightly, thumb folded horizontally across palm base | Normal frontal signing |
| **Letter S (Hard Negative)** | $\ge 8$ | Tight fist with thumb crossed horizontally across fingers | Normal frontal signing |
| **Letter M (Hard Negative)** | $\ge 8$ | Fist with thumb tucked beneath index, middle, and ring fingers | Normal frontal signing |
| **Letter N (Hard Negative)** | $\ge 8$ | Fist with thumb tucked beneath index and middle fingers | Normal frontal signing |
| **Letter T (Hard Negative)** | $\ge 8$ | Fist with thumb tucked beneath index finger only | Normal frontal signing |
| **Per-Signer Minimum Total** | **70 samples** | **30 A + 40 Hard Negatives** | |
| **Cohort Grand Total ($N=10$)** | **$\ge 700$ samples** | **$\ge 300$ A + $\ge 400$ Hard Negatives** | |

---

### 4. Capture Environment & Controls

1. **Illumination Profiles**:
   - Condition 1 (Standard): Direct diffuse indoor office lighting (300–500 lux).
   - Condition 2 (High Contrast): Warm direct directional lamp illumination.
   - Condition 3 (Low Contrast): Ambient indirect daylight.
2. **Camera Positioning**:
   - Web camera positioned at eye/chest level.
   - Hand centered in capture viewfinder occupying $25\%$–$60\%$ of vertical frame height.
3. **Capture Hygiene**:
   - Static pose holding for $\ge 500\text{ms}$ prior to capture to eliminate motion blur artifacts.
   - Visual inspection of the 21-point skeleton tracking stability before triggering sample recording.

---

### 5. Execution Procedure & Blinding

1. **Collector Initialization**:
   - Open ZeroLeak Data Studio (`frontend/src/components/ZeroLeakDataCollector.tsx`).
   - Select participant ID from cohort menu (`Signer_P26` to `Signer_P35`).
   - Verify camera resolution ($1280 \times 720$ or $640 \times 480$) and pipeline framerate ($\ge 25\text{FPS}$).
2. **Blinded / Controlled Recording**:
   - Participant refers to the visual prompt crop (`/references/crops/A.png` for positive A; standard ASL alphabet chart for negatives).
   - Collector records samples without altering posture to chase classifier confidence score.
3. **Telemetry Validation**:
   - Ensure all 21 hand landmarks are tracked.
   - Observe live diagnostics: landmark count, 42 features extracted, kinematic rule status, and latency.
4. **Data Export**:
   - Upon session completion, trigger "Export Dataset JSON" to download `zeroleak_letter_a_human_<signer_id>.json`.

---

### 6. Quantitative Success Criteria

The evaluation candidate is certified as ready for production integration only if the aggregated physical human validation dataset ($N \ge 700$) satisfies all empirical criteria:

1. **Letter A Recognition Sensitivity (Recall)**: $\ge 98.0\%$ across all signers.
2. **Hard Negative Discrimination (Specificity)**: $\ge 99.0\%$ ($FPR \le 1.0\%$) across confusable letters E, S, M, N, T.
3. **Precision**: $\ge 98.5\%$.
4. **Overall Balanced Accuracy**: $\ge 98.5\%$.
5. **Expected Calibration Error (ECE)**: $\le 3.0\%$ (calculated via predicted-class confidence binning).
6. **Inference Latency**: $\le 10.0\text{ms}$ per frame (measured on client CPU).
7. **Per-Participant Minimum Recall**: No individual signer shall exhibit Letter A recall $< 95.0\%$.

---

### 7. Stopping & Invalidation Rules

- **Stopping Condition**: Exactly 10 participants completing full 70-sample protocols without camera dropout.
- **Invalidation Condition**: Any modification of frozen weights in `letter_a_specialist_model.json`, decision threshold ($\tau=0.50$), or kinematic gate boundaries automatically voids the study and requires full re-collection.
