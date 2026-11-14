# Operation A: Physical Human Generalization Validation Protocol

**Document Version:** 1.0.0-phase1-validation-plan  
**Target:** Standardized Empirical Real-Human Validation Trial for ASL Letter A  
**Auditor:** Byte (Senior Offensive-Security & ML Verification Engineer)  
**Branch:** `experiment/letter-a-mastery`  
**Execution Milestone:** Phase 1 Final Audit  

---

## 1. Objective & Scope

As established in `LETTER_A_STATISTICAL_CONFIDENCE.md`, the current benchmark contains only 3 positive real-human Letter A captures, yielding a Clopper-Pearson 95% lower bound of $29.24\%$. 

This document defines the **Physical Human Generalization Validation Protocol** required to collect $\ge 30$ independent empirical samples across $\ge 10$ unique physical participants, achieving a statistically robust 95% CI lower bound $\ge 90.0\%$ prior to production integration.

---

## 2. Participant & Cohort Design

| Target Parameter | Requirement Specification | Justification |
| :--- | :--- | :--- |
| **Participant Count ($N_p$)** | $\ge 10$ distinct physical human signers | Eliminates idiosyncratic hand posture bias |
| **Positive 'A' Captures** | $\ge 30$ independent static captures ($\ge 3$ per signer) | Guarantees $\alpha=0.05$ lower bound $\ge 90.5\%$ |
| **Hard Negative Captures** | $\ge 40$ captures (10 each of E, S, M, T) | Verifies boundary isolation under real sensor conditions |
| **Bilateral Handedness** | Both Dominant (Right) and Non-Dominant (Left) hands | Verifies handedness invariance empirically |
| **Anatomical Diversity** | Palm scales ranging from $0.08$ to $0.24$ | Validates scale invariance across child and adult hands |
| **Lighting Environments** | 1. Daylight ($> 400$ lux)<br>2. Fluorescent Office ($250$–$350$ lux)<br>3. Low-Light Ambient ($< 50$ lux) | Audits sensor grain and landmark stability |
| **Camera Distances** | Near ($< 0.5\,\text{m}$), Medium ($0.5$–$1.0\,\text{m}$), Far ($> 1.0\,\text{m}$) | Validates MediaPipe bounding box scaling |

---

## 3. Data Collection Procedure via Zero-Leak Studio

Validation data acquisition must utilize the built-in [ZeroLeakDataCollector.tsx](file:///C:/Users/ADMIN/.gemini/antigravity/scratch/signbridge/frontend/src/components/ZeroLeakDataCollector.tsx):

1. **Protocol Consent:** Participant checks the informed consent box. Zero facial imagery, RGB video, or audio is recorded; only 21 MediaPipe coordinates and metadata are retained.
2. **Signer ID Assignment:** Assign cohort IDs (`Signer_P26` through `Signer_P35`) under the `Held-Out Test` cohort.
3. **Target Pose Verification:** Select letter 'A'. The UI evaluates the real-time suitability gate:
   - Checkmark 1: 4 fingers curled into palm.
   - Checkmark 2: Thumb upright on radial margin.
   - Checkmark 3: Thumb lateral (not crossing knuckles).
   - Checkmark 4: Knuckle non-crossing ($\ne$ S).
   - Checkmark 5: Thumb not tucked ($\ne$ M/N/T).
4. **Sample Capture:** Operator clicks "Capture Static 'A'". Landmarks and metadata are persisted.
5. **Hard Negative Capture:** Repeat process for letters E, S, M, and T under identical conditions.
6. **Dataset Export:** Export numerical JSON manifest via the "Export Benchmark JSON" control.

---

## 4. Acceptance Criteria for Production Promotion

To approve promotion of the Letter A verification gate from `experiment/letter-a-mastery` to `main`, the validation dataset must satisfy:

1. **Empirical A Recall:** $\ge 96.6\%$ ($\ge 29$ of 30 true A samples recognized).
2. **Empirical Non-A Specificity:** $\ge 97.5\%$ ($\ge 39$ of 40 hard negatives rejected).
3. **False Positive Rate:** $\le 2.5\%$.
4. **False Negative Rate:** $\le 3.4\%$.
5. **Exact 95% Clopper-Pearson Lower Bound:** $\ge 90.0\%$.

---

## 5. Audit Final Decision

```
================================================================================
AUDIT VERDICT: READY FOR HUMAN VALIDATION
================================================================================
```

- **Verdict:** **READY FOR HUMAN VALIDATION.**
- **Rationale:** The kinematic rule gate and specialist model are mathematically sound, leak-free, and fully reproducible. However, the current sample size of empirical real A captures ($n=3$) is statistically insufficient to prove population-level accuracy. Executing the physical human validation protocol above will provide the required statistical rigor for final production sign-off.
