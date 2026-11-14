# SIGNBRIDGE — ASL LETTER A SPECIALIST
## Physical Human Evaluation & Acceptance Benchmark Report (Template)

**Branch**: `experiment/letter-a-mastery`  
**Candidate Hash**: `dc7cd03a4ed647a7ec570422ed54d130bbb250d6cded42b93466ef6e967d14cf`  
**Protocol Ref**: `LETTER_A_HUMAN_STUDY_PROTOCOL.md`  
**Evaluation Session Date**: `[YYYY-MM-DD]`  
**Status**: `[PENDING DATA INTAKE / COMPLETED / CERTIFIED]`  

---

### 1. Participant Session Manifest

| Participant ID | Capture Date | Environment / Lighting | Handedness | Positive A Count | Hard Negative Count | Session Status | Export Artifact |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Signer_P26` | `[DATE]` | `[LIGHTING]` | `[R/L]` | `[COUNT >= 30]` | `[COUNT >= 40]` | `[PENDING]` | `zeroleak_P26.json` |
| `Signer_P27` | `[DATE]` | `[LIGHTING]` | `[R/L]` | `[COUNT >= 30]` | `[COUNT >= 40]` | `[PENDING]` | `zeroleak_P27.json` |
| `Signer_P28` | `[DATE]` | `[LIGHTING]` | `[R/L]` | `[COUNT >= 30]` | `[COUNT >= 40]` | `[PENDING]` | `zeroleak_P28.json` |
| `Signer_P29` | `[DATE]` | `[LIGHTING]` | `[R/L]` | `[COUNT >= 30]` | `[COUNT >= 40]` | `[PENDING]` | `zeroleak_P29.json` |
| `Signer_P30` | `[DATE]` | `[LIGHTING]` | `[R/L]` | `[COUNT >= 30]` | `[COUNT >= 40]` | `[PENDING]` | `zeroleak_P30.json` |
| `Signer_P31` | `[DATE]` | `[LIGHTING]` | `[R/L]` | `[COUNT >= 30]` | `[COUNT >= 40]` | `[PENDING]` | `zeroleak_P31.json` |
| `Signer_P32` | `[DATE]` | `[LIGHTING]` | `[R/L]` | `[COUNT >= 30]` | `[COUNT >= 40]` | `[PENDING]` | `zeroleak_P32.json` |
| `Signer_P33` | `[DATE]` | `[LIGHTING]` | `[R/L]` | `[COUNT >= 30]` | `[COUNT >= 40]` | `[PENDING]` | `zeroleak_P33.json` |
| `Signer_P34` | `[DATE]` | `[LIGHTING]` | `[R/L]` | `[COUNT >= 30]` | `[COUNT >= 40]` | `[PENDING]` | `zeroleak_P34.json` |
| `Signer_P35` | `[DATE]` | `[LIGHTING]` | `[R/L]` | `[COUNT >= 30]` | `[COUNT >= 40]` | `[PENDING]` | `zeroleak_P35.json` |
| **Total Cohort**| | | | **[>= 300]** | **[>= 400]** | | **[>= 700 SAMPLES]**|

---

### 2. Empirical Performance Metrics Summary

| Metric Name | Mathematical Definition | Minimum Acceptance Target | Empirical Result | Gate Status |
| :--- | :--- | :--- | :--- | :--- |
| **Total Physical Samples** | $N_{total} = N_A + N_{neg}$ | $\ge 700$ | `[INSERT]` | `[PASS / FAIL]` |
| **True Positives (TP)** | $\sum \mathbf{1}(\hat{y}=A \land y=A)$ | $\ge 294$ | `[INSERT]` | `[PASS / FAIL]` |
| **True Negatives (TN)** | $\sum \mathbf{1}(\hat{y}=\text{Non-A} \land y\ne A)$ | $\ge 396$ | `[INSERT]` | `[PASS / FAIL]` |
| **False Positives (FP)** | $\sum \mathbf{1}(\hat{y}=A \land y\ne A)$ | $\le 4$ | `[INSERT]` | `[PASS / FAIL]` |
| **False Negatives (FN)** | $\sum \mathbf{1}(\hat{y}=\text{Non-A} \land y=A)$ | $\le 6$ | `[INSERT]` | `[PASS / FAIL]` |
| **Sensitivity / Recall** | $TP / (TP + FN)$ | $\ge 98.0\%$ | `[INSERT %]` | `[PASS / FAIL]` |
| **Specificity** | $TN / (TN + FP)$ | $\ge 99.0\%$ | `[INSERT %]` | `[PASS / FAIL]` |
| **Precision (PPV)** | $TP / (TP + FP)$ | $\ge 98.5\%$ | `[INSERT %]` | `[PASS / FAIL]` |
| **F1-Score** | $2 \cdot \frac{\text{Prec} \cdot \text{Rec}}{\text{Prec} + \text{Rec}}$ | $\ge 98.5\%$ | `[INSERT %]` | `[PASS / FAIL]` |
| **False Positive Rate (FPR)**| $FP / (FP + TN)$ | $\le 1.0\%$ | `[INSERT %]` | `[PASS / FAIL]` |
| **False Negative Rate (FNR)**| $FN / (FN + TP)$ | $\le 2.0\%$ | `[INSERT %]` | `[PASS / FAIL]` |
| **Balanced Accuracy** | $\frac{\text{Sensitivity} + \text{Specificity}}{2}$ | $\ge 98.5\%$ | `[INSERT %]` | `[PASS / FAIL]` |
| **Expected Calibration Error (ECE)**| $\sum \frac{\|B_m\|}{N} \|\text{acc} - \text{conf}\|$ | $\le 3.0\%$ | `[INSERT %]` | `[PASS / FAIL]` |
| **Brier Calibration Score** | $\frac{1}{N} \sum (P(A \mid x) - y)^2$ | $\le 0.010$ | `[INSERT]` | `[PASS / FAIL]` |
| **Inference Latency (p95)**| 95th percentile forward pass | $\le 5.0\text{ms}$ | `[INSERT ms]`| `[PASS / FAIL]` |

---

### 3. Per-Participant Granular Analysis

| Signer ID | Hand | Target A ($N_A$) | Recalled A | A Recall (%) | Negatives ($N_{neg}$) | Rejected Neg | Neg Specificity (%) | Mean Latency (ms) | ECE (%) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Signer_P26` | `[R]` | `[30]` | `[ ]` | `[ %]` | `[40]` | `[ ]` | `[ %]` | `[ ms]` | `[ %]` | `[PASS]` |
| `Signer_P27` | `[R]` | `[30]` | `[ ]` | `[ %]` | `[40]` | `[ ]` | `[ %]` | `[ ms]` | `[ %]` | `[PASS]` |
| `Signer_P28` | `[L]` | `[30]` | `[ ]` | `[ %]` | `[40]` | `[ ]` | `[ %]` | `[ ms]` | `[ %]` | `[PASS]` |
| `Signer_P29` | `[R]` | `[30]` | `[ ]` | `[ %]` | `[40]` | `[ ]` | `[ %]` | `[ ms]` | `[ %]` | `[PASS]` |
| `Signer_P30` | `[R]` | `[30]` | `[ ]` | `[ %]` | `[40]` | `[ ]` | `[ %]` | `[ ms]` | `[ %]` | `[PASS]` |
| `Signer_P31` | `[L]` | `[30]` | `[ ]` | `[ %]` | `[40]` | `[ ]` | `[ %]` | `[ ms]` | `[ %]` | `[PASS]` |
| `Signer_P32` | `[R]` | `[30]` | `[ ]` | `[ %]` | `[40]` | `[ ]` | `[ %]` | `[ ms]` | `[ %]` | `[PASS]` |
| `Signer_P33` | `[R]` | `[30]` | `[ ]` | `[ %]` | `[40]` | `[ ]` | `[ %]` | `[ ms]` | `[ %]` | `[PASS]` |
| `Signer_P34` | `[L]` | `[30]` | `[ ]` | `[ %]` | `[40]` | `[ ]` | `[ %]` | `[ ms]` | `[ %]` | `[PASS]` |
| `Signer_P35` | `[R]` | `[30]` | `[ ]` | `[ %]` | `[40]` | `[ ]` | `[ %]` | `[ ms]` | `[ %]` | `[PASS]` |

---

### 4. Hard-Negative Discrimination Breakdown

| Confusable Letter | Tested Samples | Correctly Rejected | False Positives | Rejection Rate (%) | Primary Rejection Mechanism |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Letter E** | `[>= 80]` | `[ ]` | `[ ]` | `[ %]` | Kinematic Clause 2 (`thumb_elevated >= 0.55`) |
| **Letter S** | `[>= 80]` | `[ ]` | `[ ]` | `[ %]` | Kinematic Clause 3 & 4 (`thumb_lateral` & `cos_spread`) |
| **Letter M** | `[>= 80]` | `[ ]` | `[ ]` | `[ %]` | Kinematic Clause 5 (`dist_thumb_mid >= 0.38`) |
| **Letter N** | `[>= 80]` | `[ ]` | `[ ]` | `[ %]` | Kinematic Clause 5 (`dist_thumb_mid >= 0.38`) |
| **Letter T** | `[>= 80]` | `[ ]` | `[ ]` | `[ %]` | Kinematic Clause 5 (`dist_thumb_mid >= 0.38`) |
| **Total** | `[>= 400]` | `[ ]` | `[ ]` | `[ %]` | |

---

### 5. Kinematic Rule Gate Contribution Matrix

| Kinematic Clause | Physical Criteria | Samples Intercepted Prior to MLP | Standalone Pass Rate on True A |
| :--- | :--- | :--- | :--- |
| **1. 4 Fingers Curled** | $ext < 0.35$ for fingers 5–8, 9–12, 13–16, 17–20 | `[COUNT]` | `[ %]` |
| **2. Thumb Upright** | $thumb\_y \ge 0.55$ (radial elevation) | `[COUNT]` | `[ %]` |
| **3. Thumb Lateral** | $thumb\_x \ge 0.12$ (along radial side) | `[COUNT]` | `[ %]` |
| **4. No Cross (≠S)** | $cos\_spread < 0.15$ (vector divergence vs index) | `[COUNT]` | `[ %]` |
| **5. No Tuck (≠M/N/T)** | $dist\_thumb\_mid \ge 0.38$ (clearance vs middle knuckle) | `[COUNT]` | `[ %]` |

---

### 6. Final Certification Decision

```
[ ] APPROVED FOR PRODUCTION INTEGRATION — All empirical criteria satisfied without post-hoc modification.
[ ] CONDITIONAL REJECTION — Specific criteria failed; root-cause diagnosis required.
[ ] UNTESTED — Awaiting physical participant intake.
```

**Evaluator Signature**: `___________________________`  
**Date**: `___________________________`
