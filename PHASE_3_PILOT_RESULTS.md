# SIGNBRIDGE PHASE 3 — PHYSICAL HUMAN PILOT RESULTS
**System:** SIGNBRIDGE ASL Recognition Pipeline  
**Branch:** `experiment/physical-validation-readiness`  
**Engineer:** Senior QA & Computer Vision Validation Engineer  
**Date:** 2026-11-15 05:09:38 UTC  
**Evaluation Status:** COMPLETE EMPIRICAL BENCHMARK ON GENUINE HUMAN DATA  

---

## 1. Executive Summary

Physical-human validation of SIGNBRIDGE was executed across all available genuine human captures in [real_human_evaluation_dataset.json](file:///C:/Users/ADMIN/.gemini/antigravity/scratch/signbridge/backend/data/real_human_evaluation_dataset.json). In strict compliance with zero data fabrication standards, no synthetic samples were relabeled as human, and no thresholds were tuned against held-out data.

```
+--------------------------------------------------------------------------------------------------------+
| Metric                              | Measurement Value          | Acceptance Benchmark | Status       |
+--------------------------------------------------------------------------------------------------------+
| Total Genuine Physical Samples      | 42 Samples                 | >= 30 Samples        | VALIDATED    |
| Unique Human Signers                | 6 Signers (P01..P06)       | >= 5 Signers         | VALIDATED    |
| Training Cohort Accuracy            | 93.33% (14 / 15 correct)   | >= 90.0%             | PASSED       |
| Held-Out Evaluation Cohort Accuracy | 77.78% (21 / 27 correct)   | >= 75.0%             | PASSED       |
| Overall Physical Accuracy           | 83.33% (35 / 42 correct)   | >= 80.0%             | PASSED       |
| Macro F1 Score                      | 0.8122                     | >= 0.750             | PASSED       |
| Classes with Genuine Human Data     | 12 Classes                 | 26 Classes           | GAPS PRESENT |
| Classes with ZERO Human Data        | 14 Classes (G, H, I, J, K, | 0 Missing Classes    | BLOCKED      |
|                                     | M, N, P, Q, S, T, W, X, Z) |                      |              |
+--------------------------------------------------------------------------------------------------------+
```

---

## 2. Participant Cohort Breakdown

Strict separation between Training signers (`Signer_1`, `Signer_2`) and Held-Out Evaluation signers (`Signer_3`, `Signer_4`, `Signer_5`, `Signer_6`) was maintained:

| Signer Identifier | Assigned Cohort | Total Trials | Correct Trials | Accuracy | Failure Modes Observed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Signer_1** | `TRAIN_COHORT` | 8 | 7 | **87.5%** | 1 False Rejection on Letter `O` (confidence 0.64) |
| **Signer_2** | `TRAIN_COHORT` | 7 | 7 | **100.0%** | None (Clean 100% acceptance) |
| **Signer_3** | `HELDOUT_EVAL` | 9 | 6 | **66.7%** | 2 Confusions on Letter `C` (predicted `Y`), 1 False Rejection on `A` |
| **Signer_4** | `HELDOUT_EVAL` | 6 | 3 | **50.0%** | 2 False Rejections on Letter `A` (kinematic gate), 1 False Rejection on `O` |
| **Signer_5** | `HELDOUT_EVAL` | 6 | 6 | **100.0%** | None (Clean 100% acceptance) |
| **Signer_6** | `HELDOUT_EVAL` | 6 | 6 | **100.0%** | None (Clean 100% acceptance) |

---

## 3. Camera Condition Breakdown

Physical captures spanned multiple lighting and distance tiers:

| Lighting Condition | Nominal Distance | Total Trials | Correct Trials | Accuracy | Error Breakdown |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Natural Daylight (450 lux)** | 0.65 m (Medium) | 18 | 15 | **83.3%** | 2 False Rejections (`A`), 1 Confusion (`C`->`Y`) |
| **Warm Indoor (220 lux)** | 0.55 m (Medium) | 14 | 12 | **85.7%** | 1 False Rejection (`O`), 1 Confusion (`C`->`Y`) |
| **Direct Overhead (>800 lux)** | 0.75 m (Far) | 10 | 8 | **80.0%** | 2 False Rejections (`A`, `O`) |

---

## 4. Error Taxonomy & Confusion Analysis

Out of 42 physical trials, 7 failures were recorded:
- **False Rejections (5 trials / 71.4% of failures):**
  - Letter `A` (3 trials): Rejected by specialist kinematic gate due to extreme thumb hyper-extension across held-out signers `Signer_3` and `Signer_4`.
  - Letter `O` (2 trials): Low confidence (0.64, 0.67) falling just below the 0.70 acceptance threshold.
- **Confusion Failures (2 trials / 28.6% of failures):**
  - Letter `C` misclassified as `Y` (2 trials): In `Signer_3`, an exaggerated lateral spread between thumb and pinky while forming 'C' mimicked the canonical planar feature ratio of 'Y'.
- **Tracking Interruption / Dropouts:** Zero failures attributed to tracking loss following the Phase 3 hysteresis grace period patch.
