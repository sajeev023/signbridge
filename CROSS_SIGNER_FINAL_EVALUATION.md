# CROSS-SIGNER FINAL EVALUATION REPORT
**System:** SIGNBRIDGE ASL Recognition Pipeline  
**Branch:** `experiment/complete-alphabet-physical-validation`  
**Engineer:** Senior QA & Computer Vision Validation Engineer  
**Date:** 2026-11-15 05:47:49 UTC  
**Cohort Scope:** 6 Human Signers (Signer 1..6), 42 Genuine Physical Captures  

---

## 1. Participant-Level Generalization Performance

Signers were strictly separated into a Training Cohort (Signers 1 & 2) and an Isolated Held-Out Evaluation Cohort (Signers 3, 4, 5 & 6):

```
+--------------------------------------------------------------------------------------------------------+
| Participant ID | Cohort Role     | Total Trials | Baseline Correct | Calibrated Correct | Accuracy     |
+--------------------------------------------------------------------------------------------------------+
| Signer_1       | Training        | 8 trials     | 7 correct        | 8 correct          | 100.00%      |
| Signer_2       | Training        | 7 trials     | 7 correct        | 7 correct          | 100.00%      |
| Signer_3       | Held-Out Test   | 9 trials     | 6 correct        | 7 correct          | 77.78%       |
| Signer_4       | Held-Out Test   | 6 trials     | 5 correct        | 6 correct          | 100.00%      |
| Signer_5       | Held-Out Test   | 6 trials     | 5 correct        | 5 correct          | 83.33%       |
| Signer_6       | Held-Out Test   | 6 trials     | 5 correct        | 5 correct          | 83.33%       |
+--------------------------------------------------------------------------------------------------------+
| Training Cohort Total (S1, S2)   | 15 trials    | 14 correct (93.3%)| 15 correct         | 100.00%      |
| Held-Out Cohort Total (S3..S6)   | 27 trials    | 21 correct (77.8%)| 23 correct         | 85.19%       |
+--------------------------------------------------------------------------------------------------------+
| COMBINED PHYSICAL DATASET        | 42 trials    | 35 correct (83.3%)| 38 correct         | 90.48%       |
+--------------------------------------------------------------------------------------------------------+
```

---

## 2. Macro Performance Metrics Across Cohorts

```
+-----------------------------------+--------------------+------------------------+
| Metric                            | Baseline (T=1.062) | Calibrated (T=1.000)   |
+-----------------------------------+--------------------+------------------------+
| Overall Combined Accuracy         | 83.33% (35/42)     | 90.48% (38/42)         |
| Held-Out Test Cohort Accuracy     | 77.78% (21/27)     | 85.19% (23/27)         |
| Combined Macro Precision          | 0.8472             | 0.9028                 |
| Combined Macro Recall             | 0.8333             | 0.9048                 |
| Combined Macro F1 Score           | 0.8122             | 0.8876                 |
| Held-Out Macro F1 Score           | 0.7712             | 0.8491                 |
| False Acceptance Rate (FAR)       | 4.76% (2/42)       | 4.76% (2/42)           |
| False Rejection Rate (FRR)        | 11.90% (5/42)      | 4.76% (2/42)           |
+-----------------------------------+--------------------+------------------------+
```

---

## 3. Physical Confusion Matrix (12 Represented Classes)

Under calibrated temperature scaling ($T=1.000$):
- **A (3):** 3 predicted as A (0 confusions, 0 rejections).
- **B (2):** 2 predicted as B.
- **C (10):** 8 predicted as C, 2 confused as Y (Signer 3 lateral thumb-pinky flare).
- **D (1):** 1 predicted as D.
- **E (1):** 1 predicted as E.
- **F (1):** 1 predicted as F.
- **L (9):** 9 predicted as L.
- **O (2):** 2 predicted as O.
- **R (2):** 2 predicted as R.
- **U (1):** 1 predicted as U.
- **V (4):** 4 predicted as V.
- **Y (6):** 6 predicted as Y.

---

## 4. Camera Condition Breakdown

| Condition Variable | Category | Total Captures | Correct | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **Lighting** | Natural Daylight (450 lux) | 26 | 24 | 92.31% |
| | Warm Indoor (300 lux) | 8 | 8 | 100.00% |
| | Overhead Florescent (500 lux) | 8 | 6 | 75.00% |
| **Distance** | Near (0.55m) | 14 | 13 | 92.86% |
| | Medium (0.65m) | 16 | 15 | 93.75% |
| | Far (0.75m) | 12 | 10 | 83.33% |
| **Orientation** | Frontal (0 deg) | 34 | 31 | 91.18% |
| | Yaw/Pitch Offset (+/- 15 deg) | 8 | 7 | 87.50% |

---

## 5. End-to-End Pipeline Latency

Benchmarked across 1,000 iterations on standard x64 execution:
- Mean Latency: **8.31 ms** (120 FPS throughput capability).
- 50th Percentile (Median): **7.85 ms**.
- 95th Percentile: **11.20 ms**.
- 99th Percentile: **14.80 ms**.
- Max Latency: **18.20 ms**.
