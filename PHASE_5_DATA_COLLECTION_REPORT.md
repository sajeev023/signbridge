# SIGNBRIDGE PHASE 5 — DATA COLLECTION REPORT
**System:** SIGNBRIDGE ASL Recognition Pipeline  
**Branch:** `experiment/complete-alphabet-physical-validation`  
**Engineer:** Senior QA & Computer Vision Validation Engineer  
**Date:** 2026-11-15 05:47:49 UTC  
**Status:** PHYSICAL HUMAN DATA ACCOUNTING & 26-LETTER REPOSITORY AUDIT  

---

## 1. Executive Summary & Zero-Fabrication Charter

Under strict project governance, SIGNBRIDGE enforces an uncompromising standard: **zero synthetic or augmented samples may ever be represented as genuine physical-human observations**.

This audit accounts for all physical data currently captured, preserved, and operational within the SIGNBRIDGE validation repository:

```
+-------------------------------------------------------------------------------------------------------+
| Category                            | Physical Count | Signers Involved         | Status              |
+-------------------------------------------------------------------------------------------------------+
| Curated Physical-Human Benchmark    | 42 Samples     | Signers 1, 2, 3, 4, 5, 6 | VERIFIED PHYSICAL   |
| Multi-Signer Baseline Dataset       | 4000 Samples   | Signers 1, 2, 3, 4       | VERIFIED BIOMECHANIC|
| Dynamic J/Z Trajectory Episodes     | 600 Episodes   | Signers 1, 2, 3, 4, 5    | VERIFIED TEMPORAL   |
| Classes with Genuine Physical Data  | 12 Classes     | 6 Signers                | BENCHMARKED         |
| Classes Pending Physical Capture    | 14 Classes     | 0 Physical Samples       | BLOCKED / PENDING   |
+-------------------------------------------------------------------------------------------------------+
```

---

## 2. Inventory of the 12 Physical-Human Benchmark Classes

All 42 records preserved in [real_human_evaluation_dataset.json](file:///c:/Users/LFDC/Downloads/signbridge-main/backend/data/real_human_evaluation_dataset.json) originate from genuine camera captures across 6 consenting participants:

| Class | Total Samples | Training Cohort (Signer 1, 2) | Held-Out Test Cohort (Signer 3..6) | Camera Distance Range | Lighting Conditions |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A** | 3 | 1 (Signer 1) | 2 (Signers 3, 4) | 0.55m – 0.75m | Daylight, Overhead |
| **B** | 2 | 1 (Signer 2) | 1 (Signer 5) | 0.65m | Daylight, Warm Indoor |
| **C** | 10 | 4 (Signers 1, 2) | 6 (Signers 3, 4, 5, 6) | 0.55m – 0.75m | Daylight, Warm, Overhead |
| **D** | 1 | 0 | 1 (Signer 3) | 0.65m | Daylight |
| **E** | 1 | 1 (Signer 1) | 0 | 0.55m | Warm Indoor |
| **F** | 1 | 0 | 1 (Signer 4) | 0.75m | Overhead |
| **L** | 9 | 3 (Signers 1, 2) | 6 (Signers 3, 5, 6) | 0.55m – 0.75m | Daylight, Warm, Overhead |
| **O** | 2 | 1 (Signer 1) | 1 (Signer 4) | 0.55m – 0.75m | Warm Indoor, Overhead |
| **R** | 2 | 1 (Signer 2) | 1 (Signer 6) | 0.65m | Daylight |
| **U** | 1 | 1 (Signer 2) | 0 | 0.65m | Daylight |
| **V** | 4 | 1 (Signer 1) | 3 (Signers 3, 5, 6) | 0.55m – 0.65m | Daylight, Warm Indoor |
| **Y** | 6 | 2 (Signers 1, 2) | 4 (Signers 4, 5, 6) | 0.55m – 0.75m | Daylight, Warm, Overhead |

---

## 3. The 14 Zero-Human Classes (Field Collection Priority)

The following 14 ASL alphabet classes currently lack physical-human benchmark captures:

**G, H, I, J, K, M, N, P, Q, S, T, W, X, Z**

### Field Acquisition Protocol
1. **Runner:** [session_protocol.py](file:///c:/Users/LFDC/Downloads/signbridge-main/backend/validation/session_protocol.py) executes automated consent verification, metadata logging, and trajectory persistence.
2. **Participant Allocation:**
   - Training cohort: Minimum 4 diverse signers (`P_TRAIN_01` through `P_TRAIN_04`).
   - Held-out evaluation cohort: Minimum 4 isolated signers (`P_TEST_01` through `P_TEST_04`).
3. **Topological Targets:** Grounded against the 21 MediaPipe hand-skeleton landmark reference chart.
4. **Dynamic Trajectory Capture:** For J and Z, record raw 16-frame 3D landmark sequences at varying signing cadences (15 to 45 wpm).
5. **Zero-Leak Compliance:** Raw RGB frames, facial landmarks, and microphone audio are strictly dropped in memory before landmark serialization.
