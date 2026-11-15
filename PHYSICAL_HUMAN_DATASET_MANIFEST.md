# PHYSICAL HUMAN DATASET MANIFEST
**System:** SIGNBRIDGE Recognition Engine  
**Dataset ID:** `DS-ASL-PHYSICAL-HUMAN-2026-V1`  
**Storage Location:** `backend/data/real_human_evaluation_dataset.json`  
**Engineer:** Senior QA & Computer Vision Validation Engineer  
**Date:** 2026-11-15 05:09:38 UTC  

---

## 1. Dataset Provenance & Privacy Governance

1. **Zero Raw Media Storage:** The dataset contains exclusively 21 $(x, y, z)$ MediaPipe skeletal hand landmarks and pre-computed 42 continuous canonical features. No raw video streams, RGB images, face meshes, or audio files are stored.
2. **Pseudonymization:** All signers are indexed strictly via pseudonyms (`Signer_1` through `Signer_6`).
3. **Cohort Isolation:** Signers 1 and 2 are designated for training analysis; Signers 3, 4, 5, and 6 are strictly reserved for held-out evaluation.
4. **Anti-Fabrication Attestation:** All 42 records represent physical human signers. Zero synthetic generator samples are present.

---

## 2. Sample Inventory Summary

| Alphabet Label | Modality | Total Human Samples | Training Cohort | Held-Out Cohort | Provenance Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A** | Static | 3 | 1 (Signer 1) | 2 (Signer 3, 4) | VERIFIED PHYSICAL |
| **B** | Static | 2 | 1 (Signer 2) | 1 (Signer 5) | VERIFIED PHYSICAL |
| **C** | Static | 10 | 4 (Signer 1, 2) | 6 (Signer 3, 4, 5, 6) | VERIFIED PHYSICAL |
| **D** | Static | 1 | 0 | 1 (Signer 3) | VERIFIED PHYSICAL |
| **E** | Static | 1 | 1 (Signer 1) | 0 | VERIFIED PHYSICAL |
| **F** | Static | 1 | 0 | 1 (Signer 4) | VERIFIED PHYSICAL |
| **L** | Static | 9 | 3 (Signer 1, 2) | 6 (Signer 3, 5, 6) | VERIFIED PHYSICAL |
| **O** | Static | 2 | 1 (Signer 1) | 1 (Signer 4) | VERIFIED PHYSICAL |
| **R** | Static | 2 | 1 (Signer 2) | 1 (Signer 6) | VERIFIED PHYSICAL |
| **U** | Static | 1 | 1 (Signer 2) | 0 | VERIFIED PHYSICAL |
| **V** | Static | 4 | 1 (Signer 1) | 3 (Signer 3, 5, 6) | VERIFIED PHYSICAL |
| **Y** | Static | 6 | 2 (Signer 1, 2) | 4 (Signer 4, 5, 6) | VERIFIED PHYSICAL |
| **TOTAL REPRESENTED** | - | **42 Samples** | **15 Samples** | **27 Samples** | **12 / 26 Classes** |

---

## 3. Explicit Gap Analysis (14 Uncollected Classes)

The following 14 classes currently have **0 genuine physical human samples** in the repository:

`G, H, I, J, K, M, N, P, Q, S, T, W, X, Z`

Under project governance, these classes are flagged as **BLOCKED / PENDING FIELD COLLECTION**. They cannot be certified for production release until genuine human participant sessions are captured using the protocol defined in `PHYSICAL_VALIDATION_PROTOCOL.md`.
