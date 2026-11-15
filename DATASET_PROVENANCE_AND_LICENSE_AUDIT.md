# DATASET PROVENANCE AND LICENSE AUDIT
**System:** SIGNBRIDGE Recognition Engine  
**Branch:** `experiment/open-source-integration`  
**Engineer:** Byte (Senior Systems & ML Architect)  
**Date:** 2026-11-15 03:48:08 UTC  

---

## 1. Open-Source License & Legal Clearance

| Repository | Declared License | SPX Identifier | Commercial Use | Distribution Rights | Action Taken |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **kinivi** | Apache License 2.0 | `Apache-2.0` | **PERMITTED** | Permitted with attribution | Code patterns adapted; attribution placed in `NOTICE.md` and module docstrings. Pretrained weights rejected. |
| **nicknochnack (Action)** | None (Default Copyright) | `NOASSERTION` | **RESTRICTED** | Unauthorized | Binary weights (`action.h5`) **STRICTLY EXCLUDED**. Algorithmic hysteresis pattern reimplemented cleanly. |
| **nicknochnack (Object)** | None (Default Copyright) | `NOASSERTION` | **RESTRICTED** | Unauthorized | Completely excluded from repository. |
| **pianalytix** | None (Default Copyright) | `NOASSERTION` | **RESTRICTED** | Unauthorized | Completely excluded from repository. |

---

## 2. Dataset Provenance and Quality Audit

### 2.1 Kinivi Point History & Keypoint Datasets
- **Source:** Synthetic and self-recorded webcam landmarks (`keypoint.csv`, `point_history.csv`).
- **Signer Count:** 1 signer.
- **Recording Conditions:** Single desktop webcam, static indoor lighting.
- **Risk Assessment:** High risk of overfitting to the author's hand proportions.
- **Decision:** Dataset excluded. SIGNBRIDGE uses multi-signer datasets.

### 2.2 Pianalytix Sign Language Dataset
- **Source:** 90 video sequences recorded across 3 classes (`A`, `B`, `C`), 30 sequences each, 30 frames per sequence.
- **Signer Count:** 1 signer (Pianalytix instructor).
- **Critical Data Leakage Issue:** The training script performs a standard `train_test_split(X, y, test_size=0.05)` randomly across samples. Because samples were collected consecutively in a single 3-minute recording session, adjacent frames with identical background, camera distance, and posture are leaked across train and test splits, producing synthetic 100% test accuracy that collapses in the wild.
- **Decision:** **COMPLETELY REJECTED.**

### 2.3 Nick Nochnack Action Detection Dataset
- **Source:** 90 video sequences (30 sequences per action: `hello`, `thanks`, `iloveyou`), 30 frames each.
- **Signer Count:** 1 signer.
- **Decision:** Irrelevant to fingerspelling alphabet; excluded.

---

## 3. Zero Data Fabrication Attestation

1. **Zero External Data Conversion:** Under no circumstances was tutorial data from Pianalytix, Nick Nochnack, or Kinivi converted or labeled as "genuine human validation data" in SIGNBRIDGE.
2. **Benchmark Integrity:** All evaluation datasets in `backend/titan/datasets/asl_alphabet/metadata/` remain unpolluted.
3. **Privacy Compliance:** No external face mesh data, RGB video frames, or personally identifiable imagery has been committed to the repository.
