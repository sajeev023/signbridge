# SIGNBRIDGE REGRESSION TEST REPORT
**System:** SIGNBRIDGE Recognition Engine  
**Branch:** `experiment/open-source-integration`  
**Engineer:** Byte (Senior Systems & ML Architect)  
**Date:** 2026-11-15 03:48:08 UTC  
**Status:** ALL TESTS PASSING (0 REGRESSIONS)  

---

## 1. Test Execution Summary

| Test Category | Suite File | Tests Ran | Passed | Failed | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Open-Source Integration** | `test_open_source_integration.py` | 4 | 4 | 0 | **PASSED** |
| **Letter A Failure Injection** | `test_letter_a_failure_injection.py` | 14 | 14 | 0 | **PASSED** |
| **Virtual Alphabet Generator** | `test_virtual_alphabet_suite.py` | 18 | 18 | 0 | **PASSED** |
| **Kinematic Rules** | `test_kinematic_rules.py` | 22 | 22 | 0 | **PASSED** |
| **Multi-Signer Generalization** | `test_multisigner_pipeline.py` | 16 | 16 | 0 | **PASSED** |
| **Titan Architecture** | `test_titan_suite.py` | 24 | 24 | 0 | **PASSED** |
| **Feature Extraction & Core** | `test_alphabet_core.py` | 33 | 33 | 0 | **PASSED** |
| **TOTAL BACKEND SUITE** | `pytest backend/tests/` | **131** | **131** | **0** | **100% PASS** |

Execution Duration: 36.69 seconds.

---

## 2. Frontend Build Verification

- **Command:** `npm run build` inside `frontend/`
- **Compiler:** TypeScript 5.x (`tsc -b`) & Vite 8.3.8
- **Result:**
  - `dist/index.html` (1.42 kB)
  - `dist/assets/index-G-qUWwfW.css` (54.40 kB)
  - `dist/assets/index-VDOQ1Efg.js` (2,458.08 kB)
  - Modules Transformed: 2,103
  - Build Duration: 1.59 seconds
  - **Errors: 0 | Warnings: 0**

---

## 3. Frozen Candidate Hash Verification

Every frozen artifact specified in `backend/letter_a_frozen_manifest.json` was verified against its on-disk SHA-256 digest:

| File Path | Expected SHA-256 | Actual SHA-256 | Size (Bytes) | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| `backend/saved_models/letter_a_specialist_model.json` | `dc7cd03a...` | `dc7cd03a...` | 65,980 | **MATCH (BIT-FOR-BIT)** |
| `backend/train_letter_a_specialist.py` | `29347b55...` | `29347b55...` | 16,558 | **MATCH (BIT-FOR-BIT)** |
| `backend/alphabet_feature_extractor.py` | `a6e0a483...` | `a6e0a483...` | 8,752 | **MATCH (BIT-FOR-BIT)** |
| `frontend/src/services/alphabetInferenceEngine.ts` | `2e34e9bb...` | `2e34e9bb...` | 21,906 | **MATCH (BIT-FOR-BIT)** |
| `backend/data/alphabet_dataset_multisigner.json` | `0cfb68c6...` | `0cfb68c6...` | 17,343,481 | **MATCH (BIT-FOR-BIT)** |
| `backend/titan/datasets/asl_alphabet/metadata/dataset_manifest.json` | `3fc93a0f...` | `3fc93a0f...` | 1,103 | **MATCH (BIT-FOR-BIT)** |
| `backend/titan/datasets/asl_alphabet/metadata/real_human_benchmark.json` | `997434c7...` | `997434c7...` | 164,181 | **MATCH (BIT-FOR-BIT)** |
