# Operation A: Dataset Provenance & Record Verification Audit

**Audit Target:** All data assets utilized in ASL Letter A training, calibration, and benchmarking  
**Auditor:** Byte (Senior Offensive-Security & ML Verification Engineer)  
**Branch:** `experiment/letter-a-mastery`  
**Execution Date:** 2026-11-14  

---

## 1. Provenance Inventory Summary

Every record in the training, validation, testing, and benchmark pipelines was traced back to its raw storage format, origin cohort, participant ID, and transformation history.

| Dataset Asset Path | Total Records | Positive (A) | Negative | Provenance Type | Signer IDs | Participant Split Isolation |
| :--- | :---: | :---: | :---: | :--- | :--- | :--- |
| `backend/data/alphabet_dataset_multisigner.json` | 4,000 | 160 | 3,840 | Kinematic Multi-Signer Model + Biomechanical Parametric Jitter | Signer_1, Signer_2, Signer_3, Signer_4 | Strict: Train (S1, S2), Val (S3), Test (S4) |
| `backend/titan/datasets/asl_alphabet/metadata/real_human_benchmark.json` | 42 | 3 | 39 | Empirical Real-Camera MediaPipe Extractions | Signer_1, Signer_2, Signer_3, Signer_4, Signer_5, Signer_6 | Independent Held-Out Benchmark |
| `evaluation/real_world_landmarks_dataset.json` | 36 | 4 | 32 | Empirical Multi-Environment Video Captures | Signer_1, Signer_2, Signer_3, Signer_4 | Isolated Environmental Benchmark |
| `backend/titan/datasets/asl_alphabet/splits/train.json` | 2,240 | 80 | 2,160 | TITAN Multi-Signer Pipeline | Signer_1, Signer_2 | Train Split Only |
| `backend/titan/datasets/asl_alphabet/splits/val.json` | 1,120 | 40 | 1,080 | TITAN Multi-Signer Pipeline | Signer_3 | Validation Split Only |
| `backend/titan/datasets/asl_alphabet/splits/test.json` | 1,240 | 40 | 1,200 | TITAN Multi-Signer Pipeline | Signer_4, Signer_5 | Test Split Only |

---

## 2. Granular Record Analysis by Participant

### 2.1 Multi-Signer Dataset (`alphabet_dataset_multisigner.json`)
- **Total Records:** 4,000 samples.
- **Dimensionality:** 42 continuous normalized floats per sample.
- **Participant Partitioning:**
  - `Signer_1`: 1,000 samples (40 per class across 25 classes). Assigned strictly to `train`.
  - `Signer_2`: 1,000 samples (40 per class across 25 classes). Assigned strictly to `train`.
  - `Signer_3`: 1,000 samples (40 per class across 25 classes). Assigned strictly to `val`.
  - `Signer_4`: 1,000 samples (40 per class across 25 classes). Assigned strictly to `test`.
- **Class Balance:** Perfectly uniform at 160 samples per class across all 25 classes (24 alphabet letters + NEUTRAL).
- **Synthetic vs Human Classification:** While labeled with participant IDs, the 4,000 samples in this dataset are **kinematically modeled parametric variations** derived from seed biomechanical postures with injected joint angular variations, scale shifts ($0.15$–$0.22$), and 3D rotations ($\theta \in [-25^\circ, +25^\circ]$). They are not raw webcam frames.

### 2.2 Empirical Real-Human Benchmark (`real_human_benchmark.json`)
- **Total Records:** 42 samples.
- **Provenance:** Raw 21-landmark extractions from actual human signers in physical camera environments.
- **Participant Distribution:**
  - `Signer_1`: 8 samples (`A`: 1, `E`: 1, `L`: 5, `R`: 1)
  - `Signer_2`: 7 samples (`B`: 1, `F`: 1, `L`: 2, `U`: 1, `V`: 2)
  - `Signer_3`: 9 samples (`A`: 1, `C`: 2, `L`: 2, `R`: 1, `V`: 2, `Y`: 1)
  - `Signer_4`: 6 samples (`A`: 1, `B`: 1, `D`: 1, `O`: 2, `Y`: 1)
  - `Signer_5`: 6 samples (`C`: 4, `Y`: 2)
  - `Signer_6`: 6 samples (`C`: 4, `Y`: 2)
- **Positive A Samples (Audit of All 3 Records):**
  1. `EMPIRICAL_REAL_S_01_A` (Signer_1): Standard daylight, frontal angle. Real human capture.
  2. `EMPIRICAL_REAL_KIN_03_A` (Signer_3): Natural daylight, slight thumb self-occlusion. Real human capture.
  3. `EMPIRICAL_REAL_ENV_01_A` (Signer_4): Low-light ambient (45 lux, sensor noise). Real human capture.
- **Negative Sample Diversity:** 39 negative samples covering 11 classes (`B`, `C`, `D`, `E`, `F`, `L`, `O`, `R`, `U`, `V`, `Y`) across 6 independent real people.

---

## 3. Discrepancies & Audit Findings

1. **Synthetic vs Empirical Distinction:** The multi-signer dataset is synthetic/parametric, whereas the 42-record benchmark is empirical human data. Previous reports did not clearly distinguish parametric multi-signer sets from empirical sensor recordings.
2. **Real A Sample Scarcity:** There are only **3 empirical single-hand ASL Letter A captures** in the current evaluation benchmarks. While all 3 are genuine human captures, 3 samples are insufficient to establish population-level statistical confidence (analyzed in `LETTER_A_STATISTICAL_CONFIDENCE.md`).
3. **Absence of Facial & RGB Data:** All audited records store exclusively numerical 21 3D coordinates and 42 biomechanical features. Zero RGB imagery, facial landmarks, or PII are retained.
