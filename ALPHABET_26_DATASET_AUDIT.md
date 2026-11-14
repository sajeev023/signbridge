# SIGNBRIDGE — OPERATION ALPHABET MASTERY
## Complete Dataset Provenance and Forensic Integrity Audit

**Audit Scope:** Every dataset used for training, validation, testing, and benchmarking of the ASL 26-letter manual alphabet recognition system.  
**Audit Standard:** Zero fabricated samples, strict cryptographic hashing, complete split separation, participant tracking.

---

## 1. Master Dataset Inventory

| Dataset File | File Size | Record Count | Modality / Format | Target Classes | Data Provenance |
| :--- | :---: | :---: | :--- | :--- | :--- |
| `backend/data/alphabet_dataset_multisigner.json` | 17.34 MB | 4,000 | 42 Biomechanical Features + 21 Landmarks | 25 Static Classes (A–Y excl. J, Z + NEUTRAL) | Synthetic Biomechanical Multi-Signer Generation |
| `backend/data/real_human_evaluation_dataset.json` | 164.9 KB | 42 | 42 Biomechanical Features + 21 Landmarks | 12 Classes (A, B, C, D, E, F, L, O, R, U, V, Y) | Genuine Physical Real-Human Video Captures |
| `backend/data/dynamic_jz_trajectories.json` | 1.55 MB | 600 | 32 Temporal Features + Point Sequences | 3 Dynamic Classes (J, Z, NEUTRAL) | Synthetic Kinematic Trajectory Generation |
| `backend/titan/datasets/asl_alphabet/metadata/real_human_benchmark.json` | 164.2 KB | 42 | Canonical JSON Benchmark Manifest | 12 Classes (Identical benchmark baseline) | Curated Real-Human Verification Baseline |

---

## 2. Participant & Split Distribution

### A. Synthetic Multi-Signer Dataset (`alphabet_dataset_multisigner.json`)
- **Total Records:** 4,000 samples (160 samples per class across 25 classes).
- **Participant Split Strategy:** Strict leave-one-signer-out partitioning:
  - `Signer_1` (1,000 samples) -> **Training Split** (50% of training cohort)
  - `Signer_2` (1,000 samples) -> **Training Split** (50% of training cohort)
  - `Signer_3` (1,000 samples) -> **Validation Split** (100% of validation cohort)
  - `Signer_4` (1,000 samples) -> **Held-Out Test Split** (100% of test cohort)

### B. Real-Human Benchmark Dataset (`real_human_evaluation_dataset.json`)
- **Total Records:** 42 genuine samples.
- **Participant Distribution:**
  - `Signer_1` (8 captures) -> **Training Split** (development tuning)
  - `Signer_2` (7 captures) -> **Training Split** (development tuning)
  - `Signer_3` (9 captures) -> **Held-Out Test Split** (unseen signer)
  - `Signer_4` (6 captures) -> **Held-Out Test Split** (unseen signer)
  - `Signer_5` (6 captures) -> **Held-Out Test Split** (unseen signer)
  - `Signer_6` (6 captures) -> **Held-Out Test Split** (unseen signer)
- **Held-Out Test Cohort Size:** 27 samples across 9 classes.

### C. Dynamic Trajectory Dataset (`dynamic_jz_trajectories.json`)
- **Total Episodes:** 600 episodes (200 J, 200 Z, 200 NEUTRAL).
- **Signer Allocation:** 5 synthetic signers (`Signer_1` through `Signer_5`, 120 episodes each).
- **Split Allocation:** Train (Signer 1 & 2: 240), Val (Signer 3: 120), Held-Out Test (Signer 4 & 5: 240).

---

## 3. Comprehensive Per-Class Provenance Table (A through Z)

| Class | Modality | Synthetic Samples (Tr/Val/Te) | Real Human Captures | Real Test Support | Unique Signers | Evidence Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **A** | Static | 80 / 40 / 40 | 3 | 2 | 4 | REAL-HUMAN VALIDATION AVAILABLE |
| **B** | Static | 80 / 40 / 40 | 2 | 1 | 4 | REAL-HUMAN VALIDATION AVAILABLE |
| **C** | Static | 80 / 40 / 40 | 10 | 10 | 6 | REAL-HUMAN VALIDATION AVAILABLE |
| **D** | Static | 80 / 40 / 40 | 1 | 1 | 4 | REAL-HUMAN VALIDATION INSUFFICIENT |
| **E** | Static | 80 / 40 / 40 | 1 | 0 | 4 | REAL-HUMAN VALIDATION INSUFFICIENT |
| **F** | Static | 80 / 40 / 40 | 1 | 0 | 4 | REAL-HUMAN VALIDATION INSUFFICIENT |
| **G** | Static | 80 / 40 / 40 | 0 | 0 | 4 | SYNTHETIC VALIDATION ONLY |
| **H** | Static | 80 / 40 / 40 | 0 | 0 | 4 | SYNTHETIC VALIDATION ONLY |
| **I** | Static | 80 / 40 / 40 | 0 | 0 | 4 | SYNTHETIC VALIDATION ONLY |
| **J** | Dynamic | 80 / 40 / 80 | 0 | 0 | 5 | SYNTHETIC VALIDATION ONLY |
| **K** | Static | 80 / 40 / 40 | 0 | 0 | 4 | SYNTHETIC VALIDATION ONLY |
| **L** | Static | 80 / 40 / 40 | 9 | 2 | 4 | REAL-HUMAN VALIDATION AVAILABLE |
| **M** | Static | 80 / 40 / 40 | 0 | 0 | 4 | SYNTHETIC VALIDATION ONLY |
| **N** | Static | 80 / 40 / 40 | 0 | 0 | 4 | SYNTHETIC VALIDATION ONLY |
| **O** | Static | 80 / 40 / 40 | 2 | 2 | 4 | REAL-HUMAN VALIDATION AVAILABLE |
| **P** | Static | 80 / 40 / 40 | 0 | 0 | 4 | SYNTHETIC VALIDATION ONLY |
| **Q** | Static | 80 / 40 / 40 | 0 | 0 | 4 | SYNTHETIC VALIDATION ONLY |
| **R** | Static | 80 / 40 / 40 | 2 | 1 | 4 | REAL-HUMAN VALIDATION AVAILABLE |
| **S** | Static | 80 / 40 / 40 | 0 | 0 | 4 | SYNTHETIC VALIDATION ONLY |
| **T** | Static | 80 / 40 / 40 | 0 | 0 | 4 | SYNTHETIC VALIDATION ONLY |
| **U** | Static | 80 / 40 / 40 | 1 | 0 | 4 | REAL-HUMAN VALIDATION INSUFFICIENT |
| **V** | Static | 80 / 40 / 40 | 4 | 2 | 4 | REAL-HUMAN VALIDATION AVAILABLE |
| **W** | Static | 80 / 40 / 40 | 0 | 0 | 4 | SYNTHETIC VALIDATION ONLY |
| **X** | Static | 80 / 40 / 40 | 0 | 0 | 4 | SYNTHETIC VALIDATION ONLY |
| **Y** | Static | 80 / 40 / 40 | 6 | 6 | 6 | REAL-HUMAN VALIDATION AVAILABLE |
| **Z** | Dynamic | 80 / 40 / 80 | 0 | 0 | 5 | SYNTHETIC VALIDATION ONLY |
| **NEUTRAL**| Static | 80 / 40 / 40 | 0 | 0 | 4 | SYNTHETIC VALIDATION ONLY |

---

## 4. Cryptographic Integrity & Leakage Verification

- **Total Unique Sample Hashes:** 4,038
- **Duplicate Hash Groups:** 2 (within synthetic noise margin; zero cross-split overlap)
- **Cross-Split Leakage Groups:** **0**
- **Participant Overlap:** Strict zero-overlap between training signers (`Signer_1`, `Signer_2`) and evaluation signers (`Signer_3` through `Signer_6`).
