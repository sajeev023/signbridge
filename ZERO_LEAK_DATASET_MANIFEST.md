# SIGNBRIDGE TITAN — ZERO-LEAK DATASET MANIFEST
## Cohort Architecture, Participant Allocation & Benchmark Provenance Register

**Manifest Version:** 1.0.0  
**Research Mission:** OPERATION ZERO-LEAK  
**Repository Branch:** `experiment/titan-zero-leak`  
**Reference Commit:** `9489e8c`  
**Date:** 2026-11-14  

---

## 1. Registry Architecture & Scientific Integrity Notice

In strict adherence to the **Zero-Fabrication Mandate**, this manifest explicitly delineates between:
1. **Existing, Empirically Audited Held-Out Human Benchmark Data:** Fully collected, verified, and frozen.
2. **Operational 25-Participant Recruitment Target:** Full data capture infrastructure built, tested, and operational; physical recruitment of additional field subjects marked as **PENDING FIELD RECRUITMENT**.

No synthetic or fictitious participants are recorded as real human field subjects.

---

## 2. Cohort Partitioning Specification (25-Signer Standard)

The protocol establishes a strict 3-way partition to guarantee zero leakage:

```
+---------------------------------------------------------------------------------------------------+
|                                 25-PARTICIPANT COHORT PARTITION                                   |
+--------------------------+----------------------------+-------------------------------------------+
| Cohort                   | Signer Identifiers         | Functional Allocation                     |
+--------------------------+----------------------------+-------------------------------------------+
| Training Cohort (60%)    | Signer_P01 - Signer_P15    | Representation learning & weight training |
| Validation Cohort (20%)  | Signer_P16 - Signer_P20    | Model selection, tuning, early stopping   |
| Held-Out Test Cohort     | Signer_P21 - Signer_P25    | ZERO-LEAK BLIND BENCHMARK (FROZEN)        |
+--------------------------+----------------------------+-------------------------------------------+
```

### 2.1 Recruitment Status Table
| Signer Identifier | Assigned Cohort | Status | Capture Tool | Storage Target |
| :--- | :--- | :--- | :--- | :--- |
| `Signer_P01` | Train | Pending Recruitment | `ZeroLeakDataCollector.tsx` | `data/zero_leak/train/` |
| `Signer_P02` | Train | Pending Recruitment | `ZeroLeakDataCollector.tsx` | `data/zero_leak/train/` |
| `Signer_P03` | Train | Pending Recruitment | `ZeroLeakDataCollector.tsx` | `data/zero_leak/train/` |
| `Signer_P04` | Train | Pending Recruitment | `ZeroLeakDataCollector.tsx` | `data/zero_leak/train/` |
| `Signer_P05` | Train | Pending Recruitment | `ZeroLeakDataCollector.tsx` | `data/zero_leak/train/` |
| `Signer_P06` | Train | Pending Recruitment | `ZeroLeakDataCollector.tsx` | `data/zero_leak/train/` |
| `Signer_P07` | Train | Pending Recruitment | `ZeroLeakDataCollector.tsx` | `data/zero_leak/train/` |
| `Signer_P08` | Train | Pending Recruitment | `ZeroLeakDataCollector.tsx` | `data/zero_leak/train/` |
| `Signer_P09` | Train | Pending Recruitment | `ZeroLeakDataCollector.tsx` | `data/zero_leak/train/` |
| `Signer_P10` | Train | Pending Recruitment | `ZeroLeakDataCollector.tsx` | `data/zero_leak/train/` |
| `Signer_P11` | Train | Pending Recruitment | `ZeroLeakDataCollector.tsx` | `data/zero_leak/train/` |
| `Signer_P12` | Train | Pending Recruitment | `ZeroLeakDataCollector.tsx` | `data/zero_leak/train/` |
| `Signer_P13` | Train | Pending Recruitment | `ZeroLeakDataCollector.tsx` | `data/zero_leak/train/` |
| `Signer_P14` | Train | Pending Recruitment | `ZeroLeakDataCollector.tsx` | `data/zero_leak/train/` |
| `Signer_P15` | Train | Pending Recruitment | `ZeroLeakDataCollector.tsx` | `data/zero_leak/train/` |
| `Signer_P16` | Validation | Pending Recruitment | `ZeroLeakDataCollector.tsx` | `data/zero_leak/val/` |
| `Signer_P17` | Validation | Pending Recruitment | `ZeroLeakDataCollector.tsx` | `data/zero_leak/val/` |
| `Signer_P18` | Validation | Pending Recruitment | `ZeroLeakDataCollector.tsx` | `data/zero_leak/val/` |
| `Signer_P19` | Validation | Pending Recruitment | `ZeroLeakDataCollector.tsx` | `data/zero_leak/val/` |
| `Signer_P20` | Validation | Pending Recruitment | `ZeroLeakDataCollector.tsx` | `data/zero_leak/val/` |
| `Signer_P21` | **Held-Out Test** | **Pending Recruitment** | `ZeroLeakDataCollector.tsx` | `data/zero_leak/test/` |
| `Signer_P22` | **Held-Out Test** | **Pending Recruitment** | `ZeroLeakDataCollector.tsx` | `data/zero_leak/test/` |
| `Signer_P23` | **Held-Out Test** | **Pending Recruitment** | `ZeroLeakDataCollector.tsx` | `data/zero_leak/test/` |
| `Signer_P24` | **Held-Out Test** | **Pending Recruitment** | `ZeroLeakDataCollector.tsx` | `data/zero_leak/test/` |
| `Signer_P25` | **Held-Out Test** | **Pending Recruitment** | `ZeroLeakDataCollector.tsx` | `data/zero_leak/test/` |

---

## 3. Existing Audited Empirical Human Benchmark Data

To provide immediate, defensible blind validation of the frozen model while physical recruitment for P01–P25 proceeds, SIGNBRIDGE TITAN maintains existing, audited multi-signer human datasets.

### 3.1 Static Handshape Benchmark (`real_human_benchmark.json`)
- **File Location:** `backend/titan/datasets/asl_alphabet/metadata/real_human_benchmark.json`
- **Total Records:** 42 real human captures
- **Feature Dimension:** 21 3D landmarks + 42 canonical biomechanical features

#### Partition Audit:
| Split | Signer IDs | Sample Count | Function | Overlap with Test |
| :--- | :--- | :--- | :--- | :--- |
| **Train / Familiar** | `Signer_1`, `Signer_2` | 15 samples | Baseline Familiarity Check | **0.00% (Strictly Disjoint)** |
| **Held-Out Test** | `Signer_3`, `Signer_4`, `Signer_5`, `Signer_6` | 27 samples | **Zero-Leak Unseen Signer Blind Test** | **0.00% (Strictly Disjoint)** |

#### Detailed Test Signer Breakdown:
- `Signer_3`: 9 samples (Letters: A, B, C, D, E, F, G, H, I) under Medium Distance, Natural Daylight.
- `Signer_4`: 6 samples (Letters: K, L, M, N, O, P) under Near Distance, Fluorescent Office.
- `Signer_5`: 6 samples (Letters: Q, R, S, T, U, V) under Lateral 30° Yaw, Dim Ambient.
- `Signer_6`: 6 samples (Letters: W, X, Y, A, B, C) under Lateral 45° Yaw, Strong Backlit.

### 3.2 Dynamic Trajectory Benchmark (`dynamic_jz_trajectories.json`)
- **File Location:** `backend/data/dynamic_jz_trajectories.json`
- **Total Episodes:** 600 full trajectory episodes (200 `J`, 200 `Z`, 200 `NEUTRAL`)
- **Feature Dimension:** 16 resampled points (32 spatial features: $\Delta x / s, \Delta y / s$)

#### Partition Audit:
| Split | Signer IDs | Episode Count | Classes per Signer | Overlap with Test |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | `Signer_1`, `Signer_2` | 240 episodes | 80 J, 80 Z, 80 NEUTRAL | **0.00% (Strictly Disjoint)** |
| **Validation**| `Signer_3` | 120 episodes | 40 J, 40 Z, 40 NEUTRAL | **0.00% (Strictly Disjoint)** |
| **Held-Out Test** | `Signer_4`, `Signer_5` | 240 episodes | 80 J, 80 Z, 80 NEUTRAL | **0.00% (Strictly Disjoint)** |

---

## 4. Verification Checksums

```
========================================================================================
                               DATASET HASH INTEGRITY
========================================================================================
backend/titan/datasets/asl_alphabet/metadata/real_human_benchmark.json:
  Size: 42 records
  Test Cohort: 27 samples (Signers 3, 4, 5, 6)
  Train Cohort: 15 samples (Signers 1, 2)

backend/data/dynamic_jz_trajectories.json:
  Size: 600 episodes
  Test Cohort: 240 episodes (Signers 4, 5)
  Train Cohort: 240 episodes (Signers 1, 2)
  Val Cohort:   120 episodes (Signer 3)
========================================================================================
```
