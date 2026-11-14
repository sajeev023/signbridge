# SIGNBRIDGE — ASL LETTER A SPECIALIST
## Frozen Evaluation Candidate Manifest & Cryptographic Baseline

**Branch**: `experiment/letter-a-mastery`  
**Commit Baseline**: `a0e4577`  
**Freezing Timestamp**: `2026-11-14T17:50:00Z`  
**Status**: CRYPTOGRAPHICALLY FROZEN  

---

### 1. Freezing Policy

To guarantee empirical validity during physical human validation:
1. **Zero-Weight Modification**: Specialist neural network weights are locked. No fine-tuning, calibration re-scaling, or retraining is permitted.
2. **Zero-Threshold Modification**: Decision thresholds, temperature constants, and kinematic bounds are frozen prior to physical data intake.
3. **Invalidation Clause**: Any commit altering any file hash or threshold recorded below automatically invalidates all prior and downstream benchmark claims.

---

### 2. Cryptographic Hashes & File Manifest

All checksums computed using SHA-256 over raw binary contents:

| Component | Relative File Path | Size (Bytes) | SHA-256 Digest |
| :--- | :--- | :--- | :--- |
| **Specialist Weights** | `backend/saved_models/letter_a_specialist_model.json` | 65,980 | `dc7cd03a4ed647a7ec570422ed54d130bbb250d6cded42b93466ef6e967d14cf` |
| **Training & Rules** | `backend/train_letter_a_specialist.py` | 16,558 | `29347b5565d9f79882865a36cb0c7f612a1c88b27c759275e976a496e8946377` |
| **Feature Extractor** | `backend/alphabet_feature_extractor.py` | 8,752 | `a6e0a483df4291b8633731afd79f12546fa923e6b5d172e3bf32edf1ce4c4c72` |
| **Engine Inference** | `frontend/src/services/alphabetInferenceEngine.ts` | 21,906 | `2e34e9bb39dbe50afe1f0e651df9ef994ca4844412ab6b662d0818c9e58b8930` |
| **Multisigner Dataset**| `backend/data/alphabet_dataset_multisigner.json` | 17,343,481 | `0cfb68c66f5b752ead02f66e5d6e5e7b671d36dc11660126bc1d019010cc7ade` |
| **Dataset Manifest** | `backend/titan/datasets/asl_alphabet/metadata/dataset_manifest.json` | 1,103 | `3fc93a0f0a514b936a3123671d3ea6469166b48cf773b4b4bdcb5b948f15c13a` |
| **Human Benchmark** | `backend/titan/datasets/asl_alphabet/metadata/real_human_benchmark.json` | 164,181 | `997434c7eea07f35c356eefdcd85f7a165a8f200fe7a074062cf7e0ebcc0c2ef` |

---

### 3. Frozen Hyperparameters & Decision Thresholds

| Parameter Name | Target Scope | Frozen Value | Verification Rule |
| :--- | :--- | :--- | :--- |
| **Classification Probability Cutoff** | MLP Output | $\tau = 0.50$ | $P(A \mid x) \ge 0.50$ classified as class `A` |
| **Platt Temperature Scaling** | Logit Division | $T = 1.20$ | $\mathbf{z}_{scaled} = \mathbf{z} / 1.20$ |
| **Fist Curl Threshold (Index)** | Kinematic Gate | $ext < 0.35$ | Normalized Euclidean tip-to-wrist extension |
| **Fist Curl Threshold (Middle)** | Kinematic Gate | $ext < 0.35$ | Normalized Euclidean tip-to-wrist extension |
| **Fist Curl Threshold (Ring)** | Kinematic Gate | $ext < 0.35$ | Normalized Euclidean tip-to-wrist extension |
| **Fist Curl Threshold (Pinky)** | Kinematic Gate | $ext < 0.35$ | Normalized Euclidean tip-to-wrist extension |
| **Thumb Radial Elevation** | Kinematic Gate | $thumb\_y \ge 0.55$ | Prevents confusion with Letter E |
| **Thumb Lateral Position** | Kinematic Gate | $thumb\_x \ge 0.12$ | Prevents confusion with Letter S |
| **Thumb Bone Spread Angle** | Kinematic Gate | $cos\_spread < 0.15$ | Prevents thumb crossing knuckles (Letter S) |
| **Thumb Under-Tuck Clearance** | Kinematic Gate | $dist\_thumb\_mid \ge 0.38$ | Prevents thumb tucking into palm (Letters M/N/T) |

---

### 4. Integrity Verification Command

Run the following test to verify candidate hash matching across the repository:
```bash
pytest backend/tests/test_letter_a_collector.py -k test_frozen_candidate_manifest_integrity
```
Expected output: `1 passed`.
