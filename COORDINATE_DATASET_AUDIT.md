# SIGNBRIDGE — Coordinate Dataset Audit & Provenance Verification
**Document ID:** AUDIT-COORD-DATA-2026-V1  
**Scope:** Multi-Signer Coordinate Datasets & Genuine Physical-Human Cohorts  
**Status:** VERIFIED & SEALED  

---

## 1. Summary of Repository Landmark Datasets

The repository maintains an absolute distinction between **synthetic multi-signer coordinate datasets** and **genuine physical-human camera validation captures**.

| Dataset File | Format | Samples | Signers | Classes | Data Type | Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `backend/data/alphabet_dataset_multisigner.json` | JSON | 3,840 | 16 | 24 (A-Y exc J,Z) | Coordinate (21 LMs) | Architecture Training & Split Evaluation |
| `backend/titan/datasets/asl_alphabet/metadata/real_human_benchmark.json` | JSON | 42 | 6 (P001-P006) | 12 (A,B,C,D,E,F,L,O,R,U,V,Y) | Physical Camera Landmarks | Genuine Physical Validation Cohort |
| `backend/data/dynamic_gestures.json` | JSON | 320 | 8 | 2 (J, Z) | 16-frame Trajectories | Dynamic Sign Evaluation |

---

## 2. Multi-Signer Dataset Partitioning (3,840 Samples)

The canonical training and evaluation dataset `alphabet_dataset_multisigner.json` is strictly partitioned across non-overlapping participants to prevent identity leakage:

- **Total Samples:** 3,840
- **Samples per Class:** Exactly 160 per class across all 24 static letters.
- **Participant Split Distribution:**
  - **Train Split:** 1,920 samples (50.0%) — Signers S01–S08
  - **Validation Split:** 960 samples (25.0%) — Signers S09–S12
  - **Test Split:** 960 samples (25.0%) — Signers S13–S16 (Held-out signers)

```
Split Verification:
├── Train: 1,920 samples [80 per class x 24 classes] (Signers: S01, S02, S03, S04, S05, S06, S07, S08)
├── Val:     960 samples [40 per class x 24 classes] (Signers: S09, S10, S11, S12)
└── Test:    960 samples [40 per class x 24 classes] (Signers: S13, S14, S15, S16)
```

---

## 3. Genuine Physical-Human Dataset Audit (42 Samples)

The physical validation benchmark represents genuine camera captures from 6 human participants with zero data fabrication:

- **Total Captures:** 42 genuine samples
- **Unique Human Signers:** 6 (P001, P002, P003, P004, P005, P006)
- **Represented Classes (12):** A, B, C, D, E, F, L, O, R, U, V, Y
- **Pending Physical Classes (14):** G, H, I, J, K, M, N, P, Q, S, T, W, X, Z
- **Privacy Compliance:**
  - 100% numerical coordinates only.
  - Zero raw RGB images, facial video, or audio collected or stored.
  - Signed voluntary consent recorded in session metadata.

---

## 4. Frozen Letter A Candidate Hash Verification

All seven candidate artifacts recorded in `backend/letter_a_frozen_manifest.json` have been verified bit-for-bit identical via SHA-256:

| Artifact Path | Expected SHA-256 | Verified Status |
| :--- | :--- | :--- |
| `backend/saved_models/letter_a_specialist_model.json` | `dc7cd03a4ed647a7ec570422ed54d130bbb250d6cded42b93466ef6e967d14cf` | MATCH |
| `backend/train_letter_a_specialist.py` | `29347b5565d9f79882865a36cb0c7f612a1c88b27c759275e976a496e8946377` | MATCH |
| `backend/alphabet_feature_extractor.py` | `a6e0a483df4291b8633731afd79f12546fa923e6b5d172e3bf32edf1ce4c4c72` | MATCH |
| `frontend/src/services/alphabetInferenceEngine.ts` | `2e34e9bb39dbe50afe1f0e651df9ef994ca4844412ab6b662d0818c9e58b8930` | MATCH |
| `backend/data/alphabet_dataset_multisigner.json` | `0cfb68c66f5b752ead02f66e5d6e5e7b671d36dc11660126bc1d019010cc7ade` | MATCH |
| `backend/titan/datasets/asl_alphabet/metadata/dataset_manifest.json` | `3fc93a0f0a514b936a3123671d3ea6469166b48cf773b4b4bdcb5b948f15c13a` | MATCH |
| `backend/titan/datasets/asl_alphabet/metadata/real_human_benchmark.json` | `997434c7eea07f35c356eefdcd85f7a165a8f200fe7a074062cf7e0ebcc0c2ef` | MATCH |
