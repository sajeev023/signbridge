# SIGNBRIDGE — VIRTUAL ALPHABET INITIAL REPOSITORY AUDIT

**Audit Date:** 2026-11-14T20:44:00+05:30  
**Repository:** SIGNBRIDGE  
**Branch:** `experiment/letter-a-mastery`  
**Latest Commit:** `229f6f97c253c1c19710c28b88c8e28694edcc85` (`feat(alphabet): add 26-letter evidence acquisition suite and failure analyses`)  
**Remote Sync:** In sync with `origin/experiment/letter-a-mastery`  
**Working Tree Status:** Clean  

---

## 1. Cryptographic Baseline & Freeze Audit

### A. Letter A Frozen Candidate Manifest (`backend/letter_a_frozen_manifest.json`)
All 7 critical components verified against their frozen SHA-256 digests:

| File Path | File Size | SHA-256 Digest | Status |
|---|---|---|---|
| `backend/saved_models/letter_a_specialist_model.json` | 65,980 bytes | `dc7cd03a4ed647a7ec570422ed54d130bbb250d6cded42b93466ef6e967d14cf` | **VERIFIED** |
| `backend/train_letter_a_specialist.py` | 16,558 bytes | `29347b5565d9f79882865a36cb0c7f612a1c88b27c759275e976a496e8946377` | **VERIFIED** |
| `backend/alphabet_feature_extractor.py` | 8,752 bytes | `a6e0a483df4291b8633731afd79f12546fa923e6b5d172e3bf32edf1ce4c4c72` | **VERIFIED** |
| `frontend/src/services/alphabetInferenceEngine.ts` | 21,906 bytes | `2e34e9bb39dbe50afe1f0e651df9ef994ca4844412ab6b662d0818c9e58b8930` | **VERIFIED** |
| `backend/data/alphabet_dataset_multisigner.json` | 17,343,481 bytes | `0cfb68c66f5b752ead02f66e5d6e5e7b671d36dc11660126bc1d019010cc7ade` | **VERIFIED** |
| `backend/titan/datasets/asl_alphabet/metadata/dataset_manifest.json` | 1,103 bytes | `3fc93a0f0a514b936a3123671d3ea6469166b48cf773b4b4bdcb5b948f15c13a` | **VERIFIED** |
| `backend/titan/datasets/asl_alphabet/metadata/real_human_benchmark.json` | 164,181 bytes | `997434c7eea07f35c356eefdcd85f7a165a8f200fe7a074062cf7e0ebcc0c2ef` | **VERIFIED** |

### B. Active Model File Checksums
Active production and research model files verified:

| Active Model Artifact | Size | SHA-256 Digest | Role |
|---|---|---|---|
| `backend/saved_models/single_hand_alphabet_model.json` | 176,808 bytes | `aa672b6fffd96f4aed694c76edfc87385cb053d53a157a56bfafc3ed7ace4272` | 24-class static ASL MLP (`NEUTRAL` + A-Y without J, Z) |
| `backend/saved_models/dynamic_alphabet_model.json` | 51,881 bytes | `0f77f12a282162a34d0cb0804c53733a8ffc65e8361d3ba42265791a0b80014c` | 3-class dynamic trajectory MLP (`NEUTRAL`, `J`, `Z`) |
| `backend/saved_models/isl_alphabet_model.json` | 136,077 bytes | `191f0c4d824869e20cf2ae2615775229c082c0f5927a35b574114f4cf16ca762` | Bimanual ISLRTC alphabet model |
| `backend/saved_models/titan_asl_spatial_model.json` | 1,264,955 bytes | `aa3ff132052890b7ddc4832f5aa1e000e8e949d205bfb1f837fa7f461d16e04f` | TITAN ASL Spatial Graph Neural Network |

---

## 2. Architecture Inspection

### A. 26-Letter Inference Pipeline Topology
SIGNBRIDGE splits 26-letter ASL recognition into a coordinated dual-engine architecture:
1. **Static 24-Letter Engine (`single_hand_alphabet_model.json`):**
   - Classes: 25 (`NEUTRAL`, A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y).
   - Architecture: MLP (Input: 42 -> Hidden1: 64 -> Hidden2: 32 -> Logits: 25 -> Softmax).
   - Input: Canonical 42 continuous biomechanical features extracted from 21 MediaPipe 3D coordinates.
2. **Dynamic 2-Letter Trajectory Engine (`dynamic_alphabet_model.json`):**
   - Classes: 3 (`NEUTRAL`, `J`, `Z`).
   - Architecture: MLP (Input: 32 -> Hidden1: 32 -> Hidden2: 16 -> Logits: 3 -> Softmax).
   - Input: 16-frame rolling circular buffer resampled uniformly by arc-length, normalized by palm scale.
   - Gating: Static handshape context gate (requires `I`-pose handshape for J; requires `D`-pose handshape for Z).
3. **Specialist Binary Gate (`letter_a_specialist_model.json`):**
   - Target: Letter A vs Confusable Fist Classes (E, M, N, S, T).
   - Architecture: MLP with BatchNorm and temperature scaling ($T = 1.05$), coupled with 5 hard geometric kinematic rules:
     - 4 fingers curled ($Tip_i - MCP_i < 0.25$)
     - Thumb upright ($Y_{tip} < Y_{mcp}$)
     - Thumb lateral ($|X_{tip} - X_{index\_mcp}| < 0.35$)
     - Thumb not crossing palm ($X_{tip} < X_{mid\_mcp}$)
     - Thumb not tucked ($Z_{tip} > Z_{knuckle}$)
4. **TITAN Spatial GNN Engine (`titan_asl_spatial_model.json`):**
   - Graph representation: 21 nodes, skeletal adjacency matrix, normalized Laplacian, coordinate-invariant node features.

### B. Feature Extraction Integrity (`backend/alphabet_feature_extractor.py`)
- **42 Continuous Biomechanical Features:**
  - Index 0–4: 5 Finger Extensions (Tip-to-wrist minus MCP-to-wrist / scale).
  - Index 5–9: 5 Fingertip to Palm Center Distances / scale.
  - Index 10–14: 5 Finger Curl Metrics (Tip-to-MCP distance / scale).
  - Index 15–18: 4 Inter-finger Bone Vector Spread Cosines.
  - Index 19–23: Key Inter-fingertip Euclidean Distances / scale.
  - Index 24–28: Canonical 3D Thumb Coordinates & Opposition.
  - Index 29–31: Relative Finger Crossing Metrics (for R vs U vs V).
  - Index 32–34: Canonical Direction Vectors (for horizontal G, H vs upright D, U).
  - Index 35–37: Palm Curl & Fist Compactness.
  - Index 38–41: Structural Invariant Ratios (spread ratio, y_span, parallelism, loop closure).
- **Coordinate Transformation:**
  - Origin at joint 0 (wrist).
  - Scale: Euclidean distance $|MCP_{mid} - Wrist|$ (clamped $\ge 0.035$).
  - Orthonormal basis: $vec_y = (MCP_{mid} - Wrist)/scale$, $vec_z = (MCP_{idx} - Wrist) \times (MCP_{pinky} - Wrist)$, $vec_x = vec_y \times vec_z$.

### C. ZeroLeak Collector Integrity (`frontend/src/components/ZeroLeakDataCollector.tsx`)
- Hardened with explicit per-letter anatomical instructions for all 26 classes.
- Dynamic visual reference crops linked to `/references/crops/${selectedLetter}.png`.
- Dedicated J and Z dynamic trajectory mode with visual stroke completion bar and 16-point arc-length resampling.
- Pre-flight quality gates: MediaPipe visibility $>0.7$, boundary padding $>0.08$, non-PII SHA-256 fingerprinting.

---

## 3. Test Suites and Baseline Status

- **Existing Backend Suite:** 123 tests passing (`pytest backend/tests/ -q` executed in 20.22s, 0 failures).
- **Frontend Production Build:** Clean compilation in 2.61s (`npm run build` completed, 0 TypeScript/Vite errors).
- **Audit Conclusion:** The testing baseline is fully verified and locked. All seven candidate files remain bit-for-bit preserved.
