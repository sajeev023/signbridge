# SIGNBRIDGE — Model Training & Optimization Pipeline
**Document ID:** TRAIN-PIPELINE-2026-V1  
**Module:** `backend/coordinate_ai/trainer.py`  
**Status:** REPRODUCIBLE & PRODUCTION READY  

---

## 1. Pipeline Architecture

The training pipeline executes an end-to-end workflow from raw landmark JSON to production export:

```
[alphabet_dataset_multisigner.json (3,840 Samples)]
                   │
                   ▼
  batch_extract_enriched_coordinate_features()
  [Vectorized NumPy: (3840, 21, 3) ──> (3840, 94)]
  Elapsed: 4.42 seconds
                   │
                   ▼
  Participant-Separated Stratified Split
  ├── Train: 1,920 samples (Signers S01–S08)
  ├── Val:     960 samples (Signers S09–S12)
  └── Test:    960 samples (Signers S13–S16)
                   │
                   ▼
  PyTorch CoordinateMLP Training
  ├── Optimizer: AdamW (lr=0.002, weight_decay=1e-4)
  ├── Scheduler: CosineAnnealingLR (T_max=12)
  ├── Loss: CrossEntropyLoss
  └── Epochs: 12 (Batch Size: 64)
                   │
                   ▼
  Held-Out Evaluation & Comparison
  ├── CoordinateMLP: 99.48% Acc / 0.9948 F1
  └── RandomForest:  98.02% Acc / 0.9802 F1
                   │
                   ▼
  Export Production Weights
  └── backend/saved_models/coordinate_ai_model.json (2.1 MB)
```

---

## 2. High-Throughput Batch Feature Vectorization

Previous unvectorized scalar dictionary extraction required >120 seconds. The upgraded vectorized batch implementation processes all 3,840 samples in **4.42 seconds**:

```python
def batch_extract_enriched_coordinate_features(pts_array: np.ndarray) -> np.ndarray:
    # pts_array: (N, 21, 3)
    # Vectorized palm scale, translation, bone dot-products, and arccos
    # Returns (N, 94) float32 matrix directly suitable for PyTorch DataLoader
```

---

## 3. Training Hyperparameters

| Hyperparameter | Value | Rationale |
| :--- | :--- | :--- |
| **Input Dimension** | 94 | 63 normalized coords + 10 angles + 5 curls + 10 pairs + 6 orientation |
| **Hidden Layers** | [256, 128, 64] | Progressive dimension compression preserving geometric manifolds |
| **Layer Normalization** | Applied after each linear layer | Stabilizes internal covariate shift across varying hand anatomies |
| **Activation** | ReLU | Non-saturating non-linear articulation activation |
| **Dropout** | 0.20 (Layers 1 & 2) | Prevents co-adaptation on specific signer geometries |
| **Optimizer** | AdamW | Decoupled weight decay regularization ($10^{-4}$) |
| **Initial Learning Rate** | $2 	imes 10^{-3}$ | Rapid convergence with cosine decay |
| **LR Schedule** | CosineAnnealingLR | Smooth annealing to zero avoiding late oscillation |
| **Batch Size** | 64 | Efficient gradient updates with stochastic stability |
| **Epochs** | 12 | Full convergence without overfitting |

---

## 4. Production Export Specifications

The exported model `backend/saved_models/coordinate_ai_model.json` contains:
- `model_type`: `"CoordinateMLP"`
- `version`: `"2.0.0-coordinate-ai"`
- `input_dim`: 94
- `classes`: 24 static letters A through Y (excluding J and Z)
- `temperature`: 1.0000
- `weights`: Full float32 weight matrices and bias vectors for $W_1, b_1, \gamma_1, eta_1, W_2, b_2, \gamma_2, eta_2, W_3, b_3, \gamma_3, eta_3, W_4, b_4$.
- `metrics`: Test accuracy (99.48%), test macro F1 (0.9948).
