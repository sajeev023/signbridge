# SIGNBRIDGE TITAN — ZERO-LEAK MODEL FREEZE REGISTER
## Cryptographic Provenance, Hyperparameter Immutability & Mathematical Specification

**Register Version:** 1.0.0  
**Research Mission:** OPERATION ZERO-LEAK  
**Repository Branch:** `experiment/titan-zero-leak`  
**Reference Commit:** `9489e8c`  
**Freezing Authority:** SIGNBRIDGE Deep Learning Research Team  
**Date:** 2026-11-14  

---

## 1. Cryptographic Model Hash Register

All models evaluated under **OPERATION ZERO-LEAK** are strictly frozen. The table below provides SHA-256 cryptographic hashes for every model artifact across backend and frontend directories. Any alteration of weight parameters, biases, layer counts, or JSON ordering will invalidate these hashes.

### 1.1 SHA-256 Hash Table
| Model Artifact | File Path | File Size | SHA-256 Checksum | Freezing Status |
| :--- | :--- | :--- | :--- | :--- |
| **TITAN AK-GCN (Spatial ASL)** | `backend/saved_models/titan_asl_spatial_model.json` | 1,264,955 B | `aa3ff132052890b7ddc4832f5aa1e000e8e949d205bfb1f837fa7f461d16e04f` | **FROZEN (IMMUTABLE)** |
| **TITAN Client Browser Copy** | `frontend/src/services/titan_asl_spatial_model.json` | 1,264,955 B | `aa3ff132052890b7ddc4832f5aa1e000e8e949d205bfb1f837fa7f461d16e04f` | **FROZEN (BIT-IDENTICAL)** |
| **Baseline Pre-Gen Model** | `backend/saved_models/titan_asl_spatial_model.pre_gen.json` | 1,250,530 B | `2999532e8901dec0f5048d9e5a5bfa90ce56b23b60b5c79ba8968aa7d7485092` | **FROZEN BASELINE** |
| **SIGNBRIDGE Single-Hand** | `backend/saved_models/single_hand_alphabet_model.json` | 176,808 B | `aa672b6fffd96f4aed694c76edfc87385cb053d53a157a56bfafc3ed7ace4272` | **FROZEN PRODUCTION** |
| **Dynamic Trajectory MLP** | `backend/saved_models/dynamic_alphabet_model.json` | 51,881 B | `0f77f12a282162a34d0cb0804c53733a8ffc65e8361d3ba42265791a0b80014c` | **FROZEN DYNAMIC** |

> [!IMPORTANT]
> The backend and frontend copies of `titan_asl_spatial_model.json` share the exact cryptographic hash `aa3ff132052890b7ddc4832f5aa1e000e8e949d205bfb1f837fa7f461d16e04f`, guaranteeing 100% numerical parity between Python validation and client-side browser deployment.

---

## 2. Mathematical Architecture Specification

### 2.1 Spatial Adaptive Kinematic GCN (AK-GCN)
The frozen spatial model processes a 21-node anatomical hand skeleton graph with dual information pathways:

```
    Node Features X (21 x 6)              Auxiliary 42 Features
  [Wrist-rel (3) + Parent-rel (3)]       [Distances, Angles, Curvature]
               |                                       |
     Joint Embedding (21 x 8)                          |
               |                                       |
      Input Projection (64)                            |
               |                                       |
    +----------v----------+                            |
    | AK-GCN Layer 1      |                            |
    | A_hat + A_learned   |                            |
    | ResConn + LayerNorm |                            |
    +----------+----------+                            |
               |                                       |
    +----------v----------+                            |
    | AK-GCN Layer 2      |                            |
    | A_hat + A_learned   |                            |
    | ResConn + LayerNorm |                            |
    +----------+----------+                            |
               |                                       |
        Mean-Max Pooling                               |
        (128-dim vector)                               |
               \                                      /
                \                                    /
                 +------------->(+) <---------------+
                                 |
                          Fusion Layer (170 -> 128)
                                 |
                             GELU + LN
                                 |
                         Classifier (128 -> 27)
                                 |
                        Temperature Scaling
                              (T = 1.08)
                                 |
                       Calibrated Posterior P(y|x)
```

1. **Graph Message Passing Formula:**
   $$\tilde{A} = \hat{A} + \sigma(A_{\text{learned}})$$
   $$H^{(l+1)} = \text{LayerNorm}\left(\text{GELU}\left(\tilde{A} H^{(l)} W^{(l)} + b^{(l)}\right) + H^{(l)}\right)$$
   Where:
   - $\hat{A} = \tilde{D}^{-\frac{1}{2}} (A + I) \tilde{D}^{-\frac{1}{2}}$ is the normalized Kipf-Welling Laplacian ($21 \times 21$).
   - $A_{\text{learned}} \in \mathbb{R}^{21 \times 21}$ is the frozen learned kinematic attention matrix.
   - $W^{(l)} \in \mathbb{R}^{64 \times 64}$ are the layer projection weights.

2. **Dual Readout (Mean-Max Pooling):**
   $$h_{\text{graph}} = \left[ \frac{1}{21} \sum_{i=1}^{21} H^{(2)}_i \;\parallel\; \max_{i=1 \dots 21} H^{(2)}_i \right] \in \mathbb{R}^{128}$$

3. **Auxiliary Biomechanical Pathway:**
   A 42-dimensional vector capturing 5 fingertip-to-wrist normalized distances, 10 inter-fingertip spans, 15 joint flexion angles, and 12 palm planarity values:
   $$z_{\text{fused}} = \text{LayerNorm}\left(\text{GELU}\left(W_f [h_{\text{graph}} \parallel x_{\text{aux}}] + b_f\right)\right) \in \mathbb{R}^{128}$$
   $$\text{logits} = W_c z_{\text{fused}} + b_c \in \mathbb{R}^{27}$$

---

## 3. Frozen Calibration & Selective Rejection Parameters

The inference engine enforces strict decision boundaries for calibrated selective rejection. In accordance with OPERATION ZERO-LEAK, all values are permanently frozen:

```
+-----------------------------------+--------------------+------------------------------------------+
| Hyperparameter                    | Frozen Value       | Operational Function                     |
+-----------------------------------+--------------------+------------------------------------------+
| Temperature Scaling (T)           | 1.08               | Softmax entropy smoothing                |
| Minimum Confidence (tau_conf)     | 0.55 (55.0%)       | Minimum acceptable posterior probability |
| Maximum Entropy (tau_entropy)     | 1.35 nats          | Shannon entropy threshold                |
| Number of Output Classes          | 27 (A-Z + NEUTRAL) | Vocabulary coverage                      |
| Trajectory Buffer Length          | 16 frames          | Dynamic gesture stroke window            |
| Dynamic Palm Scale Normalizer     | ||wrist - midMCP|| | Distance-invariant coordinate scaling   |
+-----------------------------------+--------------------+------------------------------------------+
```

### 3.1 Rejection Gate Invariants
A prediction $\hat{y} = \arg\max_{c} P(y=c \mid x)$ is accepted if and only if:
$$P(y=\hat{y} \mid x) \ge 0.55 \quad \text{AND} \quad \mathcal{H}(P) \le 1.35$$
Where:
$$\mathcal{H}(P) = -\sum_{c=1}^{27} P(y=c \mid x) \ln P(y=c \mid x)$$

Any sample failing either condition is assigned `isRejected = true` with `rejectionReason: LOW_CONFIDENCE | HIGH_ENTROPY`.
