# SIGNBRIDGE TITAN: ARCHITECTURE & THEORY SPECIFICATION
## Autonomous Next-Generation Spatial-Temporal Hand Recognition Engine

**Document ID:** `SB-TITAN-ARCH-01`  
**System Class:** Independent Backup Recognition Architecture  
**Active Research Branch:** `experiment/signbridge-titan`  
**Baseline Production Reference:** Three-Engine System (`main` @ `b3b861c`)  
**Date:** November 13, 2026  
**Status:** VALIDATED RESEARCH ENGINE (Offline Verified & Exported)  

---

## 1. EXECUTIVE SUMMARY & RESEARCH OBJECTIVE

SIGNBRIDGE TITAN is an independent, backup spatial-temporal recognition architecture designed to address the representational ceiling of purely flat feature vectors and rigid heuristics.

While SIGNBRIDGE's production 42-feature regularized MLP demonstrates exceptional inference speed ($11.8\ \mu\text{s}$) and solid static accuracy ($85.19\%$ on held-out human signers), it relies on manually engineered geometric ratios (curl indices, inter-finger distances, and planar normals). It lacks:
1. **Explicit topological joint modeling**: Joint connectivity is implicitly flattened.
2. **Dynamic spatio-temporal coupling**: Handshape configuration and motion trajectories are processed in separate asynchronous pipelines.
3. **Calibrated uncertainty & OOD rejection**: Over-confident predictions can be emitted when presented with ambiguous or out-of-vocabulary gestures.

TITAN resolves these bottlenecks through a 5-stage hybrid architecture:
```
INPUT LANDMARKS (21x3)
        │
        ▼
[SPATIAL GRAPH ENCODER] ◄── Auxiliary 42-Feature Biomechanical Pathway
        │  (z_spatial ∈ R^64)
        ▼
[TEMPORAL SEQUENCE ENCODER] (H_temporal ∈ R^(Tx64), z_temporal ∈ R^64)
        │
        ▼
[CROSS-ATTENTION FUSION] (Q = z_spatial, K,V = H_temporal)
        │  (z_fused ∈ R^64)
        ▼
[CLASSIFICATION HEAD & UNCERTAINTY ESTIMATOR]
        │
        ▼
(Class Predictions, Calibrated Confidence, Predictive Entropy, Energy OOD)
```

---

## 2. SPATIAL GRAPH ENCODER (SGE)

### 2.1 Hand Skeleton Graph Topology
The hand skeleton is modeled as an undirected topological graph $G = (V, E)$ where:
- Nodes $V = \{0, 1, \dots, 20\}$ represent the 21 MediaPipe keypoints:
  - 0: Wrist root
  - Thumb: 1 (CMC), 2 (MCP), 3 (IP), 4 (Tip)
  - Index: 5 (MCP), 6 (PIP), 7 (DIP), 8 (Tip)
  - Middle: 9 (MCP), 10 (PIP), 11 (DIP), 12 (Tip)
  - Ring: 13 (MCP), 14 (PIP), 15 (DIP), 16 (Tip)
  - Pinky: 17 (MCP), 18 (PIP), 19 (DIP), 20 (Tip)
- Edges $E$ comprise 31 connections:
  - 20 anatomical bone edges along kinematic chains.
  - 5 transverse palm arch edges connecting adjacent metacarpal bases: $(5, 9), (9, 13), (13, 17), (0, 5), (0, 17)$.
  - 6 inter-digit closure edges capturing pinch and spread: $(4, 8), (8, 12), (12, 16), (16, 20), (4, 12), (5, 17)$.

### 2.2 Kipf-Welling Normalized Graph Laplacian
Let $A \in \{0, 1\}^{21 \times 21}$ be the binary symmetric adjacency matrix. Augmented with self-loops $\tilde{A} = A + I_{21}$, the degree matrix is $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$. The normalized graph propagation operator is defined as:
$$\tilde{A}_{\text{norm}} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$$
All eigenvalues of $\tilde{A}_{\text{norm}}$ satisfy $|\lambda| \le 1.0$, guaranteeing numerical stability across deep layers.

### 2.3 Node Input Representation
For landmark coordinate $\vec{p}_i = (x_i, y_i, z_i)$, palm scale is $s = \|\vec{p}_9 - \vec{p}_0\|_2 + \epsilon$. Node feature vector $\vec{x}_i \in \mathbb{R}^6$ combines:
1. Scale-invariant coordinates relative to wrist: $(\vec{p}_i - \vec{p}_0) / s \in \mathbb{R}^3$.
2. Segment displacement from parent joint in kinematic tree: $(\vec{p}_i - \vec{p}_{\text{parent}(i)}) / s \in \mathbb{R}^3$.
3. Learned anatomical joint embedding $\vec{e}_i \in \mathbb{R}^{16}$.

Input node tensor: $X \in \mathbb{R}^{21 \times 22}$.

### 2.4 GCN Layers & Dual Readout
The spatial encoder stacks 2 residual graph convolution layers:
$$H^{(l+1)} = \text{LayerNorm}\left( H^{(l)} + \text{GELU}\left( \tilde{A}_{\text{norm}} H^{(l)} W_{\text{neighbor}}^{(l)} + H^{(l)} W_{\text{self}}^{(l)} \right) \right)$$
Graph readout aggregates multi-scale global context via dual pooling (mean + max across all 21 nodes):
$$h_{\text{pool}} = \left[ \frac{1}{21}\sum_{i=1}^{21} H_i^{(2)} \;\Big\|\; \max_{i=1}^{21} H_i^{(2)} \right] \in \mathbb{R}^{128}$$
$$z_{\text{graph}} = \text{LayerNorm}\left(\text{GELU}\left( h_{\text{pool}} W_{\text{read}} \right)\right) \in \mathbb{R}^{64}$$

Auxiliary Biomechanical Pathway: The canonical 42-feature vector $\vec{f}_{42}$ is projected via a 2-layer MLP and fused via residual LayerNorm:
$$z_{\text{spatial}} = \text{LayerNorm}(z_{\text{graph}} + \text{MLP}_{\text{aux}}(\vec{f}_{42})) \in \mathbb{R}^{64}$$

---

## 3. TEMPORAL SEQUENCE ENCODER (TSE)

To eliminate the disconnect between static alphabet recognition and dynamic stroke tracking (e.g. 'J' and 'Z'), TITAN processes temporal landmark sequences over a sliding historical buffer of $T = 16$ frames.

### 3.1 Architectures Evaluated
1. **BiLSTM with Temporal Attention (Architecture B)**:
   - Evaluates bidirectional context: $\vec{h}_t = [\overrightarrow{\text{LSTM}}(x_t) ; \overleftarrow{\text{LSTM}}(x_t)] \in \mathbb{R}^{128}$.
   - Additive temporal attention pooling: $\alpha_t = \text{Softmax}(w^T \tanh(W \vec{h}_t))$.
2. **Lightweight Spatial-Temporal Transformer (Architecture C)**:
   - 2-layer Transformer encoder with $d_{\text{model}} = 64$, 4 attention heads, and sinusoidal positional encoding.
   - Causal sequence masking ensures zero lookahead during real-time camera inference:
     $$M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}$$

---

## 4. CROSS-ATTENTION SPATIAL-TEMPORAL FUSION

Rather than simple concatenation, TITAN's flagship configuration employs Multi-Head Cross-Attention (MHCA):
- **Query ($Q$)**: The static spatial graph embedding $z_{\text{spatial}} \in \mathbb{R}^{B \times 1 \times 64}$ (Querying: *What handshape is currently being articulated?*).
- **Key ($K$) & Value ($V$)**: The temporal sequence representations $H_{\text{temporal}} \in \mathbb{R}^{B \times T \times 64}$ (Querying: *Where and when did characteristic trajectory strokes occur?*).

$$\text{Attention}(Q, K, V) = \text{Softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V$$
$$z_{\text{fused}} = \text{LayerNorm}\left( z_{\text{spatial}} + \text{Dropout}(\text{MHCA}(Q, K, V)) \right)$$

This allows the network to dynamically emphasize motion cues during dynamic letters ('J', 'Z') while collapsing attention to the current frame during motionless static signs ('A', 'B', 'I', 'D').

---

## 5. TWO-HAND DUAL-STREAM EXTENSION

For bimanual signs (ISLRTC vocabulary), TITAN defines symmetric dual encoders:
- Left hand stream: $z_{\text{left}} \in \mathbb{R}^{64}$
- Right hand stream: $z_{\text{right}} \in \mathbb{R}^{64}$
- Cross-Hand Attention:
  - Left queries Right: $h_{\text{L}\to\text{R}} = \text{MHA}(z_{\text{left}}, z_{\text{right}}, z_{\text{right}})$
  - Right queries Left: $h_{\text{R}\to\text{L}} = \text{MHA}(z_{\text{right}}, z_{\text{left}}, z_{\text{left}})$
- Geometric Invariants: Computes relative 3D wrist separation vector $(\vec{w}_{\text{left}} - \vec{w}_{\text{right}})$, Euclidean distance, and palm scale ratio.
- Single-Hand Graceful Fallback: Multiplied by binary presence flags $(m_{\text{left}}, m_{\text{right}})$; automatically reduces to single-hand latent without NaN artifacts when only one hand is visible.

---

## 6. UNCERTAINTY ESTIMATION & SELECTIVE REJECTION

TITAN implements a formal rejection gate to suppress false-positive hallucination:
1. **Temperature-Scaled Calibration**: Logits scaled by validation-fitted scalar $T_{\text{cal}} > 0$:
   $$p_k = \frac{\exp(z_k / T_{\text{cal}})}{\sum_j \exp(z_j / T_{\text{cal}})}$$
2. **Predictive Entropy Gate**:
   $$H(p) = -\sum_{k=1}^K p_k \ln(p_k)$$
   If $H(p) > \tau_{\text{entropy}}$ ($1.35\ \text{nats}$), prediction is rejected as ambiguous.
3. **Out-of-Distribution Energy Score**:
   $$E(\vec{x}; T_{\text{cal}}) = -T_{\text{cal}} \ln \sum_{k=1}^K \exp(z_k / T_{\text{cal}})$$
4. **Selective Decision Boundary**:
   $$\hat{y} = \begin{cases} \arg\max_k p_k & \text{if } \max_k p_k \ge \tau_{\text{conf}} \text{ and } H(p) \le \tau_{\text{entropy}} \\ \text{NEUTRAL / UNCERTAIN} & \text{otherwise} \end{cases}$$
