# SIGNBRIDGE TITAN: ASL SPATIAL-TEMPORAL ARCHITECTURE REPORT
## Reference-Aligned Graph Convolutions, Temporal Trajectories & Confidence-Aware Fusion

**Document ID:** `SB-TITAN-ASL-ARCH-01`  
**System Class:** Spatial-Temporal Deep Learning Architecture  
**Active Research Branch:** `experiment/signbridge-titan`  
**Date:** November 13, 2026  
**Status:** SPECIFIED, IMPLEMENTED & BENCHMARKED  

---

## 1. ARCHITECTURAL OVERVIEW

The SIGNBRIDGE TITAN ASL manual alphabet system extends the canonical TITAN spatial-temporal framework to full American Sign Language ($A$ through $Z$, plus `NEUTRAL`) under the structural guidance of the Gérard Aflague Collection visual reference:

```
INPUT MEDIA-PIPE 21 3D LANDMARKS (t=0..T)
   │
   ├───> SPATIAL PATHWAY (Every Frame, < 0.07 ms)
   │     │
   │     ├── Joint Positional Embeddings (16-dim)
   │     ├── Kinematic Parent Relative Coordinates (6-dim)
   │     ├── 2-Layer Residual GCN with Kipf-Welling Laplacian
   │     ├── Dual Readout Pooling (Mean || Max = 128-dim)
   │     └── Auxiliary 42-Feature Biomechanical Pathway
   │           │
   │           └── Static Class Logits z_spatial in R^27
   │
   └───> TEMPORAL PATHWAY (Buffer T=16 Frames, J & Z Only)
         │
         ├── Fingertip Motion Tracking (Index L8 for Z, Pinky L20 for J)
         ├── 2-Layer Bidirectional LSTM with Temporal Attention
         └── Trajectory Latent Vector z_traj in R^64
               │
               v
   MOTION-GATED ARBITRATOR & FUSION
   ├── Velocity & Curvature Gate: gamma = sigma(10 * E_motion - 2.0)
   ├── Dynamic Logits: J (idx 9) & Z (idx 26) modulated by gamma
   └── Static Suppression: If gamma < 0.1, J & Z penalized by -2.5
         │
         v
   UNCERTAINTY ESTIMATION & REJECTION GATE
   ├── Temperature Scaling (T* = 1.08)
   ├── Predictive Entropy H(y|x) = -sum p_i * log(p_i)
   ├── Free Energy Score E(x) = -T * logsumexp(z / T)
   └── Selective Rejection: Suppress if P_cal < 0.55 or H > 1.35
```

---

## 2. SPATIAL GRAPH CONVOLUTIONAL ENCODER

### 2.1 Graph Laplacian Construction
The human hand skeleton is modeled as an undirected graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ with $|\mathcal{V}| = 21$ nodes (MediaPipe landmarks) and $|\mathcal{E}| = 31$ edges:
- **Anatomical Bones (20 edges):** 4 edges per finger along the kinematic chain.
- **Transverse Palm Arch (5 edges):** Inter-metacarpal arches $[5,9]$, $[9,13]$, $[13,17]$, and wrist carpal anchors $[0,5]$, $[0,17]$.
- **Inter-Digit Functional Edges (6 edges):** Tip-to-tip pinch and abduction edges $[4,8]$, $[8,12]$, $[12,16]$, $[16,20]$, $[4,12]$, and hand width $[5,17]$.

With self-loops added ($\tilde{A} = A + I_{21}$), the Kipf-Welling normalized graph Laplacian is computed as:
$$\tilde{L}_{\text{sym}} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}, \quad \tilde{D}_{ii} = \sum_{j} \tilde{A}_{ij}$$

### 2.2 Residual Graph Convolution Layer
To resolve over-smoothing across the 21 nodes while maintaining deep expressive capacity, each GCN layer incorporates an explicit self-transformation, layer normalization, GELU non-linearity, and residual skip connection:
$$H^{(l+1)} = \text{LayerNorm}\left( \text{GELU}\left( \tilde{L}_{\text{sym}} H^{(l)} W_{\text{neighbor}}^{(l)} + H^{(l)} W_{\text{self}}^{(l)} + b^{(l)} \right) \right) + H^{(l)}$$
where $W_{\text{neighbor}}, W_{\text{self}} \in \mathbb{R}^{d \times d}$ ($d=64$).

### 2.3 Dual Readout Graph Pooling
To synthesize the node-level hidden states $H \in \mathbb{R}^{21 \times 64}$ into a graph-level handshape vector, TITAN applies complementary **Mean and Max Readout Pooling**:
$$h_{\text{mean}} = \frac{1}{21} \sum_{i=0}^{20} H_i, \quad h_{\text{max}} = \max_{i \in \{0..20\}} H_i$$
$$h_{\text{pool}} = [h_{\text{mean}} \,\|\, h_{\text{max}}] \in \mathbb{R}^{128}$$
- **Mean pooling** captures the overall spatial posture centroid and palm orientation.
- **Max pooling** captures salient structural extrema (e.g., peak finger extension in `D`, `L`, `U`, `V`, `W`).

### 2.4 Auxiliary Biomechanical Feature Pathway
The pooled GCN representation is augmented with the canonical 42-feature biomechanical vector $f_{\text{aux}} \in \mathbb{R}^{42}$ via a projection subnetwork:
$$z_{\text{aux}} = \text{LayerNorm}\left( W_2 \cdot \text{GELU}(W_1 f_{\text{aux}} + b_1) + b_2 \right) \in \mathbb{R}^{64}$$
$$z_{\text{spatial}} = \text{LayerNorm}\left( \text{GELU}(W_{\text{read}} h_{\text{pool}} + b_{\text{read}}) + z_{\text{aux}} \right) \in \mathbb{R}^{64}$$

---

## 3. TEMPORAL TRAJECTORY ENCODER FOR J AND Z

Dynamic signs J and Z cannot be identified through static spatial geometry. TITAN implements a lightweight **Bidirectional LSTM with Temporal Attention Pooling**:

### 3.1 Point History Representation
A rolling circular buffer maintains the 2D coordinates of the active fingertip over a 16-frame window:
$$\vec{u}(t) = \left[ \frac{x(t) - x(0)}{s_{\text{palm}}}, \, \frac{y(t) - y(0)}{s_{\text{palm}}} \right] \in \mathbb{R}^2, \quad t \in \{0, \dots, 15\}$$
- **Sign J:** Tracks pinky tip (Landmark 20).
- **Sign Z:** Tracks index tip (Landmark 8).

### 3.2 BiLSTM & Attention Pooling
The 16-step trajectory is processed through a 2-layer BiLSTM ($h_{\text{hidden}} = 32$, bidirectional output $= 64$):
$$h_t = \text{BiLSTM}(\vec{u}(t)), \quad h_t \in \mathbb{R}^{64}$$
Temporal attention weights $\alpha_t$ focus on motion inflection points (the curve in J, the vertices in Z):
$$e_t = w_{\text{attn}}^T \tanh(W_{\text{attn}} h_t + b_{\text{attn}})$$
$$\alpha_t = \frac{\exp(e_t)}{\sum_{\tau=0}^{15} \exp(e_\tau)}$$
$$z_{\text{traj}} = \sum_{t=0}^{15} \alpha_t h_t \in \mathbb{R}^{64}$$

---

## 4. CONFIDENCE-AWARE GATED ROUTING & SELECTIVE REJECTION

### 4.1 Motion-Energy Gating Arbitrator
To prevent static signs (such as `I` or `D`) from triggering dynamic `J` or `Z` when minor hand micro-tremor occurs, TITAN deploys a sigmoid velocity gate:
$$E_{\text{motion}} = \frac{1}{16} \sum_{t=1}^{15} \|\vec{u}(t) - \vec{u}(t-1)\|_2$$
$$\gamma = \sigma(10 \cdot E_{\text{motion}} - 2.0)$$
- When $E_{\text{motion}} < 0.12$ (hand held static), $\gamma \to 0$: dynamic heads are suppressed, and the prediction is strictly resolved by the spatial GNN.
- When $E_{\text{motion}} \ge 0.25$ (deliberate motion), $\gamma \to 1$: dynamic trajectory logits modulate class `J` (idx 9) and `Z` (idx 26).

### 4.2 Uncertainty Calibration
Post-hoc temperature scaling ($T^* = 1.08$) normalizes logit over-confidence:
$$P(y = c \mid x) = \frac{\exp(z_c / T^*)}{\sum_{j=0}^{26} \exp(z_j / T^*)}$$
Predictive entropy $H(y \mid x)$ and energy score $E(x)$ detect out-of-distribution hand poses:
$$H(y \mid x) = -\sum_{c=0}^{26} P_c \ln(P_c + \epsilon)$$
$$E(x) = -T^* \ln \sum_{c=0}^{26} \exp(z_c / T^*)$$

### 4.3 Selective Rejection Rule
A prediction is gated and rejected (outputting `NEUTRAL`) if:
$$\text{Reject}(x) \iff \max_c P(y=c \mid x) < 0.55 \quad \lor \quad H(y \mid x) > 1.35$$
This suppression prevents false insertions during inter-sign transitions.
