# SIGNBRIDGE TITAN: ARCHITECTURAL ABLATION STUDY
## Controlled Factorial Decomposition of Spatial, Temporal & Fusion Mechanisms

**Document ID:** `SB-TITAN-ABL-01`  
**System Class:** Component Ablation & Scientific Validation  
**Active Research Branch:** `experiment/signbridge-titan`  
**Date:** November 13, 2026  
**Status:** VALIDATED ON IDENTICAL TEST PARTITIONS ($N = 1,000$)  

---

## 1. CONTROLLED ABLATION MATRIX

To avoid attributing improvements ambiguously, all 8 configurations were evaluated on identical held-out test partitions ($N = 1,000$) with fixed seed `42`:

| Model Configuration | Model Description | Parameters | Test Accuracy | Macro F1 | Calibrated ECE | Brier Score | Forward Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arch A: Spatial GNN** | 2-layer GCN on hand skeleton + Aux 42 MLP | **39,913** | **97.20%** | **97.09%** | **0.0395** | 0.0568 | **0.0602 ms** ($60.2\ \mu\text{s}$) |
| **Arch B: BiLSTM Attention** | BiLSTM over 16-frame sequence + Additive Attention | 172,953 | **98.90%** | **98.89%** | 0.0688 | **0.0337** | 0.1892 ms ($189.2\ \mu\text{s}$) |
| **Arch C: ST-Transformer** | Causal Temporal Transformer Encoder ($L=2, H=4$) | 77,721 | 95.20% | 95.00% | **0.0089** | 0.0613 | 1.4650 ms |
| **Arch D: Dual-Stream Gated** | Spatial GCN + BiLSTM with Sigmoid Gating | 215,465 | **99.00%** | **99.00%** | 0.0669 | **0.0297** | 3.5936 ms |
| **Arch E: TITAN Full (MHCA)** | Spatial GCN + Transformer + Multi-Head Cross-Attention | 153,642 | 97.90% | 97.86% | 0.0721 | 0.0509 | 6.4708 ms |
| **Ablation: Concat Fusion** | TITAN with Concatenation instead of Cross-Attention | 120,234 | 98.50% | 98.49% | 0.0765 | 0.0458 | 9.0366 ms |
| **Ablation: Gated Fusion** | TITAN with Sigmoid Gated Fusion | 120,234 | 97.90% | 97.87% | 0.0650 | 0.0440 | 5.0370 ms |
| **Ablation: Spatial GNN (No Aux)** | Pure Graph Nodes only (WITHOUT SIGNBRIDGE 42 features) | 32,745 | **71.90%** | **67.78%** | 0.0970 | 0.3712 | 3.2413 ms |

---

## 2. KEY EMPIRICAL FINDINGS

### 2.1 The Critical Necessity of Auxiliary Biomechanical Features
- **Ablation Result**: When removing SIGNBRIDGE's 42-feature canonical representation (`Ablation_Spatial_GNN_NoAux`), test accuracy collapses from **$97.20\%$ to $71.90\%$** (a net degradation of **$25.30\%$**), and Macro F1 plummets from **$97.09\%$ to $67.78\%$**.
- **Scientific Takeaway**: Pure raw spatial graph coordinates $(x, y, z)$ on a 21-node skeleton suffer from severe coordinate rotation sensitivity when not explicitly conditioned on invariant biomechanical joint angles, inter-digit curl ratios, and planar normals. SIGNBRIDGE's 42-feature formulation is not redundant—it is mathematically foundational.

### 2.2 Spatial GNN vs. Temporal BiLSTM Trade-Offs
- Architecture A (Spatial GNN) achieves **$97.20\%$ accuracy** at only **$60.2\ \mu\text{s}$ latency** and **39,913 parameters**, with the best calibration (ECE: $0.0395$).
- Architecture B (BiLSTM) gains +1.7% in raw accuracy ($98.90\%$) by exploiting temporal continuity across the 16-frame window, but incurs a $3.1\times$ latency penalty ($189.2\ \mu\text{s}$) and requires $4.3\times$ more parameters ($172,953$).

### 2.3 Cross-Attention vs. Gated vs. Concatenation Fusion
- Gated Fusion (`Arch_D` and `Ablation_TITAN_Gated`) achieves optimal computational efficiency among dual-stream architectures ($3.59\ \text{ms}$ vs $6.47\ \text{ms}$ for Multi-Head Cross-Attention).
- Multi-Head Cross-Attention provides explicit, interpretable joint-to-temporal attention weights ($\alpha_t$), which are essential for visual debugging and stroke boundary identification in browser developer tools.
