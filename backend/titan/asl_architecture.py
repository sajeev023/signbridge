"""
SIGNBRIDGE TITAN: ASL Manual Alphabet Spatial-Temporal Architecture
Reference: American Sign Language Alphabet Chart (Gérard Aflague Collection)

Implements:
1. Reference-Aligned Spatial Graph Neural Network (21 MediaPipe nodes, Kipf-Welling normalized Laplacian, 31 structural edges)
2. Dual Readout Graph Pooling (Mean + Max) fused with Auxiliary 42-Feature Biomechanical Pathway
3. Temporal Trajectory Encoder (BiLSTM / Causal GRU / Temporal Attention) for dynamic classes J and Z
4. Confidence-Aware Gated Fusion & Selective Rejection Engine
5. Calibration via Temperature Scaling, Predictive Entropy, and Energy-Based OOD Scoring
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.titan.graph_topology import (
    NUM_HAND_LANDMARKS,
    ALL_EDGES,
    build_adjacency_matrix,
    compute_normalized_laplacian,
    get_default_normalized_adjacency
)
from backend.titan.spatial_encoder import SpatialGraphEncoder, landmarks_to_node_features
from backend.titan.temporal_encoder import BiLSTMTemporalEncoder

ASL_VOCABULARY = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'NEUTRAL', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]
ASL_CLASS_TO_IDX = {c: i for i, c in enumerate(ASL_VOCABULARY)}
ASL_STATIC_CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'
]
ASL_DYNAMIC_CLASSES = ['J', 'Z']


class ASLReferenceFeatureExtractor:
    """
    Extracts specialized biomechanical features aligned with the Gérard Aflague reference:
    - Thumb lateral opposition (A vs S vs E)
    - Finger overlapping / tucking counts (M vs N vs T)
    - Inter-digit abduction / contact (U vs V vs W)
    - Aperture curvature (C vs O vs D vs F)
    - Inversion pitch (K vs P, G vs Q)
    """

    @staticmethod
    def compute_asl_alignment_features(landmarks: np.ndarray) -> np.ndarray:
        """
        landmarks: (21, 3) array normalized relative to wrist.
        Returns 12-dim vector of ASL reference alignment metrics.
        """
        # Distances
        wrist = landmarks[0]
        thumb_tip = landmarks[4]
        index_mcp = landmarks[5]
        index_pip = landmarks[6]
        index_tip = landmarks[8]
        middle_mcp = landmarks[9]
        middle_pip = landmarks[10]
        middle_tip = landmarks[12]
        ring_mcp = landmarks[13]
        ring_pip = landmarks[14]
        ring_tip = landmarks[16]
        pinky_mcp = landmarks[17]
        pinky_tip = landmarks[20]

        palm_scale = np.linalg.norm(index_mcp - pinky_mcp) + 1e-6

        # 1. Thumb lateral position relative to index MCP
        thumb_lateral = (thumb_tip[0] - index_mcp[0]) / palm_scale

        # 2. Thumb vertical height relative to index MCP (high for A, mid for S, low for E)
        thumb_height = (wrist[1] - thumb_tip[1]) / palm_scale

        # 3. Thumb crossing distance over middle/ring MCPs (for S, M, N, T)
        thumb_to_mid_mcp = np.linalg.norm(thumb_tip - middle_mcp) / palm_scale
        thumb_to_ring_mcp = np.linalg.norm(thumb_tip - ring_mcp) / palm_scale

        # 4. Index-Middle inter-tip distance (U=small, V=large)
        u_v_distance = np.linalg.norm(index_tip - middle_tip) / palm_scale

        # 5. Middle-Ring inter-tip distance (W=large, V=small)
        mid_ring_distance = np.linalg.norm(middle_tip - ring_tip) / palm_scale

        # 6. Aperture: Thumb-tip to Index-tip distance (C=moderate, O=zero, D=touching middle)
        aperture_index = np.linalg.norm(thumb_tip - index_tip) / palm_scale
        aperture_middle = np.linalg.norm(thumb_tip - middle_tip) / palm_scale

        # 7. Middle crossing over index (R sign: middle tip x relative to index tip)
        r_cross = (middle_tip[0] - index_tip[0]) / palm_scale

        # 8. Index hook curl (X sign: distance from index tip to index MCP vs straight length)
        x_hook_ratio = np.linalg.norm(index_tip - index_mcp) / (np.linalg.norm(index_pip - index_mcp) + np.linalg.norm(index_tip - index_pip) + 1e-6)

        # 9. Pitch angle of index vector (P and Q point downward, K and G point up/horizontal)
        index_dir_y = (index_tip[1] - index_mcp[1]) / palm_scale

        # 10. Pinky extension (I, J, Y)
        pinky_ext = np.linalg.norm(pinky_tip - wrist) / palm_scale

        # 11. Thumb-Pinky shaka span (Y sign)
        shaka_span = np.linalg.norm(thumb_tip - pinky_tip) / palm_scale

        # 12. Index extension (D, G, L, Z)
        index_ext = np.linalg.norm(index_tip - wrist) / palm_scale

        return np.array([
            thumb_lateral, thumb_height, thumb_to_mid_mcp, thumb_to_ring_mcp,
            u_v_distance, mid_ring_distance, aperture_index, aperture_middle,
            r_cross, x_hook_ratio, index_dir_y, pinky_ext, shaka_span, index_ext
        ], dtype=np.float32)


class ASLSpatialGNN(nn.Module):
    """
    Architecture 6: Reference-Aligned Spatial Graph Neural Network for ASL.
    Combines:
    - 2-Layer Residual GCN with LayerNorm and GELU
    - Dual Readout (Mean + Max Pooling = 128-dim)
    - Auxiliary Biomechanical Feature Pathway (42 features + 14 ASL-alignment features = 56-dim)
    - Linear projection to 27 ASL manual classes
    """

    def __init__(
        self,
        node_in_dim: int = 6,
        hidden_dim: int = 64,
        aux_dim: int = 42,
        num_classes: int = 27,
        dropout: float = 0.15
    ):
        super().__init__()
        self.num_classes = num_classes
        self.spatial_encoder = SpatialGraphEncoder(
            node_in_dim=node_in_dim,
            pos_embed_dim=16,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_layers=2,
            use_aux_features=True,
            aux_dim=aux_dim,
            dropout=dropout
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(
        self,
        node_features: torch.Tensor,
        adj: torch.Tensor,
        aux_features: torch.Tensor
    ) -> torch.Tensor:
        """
        node_features: (B, 21, 6)
        adj: (21, 21) or (B, 21, 21)
        aux_features: (B, aux_dim)
        Returns: logits (B, num_classes)
        """
        z_spatial = self.spatial_encoder(node_features, aux_features=aux_features, custom_adj=adj)
        logits = self.classifier(z_spatial)
        return logits


class ASLTrajectoryEncoder(nn.Module):
    """
    Temporal Trajectory Encoder for dynamic signs J and Z.
    Processes point-history sequence of active fingertip coordinates.
    Buffer length: 16 frames -> 32-dim vector (dx, dy at each step).
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.fc = nn.Linear(hidden_dim * 2, 64)

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        trajectory: (B, T, 2) where T=16 fingertip displacements
        Returns: (B, 64) trajectory embedding
        """
        lstm_out, _ = self.bilstm(trajectory)  # (B, T, 64)
        attn_scores = self.attention(lstm_out)  # (B, T, 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        pooled = torch.sum(lstm_out * attn_weights, dim=1)  # (B, 64)
        emb = F.gelu(self.fc(pooled))
        return emb


class ASLTitanHybridModel(nn.Module):
    """
    Architecture 8: Full Confidence-Aware Spatial-Temporal SIGNBRIDGE TITAN for ASL A–Z.
    Unifies:
    1. Spatial GNN for static handshape feature extraction (all 27 classes)
    2. Temporal Trajectory Encoder for motion dynamics (J, Z, NEUTRAL)
    3. Motion-gated cross-attention arbitrator
    4. Calibrated confidence and predictive uncertainty estimation
    """

    def __init__(
        self,
        node_in_dim: int = 6,
        spatial_hidden_dim: int = 64,
        aux_dim: int = 42,
        traj_input_dim: int = 2,
        num_classes: int = 27,
        dropout: float = 0.15
    ):
        super().__init__()
        self.num_classes = num_classes
        self.spatial_gnn = ASLSpatialGNN(
            node_in_dim=node_in_dim,
            hidden_dim=spatial_hidden_dim,
            aux_dim=aux_dim,
            num_classes=num_classes,
            dropout=dropout
        )
        self.trajectory_encoder = ASLTrajectoryEncoder(
            input_dim=traj_input_dim,
            hidden_dim=32,
            num_layers=2,
            dropout=dropout
        )

        # Cross-attention fusion: Spatial queries attend to temporal trajectory
        self.traj_proj = nn.Linear(64, 128)
        self.spatial_proj = nn.Linear(num_classes, 128)

        # Dynamic head for J, Z, NEUTRAL
        self.dynamic_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 3)  # [NEUTRAL, J, Z]
        )

        # Dynamic gating classifier
        self.gating_weight = nn.Parameter(torch.tensor([0.5]))

    def forward(
        self,
        node_features: torch.Tensor,
        adj: torch.Tensor,
        aux_features: torch.Tensor,
        trajectory: Optional[torch.Tensor] = None,
        motion_energy: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with dynamic routing:
        If trajectory is provided and motion_energy > threshold, fuses spatial logits with dynamic predictions.
        Otherwise, executes fast spatial GNN path.
        """
        spatial_logits = self.spatial_gnn(node_features, adj, aux_features)  # (B, 27)

        if trajectory is None:
            return spatial_logits

        # Compute trajectory embedding
        traj_emb = self.trajectory_encoder(trajectory)  # (B, 64)
        dyn_logits = self.dynamic_head(traj_emb)        # (B, 3): [NEUTRAL, J, Z]

        # Route dynamic logits to indices:
        # J_idx = 9, Z_idx = 26, NEUTRAL_idx = 14
        j_idx = ASL_CLASS_TO_IDX['J']
        z_idx = ASL_CLASS_TO_IDX['Z']

        combined_logits = spatial_logits.clone()
        if motion_energy is not None:
            # Gating based on motion energy (sigmoid gate)
            gate = torch.sigmoid(motion_energy * 10.0 - 2.0)  # active when energy > 0.2
            combined_logits[:, j_idx] = (1 - gate) * combined_logits[:, j_idx] + gate * dyn_logits[:, 1]
            combined_logits[:, z_idx] = (1 - gate) * combined_logits[:, z_idx] + gate * dyn_logits[:, 2]
        else:
            combined_logits[:, j_idx] = dyn_logits[:, 1]
            combined_logits[:, z_idx] = dyn_logits[:, 2]

        return combined_logits


class ASLUncertaintyCalibrator:
    """
    Temperature scaling, predictive entropy, and selective rejection gate for ASL classifier.
    """

    def __init__(self, temperature: float = 1.05, entropy_threshold: float = 1.35, min_confidence: float = 0.55):
        self.temperature = temperature
        self.entropy_threshold = entropy_threshold
        self.min_confidence = min_confidence

    def fit_temperature(self, logits: np.ndarray, labels: np.ndarray, lr: float = 0.01, max_iter: int = 100):
        """Fits optimal post-hoc temperature scaling parameter minimizing NLL."""
        logits_t = torch.tensor(logits, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.long)
        temp_param = nn.Parameter(torch.ones(1) * 1.0)
        optimizer = torch.optim.LBFGS([temp_param], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = logits_t / temp_param
            loss = F.cross_entropy(scaled_logits, labels_t)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        self.temperature = float(temp_param.item())
        return self.temperature

    def evaluate_uncertainty(self, logits: np.ndarray) -> Dict[str, Union[float, bool, np.ndarray]]:
        """
        Computes calibrated probability, predictive entropy, energy score, and rejection status.
        """
        scaled_logits = logits / self.temperature
        exp_l = np.exp(scaled_logits - np.max(scaled_logits))
        probs = exp_l / np.sum(exp_l)

        pred_idx = int(np.argmax(probs))
        calibrated_conf = float(probs[pred_idx])

        # Predictive entropy H(y|x)
        eps = 1e-12
        entropy = float(-np.sum(probs * np.log(probs + eps)))

        # Energy score E(x) = -T * logsumexp(z/T)
        energy = float(-self.temperature * np.log(np.sum(np.exp(scaled_logits - np.max(scaled_logits))) + eps) - np.max(scaled_logits))

        # Rejection rule
        is_rejected = False
        rejection_reason = None
        if calibrated_conf < self.min_confidence:
            is_rejected = True
            rejection_reason = f'Low Calibrated Confidence ({calibrated_conf:.3f} < {self.min_confidence:.3f})'
        elif entropy > self.entropy_threshold:
            is_rejected = True
            rejection_reason = f'High Predictive Entropy ({entropy:.3f} > {self.entropy_threshold:.3f})'

        return {
            'predicted_class_idx': pred_idx,
            'predicted_class': ASL_VOCABULARY[pred_idx],
            'calibrated_confidence': calibrated_conf,
            'predictive_entropy': entropy,
            'energy_score': energy,
            'is_rejected': is_rejected,
            'rejection_reason': rejection_reason,
            'probabilities': probs
        }
