"""
SIGNBRIDGE TITAN: Two-Hand Dual-Stream Architecture
Processes independent Left and Right hand spatial-temporal streams with cross-hand attention,
relative distance vectors, and single-hand degradation handling.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .spatial_encoder import SpatialGraphEncoder
from .temporal_encoder import TemporalTransformerEncoder


class InterHandKinematics(nn.Module):
    """Computes relative geometric invariants between left and right hands."""
    def __init__(self, out_dim: int = 32):
        super().__init__()
        # 3 (rel wrist pos) + 1 (rel wrist dist) + 1 (scale ratio) + 2 (presence flags) = 7 dims
        self.proj = nn.Sequential(
            nn.Linear(7, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(
        self,
        left_wrist: torch.Tensor,
        right_wrist: torch.Tensor,
        left_scale: torch.Tensor,
        right_scale: torch.Tensor,
        left_present: torch.Tensor,
        right_present: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            left_wrist: (B, 3)
            right_wrist: (B, 3)
            left_scale: (B, 1)
            right_scale: (B, 1)
            left_present: (B, 1)
            right_present: (B, 1)
        Returns:
            rel_feats: (B, out_dim)
        """
        rel_pos = (left_wrist - right_wrist) * (left_present * right_present)
        rel_dist = torch.norm(rel_pos, dim=-1, keepdim=True)
        scale_ratio = torch.clamp((left_scale + 1e-4) / (right_scale + 1e-4), 0.1, 10.0)

        raw = torch.cat([rel_pos, rel_dist, scale_ratio, left_present, right_present], dim=-1)
        return self.proj(raw)


class DualStreamCrossHandModel(nn.Module):
    """
    Two-Hand Spatial-Temporal Research Architecture:
    Dual encoders for Left and Right hands with bidirectional cross-hand attention.
    Gracefully degrades to single-hand when one hand is absent.
    """
    def __init__(
        self,
        num_classes: int = 10,
        d_model: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes

        # Independent spatial encoders for Left and Right
        self.left_spatial = SpatialGraphEncoder(hidden_dim=d_model, out_dim=d_model, use_aux_features=False)
        self.right_spatial = SpatialGraphEncoder(hidden_dim=d_model, out_dim=d_model, use_aux_features=False)

        # Independent temporal encoders
        self.left_temporal = TemporalTransformerEncoder(in_dim=d_model, d_model=d_model, nhead=num_heads, num_layers=1)
        self.right_temporal = TemporalTransformerEncoder(in_dim=d_model, d_model=d_model, nhead=num_heads, num_layers=1)

        # Inter-hand geometric invariants
        self.inter_hand = InterHandKinematics(out_dim=d_model // 2)

        # Cross-Hand Attention: Left queries Right, Right queries Left
        self.left_attends_right = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.right_attends_left = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)

        self.norm_l = nn.LayerNorm(d_model)
        self.norm_r = nn.LayerNorm(d_model)

        # Classification head
        combined_dim = d_model * 2 + (d_model // 2)
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(
        self,
        left_nodes: torch.Tensor,
        right_nodes: torch.Tensor,
        left_present: torch.Tensor,
        right_present: torch.Tensor,
        left_wrist: torch.Tensor,
        right_wrist: torch.Tensor,
        left_scale: torch.Tensor,
        right_scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            left_nodes: (B, 21, 6)
            right_nodes: (B, 21, 6)
            left_present: (B, 1) float 1.0 if present, 0.0 if absent
            right_present: (B, 1)
            left_wrist, right_wrist: (B, 3)
            left_scale, right_scale: (B, 1)
        Returns:
            logits: (B, num_classes)
        """
        # Encode static handshapes
        z_l = self.left_spatial(left_nodes) * left_present
        z_r = self.right_spatial(right_nodes) * right_present

        # Cross-hand attention (B, 1, d_model)
        ql = z_l.unsqueeze(1)
        qr = z_r.unsqueeze(1)

        attn_l, _ = self.left_attends_right(ql, qr, qr)
        attn_r, _ = self.right_attends_left(qr, ql, ql)

        h_l = self.norm_l(z_l + attn_l.squeeze(1) * right_present)
        h_r = self.norm_r(z_r + attn_r.squeeze(1) * left_present)

        # Inter-hand geometric features
        h_inter = self.inter_hand(left_wrist, right_wrist, left_scale, right_scale, left_present, right_present)

        # Combined representation
        combined = torch.cat([h_l, h_r, h_inter], dim=-1)
        logits = self.classifier(combined)
        return logits
