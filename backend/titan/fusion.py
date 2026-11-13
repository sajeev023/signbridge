"""
SIGNBRIDGE TITAN: Cross-Attention Spatial-Temporal Fusion
Implements Concatenation, Gated Fusion, Cross-Attention, and Multi-Head Cross-Attention (MHCA).
Enables controlled ablation experiments comparing fusion mechanisms.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatenationFusion(nn.Module):
    """Simple linear projection of concatenated spatial and temporal vectors."""
    def __init__(self, spatial_dim: int = 64, temporal_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(spatial_dim + temporal_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(
        self,
        z_spatial: torch.Tensor,
        z_temporal: torch.Tensor,
        h_seq: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # z_spatial: (B, D_s), z_temporal: (B, D_t)
        fused = self.proj(torch.cat([z_spatial, z_temporal], dim=-1))
        return fused, None


class GatedFusion(nn.Module):
    """
    Learned sigmoid gating mechanism:
    g = sigmoid(W_g * [z_spatial; z_temporal] + b_g)
    fused = g * z_spatial + (1 - g) * z_temporal
    """
    def __init__(self, spatial_dim: int = 64, temporal_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.proj_s = nn.Linear(spatial_dim, out_dim) if spatial_dim != out_dim else nn.Identity()
        self.proj_t = nn.Linear(temporal_dim, out_dim) if temporal_dim != out_dim else nn.Identity()
        self.gate_linear = nn.Linear(out_dim * 2, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        z_spatial: torch.Tensor,
        z_temporal: torch.Tensor,
        h_seq: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.proj_s(z_spatial)
        t = self.proj_t(z_temporal)
        g = torch.sigmoid(self.gate_linear(torch.cat([s, t], dim=-1)))
        fused = self.norm(g * s + (1.0 - g) * t)
        return fused, g


class CrossAttentionFusion(nn.Module):
    """
    Multi-Head Cross-Attention Fusion:
    Spatial latent serves as Query (what handshape is being checked?),
    while temporal sequence serves as Key and Value (when and where did motion occur?).
    """
    def __init__(
        self,
        spatial_dim: int = 64,
        temporal_dim: int = 64,
        out_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads

        self.proj_spatial = nn.Linear(spatial_dim, out_dim)
        self.proj_temporal = nn.Linear(temporal_dim, out_dim)

        self.mha = nn.MultiheadAttention(
            embed_dim=out_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(out_dim)
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
        )
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(
        self,
        z_spatial: torch.Tensor,
        z_temporal: torch.Tensor,
        h_seq: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_spatial: (B, D_spatial)
            z_temporal: (B, D_temporal)
            h_seq: (B, T, D_temporal)
            mask: (B, T) optional valid sequence mask
        Returns:
            fused: (B, out_dim)
            attn_weights: (B, 1, T) cross-attention weights over time
        """
        q = self.proj_spatial(z_spatial).unsqueeze(1)  # (B, 1, out_dim)
        kv = self.proj_temporal(h_seq)                  # (B, T, out_dim)

        key_padding_mask = (~mask) if mask is not None else None

        attn_out, attn_weights = self.mha(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=key_padding_mask,
        )  # attn_out: (B, 1, out_dim), attn_weights: (B, 1, T)

        h = self.norm1(q.squeeze(1) + attn_out.squeeze(1))
        fused = self.norm2(h + self.ffn(h))
        return fused, attn_weights
