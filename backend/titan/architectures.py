"""
SIGNBRIDGE TITAN: Unified Model Suite for Architecture Comparison
Implements candidate architectures A through E:
- Architecture A: Spatial GNN + MLP (Static Spatial Baseline)
- Architecture B: BiLSTM + Attention (Temporal Baseline)
- Architecture C: Spatial-Temporal Transformer (ST-Transformer)
- Architecture D: Dual-Stream Spatial GCN + BiLSTM (Gated Fusion)
- Architecture E: Hybrid Spatial GCN + Temporal Cross-Attention (TITAN Full)
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .spatial_encoder import SpatialGraphEncoder
from .temporal_encoder import BiLSTMTemporalEncoder, TemporalTransformerEncoder
from .fusion import ConcatenationFusion, GatedFusion, CrossAttentionFusion


class ArchitectureA_SpatialGNN(nn.Module):
    """
    Candidate Architecture A:
    Spatial Graph Neural Network on hand skeleton + Auxiliary 42-feature MLP.
    Focuses entirely on learned topological joint invariants for static alphabet signs.
    """
    def __init__(
        self,
        num_classes: int = 25,
        hidden_dim: int = 64,
        out_dim: int = 64,
        use_aux_features: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.spatial_encoder = SpatialGraphEncoder(
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=2,
            use_aux_features=use_aux_features,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        aux_features: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        z_spatial = self.spatial_encoder(node_features, aux_features=aux_features)
        logits = self.classifier(z_spatial)
        return logits


class ArchitectureB_BiLSTMAttention(nn.Module):
    """
    Candidate Architecture B:
    BiLSTM over temporal sequence + Learned Temporal Attention Pooling + Classifier.
    """
    def __init__(
        self,
        num_classes: int = 25,
        in_dim: int = 42,
        hidden_dim: int = 64,
        out_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.temporal_encoder = BiLSTMTemporalEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=2,
            bidirectional=True,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        temporal_sequence: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z_temporal, _, attn_weights = self.temporal_encoder(temporal_sequence, mask=mask)
        logits = self.classifier(z_temporal)
        return logits, attn_weights


class ArchitectureC_STTransformer(nn.Module):
    """
    Candidate Architecture C:
    Lightweight Spatial-Temporal Transformer with temporal self-attention & causal masking.
    """
    def __init__(
        self,
        num_classes: int = 25,
        in_dim: int = 42,
        d_model: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        causal: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.transformer_encoder = TemporalTransformerEncoder(
            in_dim=in_dim,
            d_model=d_model,
            nhead=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            causal=causal,
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(
        self,
        temporal_sequence: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z_temporal, _, attn_weights = self.transformer_encoder(temporal_sequence, mask=mask)
        logits = self.classifier(z_temporal)
        return logits, attn_weights


class ArchitectureD_DualStreamGated(nn.Module):
    """
    Candidate Architecture D:
    Dual-Stream Spatial GCN + BiLSTM with learned Sigmoid Gated Fusion.
    """
    def __init__(
        self,
        num_classes: int = 25,
        d_model: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.spatial_encoder = SpatialGraphEncoder(hidden_dim=d_model, out_dim=d_model, dropout=dropout)
        self.temporal_encoder = BiLSTMTemporalEncoder(in_dim=42, hidden_dim=d_model, out_dim=d_model, dropout=dropout)
        self.gated_fusion = GatedFusion(spatial_dim=d_model, temporal_dim=d_model, out_dim=d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        temporal_sequence: torch.Tensor,
        aux_features: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z_spatial = self.spatial_encoder(node_features, aux_features=aux_features)
        z_temporal, _, _ = self.temporal_encoder(temporal_sequence, mask=mask)
        fused, gate_values = self.gated_fusion(z_spatial, z_temporal)
        logits = self.classifier(fused)
        return logits, gate_values


class TitanHybridModel(nn.Module):
    """
    Candidate Architecture E (TITAN Full):
    Spatial Graph Encoder + Temporal Transformer Encoder + Multi-Head Cross-Attention Fusion
    + Uncertainty Estimation & Rejection Head.
    """
    def __init__(
        self,
        num_classes: int = 25,
        d_model: int = 64,
        num_heads: int = 4,
        fusion_type: str = "cross_attention",  # "concat", "gated", "cross_attention"
        causal: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model
        self.fusion_type = fusion_type

        # 1. Spatial Graph Encoder
        self.spatial_encoder = SpatialGraphEncoder(
            hidden_dim=d_model,
            out_dim=d_model,
            num_layers=2,
            use_aux_features=True,
            dropout=dropout,
        )

        # 2. Temporal Sequence Encoder
        self.temporal_encoder = TemporalTransformerEncoder(
            in_dim=42,
            d_model=d_model,
            nhead=num_heads,
            num_layers=2,
            dropout=dropout,
            causal=causal,
        )

        # 3. Fusion Module
        if fusion_type == "concat":
            self.fusion = ConcatenationFusion(spatial_dim=d_model, temporal_dim=d_model, out_dim=d_model)
        elif fusion_type == "gated":
            self.fusion = GatedFusion(spatial_dim=d_model, temporal_dim=d_model, out_dim=d_model)
        else:
            self.fusion = CrossAttentionFusion(
                spatial_dim=d_model,
                temporal_dim=d_model,
                out_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
            )

        # 4. Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        # 5. Temperature parameter for uncertainty calibration (learned/fitted post-training)
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)

    def set_temperature(self, temp: float):
        self.temperature.data.fill_(temp)

    def forward(
        self,
        node_features: torch.Tensor,
        temporal_sequence: torch.Tensor,
        aux_features: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            node_features: (B, 21, 6) current frame normalized landmarks
            temporal_sequence: (B, T, 42) historical sequence of frame features
            aux_features: (B, 42) current frame canonical features
            mask: (B, T) sequence validity mask
        Returns:
            logits: (B, num_classes)
            attention_weights: (B, 1, T) or gating weights
        """
        # Spatial graph representation
        z_spatial = self.spatial_encoder(node_features, aux_features=aux_features)

        # Temporal sequence representation
        z_temporal, h_seq, _ = self.temporal_encoder(temporal_sequence, mask=mask)

        # Cross-attention / Fusion
        if self.fusion_type == "cross_attention":
            fused, attn_weights = self.fusion(z_spatial, z_temporal, h_seq=h_seq, mask=mask)
        else:
            fused, attn_weights = self.fusion(z_spatial, z_temporal)

        logits = self.classifier(fused)
        return logits, attn_weights
