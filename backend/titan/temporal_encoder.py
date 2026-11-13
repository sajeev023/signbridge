"""
SIGNBRIDGE TITAN: Temporal Sequence Encoder (TSE)
Implements BiLSTM with temporal attention, causal GRU for streaming inference,
and lightweight Temporal Transformer Encoders with sequence masking.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal temporal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:, :T, :]


class TemporalAttentionPooling(nn.Module):
    """Learned additive attention pooling over temporal steps."""
    def __init__(self, in_features: int, hidden_dim: int = 32):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, D)
            mask: (B, T) boolean mask, True where valid, False where padded
        Returns:
            pooled: (B, D)
            weights: (B, T)
        """
        scores = self.linear2(torch.tanh(self.linear1(x))).squeeze(-1)  # (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        weights = F.softmax(scores, dim=-1)  # (B, T)
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (B, D)
        return pooled, weights


class BiLSTMTemporalEncoder(nn.Module):
    """
    Bidirectional LSTM with learned temporal attention pooling.
    Evaluated in offline/validation mode; causal version uses unidirectional LSTM.
    """
    def __init__(
        self,
        in_dim: int = 42,
        hidden_dim: int = 64,
        out_dim: int = 64,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.attn_pool = TemporalAttentionPooling(lstm_out_dim)
        self.proj = nn.Linear(lstm_out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, in_dim)
            mask: (B, T) optional boolean mask
        Returns:
            z_temporal: (B, out_dim) pooled summary representation
            seq_out: (B, T, out_dim) projected temporal sequence
            attn_weights: (B, T) temporal attention scores
        """
        lstm_out, _ = self.lstm(x)  # (B, T, lstm_out_dim)
        pooled, attn_weights = self.attn_pool(lstm_out, mask=mask)
        z_temporal = self.norm(F.gelu(self.proj(pooled)))
        seq_out = self.norm(F.gelu(self.proj(lstm_out)))
        return z_temporal, seq_out, attn_weights


class TemporalTransformerEncoder(nn.Module):
    """
    Lightweight Spatial-Temporal Transformer Encoder with multi-head self-attention.
    Supports causal masking for live real-time streaming and bidirectional mode for offline analysis.
    """
    def __init__(
        self,
        in_dim: int = 42,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.causal = causal
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn_pool = TemporalAttentionPooling(d_model)
        self.norm = nn.LayerNorm(d_model)

    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Lower-triangular boolean mask preventing future token lookahead."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, in_dim)
            mask: (B, T) True where valid, False where padded
        Returns:
            z_temporal: (B, d_model) summary representation
            seq_out: (B, T, d_model) full sequence features
            attn_weights: (B, T) pooling weights
        """
        B, T, _ = x.shape
        h = self.input_proj(x)
        h = self.pos_encoder(h)

        # Causal mask for live streaming
        causal_mask = self._generate_causal_mask(T, x.device) if self.causal else None

        # Key padding mask: PyTorch convention True = ignore
        src_key_padding_mask = (~mask) if mask is not None else None

        h_out = self.transformer(
            h,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask,
        )  # (B, T, d_model)

        pooled, attn_weights = self.attn_pool(h_out, mask=mask)
        z_temporal = self.norm(pooled)
        return z_temporal, h_out, attn_weights
