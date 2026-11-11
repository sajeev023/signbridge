import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any

class TemporalAttention(nn.Module):
    """
    Temporal Self-Attention mechanism over LSTM hidden states.
    Allows the model to dynamically weight crucial frames (e.g., peak of gesture)
    over preparatory or recovery phases.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False)
        )

    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # lstm_output: (batch_size, seq_len, hidden_dim)
        energy = self.projection(lstm_output)  # (batch_size, seq_len, 1)
        weights = F.softmax(energy, dim=1)      # (batch_size, seq_len, 1)
        context = torch.sum(weights * lstm_output, dim=1)  # (batch_size, hidden_dim)
        return context, weights.squeeze(-1)


class SignBridgeBiLSTM(nn.Module):
    """
    Spatio-Temporal Deep Learning Architecture for Indian Sign Language Recognition.
    
    Combines:
    1. Linear feature projection with LayerNorm for coordinate stabilization
    2. 1D Temporal Convolution to capture velocity and local temporal gradients
    3. 2-layer Bidirectional LSTM to capture forward and backward temporal dependencies
    4. Temporal Attention Pooling to focus on key gesture climax frames
    5. Calibrated Multilayer Perceptron classification head
    """
    def __init__(
        self,
        input_dim: int = 153,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 10,
        dropout: float = 0.3
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Spatial Feature Projection & Normalization
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )

        # 1D Temporal Convolution for local gesture velocity features
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # Bidirectional Recurrent Sequence Modeling
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Attention layer over 2 * hidden_dim (due to bidirectional)
        self.attention = TemporalAttention(hidden_dim * 2)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            logits: (batch_size, num_classes)
            attention_weights: (batch_size, seq_len)
        """
        # Feature projection
        proj = self.input_proj(x)  # (batch, seq_len, hidden_dim)

        # 1D Conv over temporal dimension: reshape to (batch, channels, seq_len)
        conv_in = proj.transpose(1, 2)
        conv_out = self.conv1d(conv_in)
        lstm_in = conv_out.transpose(1, 2)  # (batch, seq_len, hidden_dim)

        # BiLSTM processing
        lstm_out, _ = self.bilstm(lstm_in)  # (batch, seq_len, hidden_dim * 2)

        # Attention pooling
        context, attn_weights = self.attention(lstm_out)  # (batch, hidden_dim * 2)

        # Final classification
        logits = self.classifier(context)  # (batch, num_classes)
        return logits, attn_weights

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Helper to get calibrated softmax probabilities."""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
            probs = F.softmax(logits, dim=-1)
        return probs
