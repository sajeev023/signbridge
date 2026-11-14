"""
SIGNBRIDGE TITAN: Adaptive Graph Convolution & Occlusion Awareness Verification Suite
Verifies:
1. Adaptive Parameter Matrix B (21x21) initialization and properties
2. Effective Adjacency A_eff = A_norm + B computation
3. Backward compatibility with pre-adaptive weights
4. LayerNorm mathematical parity (PyTorch vs client-side formula)
5. Occlusion awareness and robust landmark clipping
6. End-to-end gradient propagation through learned adjacency
"""

import math
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.titan.asl_architecture import (
    ASLSpatialGNN,
    SpatialGraphEncoder,
    build_adjacency_matrix,
    compute_normalized_laplacian,
    get_default_normalized_adjacency,
    ASL_CLASS_TO_IDX,
    ASL_VOCABULARY
)
from backend.titan.spatial_encoder import (
    landmarks_to_node_features,
    NUM_HAND_LANDMARKS
)


def test_learned_adj_initialization():
    """Verifies that learned_adj is initialized to zeros with exact (21, 21) shape."""
    encoder = SpatialGraphEncoder(node_in_dim=6, pos_embed_dim=16, hidden_dim=64, out_dim=64)
    assert hasattr(encoder, 'learned_adj')
    assert isinstance(encoder.learned_adj, nn.Parameter)
    assert encoder.learned_adj.shape == (21, 21)
    # Must be initially zero for backward compatibility
    assert torch.all(encoder.learned_adj == 0.0)


def test_effective_adjacency_backward_compatibility():
    """Verifies that with B=0, output is numerically identical to standard Kipf-Welling GCN."""
    torch.manual_seed(42)
    encoder = SpatialGraphEncoder(node_in_dim=6, pos_embed_dim=16, hidden_dim=64, out_dim=64, dropout=0.0)
    encoder.eval()

    node_feats = torch.randn(2, 21, 6)
    aux_feats = torch.randn(2, 42)
    norm_adj = encoder.norm_adj

    # Forward with default B=0
    with torch.no_grad():
        out_default = encoder(node_feats, aux_features=aux_feats)

    # Forward without learned_adj (simulating manual override)
    with torch.no_grad():
        encoder.learned_adj.zero_()
        out_zero = encoder(node_feats, aux_features=aux_feats, custom_adj=norm_adj)

    diff = torch.max(torch.abs(out_default - out_zero)).item()
    assert diff < 1e-6, f"Difference with B=0 should be negligible, got {diff}"


def test_layernorm_javascript_parity():
    """Verifies that PyTorch LayerNorm matches the JavaScript client-side implementation."""
    dim = 64
    ln = nn.LayerNorm(dim, eps=1e-5)
    ln.eval()

    # Random test vector
    torch.manual_seed(123)
    x = torch.randn(1, dim)
    y_torch = ln(x).detach().numpy()[0]

    gamma = ln.weight.detach().numpy()
    beta = ln.bias.detach().numpy()
    x_np = x.numpy()[0]

    # JS formula
    mean = np.mean(x_np)
    var = np.var(x_np)
    inv_std = 1.0 / np.sqrt(var + 1e-5)
    y_js = ((x_np - mean) * inv_std) * gamma + beta

    max_diff = np.max(np.abs(y_torch - y_js))
    assert max_diff < 1e-5, f"LayerNorm divergence between PyTorch and JS formula: {max_diff}"


def test_occlusion_and_missing_landmarks_robustness():
    """Verifies that missing or extreme coordinates are sanitized without NaNs or Infs."""
    # Test with fewer than 21 landmarks
    incomplete = [{'x': 0.5, 'y': 0.5, 'z': 0.0}] * 10
    feats_inc = landmarks_to_node_features(incomplete)
    assert feats_inc.shape == (21, 6)
    assert np.all(feats_inc == 0.0)

    # Test with extreme outlier / NaN values
    corrupted = [{'x': float('nan') if i == 5 else (999.0 if i == 8 else 0.5), 'y': 0.5, 'z': 0.0} for i in range(21)]
    feats_corr = landmarks_to_node_features(corrupted)
    assert not np.isnan(feats_corr).any(), "NaNs must be sanitized"
    assert not np.isinf(feats_corr).any(), "Infs must be sanitized"
    assert np.max(feats_corr) <= 10.0, "Coordinates must be clipped"
    assert np.min(feats_corr) >= -10.0, "Coordinates must be clipped"


def test_adaptive_graph_gradient_flow():
    """Verifies that backpropagation computes gradients for learned_adj."""
    model = ASLSpatialGNN(node_in_dim=6, hidden_dim=64, aux_dim=42, num_classes=27, dropout=0.0)
    model.train()

    node_feats = torch.randn(4, 21, 6, requires_grad=True)
    aux_feats = torch.randn(4, 42, requires_grad=True)
    adj = model.spatial_encoder.norm_adj
    targets = torch.tensor([0, 5, 14, 26])

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()

    logits = model(node_feats, adj, aux_feats)
    loss = F.cross_entropy(logits, targets)
    loss.backward()

    # Verify gradients exist and are non-zero on learned_adj
    assert model.spatial_encoder.learned_adj.grad is not None
    grad_norm = torch.norm(model.spatial_encoder.learned_adj.grad).item()
    assert grad_norm > 0.0, "learned_adj must receive non-zero gradient update"
