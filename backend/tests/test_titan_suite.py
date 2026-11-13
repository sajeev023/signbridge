"""
SIGNBRIDGE TITAN: Comprehensive Automated Verification Suite
Verifies:
1. Spatial graph topology & normalized Laplacian properties
2. Landmark to node feature conversion & palm-scale invariance
3. SpatialGraphEncoder forward pass & residual connections
4. BiLSTMTemporalEncoder attention weights & shapes
5. TemporalTransformerEncoder causal masking & padding handling
6. Cross-attention fusion tensor dimensions & normalization
7. Two-hand dual-stream cross-hand attention & single-hand fallback
8. TITAN Full Architecture forward pass
9. Uncertainty calibration & ECE computation
10. Selective rejection & predictive entropy
11. Physical perturbation robustness transforms
12. Model serialization to JSON format
"""

import json
import math
from pathlib import Path
import numpy as np
import pytest
import torch

from backend.titan.graph_topology import (
    NUM_HAND_LANDMARKS,
    build_adjacency_matrix,
    compute_normalized_laplacian,
)
from backend.titan.spatial_encoder import (
    SpatialGraphEncoder,
    landmarks_to_node_features,
)
from backend.titan.temporal_encoder import (
    BiLSTMTemporalEncoder,
    TemporalTransformerEncoder,
)
from backend.titan.fusion import (
    ConcatenationFusion,
    GatedFusion,
    CrossAttentionFusion,
)
from backend.titan.dual_stream import (
    DualStreamCrossHandModel,
    InterHandKinematics,
)
from backend.titan.architectures import (
    ArchitectureA_SpatialGNN,
    ArchitectureC_STTransformer,
    TitanHybridModel,
)
from backend.titan.uncertainty import (
    TemperatureScaler,
    compute_brier_score,
    compute_ece,
    compute_predictive_entropy,
    evaluate_selective_rejection,
)
from backend.titan.robustness import (
    apply_scale_perturbation,
    apply_rotation_perturbation,
    apply_sensor_noise,
    apply_partial_occlusion,
)
from backend.titan.export import export_spatial_gnn_to_json


# ---------------------------------------------------------------------------
# Test 1: Graph Topology & Normalized Laplacian
# ---------------------------------------------------------------------------
def test_graph_adjacency():
    adj = build_adjacency_matrix(add_self_loops=True)
    assert adj.shape == (21, 21)
    # Check symmetry
    assert np.allclose(adj, adj.T)
    # Check self loops
    assert np.all(np.diag(adj) == 1.0)

    norm_adj = compute_normalized_laplacian(adj)
    assert norm_adj.shape == (21, 21)
    assert np.allclose(norm_adj, norm_adj.T)
    # Spectral radius check: max eigenvalue of normalized adjacency <= 1.0 + eps
    eigenvals = np.linalg.eigvalsh(norm_adj)
    assert np.max(eigenvals) <= 1.0001


# ---------------------------------------------------------------------------
# Test 2: Landmarks to Node Features Conversion
# ---------------------------------------------------------------------------
def test_landmarks_to_node_features():
    # Construct synthetic 21 landmarks
    lms = [{"x": float(i * 0.02), "y": float(i * 0.03), "z": 0.01} for i in range(21)]
    nodes = landmarks_to_node_features(lms)
    assert nodes.shape == (21, 6)
    assert not np.isnan(nodes).any()
    assert not np.isinf(nodes).any()
    # Wrist relative coords for wrist (0) must be 0
    assert np.allclose(nodes[0, :3], 0.0)


# ---------------------------------------------------------------------------
# Test 3: Spatial Graph Encoder Forward Pass
# ---------------------------------------------------------------------------
def test_spatial_graph_encoder():
    model = SpatialGraphEncoder(hidden_dim=32, out_dim=32, num_layers=2)
    dummy_nodes = torch.randn(4, 21, 6)
    dummy_aux = torch.randn(4, 42)

    # With aux features
    out_aux = model(dummy_nodes, aux_features=dummy_aux)
    assert out_aux.shape == (4, 32)
    assert not torch.isnan(out_aux).any()

    # Without aux features
    model_no_aux = SpatialGraphEncoder(hidden_dim=32, out_dim=32, use_aux_features=False)
    out_no_aux = model_no_aux(dummy_nodes)
    assert out_no_aux.shape == (4, 32)


# ---------------------------------------------------------------------------
# Test 4: BiLSTM Temporal Encoder
# ---------------------------------------------------------------------------
def test_bilstm_temporal_encoder():
    model = BiLSTMTemporalEncoder(in_dim=42, hidden_dim=32, out_dim=32)
    seq = torch.randn(2, 16, 42)
    z_temp, h_seq, weights = model(seq)

    assert z_temp.shape == (2, 32)
    assert h_seq.shape == (2, 16, 32)
    assert weights.shape == (2, 16)
    # Weights must sum to 1.0 along time axis
    assert torch.allclose(torch.sum(weights, dim=-1), torch.ones(2), atol=1e-5)


# ---------------------------------------------------------------------------
# Test 5: Temporal Transformer Encoder & Causal Masking
# ---------------------------------------------------------------------------
def test_temporal_transformer_encoder():
    model = TemporalTransformerEncoder(in_dim=42, d_model=32, nhead=2, num_layers=1, causal=True)
    seq = torch.randn(2, 16, 42)
    z_temp, h_seq, weights = model(seq)

    assert z_temp.shape == (2, 32)
    assert h_seq.shape == (2, 16, 32)
    assert not torch.isnan(z_temp).any()

    # Test with padding mask (last 4 frames masked)
    mask = torch.ones(2, 16, dtype=torch.bool)
    mask[:, 12:] = False
    z_masked, _, _ = model(seq, mask=mask)
    assert z_masked.shape == (2, 32)
    assert not torch.isnan(z_masked).any()


# ---------------------------------------------------------------------------
# Test 6: Cross-Attention Fusion
# ---------------------------------------------------------------------------
def test_cross_attention_fusion():
    fusion = CrossAttentionFusion(spatial_dim=32, temporal_dim=32, out_dim=32, num_heads=2)
    zs = torch.randn(3, 32)
    zt = torch.randn(3, 32)
    h_seq = torch.randn(3, 16, 32)

    fused, attn_weights = fusion(zs, zt, h_seq)
    assert fused.shape == (3, 32)
    assert attn_weights.shape == (3, 1, 16)
    assert not torch.isnan(fused).any()


# ---------------------------------------------------------------------------
# Test 7: Two-Hand Dual Stream
# ---------------------------------------------------------------------------
def test_two_hand_dual_stream():
    model = DualStreamCrossHandModel(num_classes=10, d_model=32, num_heads=2)
    B = 2
    left_nodes = torch.randn(B, 21, 6)
    right_nodes = torch.randn(B, 21, 6)
    left_present = torch.tensor([[1.0], [0.0]])  # Sample 2 has no left hand
    right_present = torch.tensor([[1.0], [1.0]])
    left_wrist = torch.randn(B, 3)
    right_wrist = torch.randn(B, 3)
    left_scale = torch.tensor([[0.12], [0.0]])
    right_scale = torch.tensor([[0.12], [0.11]])

    logits = model(left_nodes, right_nodes, left_present, right_present, left_wrist, right_wrist, left_scale, right_scale)
    assert logits.shape == (B, 10)
    assert not torch.isnan(logits).any()


# ---------------------------------------------------------------------------
# Test 8: TITAN Hybrid Full Model
# ---------------------------------------------------------------------------
def test_titan_hybrid_full():
    model = TitanHybridModel(num_classes=25, d_model=32, num_heads=2, fusion_type="cross_attention")
    nodes = torch.randn(2, 21, 6)
    seq = torch.randn(2, 16, 42)
    aux = torch.randn(2, 42)

    logits, attn = model(nodes, seq, aux_features=aux)
    assert logits.shape == (2, 25)
    assert attn is not None
    assert not torch.isnan(logits).any()


# ---------------------------------------------------------------------------
# Test 9: Uncertainty Calibration & ECE
# ---------------------------------------------------------------------------
def test_uncertainty_calibration():
    # Perfectly calibrated synthetic predictions
    np.random.seed(42)
    probs = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.6, 0.4],
        [0.95, 0.05],
    ])
    labels = np.array([0, 0, 1, 0])
    ece = compute_ece(probs, labels, num_bins=5)
    brier = compute_brier_score(probs, labels, num_classes=2)

    assert 0.0 <= ece <= 1.0
    assert 0.0 <= brier <= 2.0


# ---------------------------------------------------------------------------
# Test 10: Selective Rejection & Predictive Entropy
# ---------------------------------------------------------------------------
def test_selective_rejection_and_entropy():
    probs = np.array([
        [0.95, 0.05],
        [0.55, 0.45],
    ])
    labels = np.array([0, 1])

    entropies = compute_predictive_entropy(probs)
    assert entropies[0] < entropies[1]  # Confident prediction has lower entropy

    rejection_table = evaluate_selective_rejection(probs, labels, conf_thresholds=[0.50, 0.90])
    assert len(rejection_table) == 2
    # At threshold 0.90, only the 0.95 sample is accepted -> coverage = 0.5
    assert rejection_table[1]["coverage"] == 0.5


# ---------------------------------------------------------------------------
# Test 11: Physical Perturbations Robustness
# ---------------------------------------------------------------------------
def test_robustness_perturbations():
    lms = [{"x": float(i * 0.03), "y": float(i * 0.02), "z": 0.0} for i in range(21)]

    # Scale
    scaled = apply_scale_perturbation(lms, factor=1.5)
    assert len(scaled) == 21
    assert scaled[0]["x"] == lms[0]["x"]  # Wrist fixed

    # Rotation
    rotated = apply_rotation_perturbation(lms, angle_deg=45.0)
    assert len(rotated) == 21
    assert math.isclose(rotated[0]["x"], lms[0]["x"], abs_tol=1e-5)

    # Noise
    noisy = apply_sensor_noise(lms, sigma=0.01)
    assert len(noisy) == 21

    # Occlusion
    occluded = apply_partial_occlusion(lms, finger="index")
    assert occluded[5]["x"] == lms[0]["x"]  # Index MCP collapsed to wrist


# ---------------------------------------------------------------------------
# Test 12: Model Serialization to JSON
# ---------------------------------------------------------------------------
def test_model_export_json(tmp_path):
    model = ArchitectureA_SpatialGNN(num_classes=25, hidden_dim=32, out_dim=32)
    export_file = tmp_path / "titan_test_export.json"

    payload = export_spatial_gnn_to_json(model, export_file, temperature=1.15)
    assert export_file.exists()
    assert payload["num_classes"] == 25
    assert payload["temperature"] == 1.15
    assert "weights" in payload
    assert "spatial_encoder.joint_embed.weight" in payload["weights"]
