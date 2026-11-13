"""
Automated Test Suite for Dynamic J/Z Point-History Trajectory Engine
Validates buffer continuity, palm-scale distance invariance, variable speed resampling,
anatomical handshape context gating, and neural inference parity.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pytest

from backend.trajectory_point_history_engine import (
    PointHistoryBuffer,
    resample_trajectory_2d,
    calculate_trajectory_path_length,
    is_j_handshape_context,
    is_z_handshape_context,
    TRAJECTORY_BUFFER_LENGTH,
    TRAJECTORY_FEATURE_SIZE
)


@pytest.fixture
def workspace_root():
    return Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def dynamic_model_data(workspace_root):
    model_path = workspace_root / "backend" / "saved_models" / "dynamic_alphabet_model.json"
    assert model_path.exists(), f"Dynamic model artifact missing at {model_path}"
    with open(model_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def sample_trajectories(workspace_root):
    data_path = workspace_root / "backend" / "data" / "dynamic_jz_trajectories.json"
    assert data_path.exists(), f"Trajectory dataset missing at {data_path}"
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_buffer_continuity():
    """Verifies that time gaps (>300ms) or hand ID switches break continuity."""
    buf = PointHistoryBuffer(maxlen=16)

    # Clean sequence of 16 frames spaced by 33ms
    for i in range(16):
        buf.append({"x": 0.5, "y": 0.5}, palm_scale=0.1, timestamp_ms=i * 33.3, hand_id=1)
    assert buf.is_continuous(max_gap_ms=300.0) is True

    # Hand ID switch
    buf.reset()
    for i in range(15):
        buf.append({"x": 0.5, "y": 0.5}, palm_scale=0.1, timestamp_ms=i * 33.3, hand_id=1)
    buf.append({"x": 0.5, "y": 0.5}, palm_scale=0.1, timestamp_ms=15 * 33.3, hand_id=2)
    assert buf.is_continuous(max_gap_ms=300.0) is False

    # Timestamp gap
    buf.reset()
    for i in range(10):
        buf.append({"x": 0.5, "y": 0.5}, palm_scale=0.1, timestamp_ms=i * 33.3, hand_id=1)
    buf.append({"x": 0.5, "y": 0.5}, palm_scale=0.1, timestamp_ms=1000.0, hand_id=1)  # Gap of ~670ms
    assert buf.is_continuous(max_gap_ms=300.0) is False


def test_palm_scale_distance_invariance():
    """
    Verifies that the same physical hand stroke executed close to camera
    (scale=0.16) vs far from camera (scale=0.08) produces identical normalized features.
    """
    buf_close = PointHistoryBuffer(maxlen=16)
    buf_far = PointHistoryBuffer(maxlen=16)

    scale_close = 0.16
    scale_far = 0.08

    # Parabolic hook trajectory
    for i in range(16):
        t = i / 15.0
        # Physical displacement in palm-relative units
        phys_dx = -0.5 * (t ** 2)
        phys_dy = 1.0 * t

        # Projected camera screen displacement
        pt_close = {"x": 0.5 + phys_dx * scale_close, "y": 0.5 + phys_dy * scale_close}
        pt_far = {"x": 0.5 + phys_dx * scale_far, "y": 0.5 + phys_dy * scale_far}

        buf_close.append(pt_close, palm_scale=scale_close, timestamp_ms=i * 33.3)
        buf_far.append(pt_far, palm_scale=scale_far, timestamp_ms=i * 33.3)

    feat_close = buf_close.extract_normalized_features()
    feat_far = buf_far.extract_normalized_features()

    assert feat_close.shape == (32,)
    assert feat_far.shape == (32,)
    # Normalized feature representations must be mathematically identical within float precision
    np.testing.assert_allclose(feat_close, feat_far, atol=1e-4)


def test_variable_speed_resampling():
    """
    Verifies that a fast stroke (8 frames) and a slow stroke (24 frames)
    tracing the exact same geometric path resample to nearly identical 16-point representations.
    """
    # Define linear stroke from (0,0) to (1, 1)
    fast_pts = np.array([[t, t] for t in np.linspace(0, 1, 8)], dtype=np.float32)
    slow_pts = np.array([[t, t] for t in np.linspace(0, 1, 24)], dtype=np.float32)

    resampled_fast = resample_trajectory_2d(fast_pts, target_points=16)
    resampled_slow = resample_trajectory_2d(slow_pts, target_points=16)

    assert resampled_fast.shape == (16, 2)
    assert resampled_slow.shape == (16, 2)
    np.testing.assert_allclose(resampled_fast, resampled_slow, atol=1e-3)


def test_handshape_context_gating():
    """
    Verifies that 'J' handshape context only accepts pinky extended (I-shape)
    and 'Z' handshape context only accepts index extended (D-shape).
    """
    # 1. Construct synthetic 'I' pose (pinky extended, others curled)
    landmarks_i = [{"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(21)]
    landmarks_i[0] = {"x": 0.5, "y": 0.8, "z": 0.0}   # Wrist
    landmarks_i[9] = {"x": 0.5, "y": 0.7, "z": 0.0}   # Mid MCP (scale = 0.1)
    landmarks_i[5] = {"x": 0.45, "y": 0.7, "z": 0.0}  # Index MCP
    landmarks_i[8] = {"x": 0.45, "y": 0.72, "z": 0.0} # Index Tip (curled)
    landmarks_i[12] = {"x": 0.5, "y": 0.72, "z": 0.0} # Mid Tip (curled)
    landmarks_i[17] = {"x": 0.55, "y": 0.7, "z": 0.0} # Pinky MCP
    landmarks_i[20] = {"x": 0.55, "y": 0.55, "z": 0.0} # Pinky Tip (extended)

    palm_scale = 0.10
    assert is_j_handshape_context(landmarks_i, palm_scale) is True
    assert is_z_handshape_context(landmarks_i, palm_scale) is False

    # 2. Construct synthetic 'D' pose (index extended, others curled)
    landmarks_d = [{"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(21)]
    landmarks_d[0] = {"x": 0.5, "y": 0.8, "z": 0.0}   # Wrist
    landmarks_d[9] = {"x": 0.5, "y": 0.7, "z": 0.0}   # Mid MCP
    landmarks_d[5] = {"x": 0.45, "y": 0.7, "z": 0.0}  # Index MCP
    landmarks_d[8] = {"x": 0.45, "y": 0.55, "z": 0.0} # Index Tip (extended)
    landmarks_d[12] = {"x": 0.5, "y": 0.72, "z": 0.0} # Mid Tip (curled)
    landmarks_d[17] = {"x": 0.55, "y": 0.7, "z": 0.0} # Pinky MCP
    landmarks_d[20] = {"x": 0.55, "y": 0.72, "z": 0.0} # Pinky Tip (curled)

    assert is_j_handshape_context(landmarks_d, palm_scale) is False
    assert is_z_handshape_context(landmarks_d, palm_scale) is True

    # 3. Construct synthetic fist pose ('A'/'S')
    landmarks_fist = [{"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(21)]
    landmarks_fist[0] = {"x": 0.5, "y": 0.8, "z": 0.0}
    landmarks_fist[9] = {"x": 0.5, "y": 0.7, "z": 0.0}
    landmarks_fist[5] = {"x": 0.45, "y": 0.7, "z": 0.0}
    landmarks_fist[8] = {"x": 0.45, "y": 0.72, "z": 0.0}
    landmarks_fist[12] = {"x": 0.5, "y": 0.72, "z": 0.0}
    landmarks_fist[17] = {"x": 0.55, "y": 0.7, "z": 0.0}
    landmarks_fist[20] = {"x": 0.55, "y": 0.72, "z": 0.0}

    assert is_j_handshape_context(landmarks_fist, palm_scale) is False
    assert is_z_handshape_context(landmarks_fist, palm_scale) is False


def test_dynamic_mlp_forward_pass_and_shapes(dynamic_model_data):
    """Verifies weight matrix dimensions, activation outputs, and softmax normalization."""
    weights = dynamic_model_data["weights"]
    w1 = np.array(weights["w1"], dtype=np.float32)  # (32, 32)
    b1 = np.array(weights["b1"], dtype=np.float32)  # (32,)
    w2 = np.array(weights["w2"], dtype=np.float32)  # (16, 32)
    b2 = np.array(weights["b2"], dtype=np.float32)  # (16,)
    w3 = np.array(weights["w3"], dtype=np.float32)  # (3, 16)
    b3 = np.array(weights["b3"], dtype=np.float32)  # (3,)

    assert w1.shape == (32, 32)
    assert b1.shape == (32,)
    assert w2.shape == (16, 32)
    assert b2.shape == (16,)
    assert w3.shape == (3, 16)
    assert b3.shape == (3,)

    # Forward pass on zero input
    x0 = np.zeros(32, dtype=np.float32)
    z1 = np.maximum(0, w1 @ x0 + b1)
    z2 = np.maximum(0, w2 @ z1 + b2)
    logits = w3 @ z2 + b3
    probs = np.exp(logits - np.max(logits))
    probs /= np.sum(probs)

    assert len(probs) == 3
    assert math.isclose(float(np.sum(probs)), 1.0, rel_tol=1e-5)


def test_dataset_episodes_inference_fidelity(dynamic_model_data, sample_trajectories):
    """
    Runs dynamic MLP on sample episodes of J, Z, and NEUTRAL from the test cohort
    and verifies high recognition accuracy (>95%).
    """
    weights = dynamic_model_data["weights"]
    w1 = np.array(weights["w1"], dtype=np.float32)
    b1 = np.array(weights["b1"], dtype=np.float32)
    w2 = np.array(weights["w2"], dtype=np.float32)
    b2 = np.array(weights["b2"], dtype=np.float32)
    w3 = np.array(weights["w3"], dtype=np.float32)
    b3 = np.array(weights["b3"], dtype=np.float32)
    classes = dynamic_model_data["classes"]

    # Filter test cohort signers (Signer_4 and Signer_5)
    episodes = sample_trajectories["episodes"]
    test_episodes = [ep for ep in episodes if ep["signer_id"] in ["Signer_4", "Signer_5"]]
    assert len(test_episodes) == 240

    correct = 0
    for ep in test_episodes:
        pts = np.array([[p["x"], p["y"]] for p in ep["raw_trajectory"]], dtype=np.float32)
        scale = float(ep["palm_scale"])
        rel_pts = (pts - pts[0]) / scale
        resampled = resample_trajectory_2d(rel_pts, target_points=16).flatten()

        z1 = np.maximum(0, w1 @ resampled + b1)
        z2 = np.maximum(0, w2 @ z1 + b2)
        logits = w3 @ z2 + b3
        probs = np.exp(logits - np.max(logits))
        probs /= np.sum(probs)

        pred_class = classes[int(np.argmax(probs))]
        if pred_class == ep["label"]:
            correct += 1

    accuracy = (correct / len(test_episodes)) * 100.0
    assert accuracy >= 95.0, f"Expected dynamic trajectory test accuracy >= 95%, got {accuracy:.2f}%"
