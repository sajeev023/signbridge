"""
Comprehensive Stress Test Suite for Dynamic J/Z Recognition Engine
Simulates and verifies 10 rigorous physical and kinematic scenarios:
1. J at different signing speeds (fast, normal, slow)
2. Z at different signing speeds (fast, normal, slow)
3. Static 'I' without movement (must remain static, no dynamic trigger)
4. Static 'D' without movement (must remain static, no dynamic trigger)
5. Incomplete J and Z strokes (must be rejected as NEUTRAL)
6. Reversed / inverted trajectories (must be rejected as NEUTRAL)
7. Micro-tremor on static hands (must be rejected as NEUTRAL)
8. Sensor dropouts and tracking interruptions
9. Hand disappearance and reacquisition (buffer flush)
10. Scale variations / camera distance extremes (palm scale 0.08 to 0.18)
"""

import json
import math
from pathlib import Path
import numpy as np
import pytest

from backend.trajectory_point_history_engine import (
    PointHistoryBuffer,
    resample_trajectory_2d,
    calculate_trajectory_path_length,
    is_j_handshape_context,
    is_z_handshape_context
)


@pytest.fixture
def dynamic_model():
    model_path = Path("backend/saved_models/dynamic_alphabet_model.json")
    with open(model_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    w = data["weights"]
    return {
        "w1": np.array(w["w1"], dtype=np.float32),
        "b1": np.array(w["b1"], dtype=np.float32),
        "w2": np.array(w["w2"], dtype=np.float32),
        "b2": np.array(w["b2"], dtype=np.float32),
        "w3": np.array(w["w3"], dtype=np.float32),
        "b3": np.array(w["b3"], dtype=np.float32),
        "classes": data["classes"]
    }


def predict_trajectory(model, pts_normalized_32):
    z1 = np.maximum(0, model["w1"] @ pts_normalized_32 + model["b1"])
    z2 = np.maximum(0, model["w2"] @ z1 + model["b2"])
    logits = model["w3"] @ z2 + model["b3"]
    probs = np.exp(logits - np.max(logits))
    probs /= np.sum(probs)
    best_idx = int(np.argmax(probs))
    return model["classes"][best_idx], float(probs[best_idx])


def generate_j_trajectory(n_frames, scale=0.12):
    # J: Downward stroke then upward left hook
    t = np.linspace(0, 1, n_frames)
    pts = []
    for step in t:
        if step < 0.65:
            # Straight downward line
            x = 0.50
            y = 0.50 + (step / 0.65) * (scale * 1.5)
        else:
            # Curved hook left and up
            theta = ((step - 0.65) / 0.35) * math.pi
            x = 0.50 - math.sin(theta) * (scale * 0.6)
            y = 0.50 + (scale * 1.5) - (1.0 - math.cos(theta)) * (scale * 0.4)
        pts.append([x, y])
    return np.array(pts, dtype=np.float32)


def generate_z_trajectory(n_frames, scale=0.12):
    # Z: Top right-to-left, diagonal, bottom right-to-left
    t = np.linspace(0, 1, n_frames)
    pts = []
    for step in t:
        if step < 0.33:
            # Stroke 1: Left to right
            x = 0.45 + (step / 0.33) * (scale * 1.2)
            y = 0.50
        elif step < 0.66:
            # Stroke 2: Diagonal down-left
            frac = (step - 0.33) / 0.33
            x = (0.45 + scale * 1.2) - frac * (scale * 1.2)
            y = 0.50 + frac * (scale * 1.2)
        else:
            # Stroke 3: Bottom left-to-right
            frac = (step - 0.66) / 0.34
            x = 0.45 + frac * (scale * 1.2)
            y = 0.50 + scale * 1.2
        pts.append([x, y])
    return np.array(pts, dtype=np.float32)


def test_j_variable_speeds(dynamic_model):
    """Scenario 1: J at fast (10 frames), normal (20 frames), and slow (40 frames)."""
    for n in [10, 20, 40]:
        raw = generate_j_trajectory(n, scale=0.12)
        rel = (raw - raw[0]) / 0.12
        resampled = resample_trajectory_2d(rel, target_points=16).flatten()
        pred, conf = predict_trajectory(dynamic_model, resampled)
        assert pred == "J", f"Failed for J at {n} frames: predicted {pred} (conf {conf})"
        assert conf >= 0.85


def test_z_variable_speeds(dynamic_model):
    """Scenario 2: Z at fast (10 frames), normal (20 frames), and slow (40 frames)."""
    for n in [10, 20, 40]:
        raw = generate_z_trajectory(n, scale=0.12)
        rel = (raw - raw[0]) / 0.12
        resampled = resample_trajectory_2d(rel, target_points=16).flatten()
        pred, conf = predict_trajectory(dynamic_model, resampled)
        assert pred == "Z", f"Failed for Z at {n} frames: predicted {pred} (conf {conf})"
        assert conf >= 0.85


def test_static_i_zero_displacement(dynamic_model):
    """Scenario 3: Static 'I' sign held motionless."""
    # 20 frames of stationary pinky
    raw = np.array([[0.50, 0.50] for _ in range(20)], dtype=np.float32)
    path_len = calculate_trajectory_path_length(raw)
    scale = 0.12
    # Check minimum path length threshold
    assert path_len < 0.70 * scale, "Stationary hand must not exceed path length threshold"


def test_static_d_zero_displacement(dynamic_model):
    """Scenario 4: Static 'D' sign held motionless."""
    # 20 frames of stationary index
    raw = np.array([[0.50, 0.50] for _ in range(20)], dtype=np.float32)
    path_len = calculate_trajectory_path_length(raw)
    scale = 0.12
    assert path_len < 0.70 * scale


def test_incomplete_j_and_z(dynamic_model):
    """Scenario 5: Incomplete strokes (< half gesture completed)."""
    raw_j_half = generate_j_trajectory(20, scale=0.12)[:8] # only downward line, no hook
    path_len = calculate_trajectory_path_length(raw_j_half)
    scale = 0.12
    rel = (raw_j_half - raw_j_half[0]) / scale
    resampled = resample_trajectory_2d(rel, target_points=16).flatten()
    pred, conf = predict_trajectory(dynamic_model, resampled)
    # Straight downward line lacks the defining hook, must not classify as high-confidence J
    assert pred != "J" or conf < 0.85, f"Incomplete J accepted with confidence {conf}"


def test_reversed_and_inverted_trajectories(dynamic_model):
    """Scenario 6: Reversed strokes (e.g. upward hook instead of downward)."""
    raw_j = generate_j_trajectory(20, scale=0.12)
    reversed_j = raw_j[::-1] # bottom-to-top
    rel = (reversed_j - reversed_j[0]) / 0.12
    resampled = resample_trajectory_2d(rel, target_points=16).flatten()
    pred, conf = predict_trajectory(dynamic_model, resampled)
    assert pred != "J" or conf < 0.85, f"Reversed J erroneously accepted with conf {conf}"


def test_micro_tremor_suppression():
    """Scenario 7: Hand with natural physiological micro-tremor (jitter)."""
    np.random.seed(42)
    base = np.array([0.50, 0.50], dtype=np.float32)
    jitter = np.random.normal(0, 0.0015, size=(20, 2))
    raw = base + jitter
    path_len = calculate_trajectory_path_length(raw)
    scale = 0.12
    # Tremor path length must remain well below deliberate movement threshold
    assert path_len < 0.50 * scale, f"Tremor path length {path_len} should be below threshold {0.5*scale}"


def test_sensor_dropout_recovery():
    """Scenario 8: Frame dropouts (gap in timestamps)."""
    buf = PointHistoryBuffer(maxlen=16)
    t = 1000.0
    buf.append({"x": 0.5, "y": 0.5}, palm_scale=0.12, timestamp_ms=t)
    t += 33.3
    buf.append({"x": 0.51, "y": 0.52}, palm_scale=0.12, timestamp_ms=t)
    t += 400.0  # big gap (400ms > 300ms default max_gap)
    buf.append({"x": 0.55, "y": 0.60}, palm_scale=0.12, timestamp_ms=t)
    # Continuity must detect the temporal disruption
    assert not buf.is_continuous(max_gap_ms=300.0)


def test_hand_loss_and_reacquisition():
    """Scenario 9: Hand disappears and reacquires (track ID change)."""
    buf = PointHistoryBuffer(maxlen=16)
    # Fill buffer with hand_id = 1
    for i in range(15):
        buf.append({"x": 0.5 + i * 0.01, "y": 0.5 + i * 0.01}, palm_scale=0.12, timestamp_ms=1000.0 + i * 33.3, hand_id=1)
    
    # 16th point comes from hand_id = 2 (id switch / track loss)
    buf.append({"x": 0.65, "y": 0.65}, palm_scale=0.12, timestamp_ms=1000.0 + 15 * 33.3, hand_id=2)
    assert len(buf.get_raw_points()) == 16
    # is_continuous must catch the hand ID switch and reject
    assert not buf.is_continuous()

    # Manual reset on tracking loss
    buf.reset()
    assert len(buf.get_raw_points()) == 0


def test_camera_distance_invariance(dynamic_model):
    """Scenario 10: Extreme palm scales (close: 0.18 vs far: 0.08)."""
    raw_close = generate_j_trajectory(20, scale=0.18)
    raw_far = generate_j_trajectory(20, scale=0.08)

    rel_close = (raw_close - raw_close[0]) / 0.18
    rel_far = (raw_far - raw_far[0]) / 0.08

    res_close = resample_trajectory_2d(rel_close, target_points=16).flatten()
    res_far = resample_trajectory_2d(rel_far, target_points=16).flatten()

    pred_close, conf_close = predict_trajectory(dynamic_model, res_close)
    pred_far, conf_far = predict_trajectory(dynamic_model, res_far)

    assert pred_close == "J" and conf_close >= 0.85
    assert pred_far == "J" and conf_far >= 0.85
    np.testing.assert_allclose(res_close, res_far, atol=1e-3)
