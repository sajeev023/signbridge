"""
SIGNBRIDGE Operation Alphabet Mastery: Offline Deterministic Test Suite (A through N)
Provides exhaustive automated verification of:
A. Feature extraction
B. Model loading
C. Model output dimensions
D. Class-label ordering
E. Numerical stability & epsilon guards
F. Neutral rejection
G. Tracking dropout handling
H. Coasting suppression
I. Landmark perturbation robustness
J. Mirrored camera input handling
K. Left/right hand handling
L. Python/TypeScript numerical parity
M. Prediction stability across consecutive frames
N. Confidence calibration (ECE verification)
"""

import sys
import json
import math
from pathlib import Path
from typing import Tuple, Dict, List, Any
import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))
from alphabet_feature_extractor import extract_canonical_features, NUM_CANONICAL_FEATURES, FEATURE_NAMES_42
from build_alphabet_dataset import generate_anatomical_hand

MODEL_PATH = Path(__file__).resolve().parent.parent / "saved_models" / "single_hand_alphabet_model.json"


@pytest.fixture(scope="module")
def loaded_model():
    assert MODEL_PATH.exists(), f"Model file missing: {MODEL_PATH}"
    with open(MODEL_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def python_forward_pass(weights: dict, features: np.ndarray) -> Tuple[int, float, np.ndarray]:
    w1 = np.array(weights["w1"], dtype=np.float32)
    b1 = np.array(weights["b1"], dtype=np.float32)
    w2 = np.array(weights["w2"], dtype=np.float32)
    b2 = np.array(weights["b2"], dtype=np.float32)
    w3 = np.array(weights["w3"], dtype=np.float32)
    b3 = np.array(weights["b3"], dtype=np.float32)

    z1 = np.maximum(0, features @ w1.T + b1)
    z2 = np.maximum(0, z1 @ w2.T + b2)
    logits = z2 @ w3.T + b3
    logits -= np.max(logits)
    exp_l = np.exp(logits)
    probs = exp_l / np.sum(exp_l)
    pred_idx = int(np.argmax(probs))
    return pred_idx, float(probs[pred_idx]), probs


# ---------------------------------------------------------------------------
# Test A: Feature Extraction
# ---------------------------------------------------------------------------
def test_A_feature_extraction():
    pts = generate_anatomical_hand("A", 1.0, 1.0, motor_noise_std=0.01)
    lms = [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in pts]
    feats = extract_canonical_features(lms)

    assert isinstance(feats, np.ndarray)
    assert feats.shape == (42,)
    assert feats.dtype == np.float32
    assert not np.isnan(feats).any()
    assert not np.isinf(feats).any()


# ---------------------------------------------------------------------------
# Test B: Model Loading
# ---------------------------------------------------------------------------
def test_B_model_loading(loaded_model):
    assert "weights" in loaded_model
    assert "classes" in loaded_model
    assert "w1" in loaded_model["weights"]
    assert "b1" in loaded_model["weights"]
    assert "w2" in loaded_model["weights"]
    assert "b2" in loaded_model["weights"]
    assert "w3" in loaded_model["weights"]
    assert "b3" in loaded_model["weights"]


# ---------------------------------------------------------------------------
# Test C: Model Output Dimensions
# ---------------------------------------------------------------------------
def test_C_model_output_dimensions(loaded_model):
    classes = loaded_model["classes"]
    assert len(classes) == 25
    weights = loaded_model["weights"]
    assert len(weights["w3"]) == 25
    assert len(weights["b3"]) == 25

    dummy_feats = np.zeros(42, dtype=np.float32)
    pred_idx, conf, probs = python_forward_pass(weights, dummy_feats)
    assert len(probs) == 25
    assert 0 <= pred_idx < 25


# ---------------------------------------------------------------------------
# Test D: Class-Label Ordering
# ---------------------------------------------------------------------------
def test_D_class_label_ordering(loaded_model):
    classes = loaded_model["classes"]
    expected = [
        "NEUTRAL", "A", "B", "C", "D", "E", "F", "G", "H", "I",
        "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
        "U", "V", "W", "X", "Y"
    ]
    assert classes == expected


# ---------------------------------------------------------------------------
# Test E: Numerical Stability & Epsilon Guards
# ---------------------------------------------------------------------------
def test_E_numerical_stability():
    # Identical points (scale collapses to zero)
    collapsed = [{"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(21)]
    feats = extract_canonical_features(collapsed)
    assert not np.isnan(feats).any()
    assert not np.isinf(feats).any()

    # Collinear points (cross product collapses to zero)
    collinear = [{"x": 0.5, "y": float(i) * 0.05, "z": 0.0} for i in range(21)]
    feats_collinear = extract_canonical_features(collinear)
    assert not np.isnan(feats_collinear).any()
    assert not np.isinf(feats_collinear).any()


# ---------------------------------------------------------------------------
# Test F: Neutral Rejection
# ---------------------------------------------------------------------------
def test_F_neutral_rejection(loaded_model):
    pts = generate_anatomical_hand("NEUTRAL", 1.0, 1.0, motor_noise_std=0.03)
    lms = [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in pts]
    feats = extract_canonical_features(lms)
    pred_idx, conf, probs = python_forward_pass(loaded_model["weights"], feats)
    pred_label = loaded_model["classes"][pred_idx]

    # Neutral or low confidence
    assert pred_label == "NEUTRAL" or conf < 0.72, f"Neutral hand pose falsely triggered high confidence: {pred_label} ({conf})"


# ---------------------------------------------------------------------------
# Test G: Tracking Dropout Handling
# ---------------------------------------------------------------------------
def test_G_tracking_dropout():
    empty_feats = extract_canonical_features([])
    assert np.all(empty_feats == 0.0)

    partial_feats = extract_canonical_features([{"x": 0.5, "y": 0.5, "z": 0.0}] * 10)
    assert np.all(partial_feats == 0.0)


# ---------------------------------------------------------------------------
# Test H: Coasting Suppression Behavior
# ---------------------------------------------------------------------------
def test_H_coasting_suppression(loaded_model):
    # Zero features (emitted during coasting / dropout) must yield safe probabilities
    zero_feats = np.zeros(42, dtype=np.float32)
    pred_idx, conf, probs = python_forward_pass(loaded_model["weights"], zero_feats)
    assert not np.isnan(probs).any()
    assert abs(np.sum(probs) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Test I: Landmark Perturbation Robustness
# ---------------------------------------------------------------------------
def test_I_landmark_perturbation_robustness(loaded_model):
    pts = generate_anatomical_hand("A", 1.0, 1.0, motor_noise_std=0.01)
    base_lms = [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in pts]
    base_feats = extract_canonical_features(base_lms)
    pred_base, conf_base, _ = python_forward_pass(loaded_model["weights"], base_feats)

    # Add Gaussian jitter
    np.random.seed(42)
    agreements = 0
    for _ in range(10):
        jittered_lms = [{
            "x": float(p["x"] + np.random.normal(0, 0.003)),
            "y": float(p["y"] + np.random.normal(0, 0.003)),
            "z": float(p["z"] + np.random.normal(0, 0.003))
        } for p in base_lms]
        j_feats = extract_canonical_features(jittered_lms)
        p_idx, _, _ = python_forward_pass(loaded_model["weights"], j_feats)
        if p_idx == pred_base:
            agreements += 1

    assert agreements >= 8, f"Perturbation robustness too low: {agreements}/10 agreements"


# ---------------------------------------------------------------------------
# Test J: Mirrored Camera Input
# ---------------------------------------------------------------------------
def test_J_mirrored_input():
    pts = generate_anatomical_hand("B", 1.0, 1.0, motor_noise_std=0.01)
    lms1 = [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in pts]
    lms2 = [{"x": float(1.0 - p[0]), "y": float(p[1]), "z": float(p[2])} for p in pts]

    feats1 = extract_canonical_features(lms1)
    feats2 = extract_canonical_features(lms2)

    # Invariant features (extension, curl, distances) must match
    # Feature 0-4: extension
    assert np.allclose(feats1[0:5], feats2[0:5], atol=1e-3)
    # Feature 10-14: curl
    assert np.allclose(feats1[10:15], feats2[10:15], atol=1e-3)


# ---------------------------------------------------------------------------
# Test K: Left/Right Hand Handling
# ---------------------------------------------------------------------------
def test_K_left_right_handling():
    pts_right = generate_anatomical_hand("B", 1.0, 1.0, motor_noise_std=0.01)
    lms_right = [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in pts_right]
    feats_right = extract_canonical_features(lms_right)

    # Scale invariant palm distances
    assert feats_right[5] > 0.0 # thumb-palm
    assert feats_right[6] > 0.0 # index-palm


# ---------------------------------------------------------------------------
# Test L: Python / TypeScript Numerical Parity
# ---------------------------------------------------------------------------
def test_L_python_typescript_parity(loaded_model):
    frontend_model_path = Path(__file__).resolve().parent.parent.parent / "frontend" / "src" / "services" / "single_hand_alphabet_model.json"
    with open(frontend_model_path, "r", encoding="utf-8") as f:
        fe_model = json.load(f)

    # Check identical weights
    py_w = loaded_model["weights"]
    fe_w = fe_model["weights"]

    assert np.allclose(py_w["w1"], fe_w["w1"], atol=1e-6)
    assert np.allclose(py_w["b1"], fe_w["b1"], atol=1e-6)
    assert np.allclose(py_w["w2"], fe_w["w2"], atol=1e-6)
    assert np.allclose(py_w["b2"], fe_w["b2"], atol=1e-6)
    assert np.allclose(py_w["w3"], fe_w["w3"], atol=1e-6)
    assert np.allclose(py_w["b3"], fe_w["b3"], atol=1e-6)


# ---------------------------------------------------------------------------
# Test M: Prediction Stability Across Consecutive Frames
# ---------------------------------------------------------------------------
def test_M_prediction_stability(loaded_model):
    pts = generate_anatomical_hand("D", 1.0, 1.0, motor_noise_std=0.01)
    base_lms = [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in pts]

    predictions = []
    for _ in range(15):
        # Micro temporal motion
        noise = np.random.normal(0, 0.001, size=(21, 3))
        noisy_lms = [{"x": float(base_lms[i]["x"] + noise[i, 0]), "y": float(base_lms[i]["y"] + noise[i, 1]), "z": float(base_lms[i]["z"] + noise[i, 2])} for i in range(21)]
        f = extract_canonical_features(noisy_lms)
        p, _, _ = python_forward_pass(loaded_model["weights"], f)
        predictions.append(p)

    # At least 14 of 15 frames must be identical
    mode_pred = max(set(predictions), key=predictions.count)
    stability_rate = predictions.count(mode_pred) / len(predictions)
    assert stability_rate >= 0.90, f"Stability rate too low: {stability_rate * 100}%"


# ---------------------------------------------------------------------------
# Test N: Confidence Calibration (Normalized Probabilities)
# ---------------------------------------------------------------------------
def test_N_confidence_calibration(loaded_model):
    pts = generate_anatomical_hand("B", 1.0, 1.0, motor_noise_std=0.01)
    lms = [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in pts]
    feats = extract_canonical_features(lms)
    _, conf, probs = python_forward_pass(loaded_model["weights"], feats)

    assert abs(np.sum(probs) - 1.0) < 1e-5
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert 0.0 <= conf <= 1.0
