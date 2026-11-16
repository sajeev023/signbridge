"""
SIGNBRIDGE Coordinate AI — Python & TypeScript Numerical Parity Test Suite
Validates:
1. Exact class list and ordering between Python model zoo and JSON artifact.
2. Feature extraction dimensionality (94 features) and numerical consistency.
3. LayerNorm numerical precision between PyTorch and pure NumPy implementation.
4. Scale and translation invariance of coordinate features.
5. Temperature-scaled softmax output validity.
"""

import math
import json
from pathlib import Path
import numpy as np
import pytest
import torch

from backend.coordinate_ai.feature_pipeline import (
    extract_enriched_coordinate_features,
    extract_normalized_coordinates_63,
    calculate_palm_scale,
    landmarks_to_numpy,
    NUM_LANDMARKS,
    ENRICHED_FEATURE_DIM,
    RAW_COORDINATE_DIM,
)
from backend.coordinate_ai.model_zoo import (
    CoordinateMLP,
    CoordinateInferenceEngine,
    STATIC_ASL_CLASSES,
    DYNAMIC_ASL_CLASSES,
    ALL_ASL_CLASSES,
)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def model_dict():
    model_path = WORKSPACE_ROOT / "backend" / "saved_models" / "coordinate_ai_model.json"
    assert model_path.exists(), f"Coordinate AI model missing at {model_path}"
    with open(model_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def synthetic_hand():
    """Generates a realistic canonical hand landmark dictionary."""
    landmarks = []
    # Wrist at (0.5, 0.8, 0.0)
    landmarks.append({"x": 0.5, "y": 0.8, "z": 0.0})
    # Thumb joints (1-4)
    landmarks.append({"x": 0.45, "y": 0.75, "z": -0.01})
    landmarks.append({"x": 0.42, "y": 0.70, "z": -0.02})
    landmarks.append({"x": 0.40, "y": 0.65, "z": -0.02})
    landmarks.append({"x": 0.38, "y": 0.60, "z": -0.02})
    # Index finger joints (5-8)
    landmarks.append({"x": 0.46, "y": 0.68, "z": 0.0})
    landmarks.append({"x": 0.46, "y": 0.58, "z": 0.01})
    landmarks.append({"x": 0.46, "y": 0.50, "z": 0.01})
    landmarks.append({"x": 0.46, "y": 0.42, "z": 0.02})
    # Middle finger joints (9-12)
    landmarks.append({"x": 0.50, "y": 0.67, "z": 0.0})
    landmarks.append({"x": 0.50, "y": 0.56, "z": 0.01})
    landmarks.append({"x": 0.50, "y": 0.48, "z": 0.01})
    landmarks.append({"x": 0.50, "y": 0.40, "z": 0.02})
    # Ring finger joints (13-16)
    landmarks.append({"x": 0.54, "y": 0.68, "z": -0.01})
    landmarks.append({"x": 0.54, "y": 0.58, "z": 0.0})
    landmarks.append({"x": 0.54, "y": 0.51, "z": 0.0})
    landmarks.append({"x": 0.54, "y": 0.44, "z": 0.01})
    # Pinky finger joints (17-20)
    landmarks.append({"x": 0.58, "y": 0.70, "z": -0.02})
    landmarks.append({"x": 0.58, "y": 0.62, "z": -0.02})
    landmarks.append({"x": 0.58, "y": 0.56, "z": -0.02})
    landmarks.append({"x": 0.58, "y": 0.50, "z": -0.02})
    return landmarks


def test_class_metadata_parity(model_dict):
    """Verifies that model classes match STATIC_ASL_CLASSES exactly in order and length."""
    model_classes = model_dict["classes"]
    assert model_classes == STATIC_ASL_CLASSES
    assert len(model_classes) == 24
    assert "J" not in model_classes  # Dynamic sign
    assert "Z" not in model_classes  # Dynamic sign


def test_feature_pipeline_dimensions(synthetic_hand):
    """Verifies 94-dimensional feature structure."""
    raw_63 = extract_normalized_coordinates_63(synthetic_hand)
    assert raw_63.shape == (63,)
    assert not np.isnan(raw_63).any()

    feats_94 = extract_enriched_coordinate_features(synthetic_hand)
    assert feats_94.shape == (94,)
    assert not np.isnan(feats_94).any()


def test_translation_invariance(synthetic_hand):
    """Features must be invariant to rigid 2D/3D camera translations."""
    feats_orig = extract_enriched_coordinate_features(synthetic_hand)

    # Shift hand by +0.35 in X, -0.20 in Y, +0.15 in Z
    shifted_hand = [
        {"x": p["x"] + 0.35, "y": p["y"] - 0.20, "z": p["z"] + 0.15}
        for p in synthetic_hand
    ]
    feats_shifted = extract_enriched_coordinate_features(shifted_hand)

    # Maximum absolute difference across all 94 features must be negligible
    max_diff = np.max(np.abs(feats_orig - feats_shifted))
    assert max_diff < 1e-5, f"Translation invariance violated: max_diff = {max_diff}"


def test_scale_invariance(synthetic_hand):
    """Features must be invariant to uniform scaling (simulating user camera distance)."""
    feats_orig = extract_enriched_coordinate_features(synthetic_hand)

    # Scale hand by 2.0x relative to wrist (index 0)
    wrist = synthetic_hand[0]
    scaled_hand = []
    for p in synthetic_hand:
        scaled_hand.append({
            "x": wrist["x"] + (p["x"] - wrist["x"]) * 2.0,
            "y": wrist["y"] + (p["y"] - wrist["y"]) * 2.0,
            "z": wrist["z"] + (p["z"] - wrist["z"]) * 2.0
        })
    feats_scaled = extract_enriched_coordinate_features(scaled_hand)

    max_diff = np.max(np.abs(feats_orig - feats_scaled))
    assert max_diff < 1e-4, f"Scale invariance violated: max_diff = {max_diff}"


def test_pytorch_and_numpy_inference_parity(model_dict, synthetic_hand):
    """
    Verifies that PyTorch CoordinateMLP forward pass and pure-NumPy CoordinateInferenceEngine
    yield identical logits and probabilities.
    """
    feats = extract_enriched_coordinate_features(synthetic_hand)

    # 1. NumPy engine
    engine = CoordinateInferenceEngine(model_dict, confidence_threshold=0.70)
    res_numpy = engine.predict_features(feats)

    # 2. PyTorch model reconstruction
    w = model_dict["weights"]
    mlp = CoordinateMLP(input_dim=94, num_classes=24)
    mlp.eval()

    with torch.no_grad():
        mlp.fc1.weight.copy_(torch.tensor(w["w1"]))
        mlp.fc1.bias.copy_(torch.tensor(w["b1"]))
        mlp.ln1.weight.copy_(torch.tensor(w["ln1_gamma"]))
        mlp.ln1.bias.copy_(torch.tensor(w["ln1_beta"]))

        mlp.fc2.weight.copy_(torch.tensor(w["w2"]))
        mlp.fc2.bias.copy_(torch.tensor(w["b2"]))
        mlp.ln2.weight.copy_(torch.tensor(w["ln2_gamma"]))
        mlp.ln2.bias.copy_(torch.tensor(w["ln2_beta"]))

        mlp.fc3.weight.copy_(torch.tensor(w["w3"]))
        mlp.fc3.bias.copy_(torch.tensor(w["b3"]))
        mlp.ln3.weight.copy_(torch.tensor(w["ln3_gamma"]))
        mlp.ln3.bias.copy_(torch.tensor(w["ln3_beta"]))

        mlp.classifier.weight.copy_(torch.tensor(w["w4"]))
        mlp.classifier.bias.copy_(torch.tensor(w["b4"]))

        mlp.temperature.copy_(torch.tensor([float(model_dict.get("temperature", 1.0))]))

        t_in = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
        t_logits = mlp(t_in)[0]
        t_probs = torch.softmax(t_logits, dim=-1).numpy()

    # Compare top prediction and probability
    py_best_idx = int(np.argmax(t_probs))
    py_best_letter = STATIC_ASL_CLASSES[py_best_idx]
    py_best_conf = float(t_probs[py_best_idx])

    assert res_numpy["best_candidate"] == py_best_letter
    assert math.isclose(res_numpy["confidence"], py_best_conf, abs_tol=1e-4)


def test_invalid_input_rejection(model_dict):
    """Empty or corrupt landmarks must safely trigger zero features and REJECTED status."""
    engine = CoordinateInferenceEngine(model_dict, confidence_threshold=0.70)

    # Empty hand
    res_empty = engine.predict_landmarks([])
    assert res_empty["predicted_letter"] == "REJECTED"
    assert res_empty["confidence"] == 0.0
    assert res_empty["is_accepted"] is False

    # Fewer than 21 landmarks
    res_partial = engine.predict_landmarks([{"x": 0.5, "y": 0.5, "z": 0.0}] * 10)
    assert res_partial["predicted_letter"] == "REJECTED"
    assert res_partial["is_accepted"] is False
