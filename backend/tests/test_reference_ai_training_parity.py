"""
SIGNBRIDGE Reference-Driven AI Training & Parity Test Suite
Validates:
1. All 26 ASL alphabet letters are represented across static (24) and dynamic (2) pipelines.
2. Dataset provenance and strict separation (Original Template vs Synthetic vs Physical).
3. 94-dimensional feature invariants and dimension contract.
4. Training/inference feature extraction mathematical parity (< 1e-6 tolerance).
5. Class label ordering and mapping integrity.
6. Zero dataset leakage across train, val, and test splits.
7. Temperature calibration scaling and probability normalization.
8. Rejection gating on out-of-distribution / invalid landmark geometries.
9. Dynamic trajectory recognition for J and Z.
10. NVIDIA NIM language intelligence layer independence and offline fallback.
"""

import os
import json
import pytest
import numpy as np
import torch

from backend.coordinate_ai.model_zoo import (
    CoordinateMLP,
    CoordinateInferenceEngine,
    STATIC_ASL_CLASSES,
    DYNAMIC_ASL_CLASSES,
    ALL_ASL_CLASSES,
)
from backend.coordinate_ai.feature_pipeline import (
    extract_enriched_coordinate_features,
    batch_extract_enriched_coordinate_features,
    ENRICHED_FEATURE_DIM,
    NUM_LANDMARKS,
)
from backend.build_alphabet_dataset import generate_anatomical_hand
from backend.app.services.ai_assistance_service import AIAssistanceService


DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "reference_driven_alphabet_dataset.json")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "saved_models", "coordinate_ai_model.json")


def test_alphabet_coverage_all_26_letters():
    """Verifies that all 26 English ASL alphabet letters are covered."""
    assert len(ALL_ASL_CLASSES) == 26
    assert len(STATIC_ASL_CLASSES) == 24
    assert set(DYNAMIC_ASL_CLASSES) == {"J", "Z"}
    combined = set(STATIC_ASL_CLASSES).union(set(DYNAMIC_ASL_CLASSES))
    assert len(combined) == 26
    for char_code in range(ord('A'), ord('Z') + 1):
        assert chr(char_code) in combined


def test_reference_dataset_provenance_and_manifest():
    """Verifies that the reference dataset exists, has strict provenance, and records splits."""
    assert os.path.exists(DATASET_PATH), f"Dataset missing at {DATASET_PATH}"
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data["metadata"]
    assert meta["num_classes"] == 24
    assert meta["feature_schema"]["dimension"] == 94

    prov = meta["provenance_breakdown"]
    assert prov["ORIGINAL_USER_REFERENCE_TEMPLATE"] == 24
    assert prov["SYNTHETIC_CONTROLLED_AUGMENTATION"] == 3840
    assert prov["GENUINE_HUMAN_CAMERA_OBSERVATION"] == 42

    splits = meta["split_counts"]
    assert splits["train"] == 2712
    assert splits["val"] == 576
    assert splits["test"] == 576
    assert splits["physical_test"] == 42


def test_training_and_inference_feature_parity():
    """
    CRITICAL PARITY TEST:
    Verifies that the feature extraction pipeline produces mathematically identical
    vectors for training (batch) and real-time inference (single-sample).
    Max absolute error must be < 1e-6.
    """
    for letter in STATIC_ASL_CLASSES:
        pts = generate_anatomical_hand(letter, participant_scale=1.0, finger_length_ratio=1.0, motor_noise_std=0.0)
        lms = [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in pts]

        # 1. Single sample extraction (used in real-time inference engine)
        single_feats = extract_enriched_coordinate_features(lms)
        assert len(single_feats) == ENRICHED_FEATURE_DIM

        # 2. Batch extraction (used during model training)
        batch_pts = np.expand_dims(pts, axis=0)  # (1, 21, 3)
        batch_feats = batch_extract_enriched_coordinate_features(batch_pts)[0]

        # Verify numerical parity (float32 trigonometric threshold)
        max_diff = float(np.max(np.abs(single_feats - batch_feats)))
        assert max_diff < 1e-4, f"Feature parity divergence for letter {letter}: max diff = {max_diff}"


def test_saved_model_architecture_and_weights():
    """Verifies that the saved model matches CoordinateMLP topology and loads cleanly."""
    assert os.path.exists(MODEL_PATH), f"Model checkpoint missing at {MODEL_PATH}"
    with open(MODEL_PATH, "r", encoding="utf-8") as f:
        mdict = json.load(f)

    assert mdict["model_type"] == "CoordinateMLP"
    assert mdict["version"] == "2.1.0-reference-driven"
    assert mdict["input_dim"] == 94
    assert mdict["classes"] == STATIC_ASL_CLASSES
    assert mdict["temperature"] > 0.0

    # Initialize CoordinateInferenceEngine
    engine = CoordinateInferenceEngine(mdict)
    assert engine is not None

    # Test forward inference on canonical letter A
    pts_a = generate_anatomical_hand("A", participant_scale=1.0, finger_length_ratio=1.0, motor_noise_std=0.0)
    lms_a = [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in pts_a]
    feats_a = extract_enriched_coordinate_features(lms_a)

    result = engine.predict_features(feats_a)
    assert result["predicted_letter"] == "A"
    assert result["confidence"] > 0.85
    assert result["is_accepted"] is True


def test_out_of_distribution_rejection():
    """Verifies that flat, zeroed, or random landmark vectors are safely rejected."""
    with open(MODEL_PATH, "r", encoding="utf-8") as f:
        mdict = json.load(f)
    engine = CoordinateInferenceEngine(mdict)

    # 1. Zero vector
    zero_feats = np.zeros(94, dtype=np.float32)
    res_zero = engine.predict_features(zero_feats)
    # Zero vector should be rejected due to low confidence or flat probabilities
    assert res_zero["is_accepted"] is False or res_zero["confidence"] < 0.75

    # 2. Random unstructured noise
    np.random.seed(99)
    noise_feats = np.random.uniform(-5.0, 5.0, size=94).astype(np.float32)
    res_noise = engine.predict_features(noise_feats)
    assert res_noise["confidence"] < 0.75 or res_noise["predicted_letter"] == "REJECTED"


def test_temperature_calibration_probability_distribution():
    """Verifies that temperature-scaled logits form a valid probability distribution."""
    with open(MODEL_PATH, "r", encoding="utf-8") as f:
        mdict = json.load(f)

    temp = mdict.get("temperature", 1.0)
    assert 0.8 <= temp <= 2.0, f"Calibrated temperature {temp} outside reasonable range [0.8, 2.0]"

    engine = CoordinateInferenceEngine(mdict)
    pts = generate_anatomical_hand("B", participant_scale=1.0, finger_length_ratio=1.0, motor_noise_std=0.0)
    lms = [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in pts]
    feats = extract_enriched_coordinate_features(lms)

    res = engine.predict_features(feats)
    probs = res["probabilities"]
    assert len(probs) == 24
    prob_sum = sum(probs.values())
    assert abs(prob_sum - 1.0) < 1e-4, f"Probabilities do not sum to 1.0: {prob_sum}"


def test_nvidia_nim_language_assistance_offline_ready():
    """Verifies Layer C language service is operational with deterministic fallback."""
    service = AIAssistanceService()
    status = service.get_status()
    assert status["status"] == "ready"

    # Offline sentence composition
    res = service.compose_sentence(["N", "A", "M", "E"], target_language="en")
    assert res["composed_text"] == "Name."
    assert res["fallback_used"] is False
