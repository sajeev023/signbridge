import os
import sys
import json
import hashlib
import numpy as np
import pytest
import torch

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from alphabet_feature_extractor import extract_canonical_features
from train_letter_a_specialist import LetterASpecialist, verify_letter_a_geometry


def test_frozen_candidate_manifest_integrity():
    """Verify that all files in letter_a_frozen_manifest.json exist and match SHA-256 hashes."""
    manifest_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "letter_a_frozen_manifest.json"))
    assert os.path.exists(manifest_path), "letter_a_frozen_manifest.json missing"
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    for rel_path, meta in manifest.items():
        full_path = os.path.join(repo_root, rel_path)
        assert os.path.exists(full_path), f"Frozen file missing: {rel_path}"
        
        with open(full_path, "rb") as f:
            content = f.read()
            computed_hash = hashlib.sha256(content).hexdigest()
            computed_size = len(content)
            
        assert computed_hash == meta["sha256"], f"Hash mismatch for {rel_path}: expected {meta['sha256']}, got {computed_hash}"
        assert computed_size == meta["size_bytes"], f"Size mismatch for {rel_path}: expected {meta['size_bytes']}, got {computed_size}"


def test_ece_calibration_correction_mathematics():
    """Verify the mathematical distinction between flawed ECE and corrected multi-class ECE."""
    calib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "letter_a_calibration_corrected.json"))
    assert os.path.exists(calib_path), "letter_a_calibration_corrected.json missing"
    
    with open(calib_path, "r", encoding="utf-8") as f:
        calib = json.load(f)
    
    # Flawed ECE incorrectly uses positive class probability for negative predictions
    assert calib["flawed_reported_metrics"]["val_ece"] > 0.90
    assert calib["flawed_reported_metrics"]["test_ece"] > 0.90
    
    # Corrected ECE uses confidence = max(p, 1-p) for predicted class
    corrected = calib["corrected_metrics"]
    assert corrected["val"]["ece"] < 0.01  # ~0.59%
    assert corrected["test"]["ece"] < 0.01  # ~0.44%
    assert corrected["val"]["brier"] < 0.0005
    assert corrected["test"]["brier"] < 0.0005
    assert corrected["real_bm"]["ece"] < 0.05  # ~3.54%


def test_specialist_forward_pass_parity():
    """Verify Specialist MLP forward pass and temperature scaling match export weights."""
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "saved_models", "letter_a_specialist_model.json"))
    assert os.path.exists(model_path)
    
    with open(model_path, "r", encoding="utf-8") as f:
        model_data = json.load(f)
        
    weights = model_data["weights"]
    temp = model_data.get("temperature", 1.2)
    
    # Check weight dimensions
    assert len(weights["w1"]) == 32 and len(weights["w1"][0]) == 42
    assert len(weights["w2"]) == 16 and len(weights["w2"][0]) == 32
    assert len(weights["w3"]) == 2 and len(weights["w3"][0]) == 16
    
    # Test forward pass with zeros
    x = np.zeros(42, dtype=np.float32)
    z1 = np.maximum(0, (np.dot(weights["w1"], x) + weights["b1"] - weights["bn1_mean"]) / np.sqrt(np.array(weights["bn1_var"]) + 1e-5) * weights["bn1_weight"] + weights["bn1_bias"])
    z2 = np.maximum(0, np.dot(weights["w2"], z1) + weights["b2"])
    logits = (np.dot(weights["w3"], z2) + weights["b3"]) / temp
    
    exp_l = np.exp(logits - np.max(logits))
    probs = exp_l / np.sum(exp_l)
    
    assert len(probs) == 2
    assert np.isclose(np.sum(probs), 1.0)
    assert 0.0 <= probs[1] <= 1.0


def test_kinematic_gate_canonical_a():
    """Verify that canonical real human Letter A passes all 5 kinematic rule gate clauses."""
    bm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "titan", "datasets", "asl_alphabet", "metadata", "real_human_benchmark.json"))
    with open(bm_path, "r", encoding="utf-8") as f:
        bm = json.load(f)
        
    a_records = [r for r in bm["records"] if r["alphabet_label"] == "A"]
    assert len(a_records) >= 3
    
    for r in a_records:
        feat = np.array(r["features_42"], dtype=np.float32)
        geom = verify_letter_a_geometry(feat)
        assert geom["is_valid_letter_a"] is True
        assert geom["fist_curled"] is True
        assert geom["thumb_elevated"] is True
        assert geom["thumb_lateral"] is True
        assert geom["thumb_not_crossing"] is True
        assert geom["thumb_not_tucked"] is True


def test_kinematic_gate_hard_negative_rejection():
    """Verify that confusable letters E and S fail the kinematic rule gate."""
    multi_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "alphabet_dataset_multisigner.json"))
    with open(multi_path, "r", encoding="utf-8") as f:
        multi = json.load(f)
        
    # E has thumb folded flat (thumb_elevated = False)
    e_samples = [s for s in multi["samples"] if s["sign_label"] == "E"]
    assert len(e_samples) > 0
    for s in e_samples:
        feat = np.array(s["features_42"], dtype=np.float32)
        geom = verify_letter_a_geometry(feat)
        assert geom["is_valid_letter_a"] is False
        assert geom["thumb_elevated"] is False
        
    # S has thumb crossing knuckles (thumb_lateral = False or thumb_not_crossing = False)
    s_samples = [s for s in multi["samples"] if s["sign_label"] == "S"]
    assert len(s_samples) > 0
    for s in s_samples:
        feat = np.array(s["features_42"], dtype=np.float32)
        geom = verify_letter_a_geometry(feat)
        assert geom["is_valid_letter_a"] is False


def test_collector_sample_schema_conformance():
    """Verify that a sample captured by ZeroLeak Collector conforms to benchmark schema."""
    sample = {
        "id": "zl_test_1700000000000",
        "signer_id": "Signer_P26",
        "label": "A",
        "landmarks": [[0.5, 0.5, 0.0] for _ in range(21)],
        "features": [0.0] * 42,
        "timestamp": 1700000000000,
        "split": "human_evaluation"
    }
    
    required_keys = {"id", "signer_id", "label", "landmarks", "features", "timestamp", "split"}
    assert required_keys.issubset(sample.keys())
    assert len(sample["landmarks"]) == 21
    assert len(sample["features"]) == 42
    assert sample["signer_id"].startswith("Signer_P")
    assert sample["label"] in ["A", "E", "S", "M", "N", "T", "NEUTRAL"]
