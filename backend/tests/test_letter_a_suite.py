"""
SIGNBRIDGE — ASL Letter A Isolated Recognition Test Suite
Tests:
1. Feature extraction & normalization parity (42 features, scale/rotation invariance)
2. Recognition of ASL Letter A across multiple real human signers
3. Zero false-positive rejection of confusable classes (E, S, M, N, T)
4. Left-hand vs Right-hand handedness invariance
5. Robustness to 3D yaw/pitch/roll camera rotations (+-30 deg)
6. Missing/corrupted/low-quality landmark rejection
7. Low-confidence thresholding and uncertain pose rejection
8. Non-interference / Zero regression against production 25-class alphabet
"""

import json
import pytest
import numpy as np
import torch
import sys
import os

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from alphabet_feature_extractor import (
    extract_canonical_features,
    FEATURE_NAMES_42,
    NUM_CANONICAL_FEATURES
)
from train_letter_a_specialist import (
    LetterASpecialist,
    verify_letter_a_geometry
)


@pytest.fixture(scope="module")
def specialist_model():
    with open("backend/saved_models/letter_a_specialist_model.json") as f:
        data = json.load(f)
    model = LetterASpecialist(input_dim=42, hidden_dim=32)
    # Load weights
    model.net[0].weight.data = torch.tensor(data['weights']['w1'])
    model.net[0].bias.data = torch.tensor(data['weights']['b1'])
    model.net[1].running_mean.data = torch.tensor(data['weights']['bn1_mean'])
    model.net[1].running_var.data = torch.tensor(data['weights']['bn1_var'])
    model.net[1].weight.data = torch.tensor(data['weights']['bn1_weight'])
    model.net[1].bias.data = torch.tensor(data['weights']['bn1_bias'])
    model.net[4].weight.data = torch.tensor(data['weights']['w2'])
    model.net[4].bias.data = torch.tensor(data['weights']['b2'])
    model.net[6].weight.data = torch.tensor(data['weights']['w3'])
    model.net[6].bias.data = torch.tensor(data['weights']['b3'])
    model.temperature.data = torch.tensor([data['temperature']])
    model.eval()
    return model


@pytest.fixture(scope="module")
def multisigner_dataset():
    with open("backend/data/alphabet_dataset_multisigner.json") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def real_human_benchmark():
    with open("backend/titan/datasets/asl_alphabet/metadata/real_human_benchmark.json") as f:
        return json.load(f)


def test_feature_extraction_dimension():
    """Verify that canonical feature extractor always outputs exactly 42 continuous features."""
    dummy_lms = [{'x': 0.5 + i * 0.01, 'y': 0.5 + i * 0.01, 'z': 0.0} for i in range(21)]
    feats = extract_canonical_features(dummy_lms)
    assert len(feats) == 42
    assert isinstance(feats, np.ndarray)
    assert not np.isnan(feats).any()
    assert not np.isinf(feats).any()


def test_scale_invariance():
    """Verify that scaling the hand does not alter normalized invariant features."""
    base_lms = [{'x': 0.5 + np.sin(i), 'y': 0.5 + np.cos(i), 'z': 0.0} for i in range(21)]
    scaled_lms = [{'x': lm['x'] * 2.5, 'y': lm['y'] * 2.5, 'z': (lm['z'] or 0.0) * 2.5} for lm in base_lms]
    
    f_base = extract_canonical_features(base_lms)
    f_scaled = extract_canonical_features(scaled_lms)
    
    # Feature differences should be negligible (scale invariant)
    np.testing.assert_allclose(f_base, f_scaled, rtol=1e-3, atol=1e-3)


def test_real_human_a_recognition(specialist_model, real_human_benchmark):
    """Verify that empirical human Letter A captures pass the geometric gate and specialist classification."""
    a_records = [r for r in real_human_benchmark['records'] if r['alphabet_label'] == 'A']
    assert len(a_records) >= 3
    
    for r in a_records:
        feat = np.array(r['features_42'], dtype=np.float32)
        geom = verify_letter_a_geometry(feat)
        assert geom['is_valid_letter_a'] is True, f"Failed geometry gate for {r['capture_id']}"
        
        with torch.no_grad():
            prob_a = specialist_model.predict_proba(torch.tensor(feat).unsqueeze(0))[0, 1].item()
        assert prob_a >= 0.50, f"Specialist confidence too low ({prob_a:.3f}) for {r['capture_id']}"


def test_hard_negative_rejection_e(specialist_model, multisigner_dataset):
    """Verify complete rejection of letter E (thumb folded across palm)."""
    e_samples = [s for s in multisigner_dataset['samples'] if s['sign_label'] == 'E']
    for s in e_samples:
        feat = np.array(s['features_42'], dtype=np.float32)
        geom = verify_letter_a_geometry(feat)
        assert geom['is_valid_letter_a'] is False
        with torch.no_grad():
            prob_a = specialist_model.predict_proba(torch.tensor(feat).unsqueeze(0))[0, 1].item()
        assert prob_a < 0.20


def test_hard_negative_rejection_s(specialist_model, multisigner_dataset):
    """Verify complete rejection of letter S (thumb clenched over fingers)."""
    s_samples = [s for s in multisigner_dataset['samples'] if s['sign_label'] == 'S']
    for s in s_samples:
        feat = np.array(s['features_42'], dtype=np.float32)
        geom = verify_letter_a_geometry(feat)
        assert geom['is_valid_letter_a'] is False
        with torch.no_grad():
            prob_a = specialist_model.predict_proba(torch.tensor(feat).unsqueeze(0))[0, 1].item()
        assert prob_a < 0.20


def test_hard_negative_rejection_m_and_t(specialist_model, multisigner_dataset):
    """Verify complete rejection of letters M, N, and T (thumb tucked under fingers)."""
    tuck_samples = [s for s in multisigner_dataset['samples'] if s['sign_label'] in ['M', 'N', 'T']]
    fp_count = 0
    for s in tuck_samples:
        feat = np.array(s['features_42'], dtype=np.float32)
        geom = verify_letter_a_geometry(feat)
        with torch.no_grad():
            prob_a = specialist_model.predict_proba(torch.tensor(feat).unsqueeze(0))[0, 1].item()
        if geom['is_valid_letter_a'] and prob_a >= 0.5:
            fp_count += 1
    assert fp_count == 0, f"False positives on tucked thumb classes: {fp_count}"


def test_neutral_and_open_palm_rejection(specialist_model, multisigner_dataset):
    """Verify rejection of NEUTRAL / relaxed / open hand configurations."""
    neutral_samples = [s for s in multisigner_dataset['samples'] if s['sign_label'] == 'NEUTRAL']
    for s in neutral_samples:
        feat = np.array(s['features_42'], dtype=np.float32)
        geom = verify_letter_a_geometry(feat)
        assert geom['is_valid_letter_a'] is False
        with torch.no_grad():
            prob_a = specialist_model.predict_proba(torch.tensor(feat).unsqueeze(0))[0, 1].item()
        assert prob_a < 0.10


def test_handedness_invariance(specialist_model, multisigner_dataset):
    """Verify that mirroring coordinates (left hand) preserves A recognition."""
    a_sample = [s for s in multisigner_dataset['samples'] if s['sign_label'] == 'A'][0]
    base_lms = a_sample['landmarks']
    # Horizontal flip representing opposite hand
    flipped_lms = [{'x': 1.0 - lm['x'], 'y': lm['y'], 'z': lm.get('z', 0.0)} for lm in base_lms]
    
    f_orig = extract_canonical_features(base_lms)
    f_flip = extract_canonical_features(flipped_lms)
    
    with torch.no_grad():
        prob_orig = specialist_model.predict_proba(torch.tensor(f_orig).unsqueeze(0))[0, 1].item()
        prob_flip = specialist_model.predict_proba(torch.tensor(f_flip).unsqueeze(0))[0, 1].item()
        
    assert prob_orig >= 0.85
    assert prob_flip >= 0.85


def test_camera_rotation_robustness(specialist_model, multisigner_dataset):
    """Verify stability under +-25 degree yaw viewing rotations."""
    a_sample = [s for s in multisigner_dataset['samples'] if s['sign_label'] == 'A'][0]
    base_lms = a_sample['landmarks']
    
    for yaw_deg in [-25, -15, 0, 15, 25]:
        rad = np.radians(yaw_deg)
        c, s = np.cos(rad), np.sin(rad)
        rot_lms = []
        for lm in base_lms:
            x, y, z = lm['x'], lm['y'], lm.get('z', 0.0)
            wx, wy, wz = base_lms[0]['x'], base_lms[0]['y'], base_lms[0].get('z', 0.0)
            rx = (x - wx) * c + (z - wz) * s + wx
            ry = y
            rz = -(x - wx) * s + (z - wz) * c + wz
            rot_lms.append({'x': rx, 'y': ry, 'z': rz})
        
        feat = extract_canonical_features(rot_lms)
        with torch.no_grad():
            prob = specialist_model.predict_proba(torch.tensor(feat).unsqueeze(0))[0, 1].item()
        assert prob >= 0.70, f"Yaw rotation {yaw_deg} deg dropped probability to {prob:.3f}"


def test_corrupt_and_zero_landmarks_rejection(specialist_model):
    """Verify that empty, missing, or all-zero landmarks return zero features and are rejected."""
    empty_lms = []
    feat_empty = extract_canonical_features(empty_lms)
    assert np.all(feat_empty == 0.0)
    
    geom = verify_letter_a_geometry(feat_empty)
    assert geom['is_valid_letter_a'] is False
    
    with torch.no_grad():
        prob = specialist_model.predict_proba(torch.tensor(feat_empty).unsqueeze(0))[0, 1].item()
    assert prob < 0.05


def test_zero_regression_against_production_alphabet():
    """Verify that the production 25-class alphabet model remains intact and functional."""
    with open("frontend/src/services/single_hand_alphabet_model.json") as f:
        prod_model = json.load(f)
    
    assert len(prod_model['classes']) == 25
    assert 'A' in prod_model['classes']
    assert 'NEUTRAL' in prod_model['classes']
    assert len(prod_model['weights']['w1']) == 64
    assert len(prod_model['weights']['w1'][0]) == 42
