"""
SIGNBRIDGE Biomechanical Feature Extraction & Parity Test Suite
Validates 42-feature contract, 3D translation & scale invariance, and numerical stability.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))
from alphabet_feature_extractor import extract_canonical_features, NUM_CANONICAL_FEATURES, FEATURE_NAMES_42


def generate_mock_hand(scale: float = 1.0, offset_x: float = 0.0, offset_y: float = 0.0, offset_z: float = 0.0):
    """Generates an anatomically structured 21-landmark mock hand."""
    landmarks = []
    # Joint topology with realistic relative positions
    base_coords = [
        [0.50, 0.80, 0.00],  # 0: wrist
        [0.45, 0.72, 0.01],  # 1: thumb cmc
        [0.40, 0.65, 0.02],  # 2: thumb mcp
        [0.37, 0.58, 0.02],  # 3: thumb ip
        [0.35, 0.52, 0.03],  # 4: thumb tip
        [0.46, 0.55, 0.01],  # 5: index mcp
        [0.45, 0.45, 0.01],  # 6: index pip
        [0.44, 0.38, 0.01],  # 7: index dip
        [0.44, 0.32, 0.02],  # 8: index tip
        [0.50, 0.54, 0.00],  # 9: mid mcp
        [0.50, 0.43, 0.00],  # 10: mid pip
        [0.50, 0.35, 0.00],  # 11: mid dip
        [0.50, 0.28, 0.01],  # 12: mid tip
        [0.54, 0.56, -0.01], # 13: ring mcp
        [0.55, 0.46, -0.01], # 14: ring pip
        [0.55, 0.39, -0.01], # 15: ring dip
        [0.55, 0.33, -0.01], # 16: ring tip
        [0.58, 0.60, -0.02], # 17: pinky mcp
        [0.60, 0.52, -0.02], # 18: pinky pip
        [0.61, 0.46, -0.02], # 19: pinky dip
        [0.62, 0.40, -0.02], # 20: pinky tip
    ]
    for p in base_coords:
        landmarks.append({
            "x": float((p[0] - 0.50) * scale + 0.50 + offset_x),
            "y": float((p[1] - 0.80) * scale + 0.80 + offset_y),
            "z": float(p[2] * scale + offset_z)
        })
    return landmarks


def test_feature_dimension_and_naming():
    """Verify exactly 42 features are produced and match the feature names list."""
    assert NUM_CANONICAL_FEATURES == 42
    assert len(FEATURE_NAMES_42) == 42

    hand = generate_mock_hand()
    feats = extract_canonical_features(hand)
    assert isinstance(feats, np.ndarray)
    assert feats.shape == (42,)
    assert feats.dtype == np.float32


def test_numerical_stability():
    """Ensure no NaN, Infinite, or corrupt values even with degenerate inputs."""
    # Degenerate zero hand
    zero_hand = [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(21)]
    zero_feats = extract_canonical_features(zero_hand)
    assert not np.isnan(zero_feats).any(), "Found NaN in zero-hand features"
    assert not np.isinf(zero_feats).any(), "Found Inf in zero-hand features"

    # Incomplete hand
    incomplete = zero_hand[:15]
    inc_feats = extract_canonical_features(incomplete)
    assert np.all(inc_feats == 0.0)

    # Empty hand
    empty_feats = extract_canonical_features([])
    assert np.all(empty_feats == 0.0)


def test_translation_invariance():
    """Verify that shifting the hand anywhere in 3D camera coordinates does not alter features."""
    hand1 = generate_mock_hand(scale=1.0, offset_x=0.0, offset_y=0.0, offset_z=0.0)
    hand2 = generate_mock_hand(scale=1.0, offset_x=0.25, offset_y=-0.35, offset_z=0.15)

    feats1 = extract_canonical_features(hand1)
    feats2 = extract_canonical_features(hand2)

    diff = np.max(np.abs(feats1 - feats2))
    assert diff < 1e-4, f"Translation invariance violated, max diff: {diff}"


def test_scale_invariance():
    """Verify that changing hand distance / scale preserves normalized biomechanical metrics."""
    hand_small = generate_mock_hand(scale=0.6)
    hand_large = generate_mock_hand(scale=1.5)

    feats_small = extract_canonical_features(hand_small)
    feats_large = extract_canonical_features(hand_large)

    diff = np.max(np.abs(feats_small - feats_large))
    assert diff < 1e-3, f"Scale invariance violated, max diff: {diff}"

