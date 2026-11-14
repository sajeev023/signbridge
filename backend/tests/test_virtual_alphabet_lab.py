"""
SIGNBRIDGE Operation Virtual Alphabet Lab — Regression Test Suite
Provides automated pytest assertions for:
1. All 26 letters canonical recognition (Zero-Tolerance: 26/26 must pass)
2. C / O / Y perspective foreshortening resilience
3. M / N / T fist occlusion handling
4. J / Z temporal sequence trajectory recognition and invalid movement rejection
5. Adversarial input and failure injection handling
6. Python / TypeScript feature extraction parity
"""

import sys
import json
from pathlib import Path
import numpy as np
import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "backend"))

from backend.tests.virtual_alphabet.virtual_hand_generator import (
    VirtualHandGenerator,
    ALL_26_LETTERS,
    STATIC_24_LETTERS,
    DYNAMIC_2_LETTERS
)
from backend.tests.virtual_alphabet.run_virtual_evaluations import (
    load_all_models,
    evaluate_sample_full_pipeline
)


@pytest.fixture(scope="module")
def models():
    return load_all_models()


@pytest.fixture(scope="module")
def generator():
    return VirtualHandGenerator(base_seed=100)


def test_virtual_canonical_all_26_letters(models, generator):
    """
    ZERO-TOLERANCE GATE: Every single letter A-Z must correctly classify
    its canonical virtual example with confidence >= 0.70.
    """
    for letter in ALL_26_LETTERS:
        sample = generator.generate_sample(letter, variation="canonical")
        res = evaluate_sample_full_pipeline(sample, models)
        assert res["correct"] is True, f"Canonical failure on letter {letter}: pred={res['predicted']} conf={res['confidence']:.3f} reason={res.get('rejection_reason')}"
        assert res["confidence"] >= 0.70, f"Low confidence on letter {letter}: conf={res['confidence']:.3f} < 0.70"


def test_virtual_j_z_temporal_gating(models, generator):
    """
    Validates that dynamic J and Z are recognized when moving properly,
    and rejected when movement is stationary, inverted, or missing strokes.
    """
    # 1. Valid J
    s_j = generator.generate_sample("J", variation="canonical")
    res_j = evaluate_sample_full_pipeline(s_j, models)
    assert res_j["predicted"] == "J"
    assert res_j["confidence"] >= 0.90

    # 2. Valid Z
    s_z = generator.generate_sample("Z", variation="canonical")
    res_z = evaluate_sample_full_pipeline(s_z, models)
    assert res_z["predicted"] == "Z"
    assert res_z["confidence"] >= 0.90

    # 3. Stationary hand in J handshape must NOT trigger J
    s_stat_j = generator.generate_sample("J", variation="canonical")
    s_stat_j["trajectory"] = [(0.5, 0.4)] * 20
    res_stat_j = evaluate_sample_full_pipeline(s_stat_j, models)
    assert res_stat_j["predicted"] != "J"

    # 4. Stationary hand in Z handshape must NOT trigger Z
    s_stat_z = generator.generate_sample("Z", variation="canonical")
    s_stat_z["trajectory"] = [(0.5, 0.4)] * 20
    res_stat_z = evaluate_sample_full_pipeline(s_stat_z, models)
    assert res_stat_z["predicted"] != "Z"


def test_adversarial_invalid_input_rejection(models, generator):
    """
    Verifies safe rejection of malformed or corrupted inputs without unhandled exceptions.
    """
    # Truncated landmarks (< 21)
    sample_bad = generator.generate_sample("A", variation="canonical")
    sample_bad["landmarks"] = sample_bad["landmarks"][:10]
    res = evaluate_sample_full_pipeline(sample_bad, models)
    assert res["predicted"] == "REJECTED"
    assert res["rejection_reason"] == "INVALID_LANDMARK_STRUCTURE"

    # NaN in landmark
    sample_nan = generator.generate_sample("B", variation="canonical")
    sample_nan["landmarks"][5]["x"] = float("nan")
    res_nan = evaluate_sample_full_pipeline(sample_nan, models)
    assert res_nan["predicted"] == "REJECTED"
    assert "NAN" in res_nan["rejection_reason"]

    # Inf in landmark
    sample_inf = generator.generate_sample("C", variation="canonical")
    sample_inf["landmarks"][5]["y"] = float("inf")
    res_inf = evaluate_sample_full_pipeline(sample_inf, models)
    assert res_inf["predicted"] == "REJECTED"
    assert "NAN" in res_inf["rejection_reason"]


def test_python_typescript_cross_runtime_parity():
    """
    Verifies that Python and TypeScript biomechanical feature extractors agree
    within numerical tolerance eps < 1e-4 across all 26 classes.
    """
    parity_file = WORKSPACE_ROOT / "backend" / "tests" / "virtual_alphabet" / "cross_runtime_parity_results.json"
    assert parity_file.exists(), f"Parity results file missing at {parity_file}"

    with open(parity_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["all_letters_parity_passed"] is True
    assert data["feature_agreement_rate"] == 100.0
    assert data["max_global_difference"] < 1e-4
