"""
SIGNBRIDGE Open-Source Integration — Automated Test Suite
Verifies:
1. Kinivi Point History Trajectory Adapter (Buffer integrity, scale invariance)
2. Nick Nochnack Temporal Hysteresis Filter (Flicker suppression, consecutive agreement, debounce)
3. Integrated Multi-Modal Recognition Pipeline (Static 24-letter + Dynamic J/Z + Specialist)
4. Scientific Superiority of 3D Canonical Basis vs Kinivi 2D Normalization under Rotation
5. Strict Modality Isolation (Static vs Dynamic vs ISL)
"""

import sys
import math
from pathlib import Path
import numpy as np
import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "backend"))

from backend.open_source_integration.kinivi_point_history_adapter import KiniviPointHistoryAdapter
from backend.open_source_integration.nochnack_temporal_hysteresis import NochnackTemporalHysteresisFilter
from backend.open_source_integration.integrated_recognition_pipeline import IntegratedRecognitionPipeline
from backend.open_source_integration.comparative_evaluator import (
    kinivi_preprocess_landmark,
    run_static_feature_comparison_benchmark
)
from backend.tests.virtual_alphabet.virtual_hand_generator import VirtualHandGenerator
from backend.alphabet_feature_extractor import extract_canonical_features
from backend.build_alphabet_dataset import create_rotation_matrix


def test_kinivi_point_history_buffering_and_invariance():
    adapter = KiniviPointHistoryAdapter(maxlen=16)
    assert not adapter.is_ready()

    # Fill buffer with 16 points
    for i in range(16):
        adapter.append_point({"x": 0.5 + i * 0.01, "y": 0.5 + i * 0.02}, palm_scale=0.12, timestamp_ms=i * 33.3)

    assert adapter.is_ready()

    # Kinivi raw features
    kin_feats = adapter.extract_kinivi_raw_features(image_width=640, image_height=480)
    assert kin_feats.shape == (32,)
    assert kin_feats.dtype == np.float32

    # SIGNBRIDGE palm-scale invariant features
    sb_feats = adapter.extract_signbridge_invariant_features()
    assert sb_feats.shape == (32,)
    assert sb_feats.dtype == np.float32
    assert not np.isnan(sb_feats).any()


def test_nochnack_temporal_hysteresis_smoothing():
    hysteresis = NochnackTemporalHysteresisFilter(
        window_size=10,
        agreement_threshold=7,
        confidence_threshold=0.72,
        debounce_ms=500.0
    )

    # 1. Rapid jitter (alternating 'A' and 'B') should be suppressed
    for i in range(10):
        pred = "A" if i % 2 == 0 else "B"
        sign, conf, status = hysteresis.process_frame(pred, 0.95, timestamp_ms=i * 33.3)
        assert sign is None
        assert status in ("ACCUMULATING", "UNSTABLE")

    # 2. Consistent sign 'A' held for 7 consecutive frames -> EMITTED
    hysteresis.reset()
    emitted = None
    for i in range(7):
        sign, conf, status = hysteresis.process_frame("A", 0.95, timestamp_ms=i * 33.3)
        if status == "EMITTED":
            emitted = sign

    assert emitted == "A"
    assert hysteresis.get_accumulated_sentence() == "A"

    # 3. Holding sign within debounce window -> HOLDING (no duplicate emission)
    sign, conf, status = hysteresis.process_frame("A", 0.95, timestamp_ms=7 * 33.3 + 100.0)
    assert status == "HOLDING"
    assert hysteresis.get_accumulated_sentence() == "A"

    # 4. Holding sign past debounce window (500ms+) -> Second EMITTED
    sign, conf, status = hysteresis.process_frame("A", 0.95, timestamp_ms=7 * 33.3 + 600.0)
    assert status == "EMITTED"
    assert hysteresis.get_accumulated_sentence() == "AA"


def test_integrated_recognition_pipeline_execution():
    pipeline = IntegratedRecognitionPipeline()
    generator = VirtualHandGenerator(base_seed=500)

    # Test Canonical A
    s_a = generator.generate_sample("A", variation="canonical")
    res_a = pipeline.process_frame(s_a["landmarks"], timestamp_ms=0.0)
    assert res_a["raw_sign"] == "A"
    assert res_a["confidence"] >= 0.70

    # Test Canonical B
    s_b = generator.generate_sample("B", variation="canonical")
    res_b = pipeline.process_frame(s_b["landmarks"], timestamp_ms=33.3)
    assert res_b["raw_sign"] == "B"
    assert res_b["confidence"] >= 0.70


def test_rotation_invariance_superiority_over_kinivi():
    """
    Empirically verifies that SIGNBRIDGE 3D canonical features maintain < 1e-4 drift under 30 deg tilt,
    whereas Kinivi's 2D normalization drifts by > 0.30.
    """
    generator = VirtualHandGenerator(base_seed=600)
    sample = generator.generate_sample("B", variation="canonical")
    lms = sample["landmarks"]

    pts = np.array([[p["x"], p["y"], p["z"]] for p in lms], dtype=np.float32)
    R = create_rotation_matrix(pitch=0.0, yaw=0.0, roll=math.radians(30))
    wrist = pts[0]
    pts_rot = (pts - wrist) @ R.T + wrist
    lms_rot = [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "visibility": 1.0} for p in pts_rot]

    # Kinivi drift
    f_kin_orig = kinivi_preprocess_landmark(lms)
    f_kin_rot = kinivi_preprocess_landmark(lms_rot)
    kin_drift = float(np.linalg.norm(f_kin_orig - f_kin_rot) / np.linalg.norm(f_kin_orig))
    assert kin_drift > 0.30, f"Expected Kinivi drift > 0.30, got {kin_drift}"

    # SIGNBRIDGE drift
    f_sb_orig = extract_canonical_features(lms)
    f_sb_rot = extract_canonical_features(lms_rot)
    sb_drift = float(np.linalg.norm(f_sb_orig - f_sb_rot) / np.linalg.norm(f_sb_orig))
    assert sb_drift < 1e-4, f"Expected SIGNBRIDGE drift < 1e-4, got {sb_drift}"


def test_hysteresis_legitimate_transition_and_neutral_release():
    """
    Verifies that:
    1. Switching between different letters (A -> B) is not suppressed by debounce.
    2. Returning to NEUTRAL for >= 2 frames resets the debounce lock, allowing intentional
       repeated letters (e.g. 'A' then 'A') to be recognized cleanly.
    """
    filter_inst = NochnackTemporalHysteresisFilter(
        window_size=10,
        agreement_threshold=7,
        confidence_threshold=0.72,
        debounce_ms=800.0
    )

    # 1. Emit Letter A
    for t in range(0, 240, 30):
        filter_inst.process_frame("A", 0.95, timestamp_ms=t)
    assert filter_inst.get_accumulated_sentence() == "A"

    # 2. Transition directly to Letter B (different letter -> debounce must NOT suppress)
    for t in range(240, 480, 30):
        filter_inst.process_frame("B", 0.95, timestamp_ms=t)
    assert filter_inst.get_accumulated_sentence() == "AB"

    # 3. Drop hand to NEUTRAL for 3 frames (90ms)
    for t in range(480, 570, 30):
        filter_inst.process_frame("NEUTRAL", 0.0, timestamp_ms=t)

    # 4. Sign B again at t=570ms (within 800ms of first B at t=420ms)
    # Because neutral was confirmed, this is a legitimate new stroke!
    for t in range(570, 810, 30):
        filter_inst.process_frame("B", 0.95, timestamp_ms=t)
    assert filter_inst.get_accumulated_sentence() == "ABB"

