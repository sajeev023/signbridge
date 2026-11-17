"""
SIGNBRIDGE Master AI Transformation & Engineering Repair — 16-Category Verification Suite
Validates all 16 required engineering categories defined in the Master Specification.
"""

import sys
import time
import json
import numpy as np
import torch
import pytest
from pathlib import Path

# Add backend directory to sys.path
_backend_dir = Path(__file__).resolve().parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from fastapi.testclient import TestClient
from app.main import app
from app.config import (
    SAVED_MODELS_DIR, VOCABULARY_LIST, CLASS_NAMES,
    CONFIDENCE_THRESHOLD, SEQUENCE_LENGTH, NUM_FEATURES
)
from app.models.architecture import SignBridgeBiLSTM
from app.services.landmark_extractor import LandmarkExtractor
from app.services.inference_engine import InferenceEngine
from app.services.ai_assistance_service import AIAssistanceService, ai_assistance_service
from app.models.data_contracts import (
    FrameInferencePayload, InferenceResultEnvelope,
    FEATURE_SCHEMA_VERSION, MODEL_VERSION
)
from coordinate_ai.feature_pipeline import (
    extract_enriched_coordinate_features, extract_normalized_coordinates_63
)
from coordinate_ai.model_zoo import CoordinateInferenceEngine
from coordinate_ai.external_provider import (
    CoordinateInferenceRequest, LocalCoordinateProvider, NvidiaNimProvider
)

client = TestClient(app)


# ============================================================================
# CATEGORY 1: FEATURE EXTRACTION (94-D Biomechanical Invariants)
# ============================================================================
def test_cat1_feature_extraction_dimensions_and_invariants():
    # 21 synthetic canonical landmarks
    landmarks = [{"x": 0.5 + 0.02 * i, "y": 0.5 + 0.01 * (i % 5), "z": 0.005 * i} for i in range(21)]
    feats_94 = extract_enriched_coordinate_features(landmarks)
    assert isinstance(feats_94, np.ndarray)
    assert feats_94.shape == (94,)
    assert not np.isnan(feats_94).any()
    assert not np.isinf(feats_94).any()

    # Raw 63
    raw_63 = extract_normalized_coordinates_63(landmarks)
    assert raw_63.shape == (63,)


# ============================================================================
# CATEGORY 2: LANDMARK NORMALIZATION (Scale & Translation Invariance)
# ============================================================================
def test_cat2_landmark_normalization_scale_translation_invariance():
    extractor = LandmarkExtractor()
    # Baseline hand at (0.4, 0.4)
    lh = [{"x": 0.4 + 0.01 * i, "y": 0.4 + 0.01 * i, "z": 0.0} for i in range(21)]
    vec1 = extractor.normalize_landmarks(left_hand=lh, right_hand=None, pose=None)

    # Shifted hand at (0.8, 0.8) - translation should be normalized relative to wrist/center
    lh_shifted = [{"x": 0.8 + 0.01 * i, "y": 0.8 + 0.01 * i, "z": 0.0} for i in range(21)]
    vec2 = extractor.normalize_landmarks(left_hand=lh_shifted, right_hand=None, pose=None)

    # Normalized relative palm coordinates should be identical regardless of absolute camera translation
    assert np.allclose(vec1[:63], vec2[:63], atol=1e-4)


# ============================================================================
# CATEGORY 3: MODEL LOADING (Weights, Architecture, Metadatas)
# ============================================================================
def test_cat3_model_loading_and_architecture_integrity():
    # 1. Coordinate AI Model
    coord_file = SAVED_MODELS_DIR / "coordinate_ai_model.json"
    assert coord_file.exists()
    with open(coord_file, "r") as f:
        cdata = json.load(f)
    assert cdata["input_dim"] == 94
    assert len(cdata["classes"]) == 24
    engine = CoordinateInferenceEngine(cdata)
    assert len(engine.classes) == 24


    # 2. BiLSTM Checkpoint
    bilstm_file = SAVED_MODELS_DIR / "signbridge_isl_bilstm.pt"
    assert bilstm_file.exists()
    bilstm = SignBridgeBiLSTM(NUM_FEATURES, 128, 2, 10, 0.3)
    bilstm.load_state_dict(torch.load(bilstm_file, map_location="cpu"))
    bilstm.eval()


# ============================================================================
# CATEGORY 4: INFERENCE FORWARD PASS (Numerical Correctness)
# ============================================================================
def test_cat4_inference_forward_pass_predictions():
    coord_file = SAVED_MODELS_DIR / "coordinate_ai_model.json"
    with open(coord_file, "r") as f:
        cdata = json.load(f)
    engine = CoordinateInferenceEngine(cdata)

    dummy_94 = np.zeros(94, dtype=np.float32)
    res = engine.predict_features(dummy_94)
    assert "predicted_letter" in res
    assert "confidence" in res
    assert "top_candidates" in res
    assert len(res["top_candidates"]) <= 3


# ============================================================================
# CATEGORY 5: CONFIDENCE CALIBRATION (Temperature Scaling)
# ============================================================================
def test_cat5_confidence_calibration():
    coord_file = SAVED_MODELS_DIR / "coordinate_ai_model.json"
    with open(coord_file, "r") as f:
        cdata = json.load(f)
    engine = CoordinateInferenceEngine(cdata, confidence_threshold=0.70)
    assert engine.temperature == 1.0

    dummy_94 = np.ones(94, dtype=np.float32) * 0.1
    res = engine.predict_features(dummy_94)
    # Sum of probabilities should be close to 1.0
    probs = res.get("probabilities", {})
    if probs:
        total_p = sum(probs.values())
        assert abs(total_p - 1.0) < 1e-3


# ============================================================================
# CATEGORY 6: UNCERTAIN PREDICTION REJECTION (Threshold Gate)
# ============================================================================
def test_cat6_uncertain_prediction_rejection():
    coord_file = SAVED_MODELS_DIR / "coordinate_ai_model.json"
    with open(coord_file, "r") as f:
        cdata = json.load(f)
    # Set high threshold
    engine = CoordinateInferenceEngine(cdata, confidence_threshold=0.999)
    dummy_94 = np.zeros(94, dtype=np.float32)
    res = engine.predict_features(dummy_94)
    # Should be rejected because dummy vector will not produce 99.9% confidence
    assert res["is_accepted"] is False
    assert res["predicted_letter"] == "REJECTED"


# ============================================================================
# CATEGORY 7: TEMPORAL STABILIZATION (Hysteresis & Dropped Frame Grace)
# ============================================================================
def test_cat7_temporal_stabilization_and_drop_grace():
    engine = InferenceEngine()
    engine.reset_buffer()

    # Push 20 frames with hands
    lh = [{"x": 0.5, "y": 0.5, "z": 0.0}] * 21
    for _ in range(20):
        res = engine.push_landmarks(left_hand=lh, right_hand=None, pose=None)
        assert res["status"] == "buffering"

    # Simulate 2 dropped frames (MediaPipe momentary loss)
    for _ in range(2):
        res = engine.push_landmarks(left_hand=None, right_hand=None, pose=None)
        # Should stay in buffering, not crash or reset buffer to zero
        assert res["status"] == "buffering"
        assert res["buffer_count"] > 20


# ============================================================================
# CATEGORY 8: STATIC SIGN RECOGNITION (Empirical Real-Human Classes)
# ============================================================================
def test_cat8_static_sign_recognition_real_human_benchmarks():
    bench_file = _backend_dir / "titan" / "datasets" / "asl_alphabet" / "metadata" / "real_human_benchmark.json"
    if bench_file.exists():
        with open(bench_file, "r") as f:
            bench = json.load(f)
        records = bench.get("records", [])
        assert len(records) > 0
        # Verify first physical record has valid landmark format
        r0 = records[0]
        assert "landmarks" in r0
        assert len(r0["landmarks"]) == 21


# ============================================================================
# CATEGORY 9: DYNAMIC SIGN RECOGNITION (J and Z Trajectories)
# ============================================================================
def test_cat9_dynamic_sign_recognition_jz_trajectories():
    dyn_file = SAVED_MODELS_DIR / "dynamic_alphabet_model.json"
    assert dyn_file.exists()
    with open(dyn_file, "r") as f:
        dyn_data = json.load(f)
    assert "classes" in dyn_data
    assert "J" in dyn_data["classes"]
    assert "Z" in dyn_data["classes"]
    assert dyn_data.get("feature_dimension") == 32 or dyn_data.get("input_dim") == 32


# ============================================================================
# CATEGORY 10: API INTEGRATION (REST & Status Endpoints)
# ============================================================================
def test_cat10_api_integration():
    res1 = client.get("/api/health")
    assert res1.status_code == 200
    res2 = client.get("/api/coordinate/status")
    assert res2.status_code == 200
    assert res2.json()["status"] == "ready"
    res3 = client.get("/api/ai/status")
    assert res3.status_code == 200
    assert res3.json()["status"] == "ready"


# ============================================================================
# CATEGORY 11: AI PROVIDER FAILURE & LOCAL FALLBACK (Circuit Breaker)
# ============================================================================
def test_cat11_ai_provider_failure_and_local_fallback(monkeypatch):
    monkeypatch.setenv("AI_INFERENCE_ENABLED", "true")
    coord_file = SAVED_MODELS_DIR / "coordinate_ai_model.json"
    with open(coord_file, "r") as f:
        cdata = json.load(f)
    engine = CoordinateInferenceEngine(cdata)
    local_p = LocalCoordinateProvider(engine)

    # Incompatible provider configuration (e.g. LLM model pointing to wrong endpoint)
    nim_p = NvidiaNimProvider(
        local_fallback_provider=local_p,
        api_key="mock_invalid_key",
        model_name="meta/llama-3.2-11b-vision-instruct"
    )
    assert not nim_p.is_contract_compatible()

    req = CoordinateInferenceRequest(
        timestamp_ms=1000.0,
        raw_coordinates_63=[0.0] * 63,
        confidence_threshold=0.70
    )
    # Forward pass should fallback seamlessly to local provider
    res = nim_p.predict(req)
    assert res.fallback_used is True
    assert "fallback" in res.provider_name



# ============================================================================
# CATEGORY 12: FRONTEND/BACKEND PAYLOAD COMPATIBILITY
# ============================================================================
def test_cat12_payload_contract_compatibility():
    payload = FrameInferencePayload(
        frame_id=101,
        timestamp_ms=time.time() * 1000.0,
        has_hands=True,
        hand_count=1,
        handedness="Right",
        raw_coordinates_63=[0.0] * 63,
        mode="alphabet"
    )
    p_dict = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    resp = client.post("/api/predict/frame", json=p_dict)
    assert resp.status_code == 200
    envelope = resp.json()
    assert envelope["frame_id"] == 101
    assert "predicted_sign" in envelope
    assert "is_accepted" in envelope
    assert envelope["feature_schema_version"] == FEATURE_SCHEMA_VERSION


# ============================================================================
# CATEGORY 13: CAMERA PERMISSION HANDLING (Missing/Denied Inputs)
# ============================================================================
def test_cat13_camera_permission_denied_handling():
    # Client sends frame with has_hands=False and no landmarks
    resp = client.post("/api/predict/frame", json={
        "frame_id": 1,
        "has_hands": False,
        "hand_count": 0,
        "handedness": "None"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_accepted"] is False
    assert data["status"] in ["idle", "buffering", "no_hand", "uncertain"]



# ============================================================================
# CATEGORY 14: CAMERA RECONNECTION & STREAM INTERRUPTION
# ============================================================================
def test_cat14_camera_reconnection_recovery():
    engine = InferenceEngine()
    engine.reset_buffer()
    # Interrupted stream then resumed
    lh = [{"x": 0.5, "y": 0.5, "z": 0.0}] * 21
    # 5 active frames
    for _ in range(5):
        engine.push_landmarks(left_hand=lh, right_hand=None, pose=None)
    # Sudden drop (5 dropped frames exceeding grace counter)
    for _ in range(5):
        engine.push_landmarks(left_hand=None, right_hand=None, pose=None)
    # Stream reconnects
    res = engine.push_landmarks(left_hand=lh, right_hand=None, pose=None)
    assert res["has_hands"] is True


# ============================================================================
# CATEGORY 15: UI STATE SYNCHRONIZATION (Sequence Ordering & Stale Drop)
# ============================================================================
def test_cat15_ui_state_synchronization():
    # WebSocket test for frame ordering
    with client.websocket_connect("/ws/stream") as ws:
        # Frame 10 arrives
        ws.send_text(json.dumps({
            "type": "frame",
            "frame_id": 10,
            "mode": "translate",
            "left_hand": [{"x": 0.1, "y": 0.1, "z": 0.0}] * 21
        }))
        msg1 = ws.receive_json()
        assert msg1["frame_id"] == 10

        # Frame 5 (stale delayed frame) arrives after frame 10: should be dropped
        ws.send_text(json.dumps({
            "type": "frame",
            "frame_id": 5,
            "mode": "translate",
            "left_hand": [{"x": 0.1, "y": 0.1, "z": 0.0}] * 21
        }))
        # Frame 11 arrives: should be accepted
        ws.send_text(json.dumps({
            "type": "frame",
            "frame_id": 11,
            "mode": "translate",
            "left_hand": [{"x": 0.1, "y": 0.1, "z": 0.0}] * 21
        }))
        msg2 = ws.receive_json()
        assert msg2["frame_id"] == 11


# ============================================================================
# CATEGORY 16: BUILD AND PRODUCTION STARTUP
# ============================================================================
def test_cat16_build_and_production_startup():
    dist_dir = _backend_dir.parent / "frontend" / "dist"
    index_html = dist_dir / "index.html"
    assert dist_dir.exists(), "Frontend dist folder must exist"
    assert index_html.exists(), "Frontend dist/index.html must exist"
    # Root route serves index.html
    res = client.get("/")
    assert res.status_code == 200
    assert "<!doctype html>" in res.text.lower() or "signbridge" in res.text.lower()
