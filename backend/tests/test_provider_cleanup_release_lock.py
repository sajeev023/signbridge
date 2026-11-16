"""
SIGNBRIDGE Phase 8 — Provider Cleanup & Release Lock Regression Tests
Proves:
1. Local recognition works without any API key.
2. NIM is disabled by default.
3. An incompatible remote provider cannot return an accepted prediction.
4. Local confidence calibration remains active.
5. Rejection behavior remains unchanged (tau=0.70).
"""

import os
import sys
import json
import pytest
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(WORKSPACE_ROOT))

SAVED_MODELS_DIR = BASE_DIR / "saved_models"
from fastapi.testclient import TestClient

from backend.coordinate_ai.model_zoo import CoordinateInferenceEngine
from backend.coordinate_ai.feature_pipeline import (
    extract_enriched_coordinate_features,
    extract_normalized_coordinates_63
)
from backend.coordinate_ai.external_provider import (
    LocalCoordinateProvider,
    NvidiaNimProvider,
    ProviderFactory,
    CoordinateInferenceRequest,
    CoordinateInferenceResponse
)
from app.main import app


@pytest.fixture
def engine():
    model_file = SAVED_MODELS_DIR / "coordinate_ai_model.json"
    with open(model_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return CoordinateInferenceEngine(data, confidence_threshold=0.70)


@pytest.fixture
def canonical_letter_a_landmarks():
    # Canonical handshape: thumb extended along outer index, fingers curled
    return [
        {"x": 0.5, "y": 0.8, "z": 0.0},
        {"x": 0.44, "y": 0.72, "z": -0.02},
        {"x": 0.40, "y": 0.62, "z": -0.04},
        {"x": 0.38, "y": 0.52, "z": -0.05},
        {"x": 0.38, "y": 0.42, "z": -0.05},
        {"x": 0.46, "y": 0.55, "z": -0.02},
        {"x": 0.46, "y": 0.65, "z": -0.04},
        {"x": 0.46, "y": 0.72, "z": -0.03},
        {"x": 0.46, "y": 0.75, "z": -0.02},
        {"x": 0.50, "y": 0.54, "z": -0.02},
        {"x": 0.50, "y": 0.65, "z": -0.04},
        {"x": 0.50, "y": 0.72, "z": -0.03},
        {"x": 0.50, "y": 0.76, "z": -0.02},
        {"x": 0.54, "y": 0.55, "z": -0.02},
        {"x": 0.54, "y": 0.65, "z": -0.04},
        {"x": 0.54, "y": 0.72, "z": -0.03},
        {"x": 0.54, "y": 0.75, "z": -0.02},
        {"x": 0.58, "y": 0.57, "z": -0.02},
        {"x": 0.58, "y": 0.65, "z": -0.03},
        {"x": 0.58, "y": 0.71, "z": -0.03},
        {"x": 0.58, "y": 0.74, "z": -0.02},
    ]


class TestProviderCleanupAndReleaseLock:
    def test_1_local_recognition_works_without_api_key(self, canonical_letter_a_landmarks, monkeypatch):
        # Ensure zero external API keys exist
        monkeypatch.delenv("AI_API_KEY", raising=False)
        monkeypatch.setenv("AI_PROVIDER", "local")
        monkeypatch.setenv("AI_INFERENCE_ENABLED", "false")

        client = TestClient(app)
        # Verify status endpoint declares local authoritative engine
        status_res = client.get("/api/coordinate/status")
        assert status_res.status_code == 200
        data = status_res.json()
        assert data["status"] == "ready"
        assert data["authoritative_engine"] == "local_coordinate_mlp_94d"
        assert data["nim_configured"] is False
        assert data["provider"] == "local_coordinate_engine"

        # Verify prediction executes cleanly with high confidence
        pred_res = client.post("/api/coordinate/predict", json={
            "session_id": "clean_local_test",
            "landmarks": canonical_letter_a_landmarks,
            "confidence_threshold": 0.70
        })
        assert pred_res.status_code == 200
        pdata = pred_res.json()
        assert pdata["predicted_letter"] == "A"
        assert pdata["confidence"] >= 0.80
        assert pdata["is_accepted"] is True
        assert pdata["fallback_used"] is False

    def test_2_nim_disabled_by_default(self, engine, monkeypatch):
        # Unset all overrides to test factory default behavior
        monkeypatch.delenv("AI_PROVIDER", raising=False)
        monkeypatch.delenv("AI_INFERENCE_ENABLED", raising=False)
        monkeypatch.delenv("AI_API_KEY", raising=False)

        provider = ProviderFactory.get_provider(engine)
        assert isinstance(provider, LocalCoordinateProvider)
        assert provider.provider_name == "local_coordinate_engine"
        assert provider.health_check() is True

    def test_3_incompatible_remote_provider_blocked_from_prediction(self, engine, monkeypatch, canonical_letter_a_landmarks):
        # Configure an incompatible remote provider (LLaMA text/vision chat endpoint)
        monkeypatch.setenv("AI_PROVIDER", "nvidia_nim")
        monkeypatch.setenv("AI_INFERENCE_ENABLED", "true")
        monkeypatch.setenv("AI_API_KEY", "nvapi-sample-key")
        monkeypatch.setenv("AI_BASE_URL", "https://integrate.api.nvidia.com/v1")
        monkeypatch.setenv("AI_MODEL", "meta/llama-3.2-11b-vision-instruct")

        provider = ProviderFactory.get_provider(engine)
        assert isinstance(provider, NvidiaNimProvider)
        assert provider.is_contract_compatible() is False

        # Build request with 94 enriched features
        raw_63 = extract_normalized_coordinates_63(canonical_letter_a_landmarks).tolist()
        feat_94 = extract_enriched_coordinate_features(canonical_letter_a_landmarks).tolist()
        req = CoordinateInferenceRequest(
            session_id="test_contract_enforcement",
            timestamp_ms=1000.0,
            raw_coordinates_63=raw_63,
            enriched_features_94=feat_94,
            confidence_threshold=0.70
        )

        resp = provider.predict(req)
        # Incompatible remote routing must be intercepted and routed to local fallback
        assert resp.fallback_used is True
        assert "fallback" in resp.provider_name.lower()
        assert "contract mismatch" in resp.error_message.lower() or "contract" in resp.error_message.lower()
        # Must return valid calibrated local prediction (Letter A) rather than synthetic or mock prediction
        assert resp.predicted_letter == "A"
        assert resp.confidence >= 0.80

    def test_4_local_confidence_calibration_active(self, engine, canonical_letter_a_landmarks):
        feat_94 = extract_enriched_coordinate_features(canonical_letter_a_landmarks)
        res = engine.predict_features(feat_94)
        assert res["confidence"] > 0.80
        assert engine.temperature == 1.0
        # Probabilities across all 24 classes must sum to approximately 1.0
        probs = res["probabilities"]
        assert len(probs) == 24
        assert np.isclose(sum(probs.values()), 1.0, atol=1e-3)

    def test_5_rejection_behavior_strictly_enforced(self, engine, monkeypatch):
        # Test degenerate noise input
        np.random.seed(99)
        noise_94 = np.random.uniform(-0.05, 0.05, 94).astype(np.float32)

        res = engine.predict_features(noise_94)
        # Noise must fail the confidence threshold (tau=0.70)
        assert res["confidence"] < 0.70
        assert res["is_accepted"] is False
        assert res["predicted_letter"] == "REJECTED"
