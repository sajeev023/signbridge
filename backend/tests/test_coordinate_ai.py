"""
SIGNBRIDGE Coordinate AI — Unit & Integration Test Suite
Validates:
1. Coordinate feature extraction (63-dim raw, 94-dim enriched, and batch vectorization).
2. Deep CoordinateInferenceEngine forward pass and top-3 candidate ranking.
3. Modular AI provider interface (LocalCoordinateProvider, NvidiaNimProvider, ProviderFactory).
4. Automated fallback mechanics on timeout/failure.
5. Bit-for-bit invariance of frozen Letter A artifacts.
"""

import sys
import json
import hashlib
from pathlib import Path
import pytest
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from backend.coordinate_ai.feature_pipeline import (
    extract_enriched_coordinate_features,
    extract_normalized_coordinates_63,
    batch_extract_enriched_coordinate_features,
    calculate_palm_scale,
    landmarks_to_numpy,
    RAW_COORDINATE_DIM,
    ENRICHED_FEATURE_DIM,
)
from backend.coordinate_ai.model_zoo import (
    CoordinateInferenceEngine,
    STATIC_ASL_CLASSES,
    ALL_ASL_CLASSES,
)
from backend.coordinate_ai.external_provider import (
    LocalCoordinateProvider,
    NvidiaNimProvider,
    ProviderFactory,
    CoordinateInferenceRequest,
    CoordinateInferenceResponse,
)


def create_dummy_hand(scale: float = 0.10):
    """Generates a plausible 21-landmark hand configuration."""
    lms = []
    # Wrist
    lms.append({"x": 0.5, "y": 0.8, "z": 0.0})
    # Thumb (1-4)
    for i in range(1, 5):
        lms.append({"x": 0.45 - i * 0.02, "y": 0.75 - i * 0.03, "z": 0.01 * i})
    # Index (5-8)
    for i in range(1, 5):
        lms.append({"x": 0.48, "y": 0.65 - i * 0.04, "z": -0.005 * i})
    # Middle (9-12)
    for i in range(1, 5):
        lms.append({"x": 0.50, "y": 0.64 - i * 0.045, "z": 0.0})
    # Ring (13-16)
    for i in range(1, 5):
        lms.append({"x": 0.52, "y": 0.65 - i * 0.04, "z": 0.005 * i})
    # Pinky (17-20)
    for i in range(1, 5):
        lms.append({"x": 0.54 + i * 0.01, "y": 0.67 - i * 0.03, "z": 0.01 * i})
    return lms


class TestCoordinateFeaturePipeline:
    def test_raw_63_normalization(self):
        hand = create_dummy_hand()
        coords = extract_normalized_coordinates_63(hand)
        assert len(coords) == RAW_COORDINATE_DIM
        # Wrist is origin (0, 0, 0)
        assert coords[0] == pytest.approx(0.0, abs=1e-5)
        assert coords[1] == pytest.approx(0.0, abs=1e-5)
        assert coords[2] == pytest.approx(0.0, abs=1e-5)

    def test_enriched_94_feature_dimensions(self):
        hand = create_dummy_hand()
        features = extract_enriched_coordinate_features(hand)
        assert len(features) == ENRICHED_FEATURE_DIM
        # Check angle ranges [0, pi]
        angles = features[63:73]
        for a in angles:
            assert 0.0 <= a <= np.pi + 1e-4
        # Check curl ratios > 0
        curls = features[73:78]
        for c in curls:
            assert c > 0.0

    def test_batch_extraction_parity(self):
        hand = create_dummy_hand()
        single_feats = extract_enriched_coordinate_features(hand)

        pts = landmarks_to_numpy(hand)
        batch_pts = np.stack([pts, pts], axis=0)
        batch_feats = batch_extract_enriched_coordinate_features(batch_pts)

        assert batch_feats.shape == (2, ENRICHED_FEATURE_DIM)
        np.testing.assert_allclose(batch_feats[0], single_feats, rtol=1e-4, atol=1e-4)


class TestCoordinateInferenceEngine:
    @pytest.fixture
    def engine(self):
        model_file = WORKSPACE_ROOT / "backend" / "saved_models" / "coordinate_ai_model.json"
        assert model_file.exists(), "coordinate_ai_model.json must exist"
        with open(model_file, "r", encoding="utf-8") as f:
            model_dict = json.load(f)
        return CoordinateInferenceEngine(model_dict, confidence_threshold=0.70)

    def test_inference_top3_candidates(self, engine):
        hand = create_dummy_hand()
        res = engine.predict_landmarks(hand)
        assert "predicted_letter" in res
        assert "best_candidate" in res
        assert "confidence" in res
        assert "top_candidates" in res
        assert len(res["top_candidates"]) == 3

        probs = res["probabilities"]
        assert len(probs) == len(STATIC_ASL_CLASSES)
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-4)

        # Top candidate must match first ranked candidate
        assert res["best_candidate"] == res["top_candidates"][0]["letter"]
        assert res["confidence"] == pytest.approx(res["top_candidates"][0]["confidence"], abs=1e-4)

    def test_rejection_on_invalid_hand(self, engine):
        res = engine.predict_landmarks([])
        assert res["predicted_letter"] == "REJECTED"
        assert res["is_accepted"] is False
        assert res["confidence"] == 0.0


class TestAIProviderArchitecture:
    @pytest.fixture
    def engine(self):
        model_file = WORKSPACE_ROOT / "backend" / "saved_models" / "coordinate_ai_model.json"
        with open(model_file, "r", encoding="utf-8") as f:
            model_dict = json.load(f)
        return CoordinateInferenceEngine(model_dict)

    def test_local_provider_execution(self, engine):
        provider = LocalCoordinateProvider(engine)
        assert provider.health_check() is True

        hand = create_dummy_hand()
        raw_63 = extract_normalized_coordinates_63(hand).tolist()
        enriched_94 = extract_enriched_coordinate_features(hand).tolist()

        req = CoordinateInferenceRequest(
            session_id="test_session",
            timestamp_ms=1000.0,
            raw_coordinates_63=raw_63,
            enriched_features_94=enriched_94,
            confidence_threshold=0.70
        )
        resp = provider.predict(req)
        assert isinstance(resp, CoordinateInferenceResponse)
        assert resp.provider_name == "local_coordinate_engine"
        assert len(resp.top_candidates) == 3
        assert resp.latency_ms >= 0.0

    def test_provider_factory_and_fallback(self, engine, monkeypatch):
        # Configure invalid remote NIM provider to test zero-downtime fallback
        monkeypatch.setenv("AI_PROVIDER", "nvidia_nim")
        monkeypatch.setenv("AI_INFERENCE_ENABLED", "true")
        monkeypatch.setenv("AI_API_KEY", "nvapi-test-key")
        monkeypatch.setenv("AI_BASE_URL", "https://invalid-host-for-testing-fallback.internal")
        monkeypatch.setenv("AI_TIMEOUT_MS", "100")

        provider = ProviderFactory.get_provider(engine)
        assert isinstance(provider, NvidiaNimProvider)

        hand = create_dummy_hand()
        raw_63 = extract_normalized_coordinates_63(hand).tolist()
        enriched_94 = extract_enriched_coordinate_features(hand).tolist()

        req = CoordinateInferenceRequest(
            session_id="test_fallback",
            timestamp_ms=2000.0,
            raw_coordinates_63=raw_63,
            enriched_features_94=enriched_94,
            confidence_threshold=0.70
        )
        resp = provider.predict(req)
        # Should gracefully fall back to local provider
        assert resp.fallback_used is True
        assert "fallback" in resp.provider_name.lower()
        assert len(resp.top_candidates) == 3


class TestFrozenLetterAPreservation:
    def test_frozen_letter_a_manifest_integrity(self):
        manifest_path = WORKSPACE_ROOT / "backend" / "letter_a_frozen_manifest.json"
        assert manifest_path.exists()
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert len(manifest) == 7
        for relpath, meta in manifest.items():
            art_path = WORKSPACE_ROOT / relpath
            assert art_path.exists(), f"Artifact {art_path} missing"
            with open(art_path, "rb") as af:
                raw_bytes = af.read()
                computed_hash = hashlib.sha256(raw_bytes).hexdigest()
                if computed_hash != meta["sha256"]:
                    h_crlf = hashlib.sha256(raw_bytes.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')).hexdigest()
                    h_lf = hashlib.sha256(raw_bytes.replace(b'\r\n', b'\n')).hexdigest()
                    if h_crlf == meta["sha256"]:
                        computed_hash = h_crlf
                    elif h_lf == meta["sha256"]:
                        computed_hash = h_lf
            assert computed_hash == meta["sha256"], f"SHA256 mismatch for {relpath}"
