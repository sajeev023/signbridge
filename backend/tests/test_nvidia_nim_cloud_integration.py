"""
Tests for SIGNBRIDGE NVIDIA NIM Cloud AI Integration & Layer C Language Intelligence.
Validates:
1. NVIDIA_API_KEY auto-detection and provider configuration
2. Default compact model selection (Qwen 2.5 7B <= 30B)
3. Model override via NVIDIA_MODEL
4. Credential security (never exposed in status or logs)
5. Contract incompatibility rejection for Layer B CoordinateMLP (NvidiaNimProvider)
6. Rate-limiting (HTTP 429) circuit-breaker and deterministic local fallback
7. Auth error (HTTP 401) circuit-breaker and deterministic local fallback
8. Markdown code fence JSON extraction
9. Zero-credential offline operational readiness
10. FastAPI REST endpoint integration (/api/ai/compose, /api/ai/compose-sentence, /api/ai/status)
"""

import os
import json
import urllib.error
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from backend.app.services.ai_assistance_service import AIAssistanceService
from backend.coordinate_ai.model_zoo import CoordinateInferenceEngine
from backend.coordinate_ai.external_provider import (
    NvidiaNimProvider, LocalCoordinateProvider, CoordinateInferenceRequest
)
from backend.app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_nvidia_key_auto_detection_and_qwen_default(monkeypatch):
    """Verifies that setting NVIDIA_API_KEY activates nvidia_nim with Qwen 2.5 7B default."""
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-fake-key-12345")
    monkeypatch.delenv("AI_ASSIST_PROVIDER", raising=False)
    monkeypatch.delenv("AI_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_MODEL", raising=False)

    service = AIAssistanceService()
    status = service.get_status()

    assert service.provider == "nvidia_nim"
    assert service.model == "qwen/qwen2.5-7b-instruct"
    assert service.base_url == "https://integrate.api.nvidia.com/v1"
    assert status["cloud_configured"] is True
    assert status["active_provider"] == "nvidia_nim"
    # Ensure raw secret is not leaked in status response
    assert "nvapi-test-fake-key-12345" not in json.dumps(status)


def test_nvidia_model_override(monkeypatch):
    """Verifies that NVIDIA_MODEL allows overriding to another compact model like Llama 3.1 8B."""
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-fake-key")
    monkeypatch.setenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")

    service = AIAssistanceService()
    assert service.model == "meta/llama-3.1-8b-instruct"
    assert service.provider == "nvidia_nim"


def test_offline_zero_credential_readiness(monkeypatch):
    """Verifies that with no API key, the service functions 100% offline via local engine."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("AI_API_KEY", raising=False)
    monkeypatch.delenv("AI_ASSIST_PROVIDER", raising=False)

    service = AIAssistanceService()
    status = service.get_status()

    assert service.provider == "local"
    assert status["active_provider"] == "local_deterministic_engine"
    assert status["cloud_configured"] is False

    res = service.compose_sentence(["H", "E", "L", "L", "O"], target_language="en")
    assert res["composed_text"] == "Hello."
    assert res["provider"] == "local_deterministic_engine"
    assert res["fallback_used"] is False


def test_markdown_code_fence_json_extraction():
    """Verifies that _extract_json parses responses wrapped in markdown ```json ... ``` blocks."""
    raw_markdown = (
        "Here is your composed sentence:\n"
        "```json\n"
        '{\n  "composed_text": "Good morning, please help me with water."\n}\n'
        "```\n"
        "Hope this helps!"
    )
    parsed = AIAssistanceService._extract_json(raw_markdown)
    assert isinstance(parsed, dict)
    assert parsed.get("composed_text") == "Good morning, please help me with water."


def test_rate_limit_http_429_circuit_breaker_fallback(monkeypatch):
    """Verifies that HTTP 429 rate limit triggers immediate fallback to local deterministic engine."""
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-fake-key")
    service = AIAssistanceService()

    # Simulate HTTP 429 error from NVIDIA NIM endpoint
    http_429 = urllib.error.HTTPError(
        url="https://integrate.api.nvidia.com/v1/chat/completions",
        code=429,
        msg="Too Many Requests",
        hdrs={},
        fp=None
    )

    with patch("urllib.request.urlopen", side_effect=http_429):
        result = service.compose_sentence(["THANK_YOU"], target_language="en")

    assert result["fallback_used"] is True
    assert "rate limit" in result["fallback_reason"].lower()
    assert result["composed_text"] == "Thank you very much."
    assert service.fallback_requests == 1


def test_auth_error_http_401_circuit_breaker_fallback(monkeypatch):
    """Verifies that HTTP 401 unauthorized triggers clean fallback with redacted diagnostics."""
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-invalid-key")
    service = AIAssistanceService()

    http_401 = urllib.error.HTTPError(
        url="https://integrate.api.nvidia.com/v1/chat/completions",
        code=401,
        msg="Unauthorized",
        hdrs={},
        fp=None
    )

    with patch("urllib.request.urlopen", side_effect=http_401):
        result = service.compose_sentence(["WATER"], target_language="en")

    assert result["fallback_used"] is True
    assert "authentication failed" in result["fallback_reason"].lower()
    assert result["composed_text"] == "Could I please have some water?"
    # Secret must never appear in the fallback reason
    assert "nvapi-invalid-key" not in result["fallback_reason"]


def test_layer_b_coordinatemlp_contract_rejection(monkeypatch):
    """
    Verifies that passing chat/completions LLMs to NvidiaNimProvider for Layer B coordinate recognition
    is firmly rejected as contract-incompatible, protecting local CoordinateMLP authority.
    """
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-fake-key")
    monkeypatch.setenv("AI_INFERENCE_ENABLED", "true")

    model_file = os.path.join(os.path.dirname(__file__), "..", "saved_models", "coordinate_ai_model.json")
    with open(model_file, "r", encoding="utf-8") as f:
        model_dict = json.load(f)
    local_engine = CoordinateInferenceEngine(model_dict)
    local_p = LocalCoordinateProvider(local_engine)
    nim_p = NvidiaNimProvider(
        local_fallback_provider=local_p,
        model_name="qwen/qwen2.5-7b-instruct"
    )

    # Must detect that Qwen chat is NOT a 94-D CoordinateMLP tensor classifier
    assert nim_p.is_contract_compatible() is False

    req = CoordinateInferenceRequest(
        session_id="test_nim_session",
        timestamp_ms=1000.0,
        raw_coordinates_63=[0.0] * 63,
        enriched_features_94=[0.0] * 94,
        confidence_threshold=0.75
    )
    res = nim_p.predict(req)

    assert res.fallback_used is True
    assert "Incompatible remote provider contract" in res.error_message
    assert "qwen" in res.error_message.lower()


def test_api_ai_compose_endpoint(client, monkeypatch):
    """Verifies that POST /api/ai/compose and POST /api/ai/compose-sentence work via FastAPI."""
    res1 = client.post("/api/ai/compose", json={"tokens": ["W", "A", "T", "E", "R"]})
    assert res1.status_code == 200
    assert "water" in res1.json()["composed_text"].lower()

    res2 = client.post("/api/ai/compose-sentence", json={"tokens": ["HELP"]})
    assert res2.status_code == 200
    assert "help" in res2.json()["composed_text"].lower()


def test_api_ai_status_endpoint(client):
    """Verifies that GET /api/ai/status returns valid operational metadata."""
    res = client.get("/api/ai/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ready"
    assert "capabilities" in data
    assert "offline_fallback" in data["capabilities"]
