"""
SIGNBRIDGE Coordinate AI — Modular External AI Provider Interface
Supports:
1. LocalCoordinateProvider: Authoritative in-process, zero-cost, private on-device coordinate inference.
2. NvidiaNimProvider: Enterprise AI inference microservice gateway via NVIDIA NIM / Triton.
3. Automated local fallback upon network timeout, contract incompatibility, or API error.
4. Comprehensive request/response validation, latency logging, and error tracking.
"""

import os
import time
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:
    class BaseModel:
        def __init__(self, **kwargs):
            for cls in reversed(self.__class__.__mro__):
                for k, v in cls.__dict__.items():
                    if not k.startswith('_') and not callable(v):
                        setattr(self, k, v)
            for k, v in kwargs.items():
                setattr(self, k, v)

        def dict(self):
            return {k: getattr(self, k) for k in self.__dict__ if not k.startswith('_')}

    def Field(default=None, **kwargs):
        if default is ...:
            return None
        return default


VALID_COORDINATE_CLASSES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"
]


class CoordinateInferenceRequest(BaseModel):
    """Structured request schema for coordinate-based inference."""
    session_id: str = Field(default="live_webcam_session")
    timestamp_ms: float
    raw_coordinates_63: List[float] = Field(..., min_length=63, max_length=63)
    enriched_features_94: Optional[List[float]] = None
    confidence_threshold: float = Field(default=0.70, ge=0.0, le=1.0)


class CandidatePrediction(BaseModel):
    letter: str
    confidence: float


class CoordinateInferenceResponse(BaseModel):
    """Validated response schema from coordinate inference provider."""
    provider_name: str
    predicted_letter: str
    confidence: float
    is_accepted: bool
    top_candidates: List[CandidatePrediction]
    latency_ms: float
    fallback_used: bool = False
    error_message: Optional[str] = None


class BaseAIProvider(ABC):
    """Abstract interface for all SIGNBRIDGE AI inference providers."""
    @abstractmethod
    def predict(self, request: CoordinateInferenceRequest) -> CoordinateInferenceResponse:
        pass

    @abstractmethod
    def health_check(self) -> bool:
        pass


class LocalCoordinateProvider(BaseAIProvider):
    """
    Authoritative default in-process inference provider.
    Zero network latency, 100% offline, privacy-first, 94-D biomechanical CoordinateMLP.
    """
    def __init__(self, inference_engine):
        self.engine = inference_engine
        self.provider_name = "local_coordinate_engine"

    def health_check(self) -> bool:
        return self.engine is not None

    def predict(self, request: CoordinateInferenceRequest) -> CoordinateInferenceResponse:
        t0 = time.perf_counter()
        # Use enriched features if supplied, else compute from raw coordinates
        import numpy as np
        if request.enriched_features_94 and len(request.enriched_features_94) == 94:
            feats = np.array(request.enriched_features_94, dtype=np.float32)
        elif request.raw_coordinates_63 and len(request.raw_coordinates_63) == 63:
            try:
                try:
                    from backend.coordinate_ai.feature_pipeline import extract_enriched_coordinate_features
                except ImportError:
                    from coordinate_ai.feature_pipeline import extract_enriched_coordinate_features
                pts = np.array(request.raw_coordinates_63, dtype=np.float32).reshape(21, 3)
                lm = [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in pts]
                feats = extract_enriched_coordinate_features(lm)
            except Exception:
                feats = np.array(request.raw_coordinates_63, dtype=np.float32)
        else:
            feats = np.array(request.raw_coordinates_63, dtype=np.float32)

        res = self.engine.predict_features(feats)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        candidates = [
            CandidatePrediction(letter=c["letter"], confidence=c["confidence"])
            for c in res["top_candidates"]
        ]

        return CoordinateInferenceResponse(
            provider_name=self.provider_name,
            predicted_letter=res["predicted_letter"],
            confidence=res["confidence"],
            is_accepted=res["is_accepted"],
            top_candidates=candidates,
            latency_ms=round(elapsed_ms, 2),
            fallback_used=False
        )


class NvidiaNimProvider(BaseAIProvider):
    """
    NVIDIA NIM (Inference Microservice) Gateway.
    
    INFERENCE CONTRACT REQUIREMENT:
    Any external remote provider MUST host our custom 94-D CoordinateMLP model
    (or an exported TensorRT/Triton equivalent) accepting 94-D numerical biomechanical
    tensors and returning calibrated class probabilities over the 24 classes.
    
    Generic chat/completions endpoints and text/vision LLMs (e.g. LLaMA, GPT, vision-instruct)
    are fundamentally incompatible with the CoordinateMLP contract. Such endpoints are rejected
    upfront and routed to authoritative local inference with fallback telemetry.
    """
    def __init__(
        self,
        local_fallback_provider: BaseAIProvider,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout_ms: int = 2000
    ):
        self.local_fallback = local_fallback_provider
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY") or os.getenv("AI_API_KEY", "")
        self.base_url = base_url or os.getenv("NVIDIA_BASE_URL") or os.getenv("AI_BASE_URL", "https://integrate.api.nvidia.com/v1")
        self.model_name = model_name or os.getenv("NVIDIA_MODEL") or os.getenv("AI_MODEL", "qwen/qwen2.5-7b-instruct")
        self.timeout_sec = max(0.05, timeout_ms / 1000.0)
        self.provider_name = "nvidia_nim"

        # Telemetry
        self.total_requests = 0
        self.failed_requests = 0
        self.total_latency_ms = 0.0

    def is_contract_compatible(self) -> bool:
        """
        Verifies whether the configured endpoint supports the 94-D CoordinateMLP tensor contract.
        Chat/completions endpoints and text/vision LLMs are incompatible.
        """
        incompatible_tokens = ["llama", "gpt", "vision-instruct", "mistral", "qwen", "chat/completions"]
        m_lower = (self.model_name or "").lower()
        u_lower = (self.base_url or "").lower()
        if any(tok in m_lower or tok in u_lower for tok in incompatible_tokens):
            return False
        return True

    def health_check(self) -> bool:
        return bool(self.api_key and self.base_url and self.is_contract_compatible())

    def predict(self, request: CoordinateInferenceRequest) -> CoordinateInferenceResponse:
        self.total_requests += 1
        t0 = time.perf_counter()

        # 1. If external inference is disabled or unconfigured, route immediately to local provider
        if not self.api_key or os.getenv("AI_INFERENCE_ENABLED", "false").lower() != "true":
            resp = self.local_fallback.predict(request)
            resp.provider_name = "local_coordinate_engine (external disabled)"
            return resp

        # 2. Block incompatible text/chat endpoints from receiving truncated coordinate queries
        if not self.is_contract_compatible():
            self.failed_requests += 1
            fallback_res = self.local_fallback.predict(request)
            fallback_res.fallback_used = True
            fallback_res.error_message = (
                f"Incompatible remote provider contract: Configured model '{self.model_name}' "
                "is a text/chat foundation LLM and does not host the custom 94-D CoordinateMLP model. "
                "Enforcing authoritative local inference."
            )
            fallback_res.provider_name = "local_coordinate_engine (fallback: contract mismatch)"
            return fallback_res

        # 3. For a compatible remote Triton / NIM numerical tensor endpoint:
        # Pass the full 94-D enriched features
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": [
                {
                    "name": "enriched_features_94",
                    "shape": [1, 94],
                    "datatype": "FP32",
                    "data": request.enriched_features_94 or request.raw_coordinates_63
                }
            ]
        }

        try:
            import urllib.request
            req = urllib.request.Request(
                f"{self.base_url}/v2/models/{self.model_name}/infer",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as response:
                body = json.loads(response.read().decode("utf-8"))
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                self.total_latency_ms += elapsed_ms

                # Parse numerical probabilities from tensor output
                outputs = {o["name"]: o["data"] for o in body.get("outputs", [])}
                probs = outputs.get("probabilities", [])

                if not probs or len(probs) != len(VALID_COORDINATE_CLASSES):
                    raise ValueError(f"Malformed remote tensor output: expected {len(VALID_COORDINATE_CLASSES)} class probabilities.")

                import numpy as np
                probs_arr = np.array(probs, dtype=np.float32)
                top_idx = int(np.argmax(probs_arr))
                predicted = VALID_COORDINATE_CLASSES[top_idx]
                confidence = float(probs_arr[top_idx])

                # Strictly enforce calibrated confidence rejection threshold
                is_accepted = (confidence >= request.confidence_threshold)
                if not is_accepted:
                    predicted = "REJECTED"

                sorted_indices = np.argsort(probs_arr)[::-1][:3]
                candidates = [
                    CandidatePrediction(
                        letter=VALID_COORDINATE_CLASSES[i],
                        confidence=round(float(probs_arr[i]), 4)
                    )
                    for i in sorted_indices
                ]

                return CoordinateInferenceResponse(
                    provider_name=self.provider_name,
                    predicted_letter=predicted,
                    confidence=round(confidence, 4),
                    is_accepted=is_accepted,
                    top_candidates=candidates,
                    latency_ms=round(elapsed_ms, 2),
                    fallback_used=False
                )

        except Exception as e:
            self.failed_requests += 1
            # Seamless fallback to authoritative local engine
            fallback_res = self.local_fallback.predict(request)
            fallback_res.fallback_used = True
            fallback_res.error_message = f"NVIDIA NIM error ({type(e).__name__}): {str(e)[:120]}. Routed to local engine."
            fallback_res.provider_name = "local_coordinate_engine (fallback)"
            return fallback_res


class ProviderFactory:
    """Instantiates the appropriate AI provider according to environment configuration."""
    @staticmethod
    def get_provider(local_engine) -> BaseAIProvider:
        local_provider = LocalCoordinateProvider(local_engine)
        provider_type = os.getenv("AI_PROVIDER", "local").lower()

        # Incompatible remote NIM routing is disabled by default.
        # If explicitly selected, NvidiaNimProvider validates the contract and falls back to local_provider.
        if provider_type == "nvidia_nim":
            return NvidiaNimProvider(local_fallback_provider=local_provider)
        return local_provider
