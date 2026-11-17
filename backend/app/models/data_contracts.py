"""
SIGNBRIDGE Unified Data Contracts — Frame & Inference Schemas
Enforces a single, strictly validated, typed data contract across Python backend,
WebSocket streaming, REST endpoints, and React TypeScript frontend.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

FEATURE_SCHEMA_VERSION = "2.1.0-coordinate-94d"
MODEL_VERSION = "2.1.0"


class LandmarkCoord(BaseModel):
    x: float
    y: float
    z: float = 0.0


class CandidatePrediction(BaseModel):
    letter: str
    confidence: float


class FrameInferencePayload(BaseModel):
    """
    Standardized payload for single-frame inference.
    Shared by REST /api/predict/frame, /api/coordinate/predict, and WebSocket /ws/stream.
    """
    frame_id: int = Field(default=0, description="Monotonically increasing client frame sequence counter")
    timestamp_ms: float = Field(default=0.0, description="Client capture timestamp in milliseconds")
    has_hands: bool = Field(default=False, description="Whether hand landmarker detected at least one hand")
    hand_count: int = Field(default=0, ge=0, le=2, description="Number of detected hands (0, 1, or 2)")
    handedness: str = Field(default="Right", description="'Right', 'Left', 'Both', or 'None'")
    landmarks_21: Optional[List[Dict[str, float]]] = Field(default=None, description="21 normalized 3D landmarks for primary hand")
    raw_coordinates_63: Optional[List[float]] = Field(default=None, description="63 normalized coordinates [x0,y0,z0..x20,y20,z20]")
    enriched_features_94: Optional[List[float]] = Field(default=None, description="94-D biomechanical feature vector")
    feature_schema_version: str = Field(default=FEATURE_SCHEMA_VERSION, description="Feature extraction schema version")
    model_version: str = Field(default=MODEL_VERSION, description="Target model version")
    confidence_threshold: float = Field(default=0.70, ge=0.0, le=1.0, description="Minimum confidence for acceptance")
    session_id: str = Field(default="live_session", description="Client session identifier")
    mode: str = Field(default="alphabet", description="'alphabet' (ASL/Single-Hand) or 'words' (ISL vocabulary)")
    # Optional multi-hand/pose payload for ISL word vocabulary
    left_hand: Optional[List[Dict[str, float]]] = None
    right_hand: Optional[List[Dict[str, float]]] = None
    pose: Optional[List[Dict[str, float]]] = None


class InferenceResultEnvelope(BaseModel):
    """
    Standardized result envelope returned by all inference routes.
    Guarantees consistent field names, calibrated confidence, rejection status, and latency.
    """
    frame_id: int
    timestamp_ms: float
    status: str = Field(..., description="'recognized' | 'uncertain' | 'rejected' | 'buffering' | 'no_hand' | 'idle'")
    predicted_sign: str = Field(..., description="Recognized letter or word identifier, or 'REJECTED' / 'NEUTRAL'")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Calibrated confidence score")
    is_accepted: bool = Field(default=False, description="True only if confidence >= threshold and hand is tracked")
    is_dynamic: bool = Field(default=False, description="True if sign was classified via temporal trajectory engine")
    rejection_reason: Optional[str] = Field(default=None, description="Human-readable reason when is_accepted is False")
    top_candidates: List[CandidatePrediction] = Field(default_factory=list, description="Top candidate predictions with confidences")
    probabilities: Dict[str, float] = Field(default_factory=dict, description="Full or top-N probability distribution")
    feature_schema_version: str = Field(default=FEATURE_SCHEMA_VERSION)
    model_version: str = Field(default=MODEL_VERSION)
    latency_ms: float = Field(default=0.0, description="Total inference latency in milliseconds")
    translations: Optional[Dict[str, str]] = Field(default=None, description="Multilingual translations for recognized sign")
    label: Optional[str] = Field(default=None, description="Display label for recognized sign")
    category: Optional[str] = Field(default=None, description="Sign category (e.g. 'alphabet', 'greetings', 'emergency')")
    fallback_used: bool = Field(default=False, description="Whether fallback engine or local provider was used")
