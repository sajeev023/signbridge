"""
SIGNBRIDGE Coordinate AI Package
Core modules for coordinate-based alphabet recognition, feature engineering, and model inference.
"""

from backend.coordinate_ai.feature_pipeline import (
    extract_enriched_coordinate_features,
    extract_normalized_coordinates_63,
    batch_extract_enriched_coordinate_features,
    RAW_COORDINATE_DIM,
    ENRICHED_FEATURE_DIM,
)
from backend.coordinate_ai.model_zoo import (
    CoordinateMLP,
    CoordinateInferenceEngine,
    STATIC_ASL_CLASSES,
    DYNAMIC_ASL_CLASSES,
    ALL_ASL_CLASSES,
)
from backend.coordinate_ai.external_provider import (
    BaseAIProvider,
    LocalCoordinateProvider,
    NvidiaNimProvider,
    ProviderFactory,
    CandidatePrediction,
    CoordinateInferenceRequest,
    CoordinateInferenceResponse,
)

__all__ = [
    "extract_enriched_coordinate_features",
    "extract_normalized_coordinates_63",
    "batch_extract_enriched_coordinate_features",
    "RAW_COORDINATE_DIM",
    "ENRICHED_FEATURE_DIM",
    "CoordinateMLP",
    "CoordinateInferenceEngine",
    "STATIC_ASL_CLASSES",
    "DYNAMIC_ASL_CLASSES",
    "ALL_ASL_CLASSES",
    "BaseAIProvider",
    "LocalCoordinateProvider",
    "NvidiaNimProvider",
    "ProviderFactory",
    "CandidatePrediction",
    "CoordinateInferenceRequest",
    "CoordinateInferenceResponse",
]
