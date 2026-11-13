"""
SIGNBRIDGE TITAN: Autonomous Next-Generation Spatial-Temporal Recognition Research Suite
"""

from .graph_topology import (
    NUM_HAND_LANDMARKS,
    ANATOMICAL_BONES,
    PALM_CONNECTIONS,
    CROSS_FINGER_EDGES,
    build_adjacency_matrix,
    compute_normalized_laplacian,
)
from .spatial_encoder import SpatialGraphEncoder, landmarks_to_node_features
from .temporal_encoder import BiLSTMTemporalEncoder, TemporalTransformerEncoder
from .fusion import ConcatenationFusion, GatedFusion, CrossAttentionFusion
from .dual_stream import DualStreamCrossHandModel
from .architectures import (
    ArchitectureA_SpatialGNN,
    ArchitectureB_BiLSTMAttention,
    ArchitectureC_STTransformer,
    ArchitectureD_DualStreamGated,
    TitanHybridModel,
)
from .uncertainty import (
    TemperatureScaler,
    compute_ece,
    compute_brier_score,
    compute_predictive_entropy,
    compute_energy_score,
    evaluate_selective_rejection,
)
