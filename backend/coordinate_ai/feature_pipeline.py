"""
SIGNBRIDGE Coordinate AI — Coordinate Feature Engineering Pipeline
Converts 21 MediaPipe hand landmarks (x, y, z) into a structured numerical feature representation.
Provides:
1. 63-dimensional raw normalized coordinates (origin-centered at wrist, scaled by palm size).
2. Articulation angles (interior 3D angles across finger joints).
3. Fingertip extension and curl ratios.
4. Inter-fingertip distance pairs.
5. Palm orientation and normal vectors.
6. Enriched 94-dimensional invariant coordinate representation.
7. High-throughput vectorized batch extraction.
"""

import math
from typing import Dict, List, Any, Tuple
import numpy as np


NUM_LANDMARKS = 21
COORDINATES_PER_LANDMARK = 3
RAW_COORDINATE_DIM = NUM_LANDMARKS * COORDINATES_PER_LANDMARK  # 63
ENRICHED_FEATURE_DIM = 94

# Canonical joint indices (MediaPipe Hand Topology)
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MID_MCP, MID_PIP, MID_DIP, MID_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGERTIPS = [THUMB_TIP, INDEX_TIP, MID_TIP, RING_TIP, PINKY_TIP]

PAIRWISE_FINGERTIPS = [
    (THUMB_TIP, INDEX_TIP),
    (THUMB_TIP, MID_TIP),
    (THUMB_TIP, RING_TIP),
    (THUMB_TIP, PINKY_TIP),
    (INDEX_TIP, MID_TIP),
    (INDEX_TIP, RING_TIP),
    (INDEX_TIP, PINKY_TIP),
    (MID_TIP, RING_TIP),
    (MID_TIP, PINKY_TIP),
    (RING_TIP, PINKY_TIP),
]

JOINT_ANGLE_TRIPLETS = [
    (THUMB_CMC, THUMB_MCP, THUMB_IP),
    (THUMB_MCP, THUMB_IP, THUMB_TIP),
    (INDEX_MCP, INDEX_PIP, INDEX_DIP),
    (INDEX_PIP, INDEX_DIP, INDEX_TIP),
    (MID_MCP, MID_PIP, MID_DIP),
    (MID_PIP, MID_DIP, MID_TIP),
    (RING_MCP, RING_PIP, RING_DIP),
    (RING_PIP, RING_DIP, RING_TIP),
    (PINKY_MCP, PINKY_PIP, PINKY_DIP),
    (PINKY_PIP, PINKY_DIP, PINKY_TIP),
]


def landmarks_to_numpy(landmarks: List[Dict[str, float]]) -> np.ndarray:
    """Converts 21 landmark dictionaries into a (21, 3) float32 numpy array."""
    pts = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
    n = min(NUM_LANDMARKS, len(landmarks))
    for i in range(n):
        lm = landmarks[i]
        pts[i, 0] = float(lm.get("x", 0.0))
        pts[i, 1] = float(lm.get("y", 0.0))
        pts[i, 2] = float(lm.get("z", 0.0))
    return pts


def calculate_palm_scale(pts: np.ndarray) -> float:
    """Calculates palm scale as Euclidean distance between wrist and middle MCP."""
    wrist = pts[WRIST]
    mid_mcp = pts[MID_MCP]
    scale = float(np.linalg.norm(mid_mcp - wrist))
    return max(0.035, scale)


def extract_normalized_coordinates_63(landmarks: List[Dict[str, float]]) -> np.ndarray:
    """
    Extracts 63 normalized 3D coordinates:
    - Origin translated to wrist (joint 0).
    - Scaled by palm distance.
    - Preserves relative 3D geometry without angular distortion.
    """
    pts = landmarks_to_numpy(landmarks)
    wrist = pts[WRIST]
    scale = calculate_palm_scale(pts)
    norm_pts = (pts - wrist) / scale
    return norm_pts.flatten().astype(np.float32)


def compute_3d_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Computes interior 3D angle at vertex b between vectors (a - b) and (c - b)."""
    v1 = a - b
    v2 = c - b
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-7 or norm2 < 1e-7:
        return 0.0
    cosine = np.dot(v1, v2) / (norm1 * norm2)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.arccos(cosine))


def extract_articulation_angles(pts: np.ndarray) -> np.ndarray:
    """Computes 10 interior joint angles."""
    angles = [compute_3d_angle(pts[a], pts[b], pts[c]) for a, b, c in JOINT_ANGLE_TRIPLETS]
    return np.array(angles, dtype=np.float32)


def extract_curl_ratios(pts: np.ndarray, scale: float) -> np.ndarray:
    """Computes Euclidean distance from each fingertip to wrist, normalized by palm scale."""
    wrist = pts[WRIST]
    ratios = [float(np.linalg.norm(pts[tip] - wrist)) / scale for tip in FINGERTIPS]
    return np.array(ratios, dtype=np.float32)


def extract_inter_fingertip_distances(pts: np.ndarray, scale: float) -> np.ndarray:
    """Computes pairwise Euclidean distances between all 5 fingertips (10 pairs)."""
    dists = [float(np.linalg.norm(pts[i] - pts[j])) / scale for i, j in PAIRWISE_FINGERTIPS]
    return np.array(dists, dtype=np.float32)


def extract_palm_orientation(pts: np.ndarray) -> np.ndarray:
    """Computes palm normal vector and transverse axis orientation (6 values)."""
    wrist = pts[WRIST]
    v_idx = pts[INDEX_MCP] - wrist
    v_pinky = pts[PINKY_MCP] - wrist
    normal = np.cross(v_idx, v_pinky)
    norm_len = float(np.linalg.norm(normal))
    unit_normal = normal / norm_len if norm_len > 1e-7 else np.array([0.0, 0.0, 1.0], dtype=np.float32)

    mid_vec = pts[MID_MCP] - wrist
    mid_len = float(np.linalg.norm(mid_vec))
    unit_longitudinal = mid_vec / mid_len if mid_len > 1e-7 else np.array([0.0, 1.0, 0.0], dtype=np.float32)

    return np.concatenate([unit_normal, unit_longitudinal]).astype(np.float32)


def extract_enriched_coordinate_features(landmarks: List[Dict[str, float]]) -> np.ndarray:
    """
    Produces the full 94-dimensional enriched coordinate representation:
    - 63 raw normalized 3D coordinates.
    - 10 joint articulation angles.
    - 5 fingertip curl ratios.
    - 10 pairwise fingertip distances.
    - 6 palm orientation unit vectors.
    Total: 63 + 10 + 5 + 10 + 6 = 94 continuous invariant features.
    """
    if not landmarks or len(landmarks) < NUM_LANDMARKS:
        return np.zeros(ENRICHED_FEATURE_DIM, dtype=np.float32)

    pts = landmarks_to_numpy(landmarks)
    wrist = pts[WRIST]
    scale = calculate_palm_scale(pts)

    norm_pts = (pts - wrist) / scale
    raw_63 = norm_pts.flatten().astype(np.float32)
    angles_10 = extract_articulation_angles(pts)
    curls_5 = extract_curl_ratios(pts, scale)
    pairs_10 = extract_inter_fingertip_distances(pts, scale)
    orient_6 = extract_palm_orientation(pts)

    return np.concatenate([raw_63, angles_10, curls_5, pairs_10, orient_6]).astype(np.float32)


def batch_extract_enriched_coordinate_features(pts_array: np.ndarray) -> np.ndarray:
    """
    Vectorized high-throughput batch extraction of 94-dim features.
    pts_array shape: (N, 21, 3) float32
    Returns: (N, 94) float32 array
    """
    N = pts_array.shape[0]
    if N == 0:
        return np.zeros((0, ENRICHED_FEATURE_DIM), dtype=np.float32)

    wrists = pts_array[:, WRIST : WRIST + 1, :]  # (N, 1, 3)
    mid_mcps = pts_array[:, MID_MCP, :]  # (N, 3)
    scales = np.linalg.norm(mid_mcps - wrists[:, 0, :], axis=-1)  # (N,)
    scales = np.maximum(scales, 0.035)[:, None, None]  # (N, 1, 1)

    # 1. 63 Normalized coordinates
    norm_pts = (pts_array - wrists) / scales  # (N, 21, 3)
    raw_63 = norm_pts.reshape(N, 63)

    # 2. 10 Articulation angles
    scales_1d = scales[:, 0, 0]  # (N,)
    angle_cols = []
    for a, b, c in JOINT_ANGLE_TRIPLETS:
        v1 = pts_array[:, a, :] - pts_array[:, b, :]  # (N, 3)
        v2 = pts_array[:, c, :] - pts_array[:, b, :]  # (N, 3)
        dot = np.sum(v1 * v2, axis=-1)
        norm1 = np.linalg.norm(v1, axis=-1)
        norm2 = np.linalg.norm(v2, axis=-1)
        denom = np.maximum(norm1 * norm2, 1e-7)
        cos = np.clip(dot / denom, -1.0, 1.0)
        angle_cols.append(np.arccos(cos))
    angles_10 = np.stack(angle_cols, axis=1)  # (N, 10)

    # 3. 5 Curl ratios
    curl_cols = []
    for tip in FINGERTIPS:
        d = np.linalg.norm(pts_array[:, tip, :] - wrists[:, 0, :], axis=-1) / scales_1d
        curl_cols.append(d)
    curls_5 = np.stack(curl_cols, axis=1)  # (N, 5)

    # 4. 10 Pairwise fingertip distances
    pair_cols = []
    for i, j in PAIRWISE_FINGERTIPS:
        d = np.linalg.norm(pts_array[:, i, :] - pts_array[:, j, :], axis=-1) / scales_1d
        pair_cols.append(d)
    pairs_10 = np.stack(pair_cols, axis=1)  # (N, 10)

    # 5. 6 Palm orientation vectors
    v_idx = pts_array[:, INDEX_MCP, :] - wrists[:, 0, :]
    v_pinky = pts_array[:, PINKY_MCP, :] - wrists[:, 0, :]
    normal = np.cross(v_idx, v_pinky)
    norm_lens = np.linalg.norm(normal, axis=-1, keepdims=True)
    unit_normal = np.where(norm_lens > 1e-7, normal / np.maximum(norm_lens, 1e-7), np.array([0.0, 0.0, 1.0], dtype=np.float32))

    mid_vec = pts_array[:, MID_MCP, :] - wrists[:, 0, :]
    mid_lens = np.linalg.norm(mid_vec, axis=-1, keepdims=True)
    unit_mid = np.where(mid_lens > 1e-7, mid_vec / np.maximum(mid_lens, 1e-7), np.array([0.0, 1.0, 0.0], dtype=np.float32))

    orient_6 = np.concatenate([unit_normal, unit_mid], axis=-1)  # (N, 6)

    return np.concatenate([raw_63, angles_10, curls_5, pairs_10, orient_6], axis=1).astype(np.float32)
