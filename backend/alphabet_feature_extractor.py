"""
SIGNBRIDGE Canonical 42-Feature Biomechanical Hand Representation
Computes continuous, scale-, translation-, and 3D rotation-invariant features
directly from 21 MediaPipe 3D hand landmarks without arbitrary threshold cliffs.
"""

import math
from typing import Dict, List, Any, Tuple
import numpy as np

FEATURE_NAMES_42 = [
    # 0-4: 5 Finger Extensions (Tip-to-wrist minus MCP-to-wrist, normalized by palm scale)
    "ext_thumb", "ext_index", "ext_middle", "ext_ring", "ext_pinky",
    # 5-9: 5 Fingertip to Palm Center Distances
    "dist_tip_thumb_palm", "dist_tip_idx_palm", "dist_tip_mid_palm", "dist_tip_ring_palm", "dist_tip_pinky_palm",
    # 10-14: 5 Finger Curl Metrics (Tip-to-MCP distance / palm scale)
    "curl_thumb", "curl_index", "curl_middle", "curl_ring", "curl_pinky",
    # 15-18: 4 Inter-finger Bone Vector Spread Cosines
    "cos_spread_thumb_idx", "cos_spread_idx_mid", "cos_spread_mid_ring", "cos_spread_ring_pinky",
    # 19-23: Key Inter-fingertip Euclidean Distances
    "dist_tip_thumb_idx", "dist_tip_thumb_mid", "dist_tip_thumb_pinky", "dist_tip_idx_mid", "dist_tip_mid_ring",
    # 24-28: Canonical 3D Thumb Coordinates & Opposition
    "canonical_thumb_x", "canonical_thumb_y", "canonical_thumb_z", "dist_thumb_idx_mcp", "dist_thumb_ring_mcp",
    # 29-31: Relative Finger Crossing Metrics (For 'R' vs 'U' vs 'V')
    "lateral_crossing_idx_mid", "depth_crossing_idx_mid", "cross_product_sign_idx_mid",
    # 32-34: Canonical Direction Vectors (For horizontal 'G', 'H' vs upright 'D', 'U')
    "canonical_idx_dir_x", "canonical_idx_dir_y", "canonical_idx_dir_z",
    # 35-37: Palm Curl & Fist Compactness
    "curl_angle_idx_pip", "curl_angle_mid_pip", "fist_compactness",
    # 38-41: Structural Invariant Ratios
    "spread_ratio_idx_mid", "y_span_ratio", "parallelism_idx_mid", "loop_closure_thumb_idx"
]

NUM_CANONICAL_FEATURES = len(FEATURE_NAMES_42)


def dist_3d(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2))


def extract_canonical_features(landmarks: List[Dict[str, float]]) -> np.ndarray:
    """
    Extracts 42 continuous invariant biomechanical features from 21 hand landmarks.
    Landmarks is a list of 21 dicts containing 'x', 'y', and optionally 'z'.
    """
    if not landmarks or len(landmarks) < 21:
        return np.zeros(NUM_CANONICAL_FEATURES, dtype=np.float32)

    pts = np.zeros((21, 3), dtype=np.float32)
    for i in range(21):
        pts[i, 0] = landmarks[i]["x"]
        pts[i, 1] = landmarks[i]["y"]
        pts[i, 2] = landmarks[i].get("z", 0.0)

    # 1. Coordinate Frame Invariant Transformation
    # Origin: Wrist (joint 0)
    wrist = pts[0]
    mid_mcp = pts[9]
    idx_mcp = pts[5]
    pinky_mcp = pts[17]

    # Palm scale: Euclidean distance between wrist and middle MCP
    scale = max(0.035, float(np.linalg.norm(mid_mcp - wrist)))

    # Basis vectors
    # Y axis: Along the longitudinal palm axis (wrist to middle MCP)
    vec_y = (mid_mcp - wrist) / scale

    # Palm normal Z: cross product of (idx_mcp - wrist) x (pinky_mcp - wrist)
    v_idx = idx_mcp - wrist
    v_pinky = pinky_mcp - wrist
    normal = np.cross(v_idx, v_pinky)
    norm_len = float(np.linalg.norm(normal))
    if norm_len > 1e-6:
        vec_z = normal / norm_len
    else:
        vec_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # X axis: cross product of vec_y x vec_z
    vec_x = np.cross(vec_y, vec_z)
    norm_x = float(np.linalg.norm(vec_x))
    if norm_x > 1e-6:
        vec_x = vec_x / norm_x
    else:
        vec_x = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # Transform all landmarks into local orthonormal canonical coordinates
    # local_pts[i] = [ (pts[i] - wrist) . vec_x, (pts[i] - wrist) . vec_y, (pts[i] - wrist) . vec_z ] / scale
    rel_pts = (pts - wrist) / scale
    local_pts = np.zeros((21, 3), dtype=np.float32)
    local_pts[:, 0] = np.dot(rel_pts, vec_x)
    local_pts[:, 1] = np.dot(rel_pts, vec_y)
    local_pts[:, 2] = np.dot(rel_pts, vec_z)

    feat = np.zeros(NUM_CANONICAL_FEATURES, dtype=np.float32)

    # Anatomical Keypoints:
    # Tips: 4 (thumb), 8 (index), 12 (middle), 16 (ring), 20 (pinky)
    # MCPs: 2 (thumb), 5 (index), 9 (middle), 13 (ring), 17 (pinky)
    # PIPs: 3 (thumb IP), 6 (index), 10 (middle), 14 (ring), 18 (pinky)
    tips = [4, 8, 12, 16, 20]
    mcps = [2, 5, 9, 13, 17]
    pips = [3, 6, 10, 14, 18]

    # Palm center approximation
    palm_center = (pts[0] + pts[5] + pts[9] + pts[13] + pts[17]) / 5.0

    # 0-4: 5 Finger Extensions (Tip-to-wrist minus MCP-to-wrist / scale)
    for i, (tip, mcp) in enumerate(zip(tips, mcps)):
        dist_tip_wrist = dist_3d(pts[tip], pts[0])
        dist_mcp_wrist = dist_3d(pts[mcp], pts[0])
        feat[i] = (dist_tip_wrist - dist_mcp_wrist) / scale

    # 5-9: 5 Fingertip to Palm Center Distances
    for i, tip in enumerate(tips):
        feat[5 + i] = dist_3d(pts[tip], palm_center) / scale

    # 10-14: 5 Finger Curl Metrics (Tip-to-MCP distance / scale)
    for i, (tip, mcp) in enumerate(zip(tips, mcps)):
        feat[10 + i] = dist_3d(pts[tip], pts[mcp]) / scale

    # Helper function for cosine angle between vectors
    def cos_angle(u: np.ndarray, v: np.ndarray) -> float:
        mag_u = float(np.linalg.norm(u))
        mag_v = float(np.linalg.norm(v))
        if mag_u < 1e-6 or mag_v < 1e-6:
            return 0.0
        return float(np.clip(np.dot(u, v) / (mag_u * mag_v), -1.0, 1.0))

    # Bone pointing vectors: MCP to Tip
    v_thumb = pts[4] - pts[2]
    v_index = pts[8] - pts[5]
    v_mid = pts[12] - pts[9]
    v_ring = pts[16] - pts[13]
    v_pinky = pts[20] - pts[17]

    # 15-18: 4 Inter-finger Bone Vector Spread Cosines
    feat[15] = cos_angle(v_thumb, v_index)
    feat[16] = cos_angle(v_index, v_mid)
    feat[17] = cos_angle(v_mid, v_ring)
    feat[18] = cos_angle(v_ring, v_pinky)

    # 19-23: Key Inter-fingertip Euclidean Distances
    feat[19] = dist_3d(pts[4], pts[8]) / scale    # Thumb - Index
    feat[20] = dist_3d(pts[4], pts[12]) / scale   # Thumb - Middle
    feat[21] = dist_3d(pts[4], pts[20]) / scale   # Thumb - Pinky
    feat[22] = dist_3d(pts[8], pts[12]) / scale   # Index - Middle
    feat[23] = dist_3d(pts[12], pts[16]) / scale  # Middle - Ring

    # 24-28: Canonical 3D Thumb Coordinates & Opposition
    feat[24] = local_pts[4, 0]  # canonical thumb tip X
    feat[25] = local_pts[4, 1]  # canonical thumb tip Y
    feat[26] = local_pts[4, 2]  # canonical thumb tip Z (elevation over palm)
    feat[27] = dist_3d(pts[4], pts[5]) / scale   # Thumb tip to Index MCP
    feat[28] = dist_3d(pts[4], pts[13]) / scale  # Thumb tip to Ring MCP

    # 29-31: Relative Finger Crossing Metrics ('R' vs 'U' vs 'V')
    # Lateral crossing: In local coordinates, x of tip8 relative to tip12 compared to MCP offset
    delta_x_tips = local_pts[8, 0] - local_pts[12, 0]
    delta_x_mcps = local_pts[5, 0] - local_pts[9, 0]
    feat[29] = delta_x_tips / (delta_x_mcps if abs(delta_x_mcps) > 1e-4 else 1.0)
    feat[30] = local_pts[8, 2] - local_pts[12, 2]  # Depth crossing difference
    cross_2d = v_index[0] * v_mid[1] - v_index[1] * v_mid[0]
    feat[31] = float(np.sign(cross_2d)) if abs(cross_2d) > 1e-5 else 0.0

    # 32-34: Canonical Direction Vectors (Index bone in local frame)
    idx_bone_local = local_pts[8] - local_pts[5]
    idx_mag = float(np.linalg.norm(idx_bone_local))
    if idx_mag > 1e-6:
        feat[32] = idx_bone_local[0] / idx_mag
        feat[33] = idx_bone_local[1] / idx_mag
        feat[34] = idx_bone_local[2] / idx_mag
    else:
        feat[32:35] = 0.0

    # 35-37: Palm Curl & Fist Compactness
    # PIP angles: MCP - PIP - DIP
    v_idx_pip1 = pts[5] - pts[6]
    v_idx_pip2 = pts[8] - pts[6]
    feat[35] = cos_angle(v_idx_pip1, v_idx_pip2)

    v_mid_pip1 = pts[9] - pts[10]
    v_mid_pip2 = pts[12] - pts[10]
    feat[36] = cos_angle(v_mid_pip1, v_mid_pip2)

    # Fist compactness: average distance of all 5 fingertips to mid MCP (joint 9)
    avg_tip_dist = sum(dist_3d(pts[tip], pts[9]) for tip in tips) / (5.0 * scale)
    feat[37] = float(avg_tip_dist)

    # 38-41: Structural Invariant Ratios
    # Spread ratio index-middle
    feat[38] = dist_3d(pts[8], pts[12]) / max(0.01, dist_3d(pts[5], pts[9]))
    # Y-span ratio: thumb-pinky tip distance / palm scale
    feat[39] = dist_3d(pts[4], pts[20]) / scale
    # Parallelism: absolute cosine between index and middle pointing vectors
    feat[40] = cos_angle(v_index, v_mid)
    # Loop closure: thumb tip to index tip
    feat[41] = dist_3d(pts[4], pts[8]) / scale

    return feat
