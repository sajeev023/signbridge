"""
SIGNBRIDGE Point-History Temporal Engine
Adapted and upgraded from Kinivi MediaPipe Point History Classifier (Apache License 2.0).
Provides palm-scale normalized, distance-invariant, variable-speed resampled
temporal trajectory representation for ASL dynamic alphabet signs 'J' and 'Z'.

Attribution:
Original point-history circular buffer concept by Kazuhito Takahashi (@Kazuhito00)
and Nikita Kiselov (@kinivi) (https://github.com/kinivi/hand-gesture-recognition-mediapipe).
Upgraded for SIGNBRIDGE with 3D palm-scale normalization, arc-length resampling,
and static handshape context gating.
"""

import math
from collections import deque
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

TRAJECTORY_BUFFER_LENGTH = 16  # Canonical 16-frame trajectory window
TRAJECTORY_DIMENSION = 2       # 2D relative displacement in camera plane (dx, dy)
TRAJECTORY_FEATURE_SIZE = TRAJECTORY_BUFFER_LENGTH * TRAJECTORY_DIMENSION  # 32 features


class PointHistoryBuffer:
    """
    Circular trajectory buffer maintaining rolling fingertip positions with
    palm-scale distance invariance, continuity checking, and speed normalization.
    """
    def __init__(self, maxlen: int = TRAJECTORY_BUFFER_LENGTH):
        self.maxlen = maxlen
        self.buffer = deque(maxlen=maxlen)
        self.timestamps = deque(maxlen=maxlen)
        self.palm_scales = deque(maxlen=maxlen)
        self.hand_ids = deque(maxlen=maxlen)

    def append(self, point: Dict[str, float], palm_scale: float, timestamp_ms: float, hand_id: int = 0):
        self.buffer.append({"x": float(point["x"]), "y": float(point["y"]), "z": float(point.get("z", 0.0))})
        self.timestamps.append(timestamp_ms)
        self.palm_scales.append(max(0.035, float(palm_scale)))
        self.hand_ids.append(hand_id)

    def reset(self):
        self.buffer.clear()
        self.timestamps.clear()
        self.palm_scales.clear()
        self.hand_ids.clear()

    def is_continuous(self, max_gap_ms: float = 300.0) -> bool:
        """Verifies no temporal dropout or hand-identity switches during the stroke."""
        if len(self.buffer) < self.maxlen:
            return False
        # Hand ID continuity
        first_id = self.hand_ids[0]
        if any(hid != first_id for hid in self.hand_ids):
            return False
        # Timestamp continuity
        for i in range(1, len(self.timestamps)):
            if (self.timestamps[i] - self.timestamps[i - 1]) > max_gap_ms:
                return False
        return True

    def get_raw_points(self) -> List[Dict[str, float]]:
        return list(self.buffer)

    def extract_normalized_features(self) -> np.ndarray:
        """
        Extracts 32-element palm-scale normalized trajectory vector:
        1. Base origin at stroke contact point (index 0)
        2. Displacements normalized by average palm scale across the stroke
        3. Spatially resampled to exactly 16 uniform arc-length steps
        """
        if len(self.buffer) < 4:
            return np.zeros(TRAJECTORY_FEATURE_SIZE, dtype=np.float32)

        pts = np.array([[p["x"], p["y"]] for p in self.buffer], dtype=np.float32)
        avg_scale = float(np.mean(self.palm_scales))

        # Origin shift: relative to starting point
        rel_pts = (pts - pts[0]) / avg_scale

        # Arc-length spatial resampling to fixed 16 points for variable-speed invariance
        resampled = resample_trajectory_2d(rel_pts, target_points=self.maxlen)

        return resampled.flatten().astype(np.float32)


def resample_trajectory_2d(points: np.ndarray, target_points: int = 16) -> np.ndarray:
    """
    Spatially resamples a 2D trajectory of shape (N, 2) to exactly target_points
    uniform arc-length samples, achieving invariance to drawing speed.
    """
    n = len(points)
    if n == target_points:
        return points

    # Compute cumulative path length
    diffs = np.diff(points, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dist = np.insert(np.cumsum(dists), 0, 0.0)
    total_len = cum_dist[-1]

    if total_len < 1e-4:
        # Stationary point: repeat first coordinate
        return np.tile(points[0], (target_points, 1))

    # Uniform interpolation targets along arc length
    interp_dists = np.linspace(0.0, total_len, target_points)
    resampled_x = np.interp(interp_dists, cum_dist, points[:, 0])
    resampled_y = np.interp(interp_dists, cum_dist, points[:, 1])

    return np.column_stack([resampled_x, resampled_y])


def calculate_trajectory_path_length(points: np.ndarray) -> float:
    """Returns total Euclidean path length of trajectory."""
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def is_j_handshape_context(landmarks: List[Dict[str, float]], palm_scale: float) -> bool:
    """
    Validates anatomical handshape context for ASL 'J':
    Pinky must be extended, while index, middle, and ring fingers are curled.
    """
    if len(landmarks) < 21:
        return False
    scale = max(0.035, palm_scale)

    # Pinky extension (tip 20 to wrist 0 vs MCP 17 to wrist 0)
    pinky_ext = (math.dist([landmarks[20]["x"], landmarks[20]["y"]], [landmarks[0]["x"], landmarks[0]["y"]]) -
                 math.dist([landmarks[17]["x"], landmarks[17]["y"]], [landmarks[0]["x"], landmarks[0]["y"]])) / scale

    # Index and Middle curl (tip to MCP distance / scale)
    idx_curl = math.dist([landmarks[8]["x"], landmarks[8]["y"]], [landmarks[5]["x"], landmarks[5]["y"]]) / scale
    mid_curl = math.dist([landmarks[12]["x"], landmarks[12]["y"]], [landmarks[9]["x"], landmarks[9]["y"]]) / scale

    return bool(pinky_ext > 0.40 and idx_curl < 0.65 and mid_curl < 0.65)


def is_z_handshape_context(landmarks: List[Dict[str, float]], palm_scale: float) -> bool:
    """
    Validates anatomical handshape context for ASL 'Z':
    Index finger must be extended, while middle, ring, and pinky are curled.
    """
    if len(landmarks) < 21:
        return False
    scale = max(0.035, palm_scale)

    # Index extension (tip 8 to wrist 0 vs MCP 5 to wrist 0)
    idx_ext = (math.dist([landmarks[8]["x"], landmarks[8]["y"]], [landmarks[0]["x"], landmarks[0]["y"]]) -
               math.dist([landmarks[5]["x"], landmarks[5]["y"]], [landmarks[0]["x"], landmarks[0]["y"]])) / scale

    # Middle and Pinky curl
    mid_curl = math.dist([landmarks[12]["x"], landmarks[12]["y"]], [landmarks[9]["x"], landmarks[9]["y"]]) / scale
    pky_curl = math.dist([landmarks[20]["x"], landmarks[20]["y"]], [landmarks[17]["x"], landmarks[17]["y"]]) / scale

    return bool(idx_ext > 0.45 and mid_curl < 0.65 and pky_curl < 0.65)
