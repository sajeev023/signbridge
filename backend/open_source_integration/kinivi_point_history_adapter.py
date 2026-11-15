"""
SIGNBRIDGE Open-Source Integration — Kinivi Point History Adapter
Adapts and compares the Point History trajectory representation from Kazuhito Takahashi & Nikita Kiselov
(https://github.com/kinivi/hand-gesture-recognition-mediapipe, Apache License 2.0).

Enhancement:
Upgrades Kinivi's 2D screen-dimension normalization ((x - x0)/W, (y - y0)/H) to 3D palm-scale Euclidean
invariant normalization ((x - x0)/s_palm, (y - y0)/s_palm) with uniform arc-length spatial resampling.
"""

import math
from collections import deque
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

KINIVI_HISTORY_LENGTH = 16
KINIVI_RAW_FEATURE_SIZE = KINIVI_HISTORY_LENGTH * 2  # 32 features


class KiniviPointHistoryAdapter:
    """
    Adapter implementing both Kinivi's raw point-history preprocessing
    and SIGNBRIDGE's palm-scale distance-invariant point-history preprocessing.
    """
    def __init__(self, maxlen: int = KINIVI_HISTORY_LENGTH):
        self.maxlen = maxlen
        self.buffer: deque = deque(maxlen=maxlen)
        self.palm_scales: deque = deque(maxlen=maxlen)
        self.timestamps: deque = deque(maxlen=maxlen)

    def append_point(self, point: Dict[str, float], palm_scale: float, timestamp_ms: float):
        """Appends a 2D/3D fingertip point with its associated palm scale and timestamp."""
        self.buffer.append({"x": float(point["x"]), "y": float(point["y"])})
        self.palm_scales.append(max(0.035, float(palm_scale)))
        self.timestamps.append(float(timestamp_ms))

    def reset(self):
        self.buffer.clear()
        self.palm_scales.clear()
        self.timestamps.clear()

    def is_ready(self) -> bool:
        return len(self.buffer) == self.maxlen

    def extract_kinivi_raw_features(self, image_width: float = 640.0, image_height: float = 480.0) -> np.ndarray:
        """
        Extracts Kinivi's original 32-element feature vector:
        Relative to initial stroke contact point, normalized by image resolution.
        Source: pre_process_point_history() in Kinivi repo.
        """
        if len(self.buffer) < self.maxlen:
            # Pad with zeros if buffer is not full
            padding_len = self.maxlen - len(self.buffer)
            pts = list(self.buffer) + [{"x": 0.0, "y": 0.0}] * padding_len
        else:
            pts = list(self.buffer)

        base_x = pts[0]["x"]
        base_y = pts[0]["y"]

        features = []
        for p in pts:
            # Kinivi raw pixel offset normalized by screen dimensions
            dx = (p["x"] - base_x) / image_width if image_width > 0 else 0.0
            dy = (p["y"] - base_y) / image_height if image_height > 0 else 0.0
            features.extend([dx, dy])

        return np.array(features, dtype=np.float32)

    def extract_signbridge_invariant_features(self) -> np.ndarray:
        """
        Extracts SIGNBRIDGE's enhanced 32-element feature vector:
        1. Base origin at stroke contact point (index 0).
        2. Displacements normalized by average palm Euclidean scale across the stroke (distance-invariant).
        3. Spatially resampled to uniform arc-length intervals (speed-invariant).
        """
        if len(self.buffer) < self.maxlen:
            return np.zeros(KINIVI_RAW_FEATURE_SIZE, dtype=np.float32)

        pts = np.array([[p["x"], p["y"]] for p in self.buffer], dtype=np.float32)
        avg_palm_scale = float(np.mean(self.palm_scales)) if self.palm_scales else 0.10
        avg_palm_scale = max(0.035, avg_palm_scale)

        # Palm-scale normalized displacement relative to initial point
        rel_pts = (pts - pts[0]) / avg_palm_scale

        # Arc-length spatial resampling
        resampled = self._resample_arc_length(rel_pts, target_points=self.maxlen)
        return resampled.flatten().astype(np.float32)

    @staticmethod
    def _resample_arc_length(points: np.ndarray, target_points: int = 16) -> np.ndarray:
        n = len(points)
        if n == target_points:
            return points
        diffs = np.diff(points, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        cum_dist = np.insert(np.cumsum(dists), 0, 0.0)
        total_len = cum_dist[-1]

        if total_len < 1e-4:
            return np.tile(points[0], (target_points, 1))

        interp_dists = np.linspace(0.0, total_len, target_points)
        res_x = np.interp(interp_dists, cum_dist, points[:, 0])
        res_y = np.interp(interp_dists, cum_dist, points[:, 1])
        return np.column_stack([res_x, res_y])
