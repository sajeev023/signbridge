import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any

class LandmarkExtractor:
    """
    Extracts and normalizes hand and upper-body pose landmarks.
    Ensures scale and translation invariance so recognition remains reliable
    regardless of user distance or position relative to the camera.
    """
    def __init__(self):
        self.num_left_hand = 21
        self.num_right_hand = 21
        self.num_pose = 9
        self.coords_per_point = 3
        self.total_features = (self.num_left_hand + self.num_right_hand + self.num_pose) * self.coords_per_point  # 153

        # MediaPipe Holistic pose landmark indices of interest:
        # 0: nose, 11: left_shoulder, 12: right_shoulder, 13: left_elbow, 14: right_elbow,
        # 15: left_wrist, 16: right_wrist, 23: left_hip, 24: right_hip
        self.pose_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24]

        # Lazy initialized MediaPipe Holistic for server-side frame processing
        self._mp_holistic = None

    def _get_mp_holistic(self):
        if self._mp_holistic is None:
            import mediapipe as mp
            self._mp_holistic = mp.solutions.holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=1
            )
        return self._mp_holistic

    def normalize_landmarks(
        self,
        left_hand: Optional[List[Dict[str, float]]],
        right_hand: Optional[List[Dict[str, float]]],
        pose: Optional[List[Dict[str, float]]]
    ) -> np.ndarray:
        """
        Normalizes landmark coordinates for spatial scale and translation invariance.
        
        Args:
            left_hand: list of 21 dicts with keys 'x', 'y', 'z' (or None)
            right_hand: list of 21 dicts with keys 'x', 'y', 'z' (or None)
            pose: list of pose dicts with 'x', 'y', 'z' (or None)
            
        Returns:
            np.ndarray of shape (153,) with normalized coordinates.
        """
        lh_arr = np.zeros((self.num_left_hand, 3), dtype=np.float32)
        rh_arr = np.zeros((self.num_right_hand, 3), dtype=np.float32)
        pose_arr = np.zeros((self.num_pose, 3), dtype=np.float32)

        has_lh = left_hand is not None and len(left_hand) == 21
        has_rh = right_hand is not None and len(right_hand) == 21
        has_pose = pose is not None and len(pose) >= 25

        if has_lh:
            for i, p in enumerate(left_hand):
                lh_arr[i] = [p.get('x', 0.0), p.get('y', 0.0), p.get('z', 0.0)]

        if has_rh:
            for i, p in enumerate(right_hand):
                rh_arr[i] = [p.get('x', 0.0), p.get('y', 0.0), p.get('z', 0.0)]

        if has_pose:
            for idx, p_idx in enumerate(self.pose_indices):
                if p_idx < len(pose):
                    pt = pose[p_idx]
                    pose_arr[idx] = [pt.get('x', 0.0), pt.get('y', 0.0), pt.get('z', 0.0)]

        # Determine reference anchor center and scale factor
        # Preferred: midpoint of left and right shoulders (indices 1 & 2 in pose_arr)
        if has_pose and np.linalg.norm(pose_arr[1] - pose_arr[2]) > 0.02:
            center = (pose_arr[1] + pose_arr[2]) / 2.0
            scale = np.linalg.norm(pose_arr[1] - pose_arr[2])
        elif has_lh and has_rh:
            # Fallback to midpoint of both wrists
            center = (lh_arr[0] + rh_arr[0]) / 2.0
            scale = max(np.linalg.norm(lh_arr[0] - rh_arr[0]), 0.1)
        elif has_rh:
            # Single right hand: center on wrist, scale on palm span
            center = rh_arr[0]
            scale = max(np.linalg.norm(rh_arr[9] - rh_arr[0]), 0.08)
        elif has_lh:
            center = lh_arr[0]
            scale = max(np.linalg.norm(lh_arr[9] - lh_arr[0]), 0.08)
        else:
            center = np.zeros(3, dtype=np.float32)
            scale = 1.0

        scale = max(float(scale), 1e-4)

        # Normalize relative to center and scale
        if has_lh:
            lh_arr = (lh_arr - center) / scale
        if has_rh:
            rh_arr = (rh_arr - center) / scale
        if has_pose:
            pose_arr = (pose_arr - center) / scale

        # Flatten into 153 features
        feature_vector = np.concatenate([
            lh_arr.flatten(),
            rh_arr.flatten(),
            pose_arr.flatten()
        ]).astype(np.float32)

        return feature_vector

    def process_image_frame(self, bgr_frame: np.ndarray) -> np.ndarray:
        """
        Extracts landmarks directly from an OpenCV BGR image frame.
        """
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        holistic = self._get_mp_holistic()
        results = holistic.process(rgb_frame)

        lh = None
        if results.left_hand_landmarks:
            lh = [{'x': p.x, 'y': p.y, 'z': p.z} for p in results.left_hand_landmarks.landmark]

        rh = None
        if results.right_hand_landmarks:
            rh = [{'x': p.x, 'y': p.y, 'z': p.z} for p in results.right_hand_landmarks.landmark]

        pose = None
        if results.pose_landmarks:
            pose = [{'x': p.x, 'y': p.y, 'z': p.z} for p in results.pose_landmarks.landmark]

        return self.normalize_landmarks(lh, rh, pose)
