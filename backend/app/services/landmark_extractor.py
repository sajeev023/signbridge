import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2

class LandmarkExtractor:
    """
    Extracts and normalizes hand and upper-body pose landmarks.
    Ensures scale and translation invariance so recognition remains reliable
    regardless of user distance or position relative to the camera.
    
    Supports:
    1. Direct landmark dictionary normalization (from browser client MediaPipe stream)
    2. Real-time video frame decoding via MediaPipe Tasks Vision (HandLandmarker & PoseLandmarker)
    """
    def __init__(self):
        self.num_left_hand = 21
        self.num_right_hand = 21
        self.num_pose = 9
        self.coords_per_point = 3
        self.total_features = (self.num_left_hand + self.num_right_hand + self.num_pose) * self.coords_per_point  # 153

        # MediaPipe pose landmark indices:
        # 0: nose, 11: left_shoulder, 12: right_shoulder, 13: left_elbow, 14: right_elbow,
        # 15: left_wrist, 16: right_wrist, 23: left_hip, 24: right_hip
        self.pose_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24]

        # Model asset paths for server-side video processing
        self._tasks_dir = Path(__file__).resolve().parent.parent / "models" / "tasks"
        self._hand_model_path = self._tasks_dir / "hand_landmarker.task"
        self._pose_model_path = self._tasks_dir / "pose_landmarker.task"

        self._hand_detector = None
        self._pose_detector = None

    def _ensure_models_loaded(self):
        if self._hand_detector is not None and self._pose_detector is not None:
            return

        import mediapipe as mp
        from mediapipe.tasks.python import vision, BaseOptions

        if not self._hand_model_path.exists() or not self._pose_model_path.exists():
            # Lazy download if missing
            self._tasks_dir.mkdir(parents=True, exist_ok=True)
            import urllib.request
            if not self._hand_model_path.exists():
                urllib.request.urlretrieve(
                    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                    str(self._hand_model_path)
                )
            if not self._pose_model_path.exists():
                urllib.request.urlretrieve(
                    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
                    str(self._pose_model_path)
                )

        hand_options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self._hand_model_path)),
            num_hands=2,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.35
        )
        self._hand_detector = vision.HandLandmarker.create_from_options(hand_options)

        pose_options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self._pose_model_path)),
            min_pose_detection_confidence=0.35,
            min_pose_presence_confidence=0.35,
            min_tracking_confidence=0.35
        )
        self._pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

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
        Extracts and normalizes landmarks directly from an OpenCV BGR video frame.
        """
        self._ensure_models_loaded()
        import mediapipe as mp

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        hand_res = self._hand_detector.detect(mp_image)
        pose_res = self._pose_detector.detect(mp_image)

        lh_dict = None
        rh_dict = None

        if hand_res and hand_res.hand_landmarks:
            for idx, hand_landmarks in enumerate(hand_res.hand_landmarks):
                # Check handedness if available
                handedness = 'Right'
                if idx < len(hand_res.handedness) and len(hand_res.handedness[idx]) > 0:
                    handedness = hand_res.handedness[idx][0].category_name

                points = [{'x': p.x, 'y': p.y, 'z': p.z} for p in hand_landmarks]
                if handedness == 'Left':
                    lh_dict = points
                else:
                    rh_dict = points

        pose_dict = None
        if pose_res and pose_res.pose_landmarks and len(pose_res.pose_landmarks) > 0:
            pose_dict = [{'x': p.x, 'y': p.y, 'z': p.z} for p in pose_res.pose_landmarks[0]]

        return self.normalize_landmarks(lh_dict, rh_dict, pose_dict)
