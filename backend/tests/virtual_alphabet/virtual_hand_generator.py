"""
SIGNBRIDGE Virtual Alphabet Laboratory — Hand Landmark Generator
Generates anatomically grounded, kinematically valid 21-point MediaPipe hand landmark
structures for all 26 ASL alphabet letters (A through Z) across 12 controlled variations.

All generated samples are strictly marked with:
  isSynthetic: True
  is_physically_captured: False
These samples must NEVER enter physical-human benchmark datasets.
"""

import math
import copy
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

from backend.build_alphabet_dataset import generate_anatomical_hand, create_rotation_matrix
from backend.collect_trajectory_dataset import generate_j_trajectory_points, generate_z_trajectory_points

ALL_26_LETTERS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z"
]

STATIC_24_LETTERS = [c for c in ALL_26_LETTERS if c not in ("J", "Z")]
DYNAMIC_2_LETTERS = ["J", "Z"]

# Camera frame center for canonical rendering
CAMERA_CENTER = np.array([0.5, 0.7, 0.0], dtype=np.float32)


def get_base_hand_points(letter: str, scale: float = 0.15, motor_noise: float = 0.0) -> np.ndarray:
    """
    Returns 21x3 numpy array in hand local coordinates (-y is up) based on ASL kinematics.
    For J, returns the starting 'I' pose (pinky extended).
    For Z, returns the starting 'D' pose (index extended).
    """
    if letter == "J":
        static_sign = "I"
    elif letter == "Z":
        static_sign = "D"
    else:
        static_sign = letter

    # Base scale 1.0, finger ratio 1.0
    pts = generate_anatomical_hand(static_sign, participant_scale=1.0, finger_length_ratio=1.0, motor_noise_std=motor_noise)
    # Scale points to camera coordinate range (wrist to mid MCP ~ scale)
    wrist = pts[0]
    mid_mcp = pts[9]
    current_scale = float(np.linalg.norm(mid_mcp - wrist))
    if current_scale > 1e-5:
        pts = (pts / current_scale) * scale
    return pts


def to_mediapipe_landmarks(pts: np.ndarray, visibility: Optional[List[float]] = None) -> List[Dict[str, float]]:
    """Converts 21x3 coordinates to MediaPipe landmark dictionaries."""
    lms = []
    for i in range(len(pts)):
        p = pts[i]
        vis = visibility[i] if visibility is not None and i < len(visibility) else 1.0
        lms.append({
            "x": float(p[0]),
            "y": float(p[1]),
            "z": float(p[2]),
            "visibility": float(vis)
        })
    return lms


class VirtualHandGenerator:
    """
    Deterministic virtual hand generator producing valid MediaPipe hand landmarks.
    """
    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed

    def generate_sample(
        self,
        letter: str,
        variation: str = "canonical",
        seed_offset: int = 0
    ) -> Dict[str, Any]:
        """
        Generates a single virtual sample for a given letter and variation type.
        Supported variations:
        - canonical
        - small_rotation
        - moderate_rotation
        - scale_variation
        - translation
        - palm_orientation
        - left_hand
        - right_hand
        - small_noise
        - moderate_noise
        - partial_occlusion
        - invalid_configuration
        """
        seed = self.base_seed + seed_offset
        rng = np.random.RandomState(seed)

        quality_status = "VALID"
        transform_params: Dict[str, Any] = {"variation": variation, "seed": seed}

        # 1. Base anatomical points (right hand, scale ~0.15)
        base_scale = 0.15
        motor_noise = 0.005 if variation in ("small_noise", "moderate_noise") else 0.0
        pts = get_base_hand_points(letter, scale=base_scale, motor_noise=motor_noise)

        # Base rotation
        pitch, yaw, roll = 0.0, 0.0, 0.0
        scale_factor = 1.0
        shift = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        visibilities = [1.0] * 21
        is_left = False

        if variation == "canonical":
            # Pure canonical, centered at CAMERA_CENTER
            pass

        elif variation == "small_rotation":
            pitch = float(rng.uniform(-0.15, 0.15))   # ~ +-8 deg
            yaw = float(rng.uniform(-0.15, 0.15))
            roll = float(rng.uniform(-0.15, 0.15))
            transform_params.update({"pitch": pitch, "yaw": yaw, "roll": roll})

        elif variation == "moderate_rotation":
            pitch = float(rng.uniform(-0.35, 0.35))   # ~ +-20 deg
            yaw = float(rng.uniform(-0.35, 0.35))
            roll = float(rng.uniform(-0.30, 0.30))
            transform_params.update({"pitch": pitch, "yaw": yaw, "roll": roll})

        elif variation == "scale_variation":
            scale_factor = float(rng.choice([0.75, 0.90, 1.15, 1.35]))
            transform_params.update({"scale_factor": scale_factor})

        elif variation == "translation":
            shift = np.array([
                rng.uniform(-0.14, 0.14),
                rng.uniform(-0.14, 0.14),
                rng.uniform(-0.05, 0.05)
            ], dtype=np.float32)
            transform_params.update({"shift": shift.tolist()})

        elif variation == "palm_orientation":
            # Tilt away from camera (perspective yaw/pitch)
            yaw = float(rng.uniform(0.30, 0.55)) * float(rng.choice([-1, 1]))
            pitch = float(rng.uniform(-0.25, 0.25))
            transform_params.update({"yaw": yaw, "pitch": pitch})

        elif variation == "left_hand":
            is_left = True
            transform_params.update({"is_left_hand": True})

        elif variation == "right_hand":
            transform_params.update({"is_left_hand": False})

        elif variation == "small_noise":
            noise = rng.normal(0, 0.005, size=pts.shape).astype(np.float32)
            pts += noise
            transform_params.update({"noise_std": 0.005})

        elif variation == "moderate_noise":
            noise = rng.normal(0, 0.018, size=pts.shape).astype(np.float32)
            pts += noise
            quality_status = "DEGRADED"
            transform_params.update({"noise_std": 0.018})

        elif variation == "partial_occlusion":
            quality_status = "DEGRADED"
            # Occlude thumb tip (4) and middle tip (12)
            visibilities[4] = 0.15
            visibilities[12] = 0.10
            # Jitter occluded position slightly due to depth uncertainty
            pts[4] += rng.normal(0, 0.04, size=3).astype(np.float32)
            transform_params.update({"occluded_joints": [4, 12]})

        elif variation == "invalid_configuration":
            quality_status = "INVALID"
            mode = rng.choice(["nan", "zeros", "collapsed", "truncated"])
            transform_params.update({"invalid_mode": mode})
            if mode == "nan":
                pts[8, 0] = np.nan
            elif mode == "zeros":
                pts[:] = 0.0
            elif mode == "collapsed":
                # All points collapse to wrist
                pts[:] = pts[0]
            elif mode == "truncated":
                pts = pts[:12] # only 12 points instead of 21

        # Apply transformations
        if quality_status != "INVALID" or (pts.shape[0] == 21 and not np.isnan(pts).any()):
            # Rotate
            if pitch != 0.0 or yaw != 0.0 or roll != 0.0:
                R = create_rotation_matrix(pitch, yaw, roll)
                pts = pts @ R.T

            # Scale
            if scale_factor != 1.0:
                pts *= scale_factor

            # Handedness mirroring
            if is_left:
                pts[:, 0] = -pts[:, 0]

            # Translate to camera plane center
            pts += CAMERA_CENTER + shift

        landmarks = to_mediapipe_landmarks(pts, visibilities if len(pts) == 21 else None)

        sample = {
            "letter": letter,
            "generation_method": f"virtual_hand_generator_{variation}",
            "source_configuration": "ASL_MANUAL_ALPHABET_KINEMATICS",
            "transformation_params": transform_params,
            "random_seed": seed,
            "landmark_quality_status": quality_status,
            "isSynthetic": True,
            "is_physically_captured": False,
            "landmarks": landmarks
        }

        # If dynamic letter (J or Z), attach canonical trajectory sequence
        if letter in ("J", "Z"):
            scale = base_scale * scale_factor
            if letter == "J":
                traj = generate_j_trajectory_points(
                    duration_sec=1.0,
                    palm_scale=scale,
                    noise_std=0.002 if quality_status == "VALID" else 0.02
                )
            else:
                traj = generate_z_trajectory_points(
                    duration_sec=1.0,
                    palm_scale=scale,
                    noise_std=0.002 if quality_status == "VALID" else 0.02
                )
            sample["trajectory"] = traj

        return sample

    def generate_letter_suite(self, letter: str) -> List[Dict[str, Any]]:
        """Generates all 12 canonical variations for the specified letter."""
        variations = [
            "canonical",
            "small_rotation",
            "moderate_rotation",
            "scale_variation",
            "translation",
            "palm_orientation",
            "left_hand",
            "right_hand",
            "small_noise",
            "moderate_noise",
            "partial_occlusion",
            "invalid_configuration"
        ]
        samples = []
        for i, var in enumerate(variations):
            samples.append(self.generate_sample(letter, variation=var, seed_offset=i * 10))
        return samples

    def generate_full_26_alphabet_suite(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generates the comprehensive virtual evaluation set across all 26 letters."""
        suite = {}
        for letter in ALL_26_LETTERS:
            suite[letter] = self.generate_letter_suite(letter)
        return suite
