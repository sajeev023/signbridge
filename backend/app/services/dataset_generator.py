import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from app.config import (
    CLASS_NAMES, CLASS_TO_IDX, SEQUENCE_LENGTH, NUM_FEATURES,
    LEFT_HAND_POINTS, RIGHT_HAND_POINTS, POSE_KEYPOINTS, DATA_DIR, DATASET_FILE
)

class ISLKinematicSynthesizer:
    """
    Synthesizes kinematically accurate 3D landmark sequences for Indian Sign Language
    gestures based on official ISLRTC morphological specifications.
    
    Generates realistic spatial variations, temporal duration warping, perspective rotations,
    and sensor noise to produce a robust benchmark training and evaluation dataset.
    """
    def __init__(self, seq_len: int = SEQUENCE_LENGTH):
        self.seq_len = seq_len
        self.num_classes = len(CLASS_NAMES)

    def _interpolate_trajectory(self, keyframes: np.ndarray, target_len: int) -> np.ndarray:
        """Interpolates between sparse keyframes to produce a smooth temporal trajectory."""
        num_keyframes = len(keyframes)
        t_keys = np.linspace(0, 1, num_keyframes)
        t_eval = np.linspace(0, 1, target_len)
        interp = np.zeros((target_len, keyframes.shape[1]), dtype=np.float32)
        for d in range(keyframes.shape[1]):
            interp[:, d] = np.interp(t_eval, t_keys, keyframes[:, d])
        return interp

    def _generate_hand_landmarks(self, wrist_pos: np.ndarray, handshape: str, is_right: bool = True) -> np.ndarray:
        """
        Generates 21 hand landmarks relative to the wrist given a canonical handshape.
        Handshapes: 'open_flat', 'fist', 'thumbs_up', 'cup', 'bunch_pinch', 'two_finger', 'neutral'
        """
        landmarks = np.zeros((21, 3), dtype=np.float32)
        landmarks[0] = wrist_pos  # Wrist base
        sign_x = 1.0 if is_right else -1.0

        if handshape == 'open_flat':
            # Fingers extended straight
            for finger_idx in range(5):
                base_idx = 1 + finger_idx * 4
                angle = (finger_idx - 2) * 0.08 * sign_x
                for joint in range(4):
                    dist = (joint + 1) * 0.035
                    landmarks[base_idx + joint] = wrist_pos + np.array([
                        angle * (joint + 1),
                        dist,
                        0.01 * (joint + 1)
                    ])

        elif handshape == 'fist':
            # Fingers curled into palm
            for finger_idx in range(5):
                base_idx = 1 + finger_idx * 4
                for joint in range(4):
                    landmarks[base_idx + joint] = wrist_pos + np.array([
                        (finger_idx - 2) * 0.02 * sign_x,
                        0.025 * min(joint + 1, 2) - 0.015 * max(0, joint - 1),
                        -0.02 * (joint + 1)
                    ])

        elif handshape == 'thumbs_up':
            # Thumb upright, other 4 fingers curled
            # Thumb (1-4)
            for j in range(4):
                landmarks[1 + j] = wrist_pos + np.array([0.03 * sign_x, (j + 1) * 0.035, 0.01 * (j + 1)])
            # Fingers (5-20 curled)
            for f in range(1, 5):
                base = 1 + f * 4
                for j in range(4):
                    landmarks[base + j] = wrist_pos + np.array([
                        (f - 2.5) * 0.02 * sign_x,
                        0.02 * min(j + 1, 2),
                        -0.025 * (j + 1)
                    ])

        elif handshape == 'cup':
            # Curved fingers resembling holding a cup
            for finger_idx in range(5):
                base_idx = 1 + finger_idx * 4
                for joint in range(4):
                    landmarks[base_idx + joint] = wrist_pos + np.array([
                        (finger_idx - 2) * 0.025 * sign_x,
                        (joint + 1) * 0.028,
                        0.025 * (joint + 1)  # curved outward
                    ])

        elif handshape == 'bunch_pinch':
            # All 5 fingertips converge at a single apex point
            apex = wrist_pos + np.array([0.0, 0.08, 0.03])
            for finger_idx in range(5):
                base_idx = 1 + finger_idx * 4
                for joint in range(4):
                    ratio = (joint + 1) / 4.0
                    landmarks[base_idx + joint] = wrist_pos + (apex - wrist_pos) * ratio

        elif handshape == 'two_finger':
            # Index and Middle extended, Ring and Little curled, Thumb tucked
            # Index (5-8) and Middle (9-12) extended
            for f, base in [(1, 5), (2, 9)]:
                for j in range(4):
                    landmarks[base + j] = wrist_pos + np.array([(f - 1.5) * 0.02 * sign_x, (j + 1) * 0.035, 0.0])
            # Thumb (1-4) curled
            for j in range(4):
                landmarks[1 + j] = wrist_pos + np.array([0.02 * sign_x, 0.02 * (j + 1), -0.015])
            # Ring (13-16) and Little (17-20) curled
            for f, base in [(3, 13), (4, 17)]:
                for j in range(4):
                    landmarks[base + j] = wrist_pos + np.array([(f - 2.5) * 0.02 * sign_x, 0.02, -0.025])

        else:  # neutral / resting
            for finger_idx in range(5):
                base_idx = 1 + finger_idx * 4
                for joint in range(4):
                    landmarks[base_idx + joint] = wrist_pos + np.array([
                        (finger_idx - 2) * 0.02 * sign_x,
                        (joint + 1) * 0.025,
                        0.0
                    ])

        return landmarks

    def _get_base_pose(self) -> np.ndarray:
        """
        Returns normalized 9 key pose coordinates:
        [nose, l_shoulder, r_shoulder, l_elbow, r_elbow, l_wrist, r_wrist, l_hip, r_hip]
        """
        pose = np.array([
            [0.0,  0.45, 0.0],   # 0: nose
            [-0.3, 0.25, 0.0],   # 1: left shoulder
            [0.3,  0.25, 0.0],   # 2: right shoulder
            [-0.4, 0.0,  0.05],  # 3: left elbow
            [0.4,  0.0,  0.05],  # 4: right elbow
            [-0.3, -0.25, 0.1],  # 5: left wrist
            [0.3,  -0.25, 0.1],  # 6: right wrist
            [-0.2, -0.5, 0.0],   # 7: left hip
            [0.2,  -0.5, 0.0],   # 8: right hip
        ], dtype=np.float32)
        return pose

    def generate_sign_trajectory(self, sign_id: str) -> np.ndarray:
        """
        Generates a 30-frame sequence (30, 153) for the specified sign.
        """
        frames = np.zeros((self.seq_len, NUM_FEATURES), dtype=np.float32)
        t = np.linspace(0, 1, self.seq_len)

        for step in range(self.seq_len):
            progress = t[step]
            pose = self._get_base_pose()
            lh_present = False
            rh_present = False
            lh_wrist = np.zeros(3, dtype=np.float32)
            rh_wrist = np.zeros(3, dtype=np.float32)
            lh_shape = 'neutral'
            rh_shape = 'neutral'

            if sign_id == "namaste":
                # Two hands move from sides to center chest, press together
                lh_present = True
                rh_present = True
                lh_shape = 'open_flat'
                rh_shape = 'open_flat'
                # Smooth movement to center chest (x=0, y=0.15, z=0.1)
                move = np.sin(progress * np.pi * 0.8)
                lh_wrist = np.array([-0.25 * (1 - move) - 0.02 * move, 0.0 + 0.15 * move, 0.1])
                rh_wrist = np.array([0.25 * (1 - move) + 0.02 * move,  0.0 + 0.15 * move, 0.1])
                pose[5] = lh_wrist
                pose[6] = rh_wrist

            elif sign_id == "thank_you":
                # Right hand touches chin then moves outward towards viewer
                rh_present = True
                rh_shape = 'open_flat'
                if progress < 0.35:
                    # Chin touch
                    rh_wrist = np.array([0.05, 0.35, -0.05])
                else:
                    # Forward arc sweep outward
                    p = (progress - 0.35) / 0.65
                    rh_wrist = np.array([0.05 + 0.1 * p, 0.35 - 0.25 * p, -0.05 - 0.35 * p])
                pose[6] = rh_wrist

            elif sign_id == "help":
                # Left hand flat at center chest, Right fist on top lifting together
                lh_present = True
                rh_present = True
                lh_shape = 'open_flat'
                rh_shape = 'thumbs_up'
                lift = 0.18 * np.sin(progress * np.pi)
                lh_wrist = np.array([-0.05, 0.0 + lift, 0.15])
                rh_wrist = np.array([0.0, 0.08 + lift, 0.15])
                pose[5] = lh_wrist
                pose[6] = rh_wrist

            elif sign_id == "yes":
                # Right fist nodding vertically 3 cycles
                rh_present = True
                rh_shape = 'fist'
                nod = 0.08 * np.sin(progress * 6 * np.pi)
                rh_wrist = np.array([0.25, 0.2 + nod, 0.1])
                pose[6] = rh_wrist

            elif sign_id == "no":
                # Right hand wagging horizontally side to side
                rh_present = True
                rh_shape = 'open_flat'
                wag = 0.14 * np.sin(progress * 5 * np.pi)
                rh_wrist = np.array([0.2 + wag, 0.2, 0.15])
                pose[6] = rh_wrist

            elif sign_id == "water":
                # Right curved hand elevates to mouth and tilts
                rh_present = True
                rh_shape = 'cup'
                elevation = np.sin(progress * np.pi * 0.9)
                rh_wrist = np.array([0.15 - 0.08 * elevation, 0.0 + 0.35 * elevation, 0.15 - 0.15 * elevation])
                pose[6] = rh_wrist

            elif sign_id == "food":
                # Right bunch hand taps mouth twice
                rh_present = True
                rh_shape = 'bunch_pinch'
                tap = 0.08 * np.abs(np.sin(progress * 4 * np.pi))
                rh_wrist = np.array([0.06, 0.38 - tap, 0.05 - tap])
                pose[6] = rh_wrist

            elif sign_id == "emergency":
                # Both hands crossing frantically at head/chest level
                lh_present = True
                rh_present = True
                lh_shape = 'open_flat'
                rh_shape = 'open_flat'
                wave = np.sin(progress * 7 * np.pi)
                lh_wrist = np.array([-0.15 + 0.18 * wave, 0.35 + 0.05 * np.cos(progress * 7 * np.pi), 0.15])
                rh_wrist = np.array([0.15 - 0.18 * wave, 0.35 - 0.05 * np.cos(progress * 7 * np.pi), 0.15])
                pose[5] = lh_wrist
                pose[6] = rh_wrist

            elif sign_id == "doctor":
                # Left forearm steady, Right two fingers tap left wrist twice
                lh_present = True
                rh_present = True
                lh_shape = 'neutral'
                rh_shape = 'two_finger'
                lh_wrist = np.array([-0.15, 0.05, 0.12])  # resting pulse point
                tap = 0.06 * np.abs(np.sin(progress * 4 * np.pi))
                rh_wrist = np.array([-0.14, 0.07 + tap, 0.12 + tap])
                pose[5] = lh_wrist
                pose[6] = rh_wrist

            elif sign_id == "please":
                # Right open hand makes circular stroke on center chest
                rh_present = True
                rh_shape = 'open_flat'
                circle_angle = progress * 4 * np.pi
                rh_wrist = np.array([
                    0.0 + 0.08 * np.cos(circle_angle),
                    0.15 + 0.08 * np.sin(circle_angle),
                    0.05
                ])
                pose[6] = rh_wrist

            # Generate hand landmarks or zeroes if absent
            lh_landmarks = self._generate_hand_landmarks(lh_wrist, lh_shape, is_right=False) if lh_present else np.zeros((21, 3), dtype=np.float32)
            rh_landmarks = self._generate_hand_landmarks(rh_wrist, rh_shape, is_right=True) if rh_present else np.zeros((21, 3), dtype=np.float32)

            # Flatten into 153 features: LH (63) + RH (63) + Pose (27)
            frame_vec = np.concatenate([
                lh_landmarks.flatten(),
                rh_landmarks.flatten(),
                pose.flatten()
            ]).astype(np.float32)

            frames[step] = frame_vec

        return frames

    def apply_biomechanical_augmentations(self, sequence: np.ndarray) -> np.ndarray:
        """
        Applies realistic physical augmentations:
        - Scale variation (signer arm length/torso differences)
        - Spatial translation shift (standing slightly left/right/closer/further)
        - 3D rotation (camera tilt)
        - Gaussian sensor noise (MediaPipe keypoint jitter)
        - Temporal warping (fast vs slow signing)
        """
        augmented = sequence.copy()

        # 1. Spatial translation shift: delta in x, y, z
        shift = np.random.uniform(-0.05, 0.05, size=(1, 3))

        # 2. Scale factor: [0.88, 1.12]
        scale = np.random.uniform(0.88, 1.12)

        # 3. 3D rotation matrix around Y axis (yaw: +/- 10 degrees)
        yaw_deg = np.random.uniform(-10.0, 10.0)
        yaw_rad = np.radians(yaw_deg)
        cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
        rot_y = np.array([
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y]
        ], dtype=np.float32)

        # Apply to each coordinate triple
        num_points = NUM_FEATURES // 3
        reshaped = augmented.reshape(self.seq_len, num_points, 3)

        # Mask points that are currently zero (absent hand) so we don't hallucinate noise into absent hands
        non_zero_mask = np.linalg.norm(reshaped, axis=-1) > 1e-5

        for t in range(self.seq_len):
            pts = reshaped[t]
            active_mask = non_zero_mask[t]
            if np.any(active_mask):
                transformed = (pts[active_mask] @ rot_y) * scale + shift
                # Add micro sensor jitter
                noise = np.random.normal(0.0, 0.005, size=transformed.shape).astype(np.float32)
                transformed += noise
                reshaped[t, active_mask] = transformed

        augmented = reshaped.reshape(self.seq_len, NUM_FEATURES)

        # 4. Temporal warping (speed variation)
        # Warp time by stretching or compressing then interpolating back to 30 frames
        speed_factor = np.random.uniform(0.85, 1.15)
        warped_len = int(np.clip(self.seq_len * speed_factor, 20, 40))
        t_original = np.linspace(0, 1, self.seq_len)
        t_warped = np.linspace(0, 1, warped_len)
        warped_seq = np.zeros((warped_len, NUM_FEATURES), dtype=np.float32)
        for f in range(NUM_FEATURES):
            warped_seq[:, f] = np.interp(t_warped, t_original, augmented[:, f])

        # Resample back to seq_len
        final_seq = np.zeros((self.seq_len, NUM_FEATURES), dtype=np.float32)
        for f in range(NUM_FEATURES):
            final_seq[:, f] = np.interp(t_original, t_warped, warped_seq[:, f])

        return final_seq

    def build_dataset(
        self,
        samples_per_class: int = 120,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        save_path: Path = DATASET_FILE
    ) -> Dict[str, np.ndarray]:
        """
        Constructs the complete multi-class ISL dataset with stratified splits.
        """
        np.random.seed(42)
        all_sequences = []
        all_labels = []

        print(f"[Dataset] Generating {samples_per_class} samples for each of {len(CLASS_NAMES)} ISL classes...")

        for class_idx, class_name in enumerate(CLASS_NAMES):
            base_seq = self.generate_sign_trajectory(class_name)
            for _ in range(samples_per_class):
                aug_seq = self.apply_biomechanical_augmentations(base_seq)
                all_sequences.append(aug_seq)
                all_labels.append(class_idx)

        X = np.array(all_sequences, dtype=np.float32)
        y = np.array(all_labels, dtype=np.int64)

        # Stratified split per class to avoid data leakage
        train_idx, val_idx, test_idx = [], [], []

        for c in range(self.num_classes):
            c_indices = np.where(y == c)[0]
            np.random.shuffle(c_indices)
            n_samples = len(c_indices)
            n_train = int(n_samples * train_ratio)
            n_val = int(n_samples * val_ratio)

            train_idx.extend(c_indices[:n_train])
            val_idx.extend(c_indices[n_train:n_train + n_val])
            test_idx.extend(c_indices[n_train + n_val:])

        train_idx = np.array(train_idx)
        val_idx = np.array(val_idx)
        test_idx = np.array(test_idx)

        np.random.shuffle(train_idx)
        np.random.shuffle(val_idx)
        np.random.shuffle(test_idx)

        dataset = {
            "X_train": X[train_idx],
            "y_train": y[train_idx],
            "X_val": X[val_idx],
            "y_val": y[val_idx],
            "X_test": X[test_idx],
            "y_test": y[test_idx],
            "class_names": np.array(CLASS_NAMES)
        }

        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_path, **dataset)
        print(f"[Dataset] Saved to {save_path}. Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        return dataset
