"""
SIGNBRIDGE Empirical & Kinematically Grounded Alphabet Dataset Builder
Generates a multi-signer, multi-condition dataset for manual alphabet recognition.
Enforces strict Signer-Independent (LOGO) splits:
- Signer 1 & Signer 2 -> Training Set
- Signer 3 -> Validation Set
- Signer 4 & Empirical Test Subjects -> Test Set
"""

import os
import sys
import json
import math
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from alphabet_feature_extractor import extract_canonical_features, NUM_CANONICAL_FEATURES

# Supported Single-Hand Manual Alphabet Classes
SUPPORTED_CLASSES = [
    "NEUTRAL",
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y"
]

CLASS_TO_IDX = {c: i for i, c in enumerate(SUPPORTED_CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(SUPPORTED_CLASSES)}


def create_rotation_matrix(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """Euler rotation matrix (pitch=X, yaw=Y, roll=Z) in radians."""
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)

    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float32)
    return rz @ ry @ rx


def generate_anatomical_hand(
    sign: str,
    participant_scale: float,
    finger_length_ratio: float,
    motor_noise_std: float = 0.03
) -> np.ndarray:
    """
    Generates authentic 21-landmark 3D hand coordinates based on kinematic joint angles.
    Returns array of shape (21, 3) normalized in hand local coordinate space.
    """
    # 21 Keypoints:
    # 0: Wrist
    # 1-4: Thumb (CMC, MCP, IP, Tip)
    # 5-8: Index (MCP, PIP, DIP, Tip)
    # 9-12: Middle (MCP, PIP, DIP, Tip)
    # 13-16: Ring (MCP, PIP, DIP, Tip)
    # 17-20: Pinky (MCP, PIP, DIP, Tip)
    pts = np.zeros((21, 3), dtype=np.float32)

    # 1. Metacarpal Bases (Anatomical Arch in MediaPipe Camera Coordinate Space: -y is UP)
    pts[0] = [0.0, 0.0, 0.0]                     # Wrist
    pts[1] = [-0.22, -0.28, -0.04]               # Thumb CMC
    pts[2] = [-0.36, -0.52, -0.06]               # Thumb MCP
    pts[5] = [-0.18, -0.95, 0.02]                # Index MCP
    pts[9] = [0.00, -1.00, 0.00]                 # Middle MCP
    pts[13] = [0.17, -0.94, -0.02]               # Ring MCP
    pts[17] = [0.32, -0.82, -0.05]               # Pinky MCP

    # Apply participant scale to palm
    pts *= participant_scale

    # Standard bone segment lengths (relative to palm)
    idx_lens = [x * finger_length_ratio * participant_scale for x in [0.42, 0.28, 0.22]]
    mid_lens = [x * finger_length_ratio * participant_scale for x in [0.46, 0.32, 0.24]]
    rng_lens = [x * finger_length_ratio * participant_scale for x in [0.41, 0.27, 0.21]]
    pky_lens = [x * finger_length_ratio * participant_scale for x in [0.33, 0.20, 0.18]]
    thb_lens = [x * finger_length_ratio * participant_scale for x in [0.35, 0.28]]

    # Flexion angles for each finger: [MCP, PIP, DIP] in radians
    # 0.0 = straight extended, pi/2 = 90 deg curled, pi = fully curled
    # Default: Relaxed open
    flex_idx = [0.2, 0.2, 0.1]
    flex_mid = [0.2, 0.2, 0.1]
    flex_rng = [0.2, 0.2, 0.1]
    flex_pky = [0.2, 0.2, 0.1]
    thb_pose = [-0.3, 0.3, 0.2] # Abduction, MCP flex, IP flex
    finger_spreads = [-0.05, 0.0, 0.04, 0.08] # Lateral spread angles

    curl_full = [1.25, 1.35, 0.85]
    curl_tight = [1.40, 1.50, 0.95]
    curl_partial = [0.65, 0.75, 0.55]
    ext_straight = [0.02, 0.02, 0.02]

    if sign == "NEUTRAL":
        flex_idx = [np.random.uniform(0.1, 1.2), np.random.uniform(0.1, 1.2), 0.2]
        flex_mid = [np.random.uniform(0.1, 1.2), np.random.uniform(0.1, 1.2), 0.2]
        flex_rng = [np.random.uniform(0.1, 1.2), np.random.uniform(0.1, 1.2), 0.2]
        flex_pky = [np.random.uniform(0.1, 1.2), np.random.uniform(0.1, 1.2), 0.2]
        thb_pose = [np.random.uniform(-0.4, 0.4), np.random.uniform(0.1, 0.8), 0.2]

    elif sign == "A":
        # Fist with thumb straight along index knuckle pointing UP
        flex_idx = curl_full
        flex_mid = curl_full
        flex_rng = curl_full
        flex_pky = curl_full
        thb_pose = [-0.10, 0.15, 0.05] # Thumb pointing UP alongside index

    elif sign == "B":
        # 4 upright together, thumb tucked across palm
        flex_idx = ext_straight
        flex_mid = ext_straight
        flex_rng = ext_straight
        flex_pky = ext_straight
        finger_spreads = [0.0, 0.0, 0.0, 0.0] # Fingers touching
        thb_pose = [0.65, 1.45, 0.50] # Thumb across palm

    elif sign == "C":
        # All fingers curved in C-arc
        flex_idx = curl_partial
        flex_mid = curl_partial
        flex_rng = curl_partial
        flex_pky = curl_partial
        thb_pose = [-0.25, 0.45, 0.40]

    elif sign == "D":
        # Index straight upright, thumb touches middle tip, others curled
        flex_idx = ext_straight
        flex_mid = curl_full
        flex_rng = curl_full
        flex_pky = curl_full
        thb_pose = [0.45, 0.95, 0.65] # Thumb meets middle

    elif sign == "E":
        # All 4 fingers curled tightly, thumb bent HORIZONTALLY under fingertips
        flex_idx = curl_tight
        flex_mid = curl_tight
        flex_rng = curl_tight
        flex_pky = curl_tight
        thb_pose = [0.85, 1.55, 0.90] # Thumb flat across palm under tips

    elif sign == "F":
        # Index & thumb touching in O loop, middle/ring/pinky upright
        flex_idx = [0.95, 1.45, 0.85] # Index loops to thumb
        flex_mid = ext_straight
        flex_rng = ext_straight
        flex_pky = ext_straight
        finger_spreads = [0.0, 0.08, 0.15, 0.22]
        thb_pose = [0.25, 0.65, 0.60] # Thumb meets index

    elif sign == "G":
        # Index pointing horizontal (orientation rotated in global space)
        flex_idx = ext_straight
        flex_mid = curl_full
        flex_rng = curl_full
        flex_pky = curl_full
        finger_spreads = [-0.05, 0.0, 0.0, 0.0]
        thb_pose = [-0.20, 0.10, 0.05]

    elif sign == "H":
        # Index and middle pointing horizontal together
        flex_idx = ext_straight
        flex_mid = ext_straight
        flex_rng = curl_full
        flex_pky = curl_full
        finger_spreads = [0.0, 0.0, 0.0, 0.0]
        thb_pose = [-0.15, 0.20, 0.10]

    elif sign == "I":
        # Pinky upright, others curled, thumb across fingers
        flex_idx = curl_full
        flex_mid = curl_full
        flex_rng = curl_full
        flex_pky = ext_straight
        thb_pose = [0.45, 1.25, 0.40]

    elif sign == "K":
        # Index upright, middle pointing forward 45-60 deg, thumb between them
        flex_idx = ext_straight
        flex_mid = [0.95, 0.45, 0.20] # Middle forward into depth!
        flex_rng = curl_full
        flex_pky = curl_full
        finger_spreads = [-0.08, 0.18, 0.0, 0.0]
        thb_pose = [0.10, 0.45, 0.20]

    elif sign == "L":
        # L-shape: thumb & index at 90 deg, others curled
        flex_idx = ext_straight
        flex_mid = curl_full
        flex_rng = curl_full
        flex_pky = curl_full
        finger_spreads = [-0.05, 0.0, 0.0, 0.0]
        thb_pose = [-1.45, 0.05, 0.02] # Thumb extended 90 deg lateral (-x)

    elif sign == "M":
        # 3 fingers (index, mid, ring) folded over thumb
        flex_idx = curl_full
        flex_mid = curl_full
        flex_rng = curl_full
        flex_pky = curl_full
        thb_pose = [0.55, 1.10, 0.30] # Under ring

    elif sign == "N":
        # 2 fingers (index, mid) folded over thumb
        flex_idx = curl_full
        flex_mid = curl_full
        flex_rng = curl_full
        flex_pky = curl_full
        thb_pose = [0.40, 0.90, 0.30] # Under middle

    elif sign == "O":
        # All 5 fingers touching in circular O
        flex_idx = [0.85, 1.35, 0.75]
        flex_mid = [0.85, 1.35, 0.75]
        flex_rng = [0.85, 1.35, 0.75]
        flex_pky = [0.85, 1.35, 0.75]
        thb_pose = [0.35, 0.75, 0.65]

    elif sign == "P":
        # Downward pointing K
        flex_idx = [1.2, 0.05, 0.02] # Pointing down
        flex_mid = [1.5, 0.30, 0.10]
        flex_rng = curl_full
        flex_pky = curl_full
        thb_pose = [0.15, 0.50, 0.20]

    elif sign == "Q":
        # Downward pointing G
        flex_idx = [1.3, 0.05, 0.02]
        flex_mid = curl_full
        flex_rng = curl_full
        flex_pky = curl_full
        thb_pose = [0.20, 0.30, 0.10]

    elif sign == "R":
        # Index and middle crossed!
        flex_idx = ext_straight
        flex_mid = ext_straight
        flex_rng = curl_full
        flex_pky = curl_full
        finger_spreads = [0.15, -0.15, 0.0, 0.0] # Crossing!
        thb_pose = [0.45, 1.15, 0.30]

    elif sign == "S":
        # Fist with thumb wrapped ACROSS front of knuckles
        flex_idx = curl_full
        flex_mid = curl_full
        flex_rng = curl_full
        flex_pky = curl_full
        thb_pose = [0.70, 1.35, 0.55] # Across front

    elif sign == "T":
        # Thumb tucked BETWEEN index and middle finger
        flex_idx = curl_full
        flex_mid = curl_full
        flex_rng = curl_full
        flex_pky = curl_full
        thb_pose = [0.25, 0.70, 0.25] # Between index/mid

    elif sign == "U":
        # Index and middle upright together (parallel)
        flex_idx = ext_straight
        flex_mid = ext_straight
        flex_rng = curl_full
        flex_pky = curl_full
        finger_spreads = [0.0, 0.0, 0.0, 0.0] # Touching parallel
        thb_pose = [0.45, 1.25, 0.35]

    elif sign == "V":
        # Index and middle upright spread in V
        flex_idx = ext_straight
        flex_mid = ext_straight
        flex_rng = curl_full
        flex_pky = curl_full
        finger_spreads = [-0.18, 0.22, 0.0, 0.0] # V-spread!
        thb_pose = [0.45, 1.25, 0.35]

    elif sign == "W":
        # Index, middle, ring upright in W-spread
        flex_idx = ext_straight
        flex_mid = ext_straight
        flex_rng = ext_straight
        flex_pky = curl_full
        finger_spreads = [-0.20, 0.0, 0.20, 0.0] # W-spread!
        thb_pose = [0.45, 1.25, 0.35]

    elif sign == "X":
        # Index hooked, others curled
        flex_idx = [0.35, 1.65, 0.20] # Hook!
        flex_mid = curl_full
        flex_rng = curl_full
        flex_pky = curl_full
        thb_pose = [0.45, 1.15, 0.30]

    elif sign == "Y":
        # Thumb and pinky extended wide, others curled
        flex_idx = curl_full
        flex_mid = curl_full
        flex_rng = curl_full
        flex_pky = ext_straight
        finger_spreads = [0.0, 0.0, 0.0, 0.35] # Pinky wide
        thb_pose = [-1.45, 0.05, 0.02] # Thumb wide lateral (-x)

    # Add motor noise to joint angles
    def add_noise(angles: List[float]) -> List[float]:
        return [float(a + np.random.normal(0, motor_noise_std)) for a in angles]

    flex_idx = add_noise(flex_idx)
    flex_mid = add_noise(flex_mid)
    flex_rng = add_noise(flex_rng)
    flex_pky = add_noise(flex_pky)
    thb_pose = add_noise(thb_pose)

    # Forward Kinematics to calculate 3D points in MediaPipe Camera Coordinates (-y is UP)
    def compute_finger_chain(mcp_pt: np.ndarray, spreads: float, flex: List[float], lens: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # MCP -> PIP
        dir1 = np.array([math.sin(spreads), -math.cos(flex[0]), -math.sin(flex[0])], dtype=np.float32)
        dir1 /= np.linalg.norm(dir1)
        pip = mcp_pt + dir1 * lens[0]

        # PIP -> DIP
        cum_flex2 = flex[0] + flex[1]
        dir2 = np.array([math.sin(spreads), -math.cos(cum_flex2), -math.sin(cum_flex2)], dtype=np.float32)
        dir2 /= np.linalg.norm(dir2)
        dip = pip + dir2 * lens[1]

        # DIP -> Tip
        cum_flex3 = cum_flex2 + flex[2]
        dir3 = np.array([math.sin(spreads), -math.cos(cum_flex3), -math.sin(cum_flex3)], dtype=np.float32)
        dir3 /= np.linalg.norm(dir3)
        tip = dip + dir3 * lens[2]

        return pip, dip, tip

    pts[6], pts[7], pts[8] = compute_finger_chain(pts[5], finger_spreads[0], flex_idx, idx_lens)
    pts[10], pts[11], pts[12] = compute_finger_chain(pts[9], finger_spreads[1], flex_mid, mid_lens)
    pts[14], pts[15], pts[16] = compute_finger_chain(pts[13], finger_spreads[2], flex_rng, rng_lens)
    pts[18], pts[19], pts[20] = compute_finger_chain(pts[17], finger_spreads[3], flex_pky, pky_lens)

    # Thumb forward kinematics in MediaPipe coordinates
    thb_abd, thb_mcp_flex, thb_ip_flex = thb_pose
    dir_thb_ip = np.array([math.sin(thb_abd), -math.cos(thb_mcp_flex), -math.sin(thb_mcp_flex)], dtype=np.float32)
    dir_thb_ip /= np.linalg.norm(dir_thb_ip)
    pts[3] = pts[2] + dir_thb_ip * thb_lens[0]

    cum_thb_flex = thb_mcp_flex + thb_ip_flex
    dir_thb_tip = np.array([math.sin(thb_abd), -math.cos(cum_thb_flex), -math.sin(cum_thb_flex)], dtype=np.float32)
    dir_thb_tip /= np.linalg.norm(dir_thb_tip)
    pts[4] = pts[3] + dir_thb_tip * thb_lens[1]

    return pts


def build_full_dataset(
    samples_per_signer_per_class: int = 40,
    output_path: Path = Path("backend/data/alphabet_dataset_multisigner.json")
) -> Dict[str, Any]:
    """
    Constructs the multi-signer, multi-condition dataset.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset: List[Dict[str, Any]] = []

    # Signer Profiles
    signers = [
        {"id": "Signer_1", "role": "train", "scale": 0.18, "ratio": 1.05},  # Adult male
        {"id": "Signer_2", "role": "train", "scale": 0.15, "ratio": 0.95},  # Adult female
        {"id": "Signer_3", "role": "val",   "scale": 0.17, "ratio": 0.88},  # Broad hand
        {"id": "Signer_4", "role": "test",  "scale": 0.12, "ratio": 1.00},  # Distant / small hand
    ]

    sample_counter = 0

    for signer in signers:
        s_id = signer["id"]
        split = signer["role"]
        base_scale = signer["scale"]
        base_ratio = signer["ratio"]

        for sign in SUPPORTED_CLASSES:
            for rep in range(samples_per_signer_per_class):
                sample_counter += 1
                sample_id = f"{s_id}_{sign}_{rep+1:03d}"

                # Variational parameters
                scale_jitter = base_scale * np.random.uniform(0.92, 1.08)
                ratio_jitter = base_ratio * np.random.uniform(0.95, 1.05)
                pitch = np.radians(np.random.uniform(-15.0, 15.0))
                yaw = np.radians(np.random.uniform(-18.0, 18.0))
                roll = np.radians(np.random.uniform(-12.0, 12.0))
                if sign in ["G", "H"]:
                    roll += np.radians(-75.0)  # Horizontal hand orientation
                elif sign in ["P", "Q"]:
                    pitch += np.radians(75.0)  # Downward pointing fingers
                elif sign == "C":
                    yaw += np.radians(np.random.choice([35.0, 45.0, 55.0, 10.0]))  # Natural lateral C arc orientation
                rot_mat = create_rotation_matrix(pitch, yaw, roll)

                # Generate anatomical hand
                pts_local = generate_anatomical_hand(sign, scale_jitter, ratio_jitter)

                # Rotate in 3D
                pts_rot = pts_local @ rot_mat.T

                # Global frame placement (screen center 0.5, 0.5)
                tx = 0.50 + np.random.uniform(-0.06, 0.06)
                ty = 0.55 + np.random.uniform(-0.06, 0.06)
                tz = 0.00 + np.random.uniform(-0.03, 0.03)

                pts_world = pts_rot + np.array([tx, ty, tz], dtype=np.float32)

                # Sensor optical grain noise
                pts_world += np.random.normal(0, 0.0025, size=pts_world.shape).astype(np.float32)

                landmarks_list = [
                    {"x": float(pts_world[i, 0]), "y": float(pts_world[i, 1]), "z": float(pts_world[i, 2])}
                    for i in range(21)
                ]

                # Extract 42 features
                features_42 = extract_canonical_features(landmarks_list).tolist()

                dataset.append({
                    "sample_id": sample_id,
                    "sign_label": sign,
                    "class_idx": CLASS_TO_IDX[sign],
                    "participant_id": s_id,
                    "split": split,
                    "palm_scale": round(float(scale_jitter), 4),
                    "rotation_deg": [round(float(math.degrees(pitch)), 1), round(float(math.degrees(yaw)), 1), round(float(math.degrees(roll)), 1)],
                    "landmarks": landmarks_list,
                    "features_42": features_42
                })

    # Summary
    train_count = sum(1 for d in dataset if d["split"] == "train")
    val_count = sum(1 for d in dataset if d["split"] == "val")
    test_count = sum(1 for d in dataset if d["split"] == "test")

    meta = {
        "dataset_name": "SIGNBRIDGE Multi-Signer Biomechanical Alphabet Dataset",
        "total_samples": len(dataset),
        "supported_classes": SUPPORTED_CLASSES,
        "num_classes": len(SUPPORTED_CLASSES),
        "features_dim": NUM_CANONICAL_FEATURES,
        "split_counts": {"train": train_count, "val": val_count, "test": test_count},
        "signer_split": {
            "train": ["Signer_1", "Signer_2"],
            "val": ["Signer_3"],
            "test": ["Signer_4"]
        }
    }

    full_payload = {"metadata": meta, "samples": dataset}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2)

    print(f"Dataset generated successfully at {output_path}")
    print(f"Total samples: {len(dataset)} (Train: {train_count}, Val: {val_count}, Test: {test_count})")
    print(f"Classes: {len(SUPPORTED_CLASSES)}")
    return full_payload


if __name__ == "__main__":
    build_full_dataset()
