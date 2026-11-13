"""
SIGNBRIDGE Dynamic Trajectory Dataset Builder & Collection Utility
Generates and collects complete multi-speed, multi-scale, multi-signer trajectory episodes
for ASL dynamic alphabet recognition:
- 'J': Pinky-tip J-curve trajectory in 'I'-handshape context
- 'Z': Index-tip 3-stroke zigzag trajectory in 'D'-handshape context
- 'NEUTRAL': Stationary resting hand or random ambient non-alphabet movement
- 'INCOMPLETE': Aborted or partial movement strokes

Supports strict signer-independent splits:
- Training: Signer_1 & Signer_2 (Cohorts A & B)
- Validation: Signer_3 (Cohort C)
- Held-Out Test: Signer_4 & Signer_5 (Cohorts D & E)
"""

import sys
import json
import time
import math
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from trajectory_point_history_engine import (
    PointHistoryBuffer,
    resample_trajectory_2d,
    calculate_trajectory_path_length,
    TRAJECTORY_BUFFER_LENGTH,
    TRAJECTORY_FEATURE_SIZE
)
from build_alphabet_dataset import generate_anatomical_hand

TRAJECTORY_CLASSES = ["NEUTRAL", "J", "Z"]
CLASS_TO_IDX = {c: i for i, c in enumerate(TRAJECTORY_CLASSES)}


def generate_j_trajectory_points(duration_sec: float, palm_scale: float, fps: float = 30.0, noise_std: float = 0.002) -> List[Tuple[float, float]]:
    """
    Generates realistic 2D fingertip coordinates for ASL letter 'J':
    Top-to-bottom vertical descent followed by a smooth counter-clockwise hook to the left.
    """
    num_frames = max(8, int(duration_sec * fps))
    t = np.linspace(0.0, 1.0, num_frames)

    # In camera plane: x is horizontal (0..1), y is vertical down (0..1)
    # Start at (0.50, 0.40)
    start_x, start_y = 0.50, 0.38
    stroke_height = 1.35 * palm_scale
    hook_radius = 0.55 * palm_scale

    points = []
    for s in t:
        if s <= 0.60:
            # Linear downward stroke
            frac = s / 0.60
            x = start_x
            y = start_y + frac * stroke_height
        else:
            # Curve down and hook to the left
            frac = (s - 0.60) / 0.40
            angle = frac * math.pi * 0.85
            x = start_x - hook_radius * math.sin(angle)
            y = (start_y + stroke_height) + hook_radius * (1.0 - math.cos(angle))

        nx = float(np.random.normal(0, noise_std))
        ny = float(np.random.normal(0, noise_std))
        points.append((x + nx, y + ny))

    return points


def generate_z_trajectory_points(duration_sec: float, palm_scale: float, fps: float = 30.0, noise_std: float = 0.002) -> List[Tuple[float, float]]:
    """
    Generates realistic 2D fingertip coordinates for ASL letter 'Z':
    Three distinct strokes:
    1. Horizontal right
    2. Diagonal down-left
    3. Horizontal right
    """
    num_frames = max(10, int(duration_sec * fps))
    t = np.linspace(0.0, 1.0, num_frames)

    start_x, start_y = 0.42, 0.40
    width = 1.20 * palm_scale
    height = 1.20 * palm_scale

    points = []
    for s in t:
        if s <= 0.33:
            # Stroke 1: horizontal right
            frac = s / 0.33
            x = start_x + frac * width
            y = start_y
        elif s <= 0.67:
            # Stroke 2: diagonal down-left
            frac = (s - 0.33) / 0.34
            x = (start_x + width) - frac * width
            y = start_y + frac * height
        else:
            # Stroke 3: horizontal right
            frac = (s - 0.67) / 0.33
            x = start_x + frac * width
            y = start_y + height

        nx = float(np.random.normal(0, noise_std))
        ny = float(np.random.normal(0, noise_std))
        points.append((x + nx, y + ny))

    return points


def generate_neutral_trajectory_points(duration_sec: float, palm_scale: float, fps: float = 30.0, movement_type: str = "resting") -> List[Tuple[float, float]]:
    """
    Generates realistic negative control trajectories:
    - Resting hand drift (tremor)
    - Random non-sign ambient hand motion
    """
    num_frames = max(8, int(duration_sec * fps))
    start_x, start_y = 0.50, 0.50
    points = []

    if movement_type == "resting":
        # Pure tremor around a fixed point (path length < 0.2 palm scale)
        for _ in range(num_frames):
            points.append((start_x + np.random.normal(0, 0.003), start_y + np.random.normal(0, 0.003)))
    elif movement_type == "slow_drift":
        # Slow translation across screen
        for i in range(num_frames):
            frac = i / num_frames
            points.append((start_x + frac * 0.04, start_y - frac * 0.02 + np.random.normal(0, 0.002)))
    elif movement_type == "vertical_bob":
        for i in range(num_frames):
            frac = i / num_frames
            points.append((start_x + np.random.normal(0, 0.002), start_y + math.sin(frac * math.pi * 2) * 0.03))

    return points


def build_dynamic_jz_dataset():
    workspace_root = Path(__file__).resolve().parent.parent
    out_file = workspace_root / "backend" / "data" / "dynamic_jz_trajectories.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    signers = [
        {"id": "Signer_1", "split": "train", "scale_range": (0.15, 0.18), "speed_bias": 1.0},
        {"id": "Signer_2", "split": "train", "scale_range": (0.13, 0.16), "speed_bias": 0.85},
        {"id": "Signer_3", "split": "val",   "scale_range": (0.16, 0.20), "speed_bias": 1.15},
        {"id": "Signer_4", "split": "test",  "scale_range": (0.14, 0.17), "speed_bias": 0.90},
        {"id": "Signer_5", "split": "test",  "scale_range": (0.12, 0.15), "speed_bias": 1.10},
    ]

    # Speeds: fast (0.35s), normal (0.60s), slow (0.95s), deliberate (1.30s)
    base_durations = [0.35, 0.50, 0.65, 0.85, 1.10]
    episodes = []
    episode_counter = 1

    np.random.seed(42)

    for signer in signers:
        sid = signer["id"]
        split = signer["split"]
        s_min, s_max = signer["scale_range"]
        speed_factor = signer["speed_bias"]

        # Number of samples per class per signer
        # 40 J, 40 Z, 40 NEUTRAL per signer = 120 episodes * 5 signers = 600 complete episodes
        n_per_class = 40

        # 1. Generate J episodes
        for i in range(n_per_class):
            scale = float(np.random.uniform(s_min, s_max))
            dur = float(np.random.choice(base_durations) * speed_factor)
            pts_2d = generate_j_trajectory_points(dur, scale, fps=30.0, noise_std=0.002)

            # Extract normalized 32-feature vector
            rel_pts = (np.array(pts_2d) - np.array(pts_2d[0])) / scale
            resampled_32 = resample_trajectory_2d(rel_pts, target_points=TRAJECTORY_BUFFER_LENGTH).flatten().tolist()
            path_len = calculate_trajectory_path_length(np.array(pts_2d)) / scale

            episodes.append({
                "episode_id": f"DYN_J_{sid}_{i+1:03d}",
                "label": "J",
                "class_idx": 1,
                "signer_id": sid,
                "cohort_split": split,
                "duration_ms": round(dur * 1000, 1),
                "frame_count": len(pts_2d),
                "palm_scale": round(scale, 4),
                "normalized_path_length": round(path_len, 4),
                "active_fingertip": "pinky_tip (landmark 20)",
                "handshape_context": "I_pose (pinky extended)",
                "provenance": "synthetic_kinematic_trajectory",
                "raw_trajectory": [{"x": round(x, 5), "y": round(y, 5)} for x, y in pts_2d],
                "features_32": [round(v, 5) for v in resampled_32]
            })

        # 2. Generate Z episodes
        for i in range(n_per_class):
            scale = float(np.random.uniform(s_min, s_max))
            dur = float(np.random.choice(base_durations) * speed_factor)
            pts_2d = generate_z_trajectory_points(dur, scale, fps=30.0, noise_std=0.002)

            rel_pts = (np.array(pts_2d) - np.array(pts_2d[0])) / scale
            resampled_32 = resample_trajectory_2d(rel_pts, target_points=TRAJECTORY_BUFFER_LENGTH).flatten().tolist()
            path_len = calculate_trajectory_path_length(np.array(pts_2d)) / scale

            episodes.append({
                "episode_id": f"DYN_Z_{sid}_{i+1:03d}",
                "label": "Z",
                "class_idx": 2,
                "signer_id": sid,
                "cohort_split": split,
                "duration_ms": round(dur * 1000, 1),
                "frame_count": len(pts_2d),
                "palm_scale": round(scale, 4),
                "normalized_path_length": round(path_len, 4),
                "active_fingertip": "index_tip (landmark 8)",
                "handshape_context": "D_pose (index extended)",
                "provenance": "synthetic_kinematic_trajectory",
                "raw_trajectory": [{"x": round(x, 5), "y": round(y, 5)} for x, y in pts_2d],
                "features_32": [round(v, 5) for v in resampled_32]
            })

        # 3. Generate NEUTRAL episodes (resting, drifting, incomplete)
        neutral_types = ["resting", "slow_drift", "vertical_bob"]
        for i in range(n_per_class):
            scale = float(np.random.uniform(s_min, s_max))
            dur = float(np.random.choice(base_durations) * speed_factor)
            ntype = neutral_types[i % len(neutral_types)]
            pts_2d = generate_neutral_trajectory_points(dur, scale, fps=30.0, movement_type=ntype)

            rel_pts = (np.array(pts_2d) - np.array(pts_2d[0])) / scale
            resampled_32 = resample_trajectory_2d(rel_pts, target_points=TRAJECTORY_BUFFER_LENGTH).flatten().tolist()
            path_len = calculate_trajectory_path_length(np.array(pts_2d)) / scale

            episodes.append({
                "episode_id": f"DYN_NEUTRAL_{sid}_{i+1:03d}",
                "label": "NEUTRAL",
                "class_idx": 0,
                "signer_id": sid,
                "cohort_split": split,
                "duration_ms": round(dur * 1000, 1),
                "frame_count": len(pts_2d),
                "palm_scale": round(scale, 4),
                "normalized_path_length": round(path_len, 4),
                "active_fingertip": "fingertip",
                "handshape_context": "resting/neutral",
                "provenance": "synthetic_kinematic_trajectory",
                "raw_trajectory": [{"x": round(x, 5), "y": round(y, 5)} for x, y in pts_2d],
                "features_32": [round(v, 5) for v in resampled_32]
            })

    dataset_payload = {
        "metadata": {
            "dataset_name": "SIGNBRIDGE Dynamic J/Z Trajectory Dataset",
            "version": "1.0.0-point-history-canonical",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_episodes": len(episodes),
            "classes": TRAJECTORY_CLASSES,
            "feature_dimension": TRAJECTORY_FEATURE_SIZE,
            "buffer_length": TRAJECTORY_BUFFER_LENGTH,
            "splits": {
                "train": sum(1 for e in episodes if e["cohort_split"] == "train"),
                "val": sum(1 for e in episodes if e["cohort_split"] == "val"),
                "test": sum(1 for e in episodes if e["cohort_split"] == "test")
            }
        },
        "episodes": episodes
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(dataset_payload, f, indent=2)

    print(f"[SUCCESS] Built {out_file} with {len(episodes)} total trajectory episodes.")
    print(f"  Train (Signer_1, Signer_2): {dataset_payload['metadata']['splits']['train']}")
    print(f"  Val (Signer_3): {dataset_payload['metadata']['splits']['val']}")
    print(f"  Test (Signer_4, Signer_5): {dataset_payload['metadata']['splits']['test']}")

    return dataset_payload

if __name__ == "__main__":
    build_dynamic_jz_dataset()
