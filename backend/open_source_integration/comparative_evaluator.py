"""
SIGNBRIDGE Open-Source Integration — Scientific Comparative Evaluation Engine
Benchmarks SIGNBRIDGE against components adapted from:
1. Kinivi MediaPipe Hand Gesture Recognition (2D normalization vs 3D canonical basis)
2. Nick Nochnack Action Detection (Raw holistic vs temporal hysteresis smoothing)
3. Pianalytix Sign Language Detection (Raw flattened 63-coordinate vs 42-biomechanical features)
"""

import sys
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "backend"))

from backend.tests.virtual_alphabet.virtual_hand_generator import (
    VirtualHandGenerator,
    ALL_26_LETTERS,
    STATIC_24_LETTERS,
    DYNAMIC_2_LETTERS
)
from backend.build_alphabet_dataset import create_rotation_matrix, generate_anatomical_hand
from backend.alphabet_feature_extractor import extract_canonical_features
from backend.open_source_integration.kinivi_point_history_adapter import KiniviPointHistoryAdapter
from backend.open_source_integration.nochnack_temporal_hysteresis import NochnackTemporalHysteresisFilter
from backend.open_source_integration.integrated_recognition_pipeline import IntegratedRecognitionPipeline


def kinivi_preprocess_landmark(landmarks: List[Dict[str, float]], image_width: float = 640.0, image_height: float = 480.0) -> np.ndarray:
    """
    Exact implementation of Kinivi's pre_process_landmark() from app.py.
    Relative to wrist (landmark 0), normalized by max absolute value. Discards z!
    """
    coords = []
    base_x = landmarks[0]["x"] * image_width
    base_y = landmarks[0]["y"] * image_height

    for lm in landmarks:
        coords.append(lm["x"] * image_width - base_x)
        coords.append(lm["y"] * image_height - base_y)

    max_val = max(abs(x) for x in coords) if coords else 1.0
    if max_val < 1e-5:
        max_val = 1.0

    return np.array([x / max_val for x in coords], dtype=np.float32)


def pianalytix_extract_keypoints(landmarks: List[Dict[str, float]]) -> np.ndarray:
    """
    Exact implementation of Pianalytix extract_keypoints() from function.py.
    Raw un-normalized flattened (x, y, z) coordinates.
    """
    pts = []
    for lm in landmarks:
        pts.extend([lm["x"], lm["y"], lm.get("z", 0.0)])
    return np.array(pts, dtype=np.float32)


def run_static_feature_comparison_benchmark() -> Dict[str, Any]:
    """
    Benchmarks Kinivi 2D normalization vs Pianalytix raw flattening vs SIGNBRIDGE 3D canonical
    under rotation, scaling, and translation perturbations across the 24 static letters.
    """
    generator = VirtualHandGenerator(base_seed=200)
    tilts_deg = [0, 15, 30, 45]
    scales = [0.75, 1.0, 1.35]

    results = {
        "kinivi_representation": {"name": "Kinivi 2D Relative Normalization (42 coords, no z)", "scores": {}},
        "pianalytix_representation": {"name": "Pianalytix Raw Flattened Coordinates (63 values)", "scores": {}},
        "signbridge_representation": {"name": "SIGNBRIDGE 3D Orthonormal Canonical (42 features)", "scores": {}}
    }

    # Evaluate feature stability: compute correlation/invariance under tilt vs canonical
    for tilt in tilts_deg:
        kin_drifts = []
        pia_drifts = []
        sb_drifts = []

        rad = math.radians(tilt)
        R = create_rotation_matrix(pitch=0.0, yaw=0.0, roll=rad)

        for letter in STATIC_24_LETTERS:
            s_can = generator.generate_sample(letter, variation="canonical")
            lms_can = s_can["landmarks"]

            # Rotated landmarks
            pts_can = np.array([[p["x"], p["y"], p["z"]] for p in lms_can], dtype=np.float32)
            wrist = pts_can[0]
            pts_rot = (pts_can - wrist) @ R.T + wrist
            lms_rot = [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "visibility": 1.0} for p in pts_rot]

            # 1. Kinivi
            f_kin_can = kinivi_preprocess_landmark(lms_can)
            f_kin_rot = kinivi_preprocess_landmark(lms_rot)
            kin_drift = float(np.linalg.norm(f_kin_can - f_kin_rot) / (np.linalg.norm(f_kin_can) + 1e-6))
            kin_drifts.append(kin_drift)

            # 2. Pianalytix
            f_pia_can = pianalytix_extract_keypoints(lms_can)
            f_pia_rot = pianalytix_extract_keypoints(lms_rot)
            pia_drift = float(np.linalg.norm(f_pia_can - f_pia_rot) / (np.linalg.norm(f_pia_can) + 1e-6))
            pia_drifts.append(pia_drift)

            # 3. SIGNBRIDGE
            f_sb_can = extract_canonical_features(lms_can)
            f_sb_rot = extract_canonical_features(lms_rot)
            sb_drift = float(np.linalg.norm(f_sb_can - f_sb_rot) / (np.linalg.norm(f_sb_can) + 1e-6))
            sb_drifts.append(sb_drift)

        results["kinivi_representation"]["scores"][f"tilt_{tilt}deg_drift"] = float(np.mean(kin_drifts))
        results["pianalytix_representation"]["scores"][f"tilt_{tilt}deg_drift"] = float(np.mean(pia_drifts))
        results["signbridge_representation"]["scores"][f"tilt_{tilt}deg_drift"] = float(np.mean(sb_drifts))

    return results


def run_temporal_smoothing_benchmark() -> Dict[str, Any]:
    """
    Benchmarks raw frame prediction vs Nick Nochnack temporal hysteresis filter
    under injected coordinate noise to measure flicker suppression and stabilization.
    """
    pipeline = IntegratedRecognitionPipeline()
    generator = VirtualHandGenerator(base_seed=300)

    # Simulate 50 continuous frames of ASL letter 'B' with Gaussian tracking jitter
    num_frames = 50
    noise_std = 0.012

    raw_predictions = []
    filtered_emissions = []
    raw_flips = 0

    s_base = generator.generate_sample("B", variation="canonical")
    base_pts = np.array([[p["x"], p["y"], p["z"]] for p in s_base["landmarks"]], dtype=np.float32)

    pipeline.reset()

    for i in range(num_frames):
        noisy_pts = base_pts + np.random.normal(0, noise_std, size=base_pts.shape).astype(np.float32)
        lms = [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "visibility": 1.0} for p in noisy_pts]
        t_ms = i * 33.3  # 30 FPS stream

        res = pipeline.process_frame(lms, timestamp_ms=t_ms)
        raw_pred = res["raw_sign"]
        emitted = res["sign"]

        raw_predictions.append(raw_pred)
        if emitted:
            filtered_emissions.append(emitted)

        if i > 0 and raw_predictions[i] != raw_predictions[i - 1]:
            raw_flips += 1

    return {
        "num_frames": num_frames,
        "raw_prediction_flips": raw_flips,
        "raw_stability_percentage": ((num_frames - 1 - raw_flips) / (num_frames - 1)) * 100.0,
        "filtered_emissions_count": len(filtered_emissions),
        "filtered_dominant_sign": filtered_emissions[0] if filtered_emissions else None,
        "spurious_flicker_suppression_rate": 100.0 if len(set(filtered_emissions)) <= 1 else 0.0,
        "accumulated_sentence": pipeline.hysteresis_filter.get_accumulated_sentence()
    }


def run_full_comparative_audit() -> Dict[str, Any]:
    """
    Runs all comparative benchmarks and packages the analytical dataset.
    """
    print("--- Executing Open-Source Integration Comparative Benchmarks ---")
    t0 = time.perf_counter()

    static_cmp = run_static_feature_comparison_benchmark()
    temporal_cmp = run_temporal_smoothing_benchmark()

    # Latency benchmarking
    pipeline = IntegratedRecognitionPipeline()
    generator = VirtualHandGenerator(base_seed=400)
    sample = generator.generate_sample("A", variation="canonical")
    lms = sample["landmarks"]

    # Warmup
    for _ in range(20):
        pipeline.process_frame(lms, 0.0)

    latencies = []
    for i in range(100):
        t_start = time.perf_counter()
        pipeline.process_frame(lms, i * 33.3)
        latencies.append((time.perf_counter() - t_start) * 1000.0)

    mean_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    throughput_fps = 1000.0 / mean_latency if mean_latency > 0 else 0.0

    benchmark_summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_sec": time.perf_counter() - t0,
        "static_feature_comparison": static_cmp,
        "temporal_hysteresis_benchmark": temporal_cmp,
        "latency_metrics": {
            "mean_latency_ms": mean_latency,
            "p95_latency_ms": p95_latency,
            "throughput_fps": throughput_fps
        }
    }

    out_file = WORKSPACE_ROOT / "backend" / "open_source_integration" / "comparative_benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    print(f"Comparative benchmark results written to {out_file}")
    print(f"Mean latency: {mean_latency:.3f} ms ({throughput_fps:.1f} FPS)")
    return benchmark_summary


if __name__ == "__main__":
    run_full_comparative_audit()
