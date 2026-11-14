"""
SIGNBRIDGE TITAN - OPERATION ZERO-LEAK TEST SUITE
Validates model freeze cryptographic hashes, dataset isolation, zero signer leak,
calibration parameters, and inference parity.
"""

import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pytest
import torch

from backend.titan.asl_architecture import (
    ASLSpatialGNN,
    ASL_CLASS_TO_IDX,
    ASL_VOCABULARY,
    build_adjacency_matrix,
    compute_normalized_laplacian
)
from backend.titan.spatial_encoder import landmarks_to_node_features
from backend.alphabet_feature_extractor import extract_canonical_features
from backend.trajectory_point_history_engine import resample_trajectory_2d


FROZEN_HASHES = {
    'titan_asl_spatial_model.json': 'aa3ff132052890b7ddc4832f5aa1e000e8e949d205bfb1f837fa7f461d16e04f',
    'titan_asl_spatial_model.pre_gen.json': '2999532e8901dec0f5048d9e5a5bfa90ce56b23b60b5c79ba8968aa7d7485092',
    'single_hand_alphabet_model.json': 'aa672b6fffd96f4aed694c76edfc87385cb053d53a157a56bfafc3ed7ace4272',
    'dynamic_alphabet_model.json': '0f77f12a282162a34d0cb0804c53733a8ffc65e8361d3ba42265791a0b80014c'
}


def test_frozen_model_sha256_cryptographic_integrity():
    workspace = Path(__file__).resolve().parent.parent.parent
    backend_models = workspace / "backend" / "saved_models"
    frontend_models = workspace / "frontend" / "src" / "services"

    for filename, expected_hash in FROZEN_HASHES.items():
        b_path = backend_models / filename
        assert b_path.exists(), f"Frozen model {filename} missing in backend"
        actual_b_hash = hashlib.sha256(b_path.read_bytes()).hexdigest()
        assert actual_b_hash == expected_hash, f"Hash mismatch for backend {filename}: expected {expected_hash}, got {actual_b_hash}"

        # If model also lives in frontend, check frontend parity
        f_path = frontend_models / filename
        if f_path.exists():
            actual_f_hash = hashlib.sha256(f_path.read_bytes()).hexdigest()
            assert actual_f_hash == actual_b_hash, f"Frontend copy of {filename} differs from backend!"


def test_zero_signer_leak_isolation():
    workspace = Path(__file__).resolve().parent.parent.parent
    meta_path = workspace / "backend" / "titan" / "datasets" / "asl_alphabet" / "metadata" / "real_human_benchmark.json"
    assert meta_path.exists(), "Benchmark metadata missing"

    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    records = meta['records']
    train_signers = set(r['signer_id'] for r in records if r.get('cohort_split') == 'train')
    test_signers = set(r['signer_id'] for r in records if r.get('cohort_split') == 'test')

    assert len(train_signers) > 0, "No train signers defined"
    assert len(test_signers) > 0, "No test signers defined"
    overlap = train_signers.intersection(test_signers)
    assert len(overlap) == 0, f"DATA LEAKAGE DETECTED! Overlapping signers found: {overlap}"


def test_frozen_calibration_thresholds():
    workspace = Path(__file__).resolve().parent.parent.parent
    model_path = workspace / "backend" / "saved_models" / "titan_asl_spatial_model.json"
    with open(model_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    assert data.get('temperature') == 1.08, f"Temperature changed! Expected 1.08, got {data.get('temperature')}"
    assert data.get('min_confidence') == 0.55, f"Min confidence changed! Expected 0.55, got {data.get('min_confidence')}"
    assert data.get('entropy_threshold') == 1.35, f"Entropy threshold changed! Expected 1.35, got {data.get('entropy_threshold')}"
    assert len(data.get('classes', [])) == 27, "Vocabulary classes altered"


def test_blind_evaluation_held_out_accuracy_reproducibility():
    workspace = Path(__file__).resolve().parent.parent.parent
    results_path = workspace / "backend" / "saved_models" / "zero_leak_blind_evaluation_results.json"
    assert results_path.exists(), "Zero-leak blind evaluation results JSON missing"

    with open(results_path, 'r', encoding='utf-8') as f:
        res = json.load(f)

    held_out = res['held_out_test_cohort']
    frozen_acc = held_out['frozen_ak_gcn_model']['raw_accuracy']
    baseline_acc = held_out['baseline_model']['raw_accuracy']

    # Held out accuracy must be exactly 74.07% (20/27)
    assert math.isclose(frozen_acc, 20 / 27, abs_tol=1e-4), f"Expected 74.07% accuracy, got {frozen_acc}"
    # Baseline accuracy must be exactly 55.56% (15/27)
    assert math.isclose(baseline_acc, 15 / 27, abs_tol=1e-4), f"Expected 55.56% accuracy, got {baseline_acc}"
    # ECE improvement
    assert held_out['frozen_ak_gcn_model']['ece'] < held_out['baseline_model']['ece']


def test_dynamic_trajectory_model_held_out_fidelity():
    workspace = Path(__file__).resolve().parent.parent.parent
    dyn_model_path = workspace / "backend" / "saved_models" / "dynamic_alphabet_model.json"
    traj_path = workspace / "backend" / "data" / "dynamic_jz_trajectories.json"

    with open(dyn_model_path, 'r', encoding='utf-8') as f:
        model_data = json.load(f)
    with open(traj_path, 'r', encoding='utf-8') as f:
        traj_data = json.load(f)

    weights = model_data['weights']
    w1 = np.array(weights['w1'], dtype=np.float32)
    b1 = np.array(weights['b1'], dtype=np.float32)
    w2 = np.array(weights['w2'], dtype=np.float32)
    b2 = np.array(weights['b2'], dtype=np.float32)
    w3 = np.array(weights['w3'], dtype=np.float32)
    b3 = np.array(weights['b3'], dtype=np.float32)
    classes = model_data['classes']

    test_episodes = [ep for ep in traj_data['episodes'] if ep.get('cohort_split') == 'test']
    assert len(test_episodes) == 240, f"Expected 240 test episodes, found {len(test_episodes)}"

    correct = 0
    for ep in test_episodes:
        pts = np.array([[p['x'], p['y']] for p in ep['raw_trajectory']], dtype=np.float32)
        scale = float(ep['palm_scale'])
        rel_pts = (pts - pts[0]) / scale
        resampled = resample_trajectory_2d(rel_pts, target_points=16).flatten()

        z1 = np.maximum(0, w1 @ resampled + b1)
        z2 = np.maximum(0, w2 @ z1 + b2)
        logits = w3 @ z2 + b3
        pred_class = classes[int(np.argmax(logits))]
        if pred_class == ep['label']:
            correct += 1

    assert correct == len(test_episodes), f"Expected 100% dynamic test accuracy, got {correct}/{len(test_episodes)}"
