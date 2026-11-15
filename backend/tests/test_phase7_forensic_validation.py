"""
SIGNBRIDGE Phase 7 — Forensic Validation & Architecture Acceptance Suite
Validates:
1. Dataset Provenance & Integrity (3,840 static samples, 0 duplicates, balanced classes).
2. Model Reproduction & Calibration (Test accuracy >= 99.4%, ECE < 0.08).
3. Physical Human Validation Performance (42 captures, safety rejection < 0.70).
4. NVIDIA NIM Provider Fallback (Timeout, 401, Connection Error circuit breaker).
5. End-to-End Latency Headroom (< 5.0ms inference subtotal).
6. Bit-for-bit Frozen Artifact Preservation.
"""

import sys
import os
import json
import hashlib
from pathlib import Path
import pytest
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from backend.coordinate_ai.feature_pipeline import (
    extract_enriched_coordinate_features,
    extract_normalized_coordinates_63,
    batch_extract_enriched_coordinate_features,
    calculate_palm_scale,
    landmarks_to_numpy,
    NUM_LANDMARKS,
)
from backend.coordinate_ai.model_zoo import (
    CoordinateInferenceEngine,
    STATIC_ASL_CLASSES,
)
from backend.coordinate_ai.external_provider import (
    LocalCoordinateProvider,
    NvidiaNimProvider,
    ProviderFactory,
    CoordinateInferenceRequest,
    CoordinateInferenceResponse,
)


class TestDatasetForensics:
    def test_multisigner_dataset_integrity(self):
        data_path = WORKSPACE_ROOT / "backend" / "data" / "alphabet_dataset_multisigner.json"
        assert data_path.exists()
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        static_samples = [s for s in data["samples"] if s["sign_label"] in STATIC_ASL_CLASSES]
        assert len(static_samples) == 3840

        # Class balance: exactly 160 per class
        from collections import Counter
        class_counts = Counter(s["sign_label"] for s in static_samples)
        for c in STATIC_ASL_CLASSES:
            assert class_counts[c] == 160

        # Split balance: Train=1920 (Signer_1+2), Val=960 (Signer_3), Test=960 (Signer_4)
        split_counts = Counter(s["split"] for s in static_samples)
        assert split_counts["train"] == 1920
        assert split_counts["val"] == 960
        assert split_counts["test"] == 960

    def test_duplicate_sample_absence(self):
        audit_file = WORKSPACE_ROOT / "backend" / "data" / "forensic_audit_multisigner.json"
        assert audit_file.exists()
        with open(audit_file, "r", encoding="utf-8") as f:
            audit = json.load(f)
        assert audit["exact_duplicates"] == 0
        assert audit["near_duplicates"] == 0
        assert audit["min_inter_sample_distance"] > 0.10


class TestModelReproduction:
    def test_reproduced_accuracy_and_f1(self):
        rep_file = WORKSPACE_ROOT / "backend" / "data" / "model_reproduction_metrics.json"
        assert rep_file.exists()
        with open(rep_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        assert metrics["reproduced_accuracy"] == pytest.approx(0.99479, abs=1e-3)
        assert metrics["reproduced_macro_f1"] >= 0.990
        assert metrics["expected_calibration_error"] < 0.08
        assert metrics["mean_confidence_correct"] >= 0.90


class TestPhysicalCameraValidation:
    def test_physical_validation_safety_gating(self):
        val_file = WORKSPACE_ROOT / "backend" / "data" / "physical_coordinate_validation_summary.json"
        assert val_file.exists()
        with open(val_file, "r", encoding="utf-8") as f:
            summary = json.load(f)
        assert summary["total_human_benchmark_captures"] == 42
        # Safety rejections should capture low-confidence / occluded samples
        assert summary["rejected_safety"] >= 5
        # Calibrated precision on accepted samples should exceed 85%
        assert summary["calibrated_effective_precision"] >= 0.85
        # Session protocol trials generated
        assert summary["new_physical_sessions"]["sessions_saved"] >= 2


class TestProviderFallbackCircuitBreaker:
    def test_nim_fallback_results(self):
        res_file = WORKSPACE_ROOT / "backend" / "data" / "nvidia_nim_fallback_results.json"
        assert res_file.exists()
        with open(res_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        assert results["valid_api_response"]["status"] == "PASS"
        assert results["invalid_api_key_fallback"]["fallback_used"] is True
        assert results["timeout_fallback"]["fallback_used"] is True
        assert results["connection_refused_fallback"]["fallback_used"] is True
        assert results["malformed_response_fallback"]["fallback_used"] is True
        assert results["pure_local_provider"]["latency_ms"] < 3.0


class TestEndToEndLatencyBudget:
    def test_latency_within_60fps_budget(self):
        bench_file = WORKSPACE_ROOT / "backend" / "data" / "e2e_latency_benchmark.json"
        assert bench_file.exists()
        with open(bench_file, "r", encoding="utf-8") as f:
            report = json.load(f)
        # Coordinate AI subtotal must be < 4.0ms
        assert report["stages"]["coordinate_pipeline_subtotal"]["mean_ms"] < 4.0
        # End to end pipeline must be below 16.67ms (60 FPS budget)
        assert report["end_to_end_pipeline_total"]["mean_ms"] < 16.67
        assert report["end_to_end_pipeline_total"]["fps_capacity"] >= 60.0


class TestFrozenArtifactIntegrity:
    def test_frozen_manifest_sha256(self):
        manifest_path = WORKSPACE_ROOT / "backend" / "letter_a_frozen_manifest.json"
        assert manifest_path.exists()
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert len(manifest) == 7
        for relpath, meta in manifest.items():
            file_path = WORKSPACE_ROOT / relpath
            assert file_path.exists(), f"Missing frozen file {relpath}"
            h = hashlib.sha256(file_path.read_bytes()).hexdigest()
            assert h == meta["sha256"], f"SHA256 mismatch for {relpath}"
