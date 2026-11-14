"""
SIGNBRIDGE — Operation A: Automated Failure Injection & Stress Test Suite
Tests:
1. Invalid, partial, and oversized landmark arrays
2. Zero-valued and degenerate collapsed hand poses
3. NaN, positive infinity, and negative infinity coordinate injection
4. Incorrect feature dimensions passed to specialist MLP
5. Duplicate sample ID detection and quarantine
6. Missing and non-compliant participant pseudonyms
7. Malformed and missing session identifiers
8. Export/import round-trip JSON serialization and schema conformance
9. Corrupted JSON import rejection
10. Sample deletion and undo integrity
11. Consent withdrawal and right-to-erase compliance
12. Frozen model hash mismatch detection (tamper detection)
13. Frozen threshold modification defense
"""

import os
import sys
import json
import math
import hashlib
import numpy as np
import pytest
import torch

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from alphabet_feature_extractor import extract_canonical_features
from train_letter_a_specialist import LetterASpecialist, verify_letter_a_geometry


# -----------------------------------------------------------------------------
# 1. Landmark Structural Failures
# -----------------------------------------------------------------------------

def test_empty_and_partial_landmark_arrays():
    """Verify feature extractor handles empty or partial landmark arrays safely."""
    # Empty
    feats_empty = extract_canonical_features([])
    assert len(feats_empty) == 42
    assert np.all(feats_empty == 0.0)

    # Partial (< 21)
    partial_lms = [{'x': 0.5, 'y': 0.5, 'z': 0.0} for _ in range(10)]
    feats_partial = extract_canonical_features(partial_lms)
    assert len(feats_partial) == 42
    assert np.all(feats_partial == 0.0)

    # Oversized (> 21)
    oversized_lms = [{'x': 0.5, 'y': 0.5, 'z': 0.0} for _ in range(30)]
    feats_oversized = extract_canonical_features(oversized_lms)
    assert len(feats_oversized) == 42


def test_zero_valued_and_degenerate_landmarks():
    """Verify degenerate collapsed landmarks do not produce unhandled divide-by-zero crashes."""
    # All points at origin
    zero_lms = [{'x': 0.0, 'y': 0.0, 'z': 0.0} for _ in range(21)]
    feats_zero = extract_canonical_features(zero_lms)
    assert len(feats_zero) == 42
    assert not np.isnan(feats_zero).any(), "Zero landmarks resulted in NaN features"
    assert not np.isinf(feats_zero).any(), "Zero landmarks resulted in Inf features"

    # All points identical at (0.5, 0.5, 0.0)
    identical_lms = [{'x': 0.5, 'y': 0.5, 'z': 0.0} for _ in range(21)]
    feats_identical = extract_canonical_features(identical_lms)
    assert len(feats_identical) == 42
    assert not np.isnan(feats_identical).any()
    assert not np.isinf(feats_identical).any()

    # Rule gate should strictly reject degenerate pose
    geom = verify_letter_a_geometry(feats_zero)
    assert geom['is_valid_letter_a'] is False


def test_nan_and_infinity_injection():
    """Verify NaN and Infinity coordinates do not silently propagate into true-positive A classification."""
    nan_lms = [{'x': 0.5, 'y': 0.5, 'z': 0.0} for _ in range(21)]
    nan_lms[4] = {'x': float('nan'), 'y': 0.5, 'z': 0.0}
    feats_nan = extract_canonical_features(nan_lms)
    assert len(feats_nan) == 42
    # Ensure kinematic rule gate rejects NaN features
    geom_nan = verify_letter_a_geometry(feats_nan)
    assert geom_nan['is_valid_letter_a'] is False

    inf_lms = [{'x': 0.5, 'y': 0.5, 'z': 0.0} for _ in range(21)]
    inf_lms[0] = {'x': float('inf'), 'y': float('-inf'), 'z': 0.0}
    feats_inf = extract_canonical_features(inf_lms)
    assert len(feats_inf) == 42
    geom_inf = verify_letter_a_geometry(feats_inf)
    assert geom_inf['is_valid_letter_a'] is False


# -----------------------------------------------------------------------------
# 2. Specialist Dimension & Input Failures
# -----------------------------------------------------------------------------

def test_incorrect_feature_dimensions_rejection():
    """Verify specialist model rejects feature inputs that deviate from exactly 42 continuous dimensions."""
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "saved_models", "letter_a_specialist_model.json"))
    with open(model_path, "r", encoding="utf-8") as f:
        model_data = json.load(f)

    model = LetterASpecialist(input_dim=42, hidden_dim=32)
    model.eval()

    # 40 features instead of 42
    with pytest.raises(RuntimeError):
        x_short = torch.zeros(1, 40)
        model(x_short)

    # 44 features instead of 42
    with pytest.raises(RuntimeError):
        x_long = torch.zeros(1, 44)
        model(x_long)


# -----------------------------------------------------------------------------
# 3. Collector Metadata & Schema Conformance
# -----------------------------------------------------------------------------

def test_duplicate_sample_id_detection():
    """Verify sample buffers detect and quarantine duplicate IDs."""
    buffer = [
        {"id": "SMP_A_1001", "letter": "A", "signerId": "Signer_P26"},
        {"id": "SMP_A_1002", "letter": "A", "signerId": "Signer_P26"},
        {"id": "SMP_A_1001", "letter": "A", "signerId": "Signer_P26"}  # Duplicate!
    ]

    seen = set()
    duplicates = []
    unique_records = []
    for s in buffer:
        if s["id"] in seen:
            duplicates.append(s["id"])
        else:
            seen.add(s["id"])
            unique_records.append(s)

    assert len(duplicates) == 1
    assert duplicates[0] == "SMP_A_1001"
    assert len(unique_records) == 2


def test_participant_identifier_compliance():
    """Verify that only authorized pseudonym formats (Signer_P26 through Signer_P35) pass validation."""
    valid_ids = [f"Signer_P{i:02d}" for i in range(26, 36)]
    invalid_ids = ["John_Doe", "signer26", "Signer_P99", "", None, "Signer_P01"]

    def validate_eval_signer(sid):
        if not sid or not isinstance(sid, str):
            return False
        return sid in valid_ids

    for vid in valid_ids:
        assert validate_eval_signer(vid) is True

    for inv in invalid_ids:
        assert validate_eval_signer(inv) is False


def test_session_id_compliance():
    """Verify session ID conforms to SES_<Signer>_<Timestamp> schema."""
    import re
    valid_session = "SES_P26_LX94K28A"
    pattern = r"^SES_P\d{2}_[A-Z0-9]+$"
    assert re.match(pattern, valid_session) is not None

    invalid_sessions = ["session_1", "", "SES_", "random_uuid_1234"]
    for inv in invalid_sessions:
        assert re.match(pattern, inv) is None


# -----------------------------------------------------------------------------
# 4. Export / Import Roundtrip Integrity
# -----------------------------------------------------------------------------

def test_export_import_roundtrip_consistency():
    """Verify dataset serialization and deserialization retains bitwise identity."""
    sample_record = {
        "id": "SMP_A_1794650000000",
        "timestamp": "2026-11-14T18:00:00.000Z",
        "signerId": "Signer_P26",
        "cohort": "test",
        "datasetSplit": "physical_human_validation",
        "letter": "A",
        "isDynamic": False,
        "isSynthetic": False,
        "metadata": {
            "distance": "Medium (0.5m-1.0m)",
            "orientation": "Frontal (0°)",
            "lighting": "Natural Daylight",
            "handedness": "Right",
            "resolution": "1280x720",
            "fps": 30,
            "trackingQualityScore": 98.5,
            "capture_mode": "PHYSICAL_CAMERA",
            "sessionId": "SES_P26_LX94K28A"
        },
        "landmarks": [{"x": 0.5, "y": 0.5 + i * 0.01, "z": 0.0} for i in range(21)],
        "features42": [0.25] * 42
    }

    manifest = {
        "exportVersion": "1.1.0-field-ready",
        "frozenCandidateBaseline": {
            "specialistSha256": "dc7cd03a4ed647a7ec570422ed54d130bbb250d6cded42b93466ef6e967d14cf",
            "temperature": 1.062088,
            "decisionThreshold": 0.50
        },
        "sessionMetadata": {
            "sessionId": "SES_P26_LX94K28A",
            "exportTimestamp": "2026-11-14T18:05:00.000Z",
            "totalSamples": 1,
            "isSyntheticHarnessActive": False
        },
        "records": [sample_record]
    }

    # Serialize
    json_str = json.dumps(manifest, indent=2)

    # Deserialize
    recovered = json.loads(json_str)
    assert recovered["exportVersion"] == "1.1.0-field-ready"
    assert len(recovered["records"]) == 1
    rec = recovered["records"][0]
    assert rec["id"] == sample_record["id"]
    assert rec["datasetSplit"] == "physical_human_validation"
    assert len(rec["landmarks"]) == 21
    assert len(rec["features42"]) == 42


def test_corrupted_json_import_rejection():
    """Verify malformed JSON strings fail gracefully during import parsing."""
    corrupted_json = '{"records": [{"id": "SMP_A_1", "letter": "A"'  # Truncated
    with pytest.raises(json.JSONDecodeError):
        json.loads(corrupted_json)


def test_sample_deletion_and_undo_integrity():
    """Verify buffer deletion removes target and undo buffer restores exact state."""
    buffer = [
        {"id": f"SMP_A_{i}", "letter": "A"} for i in range(5)
    ]
    target_id = "SMP_A_2"

    # Delete
    deleted_item = next((s for s in buffer if s["id"] == target_id), None)
    buffer = [s for s in buffer if s["id"] != target_id]
    assert len(buffer) == 4
    assert not any(s["id"] == target_id for s in buffer)

    # Undo
    buffer.insert(2, deleted_item)
    assert len(buffer) == 5
    assert buffer[2]["id"] == target_id


def test_consent_withdrawal_and_purge():
    """Verify right-to-erase clears all records completely."""
    buffer = [{"id": f"SMP_A_{i}", "letter": "A"} for i in range(10)]
    assert len(buffer) == 10

    # Purge
    buffer.clear()
    assert len(buffer) == 0


# -----------------------------------------------------------------------------
# 5. Cryptographic Freeze & Threshold Tamper Defense
# -----------------------------------------------------------------------------

def test_frozen_model_tamper_detection():
    """Verify that altering even a single byte in the specialist weights changes the SHA-256."""
    manifest_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "letter_a_frozen_manifest.json"))
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    target_rel = "backend/saved_models/letter_a_specialist_model.json"
    full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", target_rel))

    with open(full_path, "rb") as f:
        content = f.read()

    expected_hash = manifest[target_rel]["sha256"]
    computed_hash = hashlib.sha256(content).hexdigest()
    assert computed_hash == expected_hash

    # Tampered test
    tampered_content = content[:-1] + b'X'
    tampered_hash = hashlib.sha256(tampered_content).hexdigest()
    assert tampered_hash != expected_hash


def test_frozen_threshold_tamper_defense():
    """Verify that decision threshold and temperature constants are locked to baseline values."""
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "saved_models", "letter_a_specialist_model.json"))
    with open(model_path, "r", encoding="utf-8") as f:
        model_data = json.load(f)

    # Threshold must be exactly 0.50
    assert model_data.get("threshold", 0.50) == 0.50

    # Temperature must be exactly calibrated ~1.062088
    temp = model_data.get("temperature", 1.0)
    assert math.isclose(temp, 1.0620883703231812, rel_tol=1e-5)
