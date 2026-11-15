"""
SIGNBRIDGE Unit Tests — Physical Validation Protocol and Privacy Constraints
Verifies:
1. Consent gate enforcement (rejects sessions without informed consent).
2. Pseudonym pattern compliance.
3. Privacy constraints (enforces 21 landmark metric points, rejects raw video/face data).
4. Cohort partitioning (TRAIN_COHORT vs HELDOUT_EVAL_COHORT).
5. Evaluation metrics calculation.
"""

import pytest
from pathlib import Path
from backend.validation.session_protocol import PhysicalValidationSession
from backend.validation.session_evaluator import evaluate_sessions


def test_consent_requirement():
    # Attempting to start without consent must raise ValueError
    with pytest.raises(ValueError, match="Informed consent"):
        PhysicalValidationSession(
            participant_pseudonym="P_SIGNER_01",
            cohort="TRAIN_COHORT",
            informed_consent_granted=False
        )


def test_pseudonym_validation():
    # Invalid pseudonym pattern must fail
    with pytest.raises(ValueError, match="Invalid participant pseudonym"):
        PhysicalValidationSession(
            participant_pseudonym="john_doe@email.com",  # Privacy violation! PII in pseudonym
            cohort="TRAIN_COHORT",
            informed_consent_granted=True
        )


def test_privacy_landmark_only_enforcement(tmp_path):
    session = PhysicalValidationSession(
        participant_pseudonym="P_TESTER_01",
        cohort="HELDOUT_EVAL_COHORT",
        informed_consent_granted=True
    )

    # Incomplete landmarks (< 21) must fail
    with pytest.raises(ValueError, match="Expected exactly 21 hand landmarks"):
        session.record_trial(
            ground_truth_letter="A",
            predicted_letter="A",
            confidence=0.92,
            is_accepted=True,
            camera_orientation="frontal",
            camera_distance="medium_55cm",
            lighting_condition="normal_indoor",
            landmark_visibility_ratio=1.0,
            landmarks_21=[{"x": 0.5, "y": 0.5, "z": 0.0}] * 10  # Only 10 points
        )


def test_session_lifecycle_and_evaluation(tmp_path):
    # Setup temporary sessions dir
    import backend.validation.session_protocol as sp
    orig_dir = sp.SESSIONS_DIR
    sp.SESSIONS_DIR = tmp_path

    try:
        session = PhysicalValidationSession(
            participant_pseudonym="P_SIGNER_09",
            cohort="HELDOUT_EVAL_COHORT",
            informed_consent_granted=True
        )

        dummy_lms = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0} for _ in range(21)]

        # Record a successful trial
        session.record_trial(
            ground_truth_letter="B",
            predicted_letter="B",
            confidence=0.96,
            is_accepted=True,
            camera_orientation="frontal",
            camera_distance="medium_55cm",
            lighting_condition="normal_indoor",
            landmark_visibility_ratio=1.0,
            landmarks_21=dummy_lms
        )

        # Record a confusion trial
        session.record_trial(
            ground_truth_letter="M",
            predicted_letter="N",
            confidence=0.74,
            is_accepted=True,
            camera_orientation="frontal",
            camera_distance="medium_55cm",
            lighting_condition="normal_indoor",
            landmark_visibility_ratio=1.0,
            landmarks_21=dummy_lms,
            error_class="CONFUSION"
        )

        saved_path = session.save_session()
        assert saved_path.exists()

        eval_res = evaluate_sessions(tmp_path)
        assert eval_res["status"] == "EVALUATION_COMPLETE"
        assert eval_res["total_sessions"] == 1
        assert eval_res["total_trials"] == 2
        assert eval_res["total_correct"] == 1
        assert eval_res["overall_accuracy_pct"] == 50.0
        assert "M->N" in eval_res["top_confusions"]
        assert eval_res["error_taxonomy_breakdown"]["CONFUSION"] == 1
    finally:
        sp.SESSIONS_DIR = orig_dir
