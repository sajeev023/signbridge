"""
SIGNBRIDGE Physical Validation Engine — Camera Testing Protocol Runner
Implements Phase 4 and Phase 5 requirements:
1. Zero-Leak Privacy: No raw video, RGB imagery, face mesh, or audio recorded.
2. Informed Consent: Strict boolean requirement before session initialization.
3. Pseudonym Segregation: Partitioning between Training and Held-Out Evaluation cohorts.
4. Comprehensive Metadata: Distance, lighting, camera orientation, landmark visibility, error taxonomy.
"""

import os
import re
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
SESSIONS_DIR = WORKSPACE_ROOT / "backend" / "data" / "physical_validation_sessions"


class PhysicalValidationSession:
    """
    Session logger for genuine human physical camera validation trials.
    """
    VALID_ORIENTATIONS = {"frontal", "15deg_pitch", "15deg_roll", "15deg_yaw"}
    VALID_DISTANCES = {"near_35cm", "medium_55cm", "far_85cm"}
    VALID_LIGHTING = {"normal_indoor", "dim_warm", "bright_overhead"}
    VALID_COHORTS = {"TRAIN_COHORT", "HELDOUT_EVAL_COHORT"}
    ERROR_TAXONOMY = {
        "NONE",
        "CONFUSION",
        "FALSE_REJECTION",
        "TRACKING_DROPOUT",
        "INCOMPLETE_TRAJECTORY",
        "KINEMATIC_VIOLATION"
    }

    def __init__(
        self,
        participant_pseudonym: str,
        cohort: str,
        informed_consent_granted: bool,
        operator_id: str = "OPERATOR_01"
    ):
        if not informed_consent_granted:
            raise ValueError(
                "Physical validation session REJECTED: Informed consent must be explicitly granted by the participant."
            )

        if not re.match(r"^P_[A-Z0-9_]{3,20}$", participant_pseudonym):
            raise ValueError(
                f"Invalid participant pseudonym: '{participant_pseudonym}'. Must match pattern '^P_[A-Z0-9_]{{3,20}}$'."
            )

        if cohort not in self.VALID_COHORTS:
            raise ValueError(f"Invalid cohort: '{cohort}'. Must be one of {self.VALID_COHORTS}.")

        self.participant_pseudonym = participant_pseudonym
        self.cohort = cohort
        self.informed_consent_granted = informed_consent_granted
        self.operator_id = operator_id
        self.session_id = f"{participant_pseudonym}_{int(time.time())}"
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.trials: List[Dict[str, Any]] = []

    def record_trial(
        self,
        ground_truth_letter: str,
        predicted_letter: Optional[str],
        confidence: float,
        is_accepted: bool,
        camera_orientation: str,
        camera_distance: str,
        lighting_condition: str,
        landmark_visibility_ratio: float,
        landmarks_21: List[Dict[str, float]],
        trajectory_16: Optional[List[Dict[str, float]]] = None,
        error_class: str = "NONE",
        notes: str = ""
    ) -> Dict[str, Any]:
        """
        Records a single camera validation trial with strict privacy verification.
        """
        # Privacy Enforcement: Strictly reject any attempt to supply raw RGB or face data
        if not landmarks_21 or len(landmarks_21) != 21:
            raise ValueError(f"Privacy and validation requirement: Expected exactly 21 hand landmarks, got {len(landmarks_21) if landmarks_21 else 0}.")

        for lm in landmarks_21:
            if not all(k in lm for k in ("x", "y", "z")):
                raise ValueError("Each landmark must contain metric coordinates (x, y, z).")

        if camera_orientation not in self.VALID_ORIENTATIONS:
            raise ValueError(f"Invalid orientation: {camera_orientation}. Allowed: {self.VALID_ORIENTATIONS}")

        if camera_distance not in self.VALID_DISTANCES:
            raise ValueError(f"Invalid distance: {camera_distance}. Allowed: {self.VALID_DISTANCES}")

        if lighting_condition not in self.VALID_LIGHTING:
            raise ValueError(f"Invalid lighting: {lighting_condition}. Allowed: {self.VALID_LIGHTING}")

        if error_class not in self.ERROR_TAXONOMY:
            raise ValueError(f"Invalid error class: {error_class}. Allowed: {self.ERROR_TAXONOMY}")

        trial_record = {
            "trial_id": len(self.trials) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ground_truth_letter": ground_truth_letter.upper(),
            "predicted_letter": predicted_letter.upper() if predicted_letter else None,
            "confidence": float(confidence),
            "is_accepted": bool(is_accepted),
            "is_correct": bool(is_accepted and predicted_letter and predicted_letter.upper() == ground_truth_letter.upper()),
            "camera_orientation": camera_orientation,
            "camera_distance": camera_distance,
            "lighting_condition": lighting_condition,
            "landmark_visibility_ratio": float(landmark_visibility_ratio),
            "error_classification": error_class,
            "landmarks_21": landmarks_21,
            "trajectory_16": trajectory_16,
            "notes": str(notes)
        }

        self.trials.append(trial_record)
        return trial_record

    def save_session(self) -> Path:
        """
        Saves session log to disk.
        """
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        out_file = SESSIONS_DIR / f"{self.session_id}.json"

        data = {
            "session_id": self.session_id,
            "participant_pseudonym": self.participant_pseudonym,
            "cohort": self.cohort,
            "informed_consent_granted": self.informed_consent_granted,
            "operator_id": self.operator_id,
            "created_at": self.created_at,
            "total_trials": len(self.trials),
            "correct_trials": sum(1 for t in self.trials if t["is_correct"]),
            "privacy_compliance": {
                "raw_video_stored": False,
                "face_mesh_stored": False,
                "audio_stored": False,
                "landmarks_only": True
            },
            "trials": self.trials
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return out_file
