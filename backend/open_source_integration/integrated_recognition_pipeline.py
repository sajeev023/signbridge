"""
SIGNBRIDGE Open-Source Integration — Unified Multi-Modal Recognition Pipeline
Integrates:
1. Canonical 42 Continuous Biomechanical Features (SIGNBRIDGE Core)
2. Palm-Scale Invariant Point History Buffer (adapted from Kinivi)
3. Consecutive Agreement & Temporal Hysteresis Filter (adapted from Nick Nochnack)
4. Dedicated Letter A Binary Kinematic Gate & Specialist
5. Dynamic Trajectory Kinematic Verification Gates (J & Z)
"""

import math
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from backend.alphabet_feature_extractor import extract_canonical_features
from backend.trajectory_point_history_engine import (
    is_j_handshape_context,
    is_z_handshape_context,
    verify_j_trajectory_kinematics,
    verify_z_trajectory_kinematics,
    TRAJECTORY_BUFFER_LENGTH
)
from backend.train_letter_a_specialist import verify_letter_a_geometry
from backend.open_source_integration.kinivi_point_history_adapter import KiniviPointHistoryAdapter
from backend.open_source_integration.nochnack_temporal_hysteresis import NochnackTemporalHysteresisFilter

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent


class IntegratedRecognitionPipeline:
    """
    Unified multi-engine sign recognition pipeline with open-source adaptations.
    """
    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or (WORKSPACE_ROOT / "backend" / "saved_models")
        self._load_models()

        # Open-source adapted components
        self.point_history_j = KiniviPointHistoryAdapter(maxlen=TRAJECTORY_BUFFER_LENGTH)
        self.point_history_z = KiniviPointHistoryAdapter(maxlen=TRAJECTORY_BUFFER_LENGTH)
        self.hysteresis_filter = NochnackTemporalHysteresisFilter(
            window_size=10,
            agreement_threshold=7,
            confidence_threshold=0.72,
            debounce_ms=800.0
        )

        self.last_predicted_sign = "NEUTRAL"
        self.last_confidence = 0.0

    def _load_models(self):
        with open(self.models_dir / "single_hand_alphabet_model.json", "r", encoding="utf-8") as f:
            self.static_model = json.load(f)
        with open(self.models_dir / "dynamic_alphabet_model.json", "r", encoding="utf-8") as f:
            self.dynamic_model = json.load(f)
        with open(self.models_dir / "letter_a_specialist_model.json", "r", encoding="utf-8") as f:
            self.specialist_model = json.load(f)

    def reset(self):
        self.point_history_j.reset()
        self.point_history_z.reset()
        self.hysteresis_filter.reset()
        self.last_predicted_sign = "NEUTRAL"
        self.last_confidence = 0.0

    def process_frame(
        self,
        landmarks: List[Dict[str, float]],
        timestamp_ms: float
    ) -> Dict[str, Any]:
        """
        Executes unified recognition on a single frame of 21 MediaPipe landmarks.
        """
        if not landmarks or len(landmarks) < 21:
            emitted, smoothed_conf, status = self.hysteresis_filter.process_frame(
                "REJECTED", 0.0, timestamp_ms
            )
            return {
                "sign": emitted,
                "raw_sign": "REJECTED",
                "confidence": 0.0,
                "is_dynamic": False,
                "status": "REJECTED",
                "engine": "pre_flight_quality_gate",
                "sentence": self.hysteresis_filter.get_accumulated_sentence(),
                "all_probabilities": {}
            }

        # 1. Palm scale calculation
        wrist = np.array([landmarks[0]["x"], landmarks[0]["y"], landmarks[0].get("z", 0.0)], dtype=np.float32)
        mid_mcp = np.array([landmarks[9]["x"], landmarks[9]["y"], landmarks[9].get("z", 0.0)], dtype=np.float32)
        palm_scale = max(0.035, float(np.linalg.norm(mid_mcp - wrist)))

        # 2. Update Kinivi Point History buffers for tracking dynamic letters J and Z
        pinky_tip = landmarks[20]
        index_tip = landmarks[8]
        self.point_history_j.append_point(pinky_tip, palm_scale, timestamp_ms)
        self.point_history_z.append_point(index_tip, palm_scale, timestamp_ms)

        # 3. Dynamic Trajectory Evaluation (Priority for motion-dependent signs J & Z)
        is_j_context = is_j_handshape_context(landmarks, palm_scale)
        is_z_context = is_z_handshape_context(landmarks, palm_scale)

        raw_pred = "NEUTRAL"
        raw_conf = 0.0
        engine_name = "single_hand_alphabet_model"
        is_dynamic = False
        all_probs: Dict[str, float] = {}

        if is_j_context and self.point_history_j.is_ready():
            traj_pts = np.array([[p["x"], p["y"]] for p in self.point_history_j.buffer], dtype=np.float32)
            kin_ok, _ = verify_j_trajectory_kinematics(traj_pts, palm_scale)
            if kin_ok:
                features_32 = self.point_history_j.extract_signbridge_invariant_features()
                dyn_pred, dyn_conf, _, dyn_all = self._forward_dynamic_mlp(features_32)
                if dyn_pred == "J" and dyn_conf >= 0.70:
                    raw_pred = "J"
                    raw_conf = dyn_conf
                    engine_name = "dynamic_alphabet_model (Kinivi Trajectory Adapter)"
                    is_dynamic = True
                    all_probs = dyn_all

        elif is_z_context and self.point_history_z.is_ready():
            traj_pts = np.array([[p["x"], p["y"]] for p in self.point_history_z.buffer], dtype=np.float32)
            kin_ok, _ = verify_z_trajectory_kinematics(traj_pts, palm_scale)
            if kin_ok:
                features_32 = self.point_history_z.extract_signbridge_invariant_features()
                dyn_pred, dyn_conf, _, dyn_all = self._forward_dynamic_mlp(features_32)
                if dyn_pred == "Z" and dyn_conf >= 0.70:
                    raw_pred = "Z"
                    raw_conf = dyn_conf
                    engine_name = "dynamic_alphabet_model (Kinivi Trajectory Adapter)"
                    is_dynamic = True
                    all_probs = dyn_all

        # 4. Static 24-Alphabet Evaluation (if dynamic was not triggered)
        if not is_dynamic:
            feats_42 = extract_canonical_features(landmarks)
            stat_pred, stat_conf, _, stat_all = self._forward_static_mlp(feats_42)
            raw_pred = stat_pred
            raw_conf = stat_conf
            all_probs = stat_all

            # Specialist gate for Letter A
            if raw_pred == "A":
                kin_diag = verify_letter_a_geometry(feats_42)
                if kin_diag.get("is_valid_letter_a", False):
                    prob_a = self._forward_specialist(feats_42)
                    if prob_a >= 0.70:
                        raw_pred = "A"
                        raw_conf = prob_a
                        engine_name = "letter_a_specialist_model + kinematic_gate"
                    else:
                        raw_pred = "REJECTED"
                        raw_conf = 0.0
                else:
                    raw_pred = "REJECTED"
                    raw_conf = 0.0

        # 5. Apply Nick Nochnack Temporal Hysteresis & Debounce Filter
        emitted_sign, smoothed_conf, filter_status = self.hysteresis_filter.process_frame(
            raw_pred, raw_conf, timestamp_ms
        )

        return {
            "sign": emitted_sign,
            "raw_sign": raw_pred,
            "confidence": smoothed_conf,
            "is_dynamic": is_dynamic,
            "status": filter_status,
            "engine": engine_name,
            "sentence": self.hysteresis_filter.get_accumulated_sentence(),
            "all_probabilities": all_probs
        }

    def _forward_static_mlp(self, features: np.ndarray) -> Tuple[str, float, np.ndarray, Dict[str, float]]:
        w = self.static_model["weights"]
        classes = self.static_model["classes"]
        w1, b1 = np.array(w["w1"], dtype=np.float32), np.array(w["b1"], dtype=np.float32)
        w2, b2 = np.array(w["w2"], dtype=np.float32), np.array(w["b2"], dtype=np.float32)
        w3, b3 = np.array(w["w3"], dtype=np.float32), np.array(w["b3"], dtype=np.float32)

        z1 = np.maximum(0, features @ w1.T + b1)
        z2 = np.maximum(0, z1 @ w2.T + b2)
        logits = z2 @ w3.T + b3
        logits -= np.max(logits)
        exp_l = np.exp(logits)
        probs = exp_l / np.sum(exp_l)
        idx = int(np.argmax(probs))
        all_p = {c: float(probs[i]) for i, c in enumerate(classes)}
        return classes[idx], float(probs[idx]), probs, all_p

    def _forward_dynamic_mlp(self, features: np.ndarray) -> Tuple[str, float, np.ndarray, Dict[str, float]]:
        w = self.dynamic_model["weights"]
        classes = self.dynamic_model["classes"]
        w1, b1 = np.array(w["w1"], dtype=np.float32), np.array(w["b1"], dtype=np.float32)
        w2, b2 = np.array(w["w2"], dtype=np.float32), np.array(w["b2"], dtype=np.float32)
        w3, b3 = np.array(w["w3"], dtype=np.float32), np.array(w["b3"], dtype=np.float32)

        z1 = np.maximum(0, w1 @ features + b1)
        z2 = np.maximum(0, w2 @ z1 + b2)
        logits = w3 @ z2 + b3
        logits -= np.max(logits)
        exp_l = np.exp(logits)
        probs = exp_l / np.sum(exp_l)
        idx = int(np.argmax(probs))
        all_p = {c: float(probs[i]) for i, c in enumerate(classes)}
        return classes[idx], float(probs[idx]), probs, all_p

    def _forward_specialist(self, features: np.ndarray) -> float:
        w = self.specialist_model["weights"]
        w1, b1 = np.array(w["w1"], dtype=np.float32), np.array(w["b1"], dtype=np.float32)
        bn_m, bn_v = np.array(w["bn1_mean"], dtype=np.float32), np.array(w["bn1_var"], dtype=np.float32)
        bn_w, bn_b = np.array(w["bn1_weight"], dtype=np.float32), np.array(w["bn1_bias"], dtype=np.float32)
        w2, b2 = np.array(w["w2"], dtype=np.float32), np.array(w["b2"], dtype=np.float32)
        w3, b3 = np.array(w["w3"], dtype=np.float32), np.array(w["b3"], dtype=np.float32)
        temp = float(self.specialist_model.get("temperature", 1.05))

        h1 = features @ w1.T + b1
        h1_bn = (h1 - bn_m) / np.sqrt(bn_v + 1e-5) * bn_w + bn_b
        a1 = np.maximum(0, h1_bn)
        h2 = a1 @ w2.T + b2
        a2 = np.maximum(0, h2)
        logits = (a2 @ w3.T + b3) / temp
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / np.sum(exp_l)
        return float(probs[1])
