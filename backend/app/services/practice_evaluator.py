import numpy as np
from typing import Dict, List, Optional, Any
from app.config import VOCABULARY_LIST, CLASS_TO_IDX

class PracticeEvaluator:
    """
    Evaluates a user's signing form against canonical ISL kinematic templates
    in Learning and Testing Mode. Provides granular biomechanical feedback
    and a practice accuracy score.
    """
    def __init__(self):
        self.vocabulary = {item["id"]: item for item in VOCABULARY_LIST}

    def evaluate_practice_attempt(
        self,
        target_sign: str,
        current_prediction: Dict[str, Any],
        left_hand: Optional[List[Dict[str, float]]],
        right_hand: Optional[List[Dict[str, float]]],
        pose: Optional[List[Dict[str, float]]]
    ) -> Dict[str, Any]:
        if target_sign not in self.vocabulary:
            return {"error": f"Unknown sign '{target_sign}'"}

        meta = self.vocabulary[target_sign]
        feedback_notes = []
        scores = []

        has_lh = left_hand is not None and len(left_hand) == 21
        has_rh = right_hand is not None and len(right_hand) == 21
        has_pose = pose is not None and len(pose) >= 17

        # 1. Hand configuration check
        expected_config = meta.get("hand_config", "single_handed")
        if expected_config == "two_handed":
            if has_lh and has_rh:
                scores.append(100)
                feedback_notes.append("Two-hand presence: Correct.")
            elif has_rh or has_lh:
                scores.append(40)
                feedback_notes.append("Note: This sign requires BOTH hands. Please bring your other hand into frame.")
            else:
                scores.append(0)
                feedback_notes.append("Hands not detected in frame.")
        else:
            if has_rh or has_lh:
                scores.append(100)
            else:
                scores.append(0)
                feedback_notes.append("Please place your signing hand in clear view of the camera.")

        # 2. Neural model prediction match
        pred_sign = current_prediction.get("sign_id") or current_prediction.get("predicted_sign")
        confidence = current_prediction.get("confidence", 0.0)

        if pred_sign == target_sign and confidence >= 0.70:
            scores.append(int(confidence * 100))
            feedback_notes.append(f"AI Model Match: Excellent form ({confidence*100:.1f}% confidence)!")
        elif pred_sign == target_sign:
            scores.append(int(confidence * 100))
            feedback_notes.append(f"AI Model Match: Good, but refine movement speed and hand orientation ({confidence*100:.1f}%).")
        elif pred_sign is not None:
            scores.append(max(int(confidence * 30), 15))
            feedback_notes.append(f"Form divergence: Movement resembles '{pred_sign}'. Check the reference guide.")
        else:
            scores.append(20)
            feedback_notes.append("Continue executing the complete motion arc.")

        # 3. Specific biomechanical posture hints per sign
        if target_sign == "namaste" and has_lh and has_rh:
            dist = np.linalg.norm([
                left_hand[0]["x"] - right_hand[0]["x"],
                left_hand[0]["y"] - right_hand[0]["y"]
            ])
            if dist < 0.15:
                scores.append(95)
                feedback_notes.append("Palms proximity: Well joined at chest.")
            else:
                scores.append(50)
                feedback_notes.append("Press palms closer together at center chest.")

        elif target_sign == "thank_you" and (has_rh or has_lh):
            scores.append(90)
            feedback_notes.append("Ensure smooth sweeping arc forward from chin.")

        elif target_sign == "water":
            scores.append(85)
            feedback_notes.append("Curve dominant hand slightly as if holding a vessel to mouth.")

        overall_score = int(np.mean(scores)) if scores else 0

        # Grade classification
        if overall_score >= 85:
            grade = "Excellent"
            status = "mastered"
        elif overall_score >= 65:
            grade = "Good Progress"
            status = "proficient"
        elif overall_score >= 40:
            grade = "Needs Practice"
            status = "learning"
        else:
            grade = "Beginning"
            status = "starting"

        return {
            "target_sign": target_sign,
            "target_label": meta["label"],
            "category": meta["category"],
            "accuracy_score": overall_score,
            "grade": grade,
            "status": status,
            "feedback": feedback_notes,
            "canonical_movement": meta["movement"],
            "difficulty": meta["difficulty"]
        }
