import time
from collections import deque
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch

from app.config import (
    CLASS_NAMES, VOCABULARY_LIST, SEQUENCE_LENGTH, NUM_FEATURES, NUM_CLASSES,
    CONFIDENCE_THRESHOLD, CONSECUTIVE_STABILITY_FRAMES, UNCERTAIN_FALLBACK_TEXT,
    MODEL_CHECKPOINT_PATH
)
from app.models.architecture import SignBridgeBiLSTM
from app.services.landmark_extractor import LandmarkExtractor

class InferenceEngine:
    """
    Real-Time Sliding Window Temporal Inference Engine.
    
    Maintains a FIFO queue of normalized landmark feature frames (window = 30),
    runs model forward pass, enforces confidence thresholding, and ensures
    temporal stability to prevent flickering and spurious predictions.
    """
    def __init__(self, checkpoint_path=MODEL_CHECKPOINT_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SignBridgeBiLSTM(
            input_dim=NUM_FEATURES,
            hidden_dim=128,
            num_layers=2,
            num_classes=NUM_CLASSES,
            dropout=0.3
        ).to(self.device)

        if checkpoint_path.exists():
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            self.model.eval()
            self.is_loaded = True
            print(f"[InferenceEngine] Checkpoint successfully loaded from {checkpoint_path}")
        else:
            self.is_loaded = False
            print(f"[InferenceEngine] Warning: Checkpoint not found at {checkpoint_path}")

        self.extractor = LandmarkExtractor()
        self.buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.recent_predictions = deque(maxlen=CONSECUTIVE_STABILITY_FRAMES)
        self.last_confirmed_sign = None
        self.vocabulary_lookup = {item["id"]: item for item in VOCABULARY_LIST}

    def reset_buffer(self):
        """Clears the temporal window and stability queue."""
        self.buffer.clear()
        self.recent_predictions.clear()
        self.last_confirmed_sign = None

    def push_landmarks(
        self,
        left_hand: Optional[List[Dict[str, float]]],
        right_hand: Optional[List[Dict[str, float]]],
        pose: Optional[List[Dict[str, float]]]
    ) -> Dict[str, Any]:
        """
        Receives per-frame raw landmark dictionaries, normalizes coordinates,
        appends to sliding window buffer, and evaluates prediction if window is saturated.
        """
        # Normalize incoming frame
        features = self.extractor.normalize_landmarks(left_hand, right_hand, pose)
        self.buffer.append(features)

        # Check if hands are present
        has_hands = (left_hand is not None and len(left_hand) > 0) or (right_hand is not None and len(right_hand) > 0)

        # If buffer is still filling
        if len(self.buffer) < SEQUENCE_LENGTH:
            return {
                "status": "buffering",
                "buffer_fill": len(self.buffer) / SEQUENCE_LENGTH,
                "buffer_count": len(self.buffer),
                "has_hands": has_hands,
                "message": f"Buffering gesture motion ({len(self.buffer)}/{SEQUENCE_LENGTH} frames)..."
            }

        # Buffer is saturated: perform inference
        return self._predict_window(has_hands)

    def _predict_window(self, has_hands: bool) -> Dict[str, Any]:
        """Runs forward pass on current buffer."""
        if not self.is_loaded:
            return {
                "status": "error",
                "message": "Model checkpoint not loaded on server."
            }

        # If user has dropped hands completely, do not hallucinate
        if not has_hands:
            self.recent_predictions.clear()
            return {
                "status": "idle",
                "has_hands": False,
                "message": "Waiting for hands in camera frame..."
            }

        # Convert buffer to tensor: shape (1, 30, 153)
        window_array = np.array(self.buffer, dtype=np.float32)[np.newaxis, ...]
        window_tensor = torch.tensor(window_array, dtype=torch.float32, device=self.device)

        t_start = time.perf_counter()
        with torch.no_grad():
            logits, attn_weights = self.model(window_tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            attn = attn_weights.cpu().numpy()[0].tolist()
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx])
        pred_class = CLASS_NAMES[top_idx]

        # All class probabilities dictionary for UI inspection
        all_probs = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

        # Strict Confidence Gate
        if confidence < CONFIDENCE_THRESHOLD:
            return {
                "status": "uncertain",
                "confidence": confidence,
                "predicted_sign": pred_class,
                "threshold": CONFIDENCE_THRESHOLD,
                "message": UNCERTAIN_FALLBACK_TEXT,
                "latency_ms": round(latency_ms, 2),
                "all_probabilities": all_probs,
                "attention_weights": attn
            }

        # Temporal Stability Filter
        self.recent_predictions.append(pred_class)
        stable = len(self.recent_predictions) >= 2 and (
            self.recent_predictions.count(pred_class) >= 2
        )

        metadata = self.vocabulary_lookup.get(pred_class, {})

        return {
            "status": "recognized" if stable else "evaluating",
            "is_stable": stable,
            "sign_id": pred_class,
            "label": metadata.get("label", pred_class),
            "category": metadata.get("category", "general"),
            "confidence": confidence,
            "translations": metadata.get("translations", {
                "en": metadata.get("label", pred_class)
            }),
            "description": metadata.get("description", ""),
            "latency_ms": round(latency_ms, 2),
            "timestamp": time.time(),
            "attention_weights": attn,
            "all_probabilities": all_probs
        }
