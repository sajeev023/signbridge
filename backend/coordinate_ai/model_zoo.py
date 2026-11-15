"""
SIGNBRIDGE Coordinate AI — Coordinate Model Zoo
Provides coordinate-based architectures:
1. CoordinateMLP (PyTorch deep neural network with LayerNorm & residual connection).
2. CoordinateRandomForest (Scikit-learn tree-based classifier for benchmark comparison).
3. DynamicTrajectoryBiLSTM (Temporal recurrent network for J/Z trajectories).
4. Pure NumPy / JSON forward-pass inference engine for zero-dependency client/backend execution.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier

from backend.coordinate_ai.feature_pipeline import (
    extract_enriched_coordinate_features,
    extract_normalized_coordinates_63,
    ENRICHED_FEATURE_DIM,
    RAW_COORDINATE_DIM
)

STATIC_ASL_CLASSES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"
]
DYNAMIC_ASL_CLASSES = ["J", "Z"]
ALL_ASL_CLASSES = STATIC_ASL_CLASSES + DYNAMIC_ASL_CLASSES


class CoordinateMLP(nn.Module):
    """
    Supervised coordinate-based deep neural network.
    Accepts 94-dimensional enriched coordinate features (or 63 raw normalized coordinates).
    """
    def __init__(self, input_dim: int = ENRICHED_FEATURE_DIM, num_classes: int = len(STATIC_ASL_CLASSES)):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.fc1 = nn.Linear(input_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(128, 64)
        self.ln3 = nn.LayerNorm(64)
        self.relu3 = nn.ReLU()

        self.classifier = nn.Linear(64, num_classes)
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.drop1(self.relu1(self.ln1(self.fc1(x))))
        h2 = self.drop2(self.relu2(self.ln2(self.fc2(h1))))
        h3 = self.relu3(self.ln3(self.fc3(h2)))
        logits = self.classifier(h3)
        return logits / torch.clamp(self.temperature, min=0.1, max=5.0)

    def export_to_dict(self) -> Dict[str, Any]:
        """Exports weights into standard JSON-serializable dictionary for client-side execution."""
        return {
            "model_type": "CoordinateMLP",
            "version": "2.0.0-coordinate-ai",
            "input_dim": self.input_dim,
            "classes": STATIC_ASL_CLASSES,
            "temperature": float(self.temperature.item()),
            "weights": {
                "w1": self.fc1.weight.detach().cpu().numpy().tolist(),
                "b1": self.fc1.bias.detach().cpu().numpy().tolist(),
                "ln1_gamma": self.ln1.weight.detach().cpu().numpy().tolist(),
                "ln1_beta": self.ln1.bias.detach().cpu().numpy().tolist(),
                "w2": self.fc2.weight.detach().cpu().numpy().tolist(),
                "b2": self.fc2.bias.detach().cpu().numpy().tolist(),
                "ln2_gamma": self.ln2.weight.detach().cpu().numpy().tolist(),
                "ln2_beta": self.ln2.bias.detach().cpu().numpy().tolist(),
                "w3": self.fc3.weight.detach().cpu().numpy().tolist(),
                "b3": self.fc3.bias.detach().cpu().numpy().tolist(),
                "ln3_gamma": self.ln3.weight.detach().cpu().numpy().tolist(),
                "ln3_beta": self.ln3.bias.detach().cpu().numpy().tolist(),
                "w4": self.classifier.weight.detach().cpu().numpy().tolist(),
                "b4": self.classifier.bias.detach().cpu().numpy().tolist(),
            }
        }


class DynamicTrajectoryBiLSTM(nn.Module):
    """
    Temporal coordinate model for recognizing dynamic ASL letters J and Z
    from 16-frame coordinate trajectories (16 x 32 normalized motion features).
    """
    def __init__(self, input_dim: int = 32, hidden_dim: int = 64, num_classes: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.classes = ["NEUTRAL", "J", "Z"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        out, (hn, _) = self.lstm(x)
        last_hidden = torch.cat([hn[-2], hn[-1]], dim=-1)
        return self.fc(last_hidden)


class CoordinateInferenceEngine:
    """
    High-performance, zero-latency inference engine for coordinate-based recognition.
    Supports:
    - Pure NumPy matrix operations (no PyTorch dependency required at inference time).
    - Top-3 candidate ranking with probabilities.
    - Low-confidence rejection threshold.
    - Unknown / invalid handshape rejection.
    """
    def __init__(self, model_dict: Dict[str, Any], confidence_threshold: float = 0.70):
        self.classes = model_dict.get("classes", STATIC_ASL_CLASSES)
        self.threshold = confidence_threshold
        self.temperature = float(model_dict.get("temperature", 1.0))

        w = model_dict["weights"]
        self.w1 = np.array(w["w1"], dtype=np.float32)
        self.b1 = np.array(w["b1"], dtype=np.float32)
        self.g1 = np.array(w.get("ln1_gamma", np.ones(self.w1.shape[0])), dtype=np.float32)
        self.bet1 = np.array(w.get("ln1_beta", np.zeros(self.w1.shape[0])), dtype=np.float32)

        self.w2 = np.array(w["w2"], dtype=np.float32)
        self.b2 = np.array(w["b2"], dtype=np.float32)
        self.g2 = np.array(w.get("ln2_gamma", np.ones(self.w2.shape[0])), dtype=np.float32)
        self.bet2 = np.array(w.get("ln2_beta", np.zeros(self.w2.shape[0])), dtype=np.float32)

        self.w3 = np.array(w["w3"], dtype=np.float32)
        self.b3 = np.array(w["b3"], dtype=np.float32)
        self.g3 = np.array(w.get("ln3_gamma", np.ones(self.w3.shape[0])), dtype=np.float32)
        self.bet3 = np.array(w.get("ln3_beta", np.zeros(self.w3.shape[0])), dtype=np.float32)

        self.w4 = np.array(w["w4"], dtype=np.float32)
        self.b4 = np.array(w["b4"], dtype=np.float32)

    def _layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps) * gamma + beta

    def predict_features(self, features: np.ndarray) -> Dict[str, Any]:
        """Executes forward pass given numerical coordinate feature vector."""
        # Layer 1
        h1 = features @ self.w1.T + self.b1
        h1 = self._layer_norm(h1, self.g1, self.bet1)
        a1 = np.maximum(0, h1)

        # Layer 2
        h2 = a1 @ self.w2.T + self.b2
        h2 = self._layer_norm(h2, self.g2, self.bet2)
        a2 = np.maximum(0, h2)

        # Layer 3
        h3 = a2 @ self.w3.T + self.b3
        h3 = self._layer_norm(h3, self.g3, self.bet3)
        a3 = np.maximum(0, h3)

        # Output logits
        logits = (a3 @ self.w4.T + self.b4) / max(0.1, self.temperature)
        logits -= np.max(logits)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)

        # Sort top-3 candidates
        ranked_indices = np.argsort(probs)[::-1]
        top3 = [
            {"letter": self.classes[i], "confidence": float(probs[i])}
            for i in ranked_indices[:3]
        ]

        best_idx = ranked_indices[0]
        best_letter = self.classes[best_idx]
        best_conf = float(probs[best_idx])

        is_accepted = best_conf >= self.threshold
        predicted_sign = best_letter if is_accepted else "REJECTED"

        return {
            "predicted_letter": predicted_sign,
            "best_candidate": best_letter,
            "confidence": best_conf,
            "is_accepted": is_accepted,
            "top_candidates": top3,
            "probabilities": {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}
        }

    def predict_landmarks(self, landmarks: List[Dict[str, float]]) -> Dict[str, Any]:
        """End-to-end inference from 21 MediaPipe hand landmarks."""
        if not landmarks or len(landmarks) < 21:
            return {
                "predicted_letter": "REJECTED",
                "best_candidate": None,
                "confidence": 0.0,
                "is_accepted": False,
                "top_candidates": [],
                "probabilities": {}
            }
        feats = extract_enriched_coordinate_features(landmarks)
        return self.predict_features(feats)
