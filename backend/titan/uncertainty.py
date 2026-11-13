"""
SIGNBRIDGE TITAN: Uncertainty Estimation, Calibration & Selective Rejection
Implements temperature scaling, Expected Calibration Error (ECE), Brier score,
predictive entropy, energy-based OOD detection, and false-acceptance/rejection telemetry.
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def compute_ece(probs: np.ndarray, labels: np.ndarray, num_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE) across uniform confidence bins.
    Args:
        probs: (N, K) predicted probability distribution
        labels: (N,) ground-truth class indices
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(np.float32)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    total_samples = len(labels)

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_count = np.sum(in_bin)

        if bin_count > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_count / total_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    """Computes multiclass Brier score: 1/N * sum_i sum_k (p_ik - y_ik)^2."""
    N = len(labels)
    one_hot = np.zeros((N, num_classes), dtype=np.float32)
    one_hot[np.arange(N), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def compute_predictive_entropy(probs: np.ndarray) -> np.ndarray:
    """Computes Shannon entropy H(p) = -sum p_k * log(p_k + eps) in nats."""
    eps = 1e-12
    return -np.sum(probs * np.log(probs + eps), axis=-1)


def compute_energy_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Computes free energy score E(x) = -T * log(sum(exp(logits / T)))."""
    scaled = logits / temperature
    max_l = np.max(scaled, axis=-1, keepdims=True)
    lse = max_l.squeeze(-1) + np.log(np.sum(np.exp(scaled - max_l), axis=-1))
    return -temperature * lse


class TemperatureScaler(nn.Module):
    """
    Learns optimal scalar temperature T > 0 on validation set logits via NLL minimization.
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, max_iter: int = 100) -> float:
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def eval_step():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        with torch.no_grad():
            self.temperature.clamp_(min=0.1, max=10.0)
            learned_temp = float(self.temperature.item())
        return learned_temp


def evaluate_selective_rejection(
    probs: np.ndarray,
    labels: np.ndarray,
    conf_thresholds: List[float] = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95],
) -> List[Dict[str, float]]:
    """
    Evaluates selective classification:
    Coverage (fraction of samples accepted) vs. Accuracy on accepted samples,
    plus False Acceptance Rate (FAR) and False Rejection Rate (FRR).
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    total = len(labels)

    results = []
    for tau in conf_thresholds:
        accepted = confidences >= tau
        rejected = ~accepted
        num_accepted = int(np.sum(accepted))
        num_rejected = int(np.sum(rejected))

        coverage = float(num_accepted / total) if total > 0 else 0.0

        if num_accepted > 0:
            correct_accepted = int(np.sum(predictions[accepted] == labels[accepted]))
            selective_acc = float(correct_accepted / num_accepted)
            false_accepts = int(np.sum(predictions[accepted] != labels[accepted]))
            far = float(false_accepts / num_accepted)
        else:
            selective_acc = 1.0
            far = 0.0

        if num_rejected > 0:
            # Rejection of what would have been correct predictions
            falsely_rejected = int(np.sum(predictions[rejected] == labels[rejected]))
            frr = float(falsely_rejected / total)
        else:
            frr = 0.0

        results.append({
            "threshold": tau,
            "coverage": round(coverage, 4),
            "selective_accuracy": round(selective_acc, 4),
            "far": round(far, 4),
            "frr": round(frr, 4),
            "num_accepted": num_accepted,
            "num_rejected": num_rejected,
        })

    return results
