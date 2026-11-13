"""
SIGNBRIDGE TITAN: Autonomous Training, Optimization & Calibration Pipeline
Implements reproducible multi-architecture training with AdamW, CosineAnnealingLR,
label smoothing, temperature scaling calibration, and strict signer-isolated partitions.
"""

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from backend.alphabet_feature_extractor import extract_canonical_features
from .architectures import (
    ArchitectureA_SpatialGNN,
    ArchitectureB_BiLSTMAttention,
    ArchitectureC_STTransformer,
    ArchitectureD_DualStreamGated,
    TitanHybridModel,
)
from .spatial_encoder import landmarks_to_node_features
from .uncertainty import (
    TemperatureScaler,
    compute_brier_score,
    compute_ece,
    compute_energy_score,
    compute_predictive_entropy,
    evaluate_selective_rejection,
)

# Standard Alphabet Vocabulary (25 classes: NEUTRAL + 24 static letters)
ALPHABET_CLASSES: List[str] = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M",
    "N", "NEUTRAL", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"
]
CLASS_TO_IDX = {c: i for i, c in enumerate(ALPHABET_CLASSES)}
NUM_CLASSES = len(ALPHABET_CLASSES)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TitanDataset(Dataset):
    """
    PyTorch Dataset providing:
    - node_features: (21, 6)
    - temporal_sequence: (T=16, 42)
    - aux_features: (42,)
    - label: int class index
    - signer_id: str
    - provenance: str
    """
    def __init__(
        self,
        node_features: np.ndarray,
        temporal_sequences: np.ndarray,
        aux_features: np.ndarray,
        labels: np.ndarray,
        signer_ids: List[str],
        provenance: List[str],
    ):
        self.node_features = torch.tensor(node_features, dtype=torch.float32)
        self.temporal_sequences = torch.tensor(temporal_sequences, dtype=torch.float32)
        self.aux_features = torch.tensor(aux_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.signer_ids = signer_ids
        self.provenance = provenance

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.node_features[idx],
            self.temporal_sequences[idx],
            self.aux_features[idx],
            self.labels[idx],
        )


def load_multisigner_dataset(
    dataset_path: Path,
    sequence_len: int = 16,
) -> Dict[str, Any]:
    """Loads and formats dataset from alphabet_dataset_multisigner.json."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data.get("samples", [])
    node_list = []
    seq_list = []
    aux_list = []
    labels = []
    signers = []
    provenance = []
    splits = []

    for s in samples:
        label_str = s.get("sign_label")
        if label_str not in CLASS_TO_IDX:
            continue
        c_idx = CLASS_TO_IDX[label_str]

        # 1. Node features (21, 6)
        lms = s.get("landmarks", [])
        nodes = landmarks_to_node_features(lms)

        # 2. Auxiliary features (42,)
        aux = np.array(s.get("features_42", []), dtype=np.float32)
        if len(aux) != 42:
            aux = extract_canonical_features(lms)

        # 3. Temporal sequence (T, 42) with physiological micro-jitter simulation
        # In static signs, historical sequence represents the hand held over T frames
        seq = np.tile(aux, (sequence_len, 1))
        # Add slight physiological Gaussian micro-jitter to temporal sequence
        jitter = np.random.normal(0, 0.002, seq.shape).astype(np.float32)
        seq = seq + jitter

        node_list.append(nodes)
        seq_list.append(seq)
        aux_list.append(aux)
        labels.append(c_idx)
        signers.append(s.get("participant_id", "Unknown"))
        provenance.append(s.get("provenance", "synthetic_anatomical"))
        splits.append(s.get("split", "train"))

    return {
        "nodes": np.array(node_list, dtype=np.float32),
        "sequences": np.array(seq_list, dtype=np.float32),
        "aux": np.array(aux_list, dtype=np.float32),
        "labels": np.array(labels, dtype=np.int64),
        "signers": signers,
        "provenance": provenance,
        "splits": splits,
    }


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> Dict[str, Any]:
    """Computes Top-1 Accuracy, Macro Precision, Recall, F1, and Per-Class Recall."""
    top1_acc = float(np.mean(y_true == y_pred))

    recalls = []
    precisions = []
    f1s = []
    per_class = {}

    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))

        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        recalls.append(rec)
        precisions.append(prec)
        f1s.append(f1)
        per_class[ALPHABET_CLASSES[c]] = {
            "recall": round(rec, 4),
            "precision": round(prec, 4),
            "f1": round(f1, 4),
            "support": int(tp + fn),
        }

    macro_rec = float(np.mean(recalls))
    macro_prec = float(np.mean(precisions))
    macro_f1 = float(np.mean(f1s))

    return {
        "accuracy": round(top1_acc, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
    }


def train_single_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for nodes, seq, aux, targets in loader:
        nodes = nodes.to(device)
        seq = seq.to(device)
        aux = aux.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        # Model forward handles various signatures
        if isinstance(model, ArchitectureA_SpatialGNN):
            logits = model(nodes, aux_features=aux)
        elif isinstance(model, (ArchitectureB_BiLSTMAttention, ArchitectureC_STTransformer)):
            logits, _ = model(seq)
        elif isinstance(model, ArchitectureD_DualStreamGated):
            logits, _ = model(nodes, seq, aux_features=aux)
        elif isinstance(model, TitanHybridModel):
            logits, _ = model(nodes, seq, aux_features=aux)
        else:
            logits = model(nodes, seq, aux_features=aux)

        loss = criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(targets)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    temperature: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()
    all_logits = []
    all_probs = []
    all_targets = []
    total_latency_ms = 0.0
    num_samples = 0

    for nodes, seq, aux, targets in loader:
        nodes = nodes.to(device)
        seq = seq.to(device)
        aux = aux.to(device)

        t0 = time.perf_counter()
        if isinstance(model, ArchitectureA_SpatialGNN):
            logits = model(nodes, aux_features=aux)
        elif isinstance(model, (ArchitectureB_BiLSTMAttention, ArchitectureC_STTransformer)):
            logits, _ = model(seq)
        elif isinstance(model, ArchitectureD_DualStreamGated):
            logits, _ = model(nodes, seq, aux_features=aux)
        elif isinstance(model, TitanHybridModel):
            logits, _ = model(nodes, seq, aux_features=aux)
        else:
            logits = model(nodes, seq, aux_features=aux)
        t1 = time.perf_counter()

        total_latency_ms += (t1 - t0) * 1000.0
        num_samples += len(targets)

        scaled_logits = logits / temperature
        probs = F.softmax(scaled_logits, dim=-1)

        all_logits.append(logits.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_targets.append(targets.numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    avg_latency_ms = total_latency_ms / max(1, num_samples)

    return all_logits, all_probs, all_targets, avg_latency_ms


def run_architecture_training(
    arch_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 25,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Trains, calibrates, and evaluates a specific candidate architecture."""
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(num_epochs):
        train_loss = train_single_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        # Evaluate on validation set
        val_logits, val_probs, val_targets, _ = evaluate_model(model, val_loader, device)
        val_loss = float(nn.CrossEntropyLoss()(torch.tensor(val_logits), torch.tensor(val_targets)).item())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # 1. Uncertainty calibration (Temperature Scaling on val set)
    val_logits, _, val_targets, _ = evaluate_model(model, val_loader, device)
    scaler = TemperatureScaler()
    optimal_temp = scaler.fit(torch.tensor(val_logits), torch.tensor(val_targets))

    # 2. Test Set Evaluation (Uncalibrated & Calibrated)
    test_logits, test_probs_uncal, test_targets, avg_latency = evaluate_model(model, test_loader, device, temperature=1.0)
    _, test_probs_cal, _, _ = evaluate_model(model, test_loader, device, temperature=optimal_temp)

    preds_uncal = np.argmax(test_probs_uncal, axis=1)
    preds_cal = np.argmax(test_probs_cal, axis=1)

    metrics_uncal = compute_multiclass_metrics(test_targets, preds_uncal)
    metrics_cal = compute_multiclass_metrics(test_targets, preds_cal)

    ece_uncal = compute_ece(test_probs_uncal, test_targets, num_bins=10)
    ece_cal = compute_ece(test_probs_cal, test_targets, num_bins=10)

    brier_uncal = compute_brier_score(test_probs_uncal, test_targets, NUM_CLASSES)
    brier_cal = compute_brier_score(test_probs_cal, test_targets, NUM_CLASSES)

    selective_results = evaluate_selective_rejection(test_probs_cal, test_targets)

    # Model complexity
    param_count = sum(p.numel() for p in model.parameters())

    return {
        "architecture": arch_name,
        "parameter_count": param_count,
        "optimal_temperature": round(optimal_temp, 4),
        "avg_latency_ms": round(avg_latency, 4),
        "accuracy": metrics_cal["accuracy"],
        "macro_precision": metrics_cal["macro_precision"],
        "macro_recall": metrics_cal["macro_recall"],
        "macro_f1": metrics_cal["macro_f1"],
        "ece_uncalibrated": round(ece_uncal, 4),
        "ece_calibrated": round(ece_cal, 4),
        "brier_score": round(brier_cal, 4),
        "per_class": metrics_cal["per_class"],
        "selective_rejection": selective_results,
    }
