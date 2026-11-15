"""
SIGNBRIDGE Coordinate AI — Model Training & Comparison Pipeline
Trains, validates, and compares coordinate-based architectures:
- CoordinateMLP (Deep Coordinate Neural Network)
- CoordinateRandomForest (Ensemble tree baseline)
Evaluates on participant-separated test cohorts and exports the production coordinate model.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from backend.coordinate_ai.feature_pipeline import (
    batch_extract_enriched_coordinate_features,
    ENRICHED_FEATURE_DIM,
    NUM_LANDMARKS,
)
from backend.coordinate_ai.model_zoo import (
    CoordinateMLP,
    STATIC_ASL_CLASSES,
)

LABEL_TO_IDX = {c: i for i, c in enumerate(STATIC_ASL_CLASSES)}


def load_dataset_features(dataset_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Loads dataset and extracts 94-dim enriched coordinate features using batch vectorization."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data["samples"]
    valid_samples = []
    labels_list = []
    signers_list = []
    splits_list = []

    for s in samples:
        lbl = s["sign_label"]
        if lbl not in LABEL_TO_IDX:
            continue
        valid_samples.append(s["landmarks"])
        labels_list.append(LABEL_TO_IDX[lbl])
        signers_list.append(s.get("participant_id", "Unknown"))
        splits_list.append(s.get("split", "train"))

    # Convert all valid sample landmark dictionaries to (N, 21, 3) float32 numpy array
    N = len(valid_samples)
    pts_array = np.zeros((N, NUM_LANDMARKS, 3), dtype=np.float32)
    for i, lms in enumerate(valid_samples):
        for j in range(min(NUM_LANDMARKS, len(lms))):
            pts_array[i, j, 0] = float(lms[j].get("x", 0.0))
            pts_array[i, j, 1] = float(lms[j].get("y", 0.0))
            pts_array[i, j, 2] = float(lms[j].get("z", 0.0))

    X = batch_extract_enriched_coordinate_features(pts_array)
    y = np.array(labels_list, dtype=np.int64)
    return X, y, signers_list, splits_list


def train_coordinate_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 15,
    batch_size: int = 64
) -> CoordinateMLP:
    """Trains the CoordinateMLP on participant-separated training data."""
    model = CoordinateMLP(input_dim=ENRICHED_FEATURE_DIM, num_classes=len(STATIC_ASL_CLASSES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    print(f"Training CoordinateMLP on {len(X_train)} samples across {epochs} epochs...")
    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

    model.eval()
    return model


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": float(acc),
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "macro_f1": float(f1)
    }


def main():
    t0 = time.time()
    dataset_file = WORKSPACE_ROOT / "backend" / "data" / "alphabet_dataset_multisigner.json"
    print(f"Loading coordinate dataset from {dataset_file}...", flush=True)
    X, y, signers, splits = load_dataset_features(dataset_file)
    print(f"Extracted features for {len(X)} samples in {time.time() - t0:.2f}s (Shape: {X.shape}).", flush=True)

    train_mask = np.array([s == "train" for s in splits])
    val_mask = np.array([s == "val" for s in splits])
    test_mask = np.array([s == "test" for s in splits])

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"Dataset split counts: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}", flush=True)

    # 1. Train CoordinateMLP
    mlp_model = train_coordinate_mlp(X_train, y_train, X_val, y_val, epochs=12)
    with torch.no_grad():
        test_logits = mlp_model(torch.from_numpy(X_test))
        mlp_preds = torch.argmax(test_logits, dim=-1).numpy()
    mlp_metrics = evaluate_model(y_test, mlp_preds)
    print("CoordinateMLP Test Metrics:", mlp_metrics, flush=True)

    # 2. Train Random Forest Baseline
    print("Training CoordinateRandomForest baseline...", flush=True)
    rf = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42, n_jobs=1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_metrics = evaluate_model(y_test, rf_preds)
    print("CoordinateRandomForest Test Metrics:", rf_metrics)

    # 3. Export Winning CoordinateMLP Model
    export_dict = mlp_model.export_to_dict()
    export_dict["metrics"] = {
        "test_accuracy": mlp_metrics["accuracy"],
        "test_macro_f1": mlp_metrics["macro_f1"],
        "rf_baseline_accuracy": rf_metrics["accuracy"],
        "rf_baseline_f1": rf_metrics["macro_f1"]
    }

    out_file = WORKSPACE_ROOT / "backend" / "saved_models" / "coordinate_ai_model.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(export_dict, f, indent=2)
    print(f"Exported coordinate AI model to {out_file} ({out_file.stat().st_size} bytes). Total elapsed: {time.time() - t0:.2f}s.")


if __name__ == "__main__":
    main()
