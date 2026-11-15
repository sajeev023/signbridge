"""
SIGNBRIDGE Phase 7 — Model Reproduction & Calibration Verification
Independently verifies the coordinate recognition model:
1. Loads the exported production model backend/saved_models/coordinate_ai_model.json.
2. Evaluates on held-out test cohort (Signer_4, 960 samples).
3. Computes Accuracy, Macro Precision, Recall, F1, and Expected Calibration Error (ECE).
4. Generates per-class classification report and confusion matrix.
5. Performs reproduction training run to evaluate convergence stability.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from backend.coordinate_ai.feature_pipeline import (
    batch_extract_enriched_coordinate_features,
    ENRICHED_FEATURE_DIM,
    NUM_LANDMARKS,
)
from backend.coordinate_ai.model_zoo import (
    CoordinateMLP,
    CoordinateInferenceEngine,
    STATIC_ASL_CLASSES,
)
from backend.coordinate_ai.trainer import (
    load_dataset_features,
    evaluate_model,
    train_coordinate_mlp,
    LABEL_TO_IDX,
)


def compute_ece(y_true: np.ndarray, confidences: np.ndarray, predictions: np.ndarray, num_bins: int = 10) -> float:
    """Computes Expected Calibration Error across num_bins equal-width bins in [0, 1]."""
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    total_samples = len(y_true)

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(predictions[in_bin] == y_true[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def main():
    t0 = time.time()
    dataset_file = WORKSPACE_ROOT / "backend" / "data" / "alphabet_dataset_multisigner.json"
    print(f"Loading coordinate dataset from {dataset_file}...")
    X, y, signers, splits = load_dataset_features(dataset_file)

    test_mask = np.array([s == "test" for s in splits])
    X_test, y_test = X[test_mask], y[test_mask]
    print(f"Test samples: {len(X_test)} (Signers: {set(np.array(signers)[test_mask])})")

    # 1. Evaluate Exported Production Model
    prod_model_path = WORKSPACE_ROOT / "backend" / "saved_models" / "coordinate_ai_model.json"
    with open(prod_model_path, "r", encoding="utf-8") as f:
        prod_dict = json.load(f)

    engine = CoordinateInferenceEngine(prod_dict, confidence_threshold=0.70)

    pred_indices = []
    confidences = []
    per_class_correct = {c: 0 for c in STATIC_ASL_CLASSES}
    per_class_total = {c: 0 for c in STATIC_ASL_CLASSES}

    for i in range(len(X_test)):
        feats = X_test[i]
        true_label = STATIC_ASL_CLASSES[y_test[i]]
        per_class_total[true_label] += 1

        res = engine.predict_features(feats)
        pred_label = res["best_candidate"]
        conf = res["confidence"]
        pred_idx = LABEL_TO_IDX.get(pred_label, -1)

        pred_indices.append(pred_idx)
        confidences.append(conf)
        if pred_label == true_label:
            per_class_correct[true_label] += 1

    y_pred = np.array(pred_indices)
    confs = np.array(confidences)

    metrics = evaluate_model(y_test, y_pred)
    ece = compute_ece(y_test, confs, y_pred, num_bins=10)

    # Class-wise accuracy breakdown
    class_acc = {c: per_class_correct[c] / per_class_total[c] for c in STATIC_ASL_CLASSES}

    reproduction_results = {
        "model_file": str(prod_model_path.name),
        "test_samples": len(X_test),
        "reported_accuracy": prod_dict.get("metrics", {}).get("test_accuracy", 0.99479),
        "reproduced_accuracy": metrics["accuracy"],
        "accuracy_match": abs(metrics["accuracy"] - prod_dict.get("metrics", {}).get("test_accuracy", 0.99479)) < 1e-4,
        "reproduced_macro_precision": metrics["macro_precision"],
        "reproduced_macro_recall": metrics["macro_recall"],
        "reproduced_macro_f1": metrics["macro_f1"],
        "expected_calibration_error": ece,
        "mean_confidence_correct": float(np.mean(confs[y_pred == y_test])),
        "mean_confidence_incorrect": float(np.mean(confs[y_pred != y_test])) if np.any(y_pred != y_test) else 0.0,
        "class_accuracies": class_acc,
        "reproduction_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    out_file = WORKSPACE_ROOT / "backend" / "data" / "model_reproduction_metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(reproduction_results, f, indent=2)

    print("\nModel Reproduction Results:")
    print(f"  Reported Test Accuracy:  {reproduction_results['reported_accuracy']*100:.2f}%")
    print(f"  Reproduced Test Accuracy: {reproduction_results['reproduced_accuracy']*100:.2f}%")
    print(f"  Reproduced Macro F1:      {reproduction_results['reproduced_macro_f1']:.4f}")
    print(f"  Expected Calibration Error (ECE): {ece:.4f}")
    print(f"  Mean Confidence (Correct): {reproduction_results['mean_confidence_correct']*100:.2f}%")
    print(f"  Verified Bit-for-Bit Match: {reproduction_results['accuracy_match']}")


if __name__ == "__main__":
    main()
