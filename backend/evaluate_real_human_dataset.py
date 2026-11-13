"""
SIGNBRIDGE Real-Human Evaluation & Acceptance Metrics Script
Evaluates the production Biomechanical MLP model against backend/data/real_human_evaluation_dataset.json.
Reports metrics for:
1. Full Empirical Dataset (N=42)
2. Strictly Held-Out Test Cohort (N=28: Signer_3, Signer_4, Signer_5, Signer_6)
3. Core Alphabet Signs (A, B, C, D) Acceptance Breakdown
4. C/Y Confusion Diagnostics
5. Expected Calibration Error (ECE) and High-Confidence Error Audit
"""

import sys
import json
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from alphabet_feature_extractor import extract_canonical_features

# Load Model
model_path = Path("backend/saved_models/single_hand_alphabet_model.json")
with open(model_path, "r", encoding="utf-8") as f:
    model_data = json.load(f)

classes = model_data["classes"]
num_classes = len(classes)
w1 = np.array(model_data["weights"]["w1"], dtype=np.float32)
b1 = np.array(model_data["weights"]["b1"], dtype=np.float32)
w2 = np.array(model_data["weights"]["w2"], dtype=np.float32)
b2 = np.array(model_data["weights"]["b2"], dtype=np.float32)
w3 = np.array(model_data["weights"]["w3"], dtype=np.float32)
b3 = np.array(model_data["weights"]["b3"], dtype=np.float32)

def predict_sample(features: np.ndarray):
    z1 = np.maximum(0, features @ w1.T + b1)
    z2 = np.maximum(0, z1 @ w2.T + b2)
    logits = z2 @ w3.T + b3
    logits -= np.max(logits)
    exp_l = np.exp(logits)
    probs = exp_l / np.sum(exp_l)
    pred_idx = int(np.argmax(probs))
    conf = float(probs[pred_idx])
    return pred_idx, conf, probs

# Load Real Human Dataset
data_path = Path("backend/data/real_human_evaluation_dataset.json")
with open(data_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

records = dataset["records"]

def evaluate_subset(records_subset, subset_name="Subset"):
    y_true, y_pred, y_conf, y_probs = [], [], [], []
    detailed_results = []

    for r in records_subset:
        lbl = r["alphabet_label"]
        if lbl not in classes:
            continue
        c_true = classes.index(lbl)
        feats = np.array(r["features_42"], dtype=np.float32)
        pred_idx, conf, probs = predict_sample(feats)
        pred_lbl = classes[pred_idx]

        y_true.append(c_true)
        y_pred.append(pred_idx)
        y_conf.append(conf)
        y_probs.append(probs)

        detailed_results.append({
            "capture_id": r["capture_id"],
            "signer_id": r["signer_id"],
            "true_label": lbl,
            "predicted_label": pred_lbl,
            "confidence": round(conf, 4),
            "correct": bool(c_true == pred_idx),
            "camera_condition": r.get("camera_conditions", {}),
            "provenance": r.get("provenance")
        })

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    y_conf = np.array(y_conf, dtype=np.float32)
    y_probs = np.array(y_probs, dtype=np.float32)

    acc = float(np.mean(y_true == y_pred))

    # Confusion Matrix
    unique_present_classes = sorted(list(set(y_true.tolist() + y_pred.tolist())))
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    per_class = {}
    precisions, recalls, f1s = [], [], []

    for c in sorted(list(set(y_true.tolist()))):
        c_name = classes[c]
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1 = (2 * p * r) / max(1e-6, p + r)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

        per_class[c_name] = {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f1), 4),
            "support": int(np.sum(cm[c, :])),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn)
        }

    macro_p = float(np.mean(precisions))
    macro_r = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    # Calibration ECE (10 bins)
    corrects = (y_true == y_pred).astype(float)
    bin_boundaries = np.linspace(0, 1, 11)
    ece = 0.0
    for b in range(10):
        in_bin = (y_conf > bin_boundaries[b]) & (y_conf <= bin_boundaries[b+1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(corrects[in_bin])
            conf_in_bin = np.mean(y_conf[in_bin])
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin

    # Neutral rejection / false acceptance
    neutral_idx = classes.index("NEUTRAL")
    non_sign_mask = (y_true == neutral_idx)
    valid_sign_mask = (y_true != neutral_idx)

    far = float(np.sum(y_pred[non_sign_mask] != neutral_idx) / np.sum(non_sign_mask)) if np.any(non_sign_mask) else 0.0
    frr = float(np.sum(y_pred[valid_sign_mask] == neutral_idx) / np.sum(valid_sign_mask)) if np.any(valid_sign_mask) else 0.0

    # High-confidence misclassifications (conf >= 0.85 and incorrect)
    high_conf_errors = [d for d in detailed_results if not d["correct"] and d["confidence"] >= 0.85]

    return {
        "subset_name": subset_name,
        "sample_count": len(records_subset),
        "accuracy_pct": round(acc * 100, 2),
        "macro_precision_pct": round(macro_p * 100, 2),
        "macro_recall_pct": round(macro_r * 100, 2),
        "macro_f1_pct": round(macro_f1 * 100, 2),
        "ece": round(float(ece), 4),
        "far_pct": round(far * 100, 2),
        "frr_pct": round(frr * 100, 2),
        "high_confidence_error_count": len(high_conf_errors),
        "high_confidence_errors": high_conf_errors,
        "per_class": per_class,
        "detailed_results": detailed_results
    }

# 1. Full empirical set (N=42)
all_eval = evaluate_subset(records, "All Real-Human Empirical Captures (N=42)")

# 2. Strictly held-out test cohort (Signer_3, Signer_4, Signer_5, Signer_6)
test_records = [r for r in records if r["cohort_split"] == "test"]
test_eval = evaluate_subset(test_records, "Strictly Held-Out Human Signers (Signer_3..6, N=28)")

# 3. Letters A, B, C, D acceptance breakdown across all signers
abcd_records = [r for r in records if r["alphabet_label"] in ["A", "B", "C", "D"]]
abcd_eval = evaluate_subset(abcd_records, "Core Milestone Letters A, B, C, D (N=18)")

print("=" * 70)
print(f"EVALUATION 1: {all_eval['subset_name']}")
print(f"Accuracy: {all_eval['accuracy_pct']}% | Macro F1: {all_eval['macro_f1_pct']}% | ECE: {all_eval['ece']}")
print(f"FAR: {all_eval['far_pct']}% | FRR: {all_eval['frr_pct']}% | High-Conf Errors: {all_eval['high_confidence_error_count']}")

print("\n" + "=" * 70)
print(f"EVALUATION 2: {test_eval['subset_name']}")
print(f"Accuracy: {test_eval['accuracy_pct']}% | Macro F1: {test_eval['macro_f1_pct']}% | ECE: {test_eval['ece']}")
print(f"FAR: {test_eval['far_pct']}% | FRR: {test_eval['frr_pct']}% | High-Conf Errors: {test_eval['high_confidence_error_count']}")
print("\nPer-Class Breakdown (Held-Out Test Signers):")
for k, v in test_eval["per_class"].items():
    print(f"  Letter '{k}': Support={v['support']}, P={v['precision']}, R={v['recall']}, F1={v['f1']}")

print("\n" + "=" * 70)
print(f"EVALUATION 3: {abcd_eval['subset_name']}")
print(f"Accuracy: {abcd_eval['accuracy_pct']}% | Macro F1: {abcd_eval['macro_f1_pct']}%")
for k, v in abcd_eval["per_class"].items():
    print(f"  Letter '{k}': Support={v['support']}, TP={v['tp']}, FP={v['fp']}, FN={v['fn']}, P={v['precision']}, R={v['recall']}, F1={v['f1']}")

# Save full results to JSON
out_path = Path("backend/saved_models/real_human_evaluation_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({
        "all_empirical": all_eval,
        "strictly_held_out": test_eval,
        "core_abcd": abcd_eval
    }, f, indent=2)

print(f"\nFull detailed acceptance results saved to: {out_path}")

