"""
SIGNBRIDGE 4-Fold Leave-One-Signer-Out (LOSO) Cross-Validation Suite
Evaluates true signer generalization across all 4 signer cohorts.
"""

import sys
import json
from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPClassifier

sys.path.append(str(Path(__file__).resolve().parent))
from train_and_evaluate_classifiers import compute_comprehensive_metrics

def run_loso():
    data_path = Path("backend/data/alphabet_dataset_multisigner.json")
    with open(data_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    samples = d["samples"]
    classes = d["metadata"]["supported_classes"]
    signers = sorted(list(set(s["participant_id"] for s in samples)))

    print(f"Executing 4-Fold Leave-One-Signer-Out (LOSO) Cross-Validation on: {signers}")

    loso_results = {}
    for held_out in signers:
        X_tr, y_tr = [], []
        X_te, y_te = [], []
        for s in samples:
            pid = s["participant_id"]
            f = s["features_42"]
            c = s["class_idx"]
            if pid == held_out:
                X_te.append(f)
                y_te.append(c)
            else:
                X_tr.append(f)
                y_tr.append(c)

        X_tr, y_tr = np.array(X_tr, dtype=np.float32), np.array(y_tr, dtype=int)
        X_te, y_te = np.array(X_te, dtype=np.float32), np.array(y_te, dtype=int)

        mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", max_iter=120, random_state=42)
        mlp.fit(X_tr, y_tr)
        probs = mlp.predict_proba(X_te)
        preds = np.argmax(probs, axis=1)

        metrics = compute_comprehensive_metrics(y_te, preds, probs, classes)
        loso_results[held_out] = {
            "test_samples": len(y_te),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "far_pct": metrics["far_pct"],
            "frr_pct": metrics["frr_pct"],
            "ece": metrics["ece"]
        }
        print(f"Fold [{held_out} Held-Out]: Acc={metrics['accuracy']}%, Macro F1={metrics['macro_f1']}%, ECE={metrics['ece']}")

    accs = [loso_results[k]["accuracy"] for k in loso_results]
    f1s = [loso_results[k]["macro_f1"] for k in loso_results]
    mean_acc = float(np.mean(accs))
    std_acc = float(np.std(accs))
    mean_f1 = float(np.mean(f1s))
    std_f1 = float(np.std(f1s))

    print(f"\nMean LOSO Accuracy: {mean_acc:.2f}% +/- {std_acc:.2f}%")
    print(f"Mean LOSO Macro F1: {mean_f1:.2f}% +/- {std_f1:.2f}%")

    out_path = Path("backend/saved_models/loso_evaluation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "folds": loso_results,
            "mean_accuracy": round(mean_acc, 2),
            "std_accuracy": round(std_acc, 2),
            "mean_macro_f1": round(mean_f1, 2),
            "std_macro_f1": round(std_f1, 2)
        }, f, indent=2)
    print(f"Saved LOSO results to {out_path}")

if __name__ == "__main__":
    run_loso()

