"""
SIGNBRIDGE Experiment: Kinivi-Inspired Static Regularization & Dropout Exploration
Evaluates whether input feature dropout and modified MLP layer topologies
improve generalization on SIGNBRIDGE's canonical 42-feature contract under
strict signer-independent partitions.

Evaluates 4 Candidates:
Candidate A: Production Compact MLP (42 -> 64 -> 32 -> 25, Alpha = 1e-3, No Dropout)
Candidate B: Compact MLP with Input Dropout (p = 0.15 on canonical 42 features)
Candidate C: Compact MLP with Input + Hidden Dropout (p_in = 0.15, p_hidden = 0.25)
Candidate D: Kinivi-Inspired Topology adapted to 42 features (42 -> 48 -> 24 -> 25, Alpha = 2e-3)
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPClassifier

sys.path.append(str(Path(__file__).resolve().parent))
from alphabet_feature_extractor import extract_canonical_features

CLASSES = [
    "NEUTRAL", "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y"
]

def apply_input_dropout(X: np.ndarray, drop_p: float = 0.15) -> np.ndarray:
    """Randomly zeros out drop_p fraction of input features during training."""
    mask = np.random.binomial(1, 1.0 - drop_p, size=X.shape).astype(np.float32)
    return X * mask

def evaluate_predictions(y_true, y_pred, y_probs):
    acc = float(np.mean(y_true == y_pred)) * 100
    classes_present = sorted(list(set(y_true.tolist() + y_pred.tolist())))
    f1s = []
    for c in sorted(list(set(y_true.tolist()))):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1 = (2 * p * r) / max(1e-6, p + r)
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s)) * 100 if f1s else 0.0

    # Calibration ECE (10 bins)
    conf = np.max(y_probs, axis=1) if y_probs.size > 0 else np.zeros_like(y_pred, dtype=float)
    corrects = (y_true == y_pred).astype(float)
    bin_bounds = np.linspace(0, 1, 11)
    ece = 0.0
    for b in range(10):
        in_bin = (conf > bin_bounds[b]) & (conf <= bin_bounds[b+1])
        prop = np.mean(in_bin)
        if prop > 0:
            ece += np.abs(np.mean(corrects[in_bin]) - np.mean(conf[in_bin])) * prop

    return {
        "accuracy": round(acc, 2),
        "macro_f1": round(macro_f1, 2),
        "ece": round(float(ece), 4)
    }

def run_static_fusion_experiments():
    workspace_root = Path(__file__).resolve().parent.parent

    # Load datasets
    with open(workspace_root / "backend" / "data" / "alphabet_dataset_multisigner.json", "r") as f:
        synth_data = json.load(f)
    with open(workspace_root / "backend" / "data" / "real_human_evaluation_dataset.json", "r") as f:
        real_data = json.load(f)

    train_samples = []
    val_samples = []
    test_synth_samples = []
    test_real_samples = []

    for s in synth_data["samples"]:
        lbl = s["sign_label"]
        if lbl not in CLASSES: continue
        c_idx = CLASSES.index(lbl)
        item = {"feats": s["features_42"], "class": c_idx, "label": lbl}
        if s["split"] == "train": train_samples.append(item)
        elif s["split"] == "val": val_samples.append(item)
        elif s["split"] == "test": test_synth_samples.append(item)

    for r in real_data["records"]:
        lbl = r["alphabet_label"]
        if lbl not in CLASSES: continue
        c_idx = CLASSES.index(lbl)
        item = {"feats": r["features_42"], "class": c_idx, "label": lbl, "signer": r["signer_id"]}
        if r["cohort_split"] == "train": train_samples.append(item)
        else: test_real_samples.append(item)

    X_train = np.array([s["feats"] for s in train_samples], dtype=np.float32)
    y_train = np.array([s["class"] for s in train_samples], dtype=int)

    X_val = np.array([s["feats"] for s in val_samples], dtype=np.float32)
    y_val = np.array([s["class"] for s in val_samples], dtype=int)

    X_test_s = np.array([s["feats"] for s in test_synth_samples], dtype=np.float32)
    y_test_s = np.array([s["class"] for s in test_synth_samples], dtype=int)

    X_test_r = np.array([s["feats"] for s in test_real_samples], dtype=np.float32)
    y_test_r = np.array([s["class"] for s in test_real_samples], dtype=int)

    results = {}
    np.random.seed(42)

    # Candidate A: Baseline Production Compact MLP (64, 32, alpha=1e-3)
    print("\n--- Evaluating Candidate A: Baseline Compact MLP (42 -> 64 -> 32 -> 25, Alpha=1e-3) ---")
    mlpa = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=150, alpha=1e-3, random_state=42)
    mlpa.fit(X_train, y_train)
    p_s_a = mlpa.predict_proba(X_test_s)
    p_r_a = mlpa.predict_proba(X_test_r)
    results["Candidate_A_Baseline_Compact"] = {
        "description": "Production Compact MLP (64, 32, alpha=1e-3)",
        "synthetic_test": evaluate_predictions(y_test_s, np.argmax(p_s_a, axis=1), p_s_a),
        "real_held_out_test": evaluate_predictions(y_test_r, np.argmax(p_r_a, axis=1), p_r_a)
    }

    # Candidate B: Compact MLP with Input Dropout (p = 0.15)
    print("\n--- Evaluating Candidate B: Compact MLP + Input Feature Dropout (p=0.15) ---")
    X_train_drop = apply_input_dropout(X_train, drop_p=0.15)
    mlpb = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=150, alpha=1e-3, random_state=42)
    mlpb.fit(X_train_drop, y_train)
    p_s_b = mlpb.predict_proba(X_test_s)
    p_r_b = mlpb.predict_proba(X_test_r)
    results["Candidate_B_Input_Dropout"] = {
        "description": "Compact MLP + Input Dropout (p=0.15)",
        "synthetic_test": evaluate_predictions(y_test_s, np.argmax(p_s_b, axis=1), p_s_b),
        "real_held_out_test": evaluate_predictions(y_test_r, np.argmax(p_r_b, axis=1), p_r_b)
    }

    # Candidate C: Compact MLP with Stronger Regularization (alpha=3e-3)
    print("\n--- Evaluating Candidate C: Compact MLP + Stronger L2 Regularization (alpha=3e-3) ---")
    mlpc = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=150, alpha=3e-3, random_state=42)
    mlpc.fit(X_train, y_train)
    p_s_c = mlpc.predict_proba(X_test_s)
    p_r_c = mlpc.predict_proba(X_test_r)
    results["Candidate_C_Stronger_Regularization"] = {
        "description": "Compact MLP + Stronger L2 (alpha=3e-3)",
        "synthetic_test": evaluate_predictions(y_test_s, np.argmax(p_s_c, axis=1), p_s_c),
        "real_held_out_test": evaluate_predictions(y_test_r, np.argmax(p_r_c, axis=1), p_r_c)
    }

    # Candidate D: Kinivi-Inspired Compact Topology (42 -> 48 -> 24 -> 25)
    print("\n--- Evaluating Candidate D: Kinivi-Inspired Topology (42 -> 48 -> 24 -> 25, Alpha=2e-3) ---")
    mlpd = MLPClassifier(hidden_layer_sizes=(48, 24), max_iter=150, alpha=2e-3, random_state=42)
    mlpd.fit(X_train, y_train)
    p_s_d = mlpd.predict_proba(X_test_s)
    p_r_d = mlpd.predict_proba(X_test_r)
    results["Candidate_D_Kinivi_Inspired_Topology"] = {
        "description": "Kinivi-Inspired Compact Topology (48, 24, alpha=2e-3)",
        "synthetic_test": evaluate_predictions(y_test_s, np.argmax(p_s_d, axis=1), p_s_d),
        "real_held_out_test": evaluate_predictions(y_test_r, np.argmax(p_r_d, axis=1), p_r_d)
    }

    print("\n" + "=" * 80)
    print("STATIC RECOGNITION ARCHITECTURE COMPARISON RESULTS")
    print("=" * 80)
    for name, res in results.items():
        s_acc = res["synthetic_test"]["accuracy"]
        r_acc = res["real_held_out_test"]["accuracy"]
        r_f1 = res["real_held_out_test"]["macro_f1"]
        r_ece = res["real_held_out_test"]["ece"]
        print(f"{name:<35} | Synth Acc: {s_acc:>6.2f}% | Real Acc: {r_acc:>6.2f}% | Real F1: {r_f1:>6.2f}% | ECE: {r_ece:.4f}")

    out_file = workspace_root / "backend" / "saved_models" / "static_fusion_experiment_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results

if __name__ == "__main__":
    run_static_fusion_experiments()
