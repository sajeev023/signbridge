"""
SIGNBRIDGE Scientific Experiment: Kinivi Preprocessing & Architecture Comparison
Evaluates Kinivi's coordinate normalization (relative 2D pixel coordinates / max_abs)
against SIGNBRIDGE's canonical 42-feature biomechanical invariant representation
under identical training, validation, held-out real-human test partitions, and rotation perturbation.
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPClassifier

sys.path.append(str(Path(__file__).resolve().parent))
from alphabet_feature_extractor import extract_canonical_features
from build_alphabet_dataset import create_rotation_matrix

CLASSES = [
    "NEUTRAL", "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y"
]
NUM_CLASSES = len(CLASSES)


def kinivi_preprocess_landmarks(landmarks: list) -> np.ndarray:
    """
    Kinivi's exact preprocessing function from app.py:
    1. Subtract landmark 0 (wrist) (x_0, y_0) from all 21 points
    2. Flatten to 42 elements
    3. Divide by max(abs(coordinates))
    """
    if not landmarks or len(landmarks) < 21:
        return np.zeros(42, dtype=np.float32)

    pts = np.zeros((21, 2), dtype=np.float32)
    for i in range(21):
        pts[i, 0] = landmarks[i]["x"]
        pts[i, 1] = landmarks[i]["y"]

    base_x, base_y = pts[0, 0], pts[0, 1]
    rel_pts = pts - np.array([base_x, base_y], dtype=np.float32)
    flat = rel_pts.flatten()

    max_val = np.max(np.abs(flat))
    if max_val > 1e-6:
        return flat / max_val
    return flat


def rotate_landmarks_2d(landmarks: list, angle_deg: float) -> list:
    """Rotates landmarks in image plane around wrist."""
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    rot_mat = np.array([[c, -s], [s, c]], dtype=np.float32)

    wrist_x, wrist_y = landmarks[0]["x"], landmarks[0]["y"]
    rotated = []
    for lm in landmarks:
        dx = lm["x"] - wrist_x
        dy = lm["y"] - wrist_y
        r = np.dot(rot_mat, np.array([dx, dy]))
        rotated.append({
            "x": float(r[0] + wrist_x),
            "y": float(r[1] + wrist_y),
            "z": lm.get("z", 0.0)
        })
    return rotated


def run_experiment():
    workspace_root = Path(__file__).resolve().parent.parent

    # Load multi-signer dataset
    with open(workspace_root / "backend" / "data" / "alphabet_dataset_multisigner.json", "r") as f:
        synth_data = json.load(f)

    # Load real human evaluation dataset
    with open(workspace_root / "backend" / "data" / "real_human_evaluation_dataset.json", "r") as f:
        real_data = json.load(f)

    train_synth = [s for s in synth_data["samples"] if s["split"] == "train" and s["sign_label"] in CLASSES]
    val_synth = [s for s in synth_data["samples"] if s["split"] == "val" and s["sign_label"] in CLASSES]
    test_synth = [s for s in synth_data["samples"] if s["split"] == "test" and s["sign_label"] in CLASSES]

    train_real = [r for r in real_data["records"] if r["cohort_split"] == "train" and r["alphabet_label"] in CLASSES]
    test_real = [r for r in real_data["records"] if r["cohort_split"] == "test" and r["alphabet_label"] in CLASSES]

    print(f"Data Loaded: Train (Synth: {len(train_synth)}, Real: {len(train_real)}), Val ({len(val_synth)}), Test Synth ({len(test_synth)}), Held-Out Real Test ({len(test_real)})")

    # Extract Features for Representation 1: Kinivi (2D Rel / MaxAbs)
    X_train_kinivi = []
    y_train = []
    for s in train_synth:
        X_train_kinivi.append(kinivi_preprocess_landmarks(s["landmarks"]))
        y_train.append(CLASSES.index(s["sign_label"]))
    for r in train_real:
        X_train_kinivi.append(kinivi_preprocess_landmarks(r["landmarks"]))
        y_train.append(CLASSES.index(r["alphabet_label"]))

    X_train_kinivi = np.array(X_train_kinivi, dtype=np.float32)
    y_train = np.array(y_train, dtype=int)

    X_test_synth_kinivi = np.array([kinivi_preprocess_landmarks(s["landmarks"]) for s in test_synth], dtype=np.float32)
    y_test_synth = np.array([CLASSES.index(s["sign_label"]) for s in test_synth], dtype=int)

    X_test_real_kinivi = np.array([kinivi_preprocess_landmarks(r["landmarks"]) for r in test_real], dtype=np.float32)
    y_test_real = np.array([CLASSES.index(r["alphabet_label"]) for r in test_real], dtype=int)

    # Extract Features for Representation 2: SIGNBRIDGE (Canonical 42 Biomechanical)
    X_train_signbridge = []
    for s in train_synth:
        X_train_signbridge.append(extract_canonical_features(s["landmarks"]))
    for r in train_real:
        X_train_signbridge.append(extract_canonical_features(r["landmarks"]))
    X_train_signbridge = np.array(X_train_signbridge, dtype=np.float32)

    X_test_synth_signbridge = np.array([extract_canonical_features(s["landmarks"]) for s in test_synth], dtype=np.float32)
    X_test_real_signbridge = np.array([extract_canonical_features(r["landmarks"]) for r in test_real], dtype=np.float32)

    # Rotation Robustness Test Sets (20 deg and 40 deg in-plane rotation)
    rot_20_kinivi = np.array([kinivi_preprocess_landmarks(rotate_landmarks_2d(r["landmarks"], 20.0)) for r in test_real], dtype=np.float32)
    rot_20_signbridge = np.array([extract_canonical_features(rotate_landmarks_2d(r["landmarks"], 20.0)) for r in test_real], dtype=np.float32)

    rot_40_kinivi = np.array([kinivi_preprocess_landmarks(rotate_landmarks_2d(r["landmarks"], 40.0)) for r in test_real], dtype=np.float32)
    rot_40_signbridge = np.array([extract_canonical_features(rotate_landmarks_2d(r["landmarks"], 40.0)) for r in test_real], dtype=np.float32)

    results = {}

    # MODEL A: Kinivi Preprocessing + Kinivi Architecture (Dense(20) -> Dense(10))
    print("\n--- Training Model A: Kinivi Preprocessing + Kinivi Exact MLP (42 -> 20 -> 10 -> 25) ---")
    mlp_kinivi_orig = MLPClassifier(hidden_layer_sizes=(20, 10), max_iter=150, alpha=1e-3, random_state=42)
    mlp_kinivi_orig.fit(X_train_kinivi, y_train)

    acc_synth_a = float(np.mean(mlp_kinivi_orig.predict(X_test_synth_kinivi) == y_test_synth)) * 100
    acc_real_a = float(np.mean(mlp_kinivi_orig.predict(X_test_real_kinivi) == y_test_real)) * 100
    acc_rot20_a = float(np.mean(mlp_kinivi_orig.predict(rot_20_kinivi) == y_test_real)) * 100
    acc_rot40_a = float(np.mean(mlp_kinivi_orig.predict(rot_40_kinivi) == y_test_real)) * 100

    results["Kinivi_Orig_Architecture"] = {
        "preprocessing": "Raw 2D Rel Coords (MaxAbs norm)",
        "topology": "42 -> 20 -> 10 -> 25",
        "synth_test_acc": round(acc_synth_a, 2),
        "real_held_out_acc": round(acc_real_a, 2),
        "rot_20deg_acc": round(acc_rot20_a, 2),
        "rot_40deg_acc": round(acc_rot40_a, 2)
    }
    print(f"Model A: Synth Acc={acc_synth_a:.2f}%, Real Held-Out Acc={acc_real_a:.2f}%, Rot 20deg={acc_rot20_a:.2f}%, Rot 40deg={acc_rot40_a:.2f}%")

    # MODEL B: Kinivi Preprocessing + SIGNBRIDGE Compact MLP (42 -> 64 -> 32 -> 25)
    print("\n--- Training Model B: Kinivi Preprocessing + Compact MLP (42 -> 64 -> 32 -> 25) ---")
    mlp_kinivi_scaled = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=150, alpha=1e-3, random_state=42)
    mlp_kinivi_scaled.fit(X_train_kinivi, y_train)

    acc_synth_b = float(np.mean(mlp_kinivi_scaled.predict(X_test_synth_kinivi) == y_test_synth)) * 100
    acc_real_b = float(np.mean(mlp_kinivi_scaled.predict(X_test_real_kinivi) == y_test_real)) * 100
    acc_rot20_b = float(np.mean(mlp_kinivi_scaled.predict(rot_20_kinivi) == y_test_real)) * 100
    acc_rot40_b = float(np.mean(mlp_kinivi_scaled.predict(rot_40_kinivi) == y_test_real)) * 100

    results["Kinivi_Preprocessing_Larger_MLP"] = {
        "preprocessing": "Raw 2D Rel Coords (MaxAbs norm)",
        "topology": "42 -> 64 -> 32 -> 25",
        "synth_test_acc": round(acc_synth_b, 2),
        "real_held_out_acc": round(acc_real_b, 2),
        "rot_20deg_acc": round(acc_rot20_b, 2),
        "rot_40deg_acc": round(acc_rot40_b, 2)
    }
    print(f"Model B: Synth Acc={acc_synth_b:.2f}%, Real Held-Out Acc={acc_real_b:.2f}%, Rot 20deg={acc_rot20_b:.2f}%, Rot 40deg={acc_rot40_b:.2f}%")

    # MODEL C: SIGNBRIDGE Canonical 42 Biomechanical Features + Compact MLP (42 -> 64 -> 32 -> 25)
    print("\n--- Training Model C: SIGNBRIDGE Canonical 42 Biomechanical Features + Compact MLP ---")
    mlp_sb = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=150, alpha=1e-3, random_state=42)
    mlp_sb.fit(X_train_signbridge, y_train)

    acc_synth_c = float(np.mean(mlp_sb.predict(X_test_synth_signbridge) == y_test_synth)) * 100
    acc_real_c = float(np.mean(mlp_sb.predict(X_test_real_signbridge) == y_test_real)) * 100
    acc_rot20_c = float(np.mean(mlp_sb.predict(rot_20_signbridge) == y_test_real)) * 100
    acc_rot40_c = float(np.mean(mlp_sb.predict(rot_40_signbridge) == y_test_real)) * 100

    results["SIGNBRIDGE_Biomechanical_Canonical"] = {
        "preprocessing": "Canonical 42 Invariant Biomechanical Features",
        "topology": "42 -> 64 -> 32 -> 25",
        "synth_test_acc": round(acc_synth_c, 2),
        "real_held_out_acc": round(acc_real_c, 2),
        "rot_20deg_acc": round(acc_rot20_c, 2),
        "rot_40deg_acc": round(acc_rot40_c, 2)
    }
    print(f"Model C: Synth Acc={acc_synth_c:.2f}%, Real Held-Out Acc={acc_real_c:.2f}%, Rot 20deg={acc_rot20_c:.2f}%, Rot 40deg={acc_rot40_c:.2f}%")

    # Save experiment results
    out_file = workspace_root / "backend" / "saved_models" / "kinivi_comparison_experiment_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nExperiment complete. Results saved to: {out_file}")
    return results

if __name__ == "__main__":
    run_experiment()

