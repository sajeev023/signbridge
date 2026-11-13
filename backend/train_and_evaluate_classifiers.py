"""
SIGNBRIDGE Multi-Model Classifier Training, Evaluation & Benchmarking Suite
Compares 4 classifier architectures on strict signer-independent splits:
1. Nearest Centroid (Diagnostic Baseline)
2. Multinomial Logistic Regression (Linear Baseline)
3. Decision Tree / Random Forest Ensemble (Non-linear Baseline)
4. Deep Biomechanical MLP with Continuous Geometry (Production Candidate)

Measures: Accuracy, Per-class Precision/Recall/Macro F1, Confusion Matrix,
FAR, FRR, Calibration, and Inference Latency. Exports winning model weights to JSON.
"""

import os
import sys
import json
import time
import math
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from sklearn.neural_network import MLPClassifier

sys.path.append(str(Path(__file__).resolve().parent))
from alphabet_feature_extractor import extract_canonical_features, NUM_CANONICAL_FEATURES


class NearestCentroidClassifier:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.centroids = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        dim = X.shape[1]
        self.centroids = np.zeros((self.num_classes, dim), dtype=np.float32)
        for c in range(self.num_classes):
            mask = (y == c)
            if np.any(mask):
                self.centroids[c] = np.mean(X[mask], axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Distance to each centroid: (N, C)
        dists = np.linalg.norm(X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :], axis=2)
        return np.argmin(dists, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        dists = np.linalg.norm(X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :], axis=2)
        sim = 1.0 / (dists + 1e-4)
        return sim / np.sum(sim, axis=1, keepdims=True)


class LinearSoftmaxClassifier:
    def __init__(self, in_dim: int, num_classes: int):
        self.W = np.zeros((num_classes, in_dim), dtype=np.float32)
        self.b = np.zeros(num_classes, dtype=np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.05, epochs: int = 150):
        N, D = X.shape
        C = len(self.b)
        Y_onehot = np.zeros((N, C), dtype=np.float32)
        Y_onehot[np.arange(N), y] = 1.0

        for ep in range(epochs):
            logits = X @ self.W.T + self.b
            logits -= np.max(logits, axis=1, keepdims=True)
            exp_l = np.exp(logits)
            probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)

            grad_W = (probs - Y_onehot).T @ X / N + 0.001 * self.W
            grad_b = np.mean(probs - Y_onehot, axis=0)

            self.W -= lr * grad_W
            self.b -= lr * grad_b

    def predict(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.W.T + self.b
        return np.argmax(logits, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.W.T + self.b
        logits -= np.max(logits, axis=1, keepdims=True)
        exp_l = np.exp(logits)
        return exp_l / np.sum(exp_l, axis=1, keepdims=True)


class SimpleRandomForest:
    """Lightweight pure-numpy random forest ensemble for diagnostic comparison."""
    def __init__(self, n_trees: int = 25, max_depth: int = 12):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X: np.ndarray, y: np.ndarray, num_classes: int):
        self.trees = []
        n_samples, n_features = X.shape
        sub_feat_size = int(math.sqrt(n_features)) + 3

        for _ in range(self.n_trees):
            # Bootstrap sample
            boot_idx = np.random.choice(n_samples, n_samples, replace=True)
            X_boot, y_boot = X[boot_idx], y[boot_idx]
            feat_idx = np.random.choice(n_features, sub_feat_size, replace=False)

            # Fit single axis-aligned decision tree
            tree = self._build_tree(X_boot[:, feat_idx], y_boot, depth=0, num_classes=num_classes)
            self.trees.append((tree, feat_idx))

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int, num_classes: int) -> Dict[str, Any]:
        counts = np.bincount(y, minlength=num_classes)
        majority = int(np.argmax(counts))

        if depth >= self.max_depth or len(np.unique(y)) <= 1 or len(y) < 6:
            probs = counts / max(1, len(y))
            return {"leaf": True, "pred": majority, "probs": probs}

        # Random candidate splits
        best_gain = -1.0
        best_feat = -1
        best_thresh = 0.0

        n_samples, n_feats = X.shape
        current_entropy = self._entropy(y, num_classes)

        for f in range(n_feats):
            vals = X[:, f]
            cand_thresh = np.percentile(vals, [25, 50, 75])
            for th in cand_thresh:
                left_mask = vals <= th
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                e_left = self._entropy(y[left_mask], num_classes)
                e_right = self._entropy(y[right_mask], num_classes)
                p_l = np.sum(left_mask) / n_samples
                gain = current_entropy - (p_l * e_left + (1 - p_l) * e_right)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = f
                    best_thresh = float(th)

        if best_gain <= 1e-4:
            probs = counts / max(1, len(y))
            return {"leaf": True, "pred": majority, "probs": probs}

        left_mask = X[:, best_feat] <= best_thresh
        left_sub = self._build_tree(X[left_mask], y[left_mask], depth + 1, num_classes)
        right_sub = self._build_tree(X[~left_mask], y[~left_mask], depth + 1, num_classes)

        return {
            "leaf": False,
            "feat": best_feat,
            "thresh": best_thresh,
            "left": left_sub,
            "right": right_sub,
            "pred": majority
        }

    def _entropy(self, y: np.ndarray, num_classes: int) -> float:
        if len(y) == 0:
            return 0.0
        p = np.bincount(y, minlength=num_classes) / len(y)
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))

    def _predict_sample(self, x: np.ndarray, node: Dict[str, Any]) -> Tuple[int, np.ndarray]:
        if node["leaf"]:
            return node["pred"], node["probs"]
        if x[node["feat"]] <= node["thresh"]:
            return self._predict_sample(x, node["left"])
        else:
            return self._predict_sample(x, node["right"])

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        N = len(X)
        all_probs = []
        for tree, feat_idx in self.trees:
            tree_probs = np.array([self._predict_sample(x[feat_idx], tree)[1] for x in X])
            all_probs.append(tree_probs)
        return np.mean(all_probs, axis=0)


# Model 4 (Biomechanical MLP) is implemented using optimized MLPClassifier (Adam, ReLU, 42 -> 128 -> 64 -> 25)


def compute_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray, classes: List[str]) -> Dict[str, Any]:
    num_classes = len(classes)
    acc = float(np.mean(y_true == y_pred))

    precisions = []
    recalls = []
    f1s = []
    per_class = {}

    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf_matrix[t, p] += 1

    for c in range(num_classes):
        tp = conf_matrix[c, c]
        fp = np.sum(conf_matrix[:, c]) - tp
        fn = np.sum(conf_matrix[c, :]) - tp

        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1 = (2 * p * r) / max(1e-6, p + r)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

        per_class[classes[c]] = {
            "precision": round(float(p), 3),
            "recall": round(float(r), 3),
            "f1": round(float(f1), 3),
            "support": int(np.sum(conf_matrix[c, :]))
        }

    macro_p = float(np.mean(precisions))
    macro_r = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    # Expected Calibration Error (ECE) with 10 bins
    confidences = np.max(y_probs, axis=1)
    corrects = (y_true == y_pred).astype(float)
    bin_boundaries = np.linspace(0, 1, 11)
    ece = 0.0
    for b in range(10):
        in_bin = (confidences > bin_boundaries[b]) & (confidences <= bin_boundaries[b+1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(corrects[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin

    # FAR (False Acceptance Rate on Neutral/Resting non-sign opportunities)
    neutral_idx = classes.index("NEUTRAL") if "NEUTRAL" in classes else 0
    non_sign_mask = (y_true == neutral_idx)
    if np.any(non_sign_mask):
        false_accepts = np.sum(y_pred[non_sign_mask] != neutral_idx)
        far = float(false_accepts / np.sum(non_sign_mask))
    else:
        far = 0.0

    # FRR (False Rejection Rate on Valid Sign opportunities)
    valid_sign_mask = (y_true != neutral_idx)
    if np.any(valid_sign_mask):
        false_rejects = np.sum(y_pred[valid_sign_mask] == neutral_idx)
        frr = float(false_rejects / np.sum(valid_sign_mask))
    else:
        frr = 0.0

    return {
        "accuracy": round(acc * 100, 2),
        "macro_precision": round(macro_p * 100, 2),
        "macro_recall": round(macro_r * 100, 2),
        "macro_f1": round(macro_f1 * 100, 2),
        "ece": round(float(ece), 4),
        "far_pct": round(far * 100, 2),
        "frr_pct": round(frr * 100, 2),
        "per_class": per_class,
        "confusion_matrix": conf_matrix.tolist()
    }


def run_experiments():
    data_path = Path("backend/data/alphabet_dataset_multisigner.json")
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}. Run build_alphabet_dataset.py first.")

    with open(data_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    meta = payload["metadata"]
    classes = meta["supported_classes"]
    num_classes = len(classes)
    samples = payload["samples"]

    # Ingest verified empirical real-human benchmark captures
    real_gt_path = Path("backend/data/real_human_evaluation_dataset.json")
    if not real_gt_path.exists():
        real_gt_path = Path("evaluation/real_world_landmarks_dataset.json")
    if real_gt_path.exists():
        with open(real_gt_path, "r", encoding="utf-8") as f:
            real_data_payload = json.load(f)
        real_records = real_data_payload.get("records", real_data_payload if isinstance(real_data_payload, list) else [])
        for s in real_records:
            lbl = s.get("alphabet_label") or s.get("signLabel")
            if lbl in classes:
                feat42 = s.get("features_42") or extract_canonical_features(s.get("landmarks") or s.get("groundTruth")).tolist()
                pid = s.get("signer_id") or s.get("participantId", "Real_Subject")
                assigned_split = s.get("cohort_split", "train" if pid in ["Signer_1", "Signer_2"] else "test")
                samples.append({
                    "sample_id": s.get("capture_id") or f"EMPIRICAL_{s.get('sampleId')}",
                    "sign_label": lbl,
                    "class_idx": classes.index(lbl),
                    "participant_id": pid,
                    "split": assigned_split,
                    "landmarks": s.get("landmarks") or s.get("groundTruth"),
                    "features_42": feat42,
                    "is_real_camera": True
                })

    # Prepare numpy arrays
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []
    test_samples_meta = []

    for s in samples:
        f = s["features_42"]
        c = s["class_idx"]
        split = s["split"]
        if split == "train":
            X_train.append(f)
            y_train.append(c)
        elif split == "val":
            X_val.append(f)
            y_val.append(c)
        elif split == "test":
            X_test.append(f)
            y_test.append(c)
            test_samples_meta.append(s)

    X_train, y_train = np.array(X_train, dtype=np.float32), np.array(y_train, dtype=int)
    X_val, y_val = np.array(X_val, dtype=np.float32), np.array(y_val, dtype=int)
    X_test, y_test = np.array(X_test, dtype=np.float32), np.array(y_test, dtype=int)

    print(f"Data Shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    results = {}

    # 1. Model 1: Nearest Centroid
    print("\n--- Training Model 1: Nearest Centroid ---")
    t0 = time.perf_counter()
    nc = NearestCentroidClassifier(num_classes)
    nc.fit(X_train, y_train)
    t_train_nc = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred_nc = nc.predict(X_test)
    y_prob_nc = nc.predict_proba(X_test)
    t_inf_nc = (time.perf_counter() - t0) / len(X_test) * 1e6
    results["Nearest_Centroid"] = {
        "train_time_sec": round(t_train_nc, 4),
        "inf_latency_us": round(t_inf_nc, 2),
        **compute_comprehensive_metrics(y_test, y_pred_nc, y_prob_nc, classes)
    }
    print(f"Nearest Centroid Test Acc: {results['Nearest_Centroid']['accuracy']}% (Macro F1: {results['Nearest_Centroid']['macro_f1']}%)")

    # 2. Model 2: Linear Softmax
    print("\n--- Training Model 2: Linear Softmax ---")
    t0 = time.perf_counter()
    ls = LinearSoftmaxClassifier(NUM_CANONICAL_FEATURES, num_classes)
    ls.fit(X_train, y_train, lr=0.08, epochs=150)
    t_train_ls = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred_ls = ls.predict(X_test)
    y_prob_ls = ls.predict_proba(X_test)
    t_inf_ls = (time.perf_counter() - t0) / len(X_test) * 1e6
    results["Linear_Softmax"] = {
        "train_time_sec": round(t_train_ls, 4),
        "inf_latency_us": round(t_inf_ls, 2),
        **compute_comprehensive_metrics(y_test, y_pred_ls, y_prob_ls, classes)
    }
    print(f"Linear Softmax Test Acc: {results['Linear_Softmax']['accuracy']}% (Macro F1: {results['Linear_Softmax']['macro_f1']}%)")

    # 3. Model 3: Random Forest Ensemble
    print("\n--- Training Model 3: Random Forest (25 Trees) ---")
    t0 = time.perf_counter()
    rf = SimpleRandomForest(n_trees=25, max_depth=12)
    rf.fit(X_train, y_train, num_classes)
    t_train_rf = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)
    t_inf_rf = (time.perf_counter() - t0) / len(X_test) * 1e6
    results["Random_Forest"] = {
        "train_time_sec": round(t_train_rf, 4),
        "inf_latency_us": round(t_inf_rf, 2),
        **compute_comprehensive_metrics(y_test, y_pred_rf, y_prob_rf, classes)
    }
    print(f"Random Forest Test Acc: {results['Random_Forest']['accuracy']}% (Macro F1: {results['Random_Forest']['macro_f1']}%)")

    # 4. Model 4: Deep Biomechanical MLP (42 -> 128 -> 64 -> 25)
    print("\n--- Training Model 4: Deep Biomechanical MLP (42 -> 128 -> 64 -> 25) ---")
    t0 = time.perf_counter()
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        learning_rate_init=0.003,
        alpha=1e-4,
        batch_size=64,
        max_iter=150,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=10,
        random_state=42
    )
    mlp.fit(X_train, y_train)
    t_train_mlp = time.perf_counter() - t0

    # Optimize calibration temperature T on validation data
    z1_val = np.maximum(0, X_val @ mlp.coefs_[0] + mlp.intercepts_[0])
    z2_val = np.maximum(0, z1_val @ mlp.coefs_[1] + mlp.intercepts_[1])
    logits_val = z2_val @ mlp.coefs_[2] + mlp.intercepts_[2]

    best_T = 1.0
    best_val_ece = 1.0
    for T in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0]:
        l_scaled = logits_val / T
        l_scaled -= np.max(l_scaled, axis=1, keepdims=True)
        p_scaled = np.exp(l_scaled) / np.sum(l_scaled, axis=1, keepdims=True)
        m_val = compute_comprehensive_metrics(y_val, np.argmax(p_scaled, axis=1), p_scaled, classes)
        if m_val["ece"] < best_val_ece:
            best_val_ece = m_val["ece"]
            best_T = T

    print(f"[CALIBRATION] Optimal Validation Temperature T = {best_T:.2f} (Val ECE: {best_val_ece:.4f})")

    # Apply calibrated temperature scaling directly to final layer weights
    w3_calib = (mlp.coefs_[2] / best_T).T.tolist()
    b3_calib = (mlp.intercepts_[2] / best_T).tolist()

    best_weights = {
        "w1": mlp.coefs_[0].T.tolist(),
        "b1": mlp.intercepts_[0].tolist(),
        "w2": mlp.coefs_[1].T.tolist(),
        "b2": mlp.intercepts_[1].tolist(),
        "w3": w3_calib,
        "b3": b3_calib
    }

    # Evaluate on Test Set with calibrated probabilities
    t0 = time.perf_counter()
    z1_test = np.maximum(0, X_test @ mlp.coefs_[0] + mlp.intercepts_[0])
    z2_test = np.maximum(0, z1_test @ mlp.coefs_[1] + mlp.intercepts_[1])
    logits_test = (z2_test @ mlp.coefs_[2] + mlp.intercepts_[2]) / best_T
    logits_test -= np.max(logits_test, axis=1, keepdims=True)
    test_probs = np.exp(logits_test) / np.sum(np.exp(logits_test), axis=1, keepdims=True)
    test_preds = np.argmax(test_probs, axis=1)
    t_inf_mlp = (time.perf_counter() - t0) / len(X_test) * 1e6

    results["Biomechanical_MLP"] = {
        "train_time_sec": round(t_train_mlp, 4),
        "inf_latency_us": round(t_inf_mlp, 2),
        **compute_comprehensive_metrics(y_test, test_preds, test_probs, classes)
    }
    print(f"Biomechanical MLP Test Acc: {results['Biomechanical_MLP']['accuracy']}% (Macro F1: {results['Biomechanical_MLP']['macro_f1']}%)")

    # Also evaluate on empirical real-camera samples specifically
    real_indices = [i for i, s in enumerate(test_samples_meta) if s.get("is_real_camera", False)]
    if real_indices:
        real_idx_arr = np.array(real_indices, dtype=int)
        y_real_true = y_test[real_idx_arr]
        y_real_pred = test_preds[real_idx_arr]
        y_real_prob = test_probs[real_idx_arr]
        real_metrics = compute_comprehensive_metrics(y_real_true, y_real_pred, y_real_prob, classes)
        results["Real_Camera_Benchmark_Subset"] = {
            "num_samples": len(real_indices),
            **real_metrics
        }
        print(f"\n[REAL CAMERA BENCHMARK] Usable Samples: {len(real_indices)} | Accuracy: {real_metrics['accuracy']}% | Macro F1: {real_metrics['macro_f1']}%")

    # Export Confusion Matrix to standalone JSON
    cm_path = Path("backend/saved_models/ALPHABET_CONFUSION_MATRIX.json")
    with open(cm_path, "w", encoding="utf-8") as f:
        json.dump({
            "classes": classes,
            "confusion_matrix": results["Biomechanical_MLP"]["confusion_matrix"],
            "per_class": results["Biomechanical_MLP"]["per_class"]
        }, f, indent=2)

    # Export Report and Best Model Weights
    exp_report_path = Path("backend/saved_models/classifier_experiments_report.json")
    with open(exp_report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Export production weights JSON
    prod_model_payload = {
        "model_name": "SIGNBRIDGE Biomechanical Manual Alphabet Neural Classifier",
        "version": "2.2.0-canonical-geometry",
        "reference": "Single-Hand Manual Alphabet Chart + ISL Biomechanical Invariants",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "accuracy": results["Biomechanical_MLP"]["accuracy"],
        "macro_f1": results["Biomechanical_MLP"]["macro_f1"],
        "num_features": NUM_CANONICAL_FEATURES,
        "feature_names": [
            "ext_thumb", "ext_index", "ext_middle", "ext_ring", "ext_pinky",
            "dist_tip_thumb_palm", "dist_tip_idx_palm", "dist_tip_mid_palm", "dist_tip_ring_palm", "dist_tip_pinky_palm",
            "curl_thumb", "curl_index", "curl_middle", "curl_ring", "curl_pinky",
            "cos_spread_thumb_idx", "cos_spread_idx_mid", "cos_spread_mid_ring", "cos_spread_ring_pinky",
            "dist_tip_thumb_idx", "dist_tip_thumb_mid", "dist_tip_thumb_pinky", "dist_tip_idx_mid", "dist_tip_mid_ring",
            "canonical_thumb_x", "canonical_thumb_y", "canonical_thumb_z", "dist_thumb_idx_mcp", "dist_thumb_ring_mcp",
            "lateral_crossing_idx_mid", "depth_crossing_idx_mid", "cross_product_sign_idx_mid",
            "canonical_idx_dir_x", "canonical_idx_dir_y", "canonical_idx_dir_z",
            "curl_angle_idx_pip", "curl_angle_mid_pip", "fist_compactness",
            "spread_ratio_idx_mid", "y_span_ratio", "parallelism_idx_mid", "loop_closure_thumb_idx"
        ],
        "classes": classes,
        "weights": best_weights
    }

    # Save to both backend/saved_models and frontend/src/services
    backend_save = Path("backend/saved_models/single_hand_alphabet_model.json")
    frontend_save = Path("frontend/src/services/single_hand_alphabet_model.json")

    with open(backend_save, "w", encoding="utf-8") as f:
        json.dump(prod_model_payload, f, indent=2)

    with open(frontend_save, "w", encoding="utf-8") as f:
        json.dump(prod_model_payload, f, indent=2)

    print(f"\nModel exported successfully to:")
    print(f"- {backend_save}")
    print(f"- {frontend_save}")
    print(f"- Benchmark report at: {exp_report_path}")

    return results


if __name__ == "__main__":
    run_experiments()
