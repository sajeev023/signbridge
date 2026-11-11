import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import (
    CLASS_NAMES, SEQUENCE_LENGTH, NUM_FEATURES, NUM_CLASSES,
    MODEL_CHECKPOINT_PATH, MODEL_METADATA_PATH, DATASET_FILE
)
from app.models.architecture import SignBridgeBiLSTM

def run_evaluation():
    print("=" * 70)
    print(" SIGNBRIDGE EXPERIMENTAL VALIDATION & PERFORMANCE EVALUATION")
    print("=" * 70)

    if not DATASET_FILE.exists():
        raise FileNotFoundError(f"Dataset file {DATASET_FILE} does not exist. Run train.py first.")

    if not MODEL_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint {MODEL_CHECKPOINT_PATH} does not exist. Run train.py first.")

    dataset = np.load(DATASET_FILE, allow_pickle=True)
    X_test = dataset["X_test"]
    y_test = dataset["y_test"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation Hardware: {device}")
    print(f"Test Set Volume: {len(X_test)} dynamic sequences across {NUM_CLASSES} classes")

    # Load model
    model = SignBridgeBiLSTM(
        input_dim=NUM_FEATURES,
        hidden_dim=128,
        num_layers=2,
        num_classes=NUM_CLASSES,
        dropout=0.3
    ).to(device)

    model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH, map_location=device))
    model.eval()

    # Inference & Latency benchmarking
    latencies_ms = []
    y_pred = []
    y_probs = []

    # Warmup
    dummy = torch.zeros(1, SEQUENCE_LENGTH, NUM_FEATURES, device=device)
    for _ in range(10):
        _ = model(dummy)

    print("\nRunning test evaluation and measuring per-sequence inference latency...")
    with torch.no_grad():
        for i in range(len(X_test)):
            sample = torch.tensor(X_test[i:i+1], dtype=torch.float32, device=device)
            t0 = time.perf_counter()
            logits, attn = model(sample)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)
            pred_class = int(np.argmax(probs))
            y_pred.append(pred_class)
            y_probs.append(probs)

    y_pred = np.array(y_pred)
    latencies_ms = np.array(latencies_ms)

    # Metrics computation
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=list(range(NUM_CLASSES)), average=None, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro', zero_division=0
    )
    conf_mat = confusion_matrix(y_test, y_pred, labels=list(range(NUM_CLASSES)))

    mean_lat = float(np.mean(latencies_ms))
    p95_lat = float(np.percentile(latencies_ms, 95))
    min_lat = float(np.min(latencies_ms))
    max_lat = float(np.max(latencies_ms))

    print(f"\n--- OVERALL METRICS ---")
    print(f"Overall Accuracy:  {acc * 100:.2f}%")
    print(f"Macro Precision:   {macro_p * 100:.2f}%")
    print(f"Macro Recall:      {macro_r * 100:.2f}%")
    print(f"Macro F1-Score:    {macro_f1 * 100:.2f}%")
    print(f"\n--- LATENCY BENCHMARK ---")
    print(f"Mean Latency:      {mean_lat:.2f} ms")
    print(f"95th Percentile:   {p95_lat:.2f} ms")
    print(f"Throughput:        {1000.0 / mean_lat:.1f} windows/sec")

    print("\n--- PER-CLASS PERFORMANCE TABLE ---")
    header = f"{'Class ID':<14} | {'Sign Label':<16} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}"
    print(header)
    print("-" * len(header))
    per_class_results = []
    for i, name in enumerate(CLASS_NAMES):
        row_str = f"{name:<14} | {CLASS_NAMES[i]:<16} | {precision[i]*100:>8.2f}% | {recall[i]*100:>8.2f}% | {f1[i]*100:>8.2f}% | {support[i]:>8}"
        print(row_str)
        per_class_results.append({
            "class_id": name,
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i])
        })

    print("\n--- CONFUSION MATRIX ---")
    print(conf_mat)

    # Compile experiment output object
    evaluation_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "hardware": str(device),
        "test_samples": len(y_test),
        "overall_accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "latency_ms": {
            "mean": mean_lat,
            "p95": p95_lat,
            "min": min_lat,
            "max": max_lat
        },
        "per_class": per_class_results,
        "confusion_matrix": conf_mat.tolist()
    }

    results_path = Path(__file__).resolve().parent / "saved_models" / "evaluation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2)
    print(f"\nDetailed evaluation results saved to {results_path}")

    return evaluation_results

if __name__ == "__main__":
    run_evaluation()
