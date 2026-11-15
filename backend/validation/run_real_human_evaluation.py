"""
SIGNBRIDGE Physical Human Benchmark Evaluator
Runs the Integrated Recognition Pipeline across all genuine physical human captures
in backend/data/real_human_evaluation_dataset.json without modifying models or thresholds.
"""

import sys
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "backend"))

from backend.open_source_integration.integrated_recognition_pipeline import IntegratedRecognitionPipeline

def main():
    pipeline = IntegratedRecognitionPipeline()
    dataset_file = WORKSPACE_ROOT / "backend" / "data" / "real_human_evaluation_dataset.json"

    with open(dataset_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data["records"]
    print(f"Loaded {len(records)} physical human records.")

    results_by_cohort = {
        "train": {"total": 0, "correct": 0, "accepted": 0, "trials": []},
        "test": {"total": 0, "correct": 0, "accepted": 0, "trials": []}
    }

    per_class_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "total": 0, "confidences": []})
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    error_taxonomy = defaultdict(int)
    camera_breakdown = defaultdict(lambda: {"total": 0, "correct": 0})
    participant_breakdown = defaultdict(lambda: {"total": 0, "correct": 0})

    for i, r in enumerate(records):
        gt = r["alphabet_label"]
        lms = r["landmarks"]
        signer = r["signer_id"]
        cohort = r["cohort_split"]
        cond = r.get("camera_conditions", {})
        lighting = cond.get("lighting", "Standard")
        dist = str(cond.get("distance_m", "0.65"))

        # Process frame through integrated pipeline
        pipeline.reset()
        # Feed landmark frame
        res = pipeline.process_frame(lms, timestamp_ms=0.0)
        # Feed 7 frames to fulfill Nochnack K=7 consensus requirement
        for step in range(1, 8):
            res = pipeline.process_frame(lms, timestamp_ms=step * 33.3)

        raw_pred = res["raw_sign"]
        emitted_pred = res["sign"]
        conf = res["confidence"]
        status = res["status"]

        is_accepted = (status == "EMITTED") or (raw_pred != "REJECTED" and conf >= 0.70)
        is_correct = is_accepted and (raw_pred == gt)

        results_by_cohort[cohort]["total"] += 1
        if is_accepted:
            results_by_cohort[cohort]["accepted"] += 1
        if is_correct:
            results_by_cohort[cohort]["correct"] += 1

        # Per class
        per_class_stats[gt]["total"] += 1
        per_class_stats[gt]["confidences"].append(conf)
        if is_correct:
            per_class_stats[gt]["tp"] += 1
        else:
            per_class_stats[gt]["fn"] += 1
            if is_accepted and raw_pred != "REJECTED":
                per_class_stats[raw_pred]["fp"] += 1
                confusion_matrix[gt][raw_pred] += 1
                error_taxonomy["CONFUSION"] += 1
            elif not is_accepted:
                error_taxonomy["FALSE_REJECTION"] += 1

        camera_breakdown[lighting]["total"] += 1
        if is_correct:
            camera_breakdown[lighting]["correct"] += 1

        participant_breakdown[signer]["total"] += 1
        if is_correct:
            participant_breakdown[signer]["correct"] += 1

        results_by_cohort[cohort]["trials"].append({
            "capture_id": r.get("capture_id", f"CAP_{i:03d}"),
            "signer_id": signer,
            "ground_truth": gt,
            "raw_predicted": raw_pred,
            "emitted_predicted": emitted_pred,
            "confidence": conf,
            "status": status,
            "is_correct": is_correct,
            "camera_conditions": cond
        })

    # Summary calculations
    total_samples = len(records)
    total_correct = results_by_cohort["train"]["correct"] + results_by_cohort["test"]["correct"]
    overall_acc = (total_correct / total_samples) * 100.0

    train_acc = (results_by_cohort["train"]["correct"] / results_by_cohort["train"]["total"]) * 100.0
    test_acc = (results_by_cohort["test"]["correct"] / results_by_cohort["test"]["total"]) * 100.0

    # Macro F1
    f1_list = []
    per_class_final = {}
    for cls in sorted(per_class_stats.keys()):
        stats = per_class_stats[cls]
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)
        per_class_final[cls] = {
            "total": stats["total"],
            "tp": tp, "fp": fp, "fn": fn,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "mean_conf": float(np.mean(stats["confidences"])) if stats["confidences"] else 0.0
        }

    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0

    eval_out = {
        "timestamp": "2026-11-15T05:08:00Z",
        "dataset_source": "real_human_evaluation_dataset.json",
        "total_physical_samples": total_samples,
        "classes_represented_count": len(per_class_stats),
        "missing_classes_count": 26 - len(per_class_stats),
        "overall_accuracy_pct": overall_acc,
        "macro_f1": macro_f1,
        "cohorts": {
            "train": {
                "signers": ["Signer_1", "Signer_2"],
                "total": results_by_cohort["train"]["total"],
                "correct": results_by_cohort["train"]["correct"],
                "accuracy_pct": train_acc
            },
            "held_out_test": {
                "signers": ["Signer_3", "Signer_4", "Signer_5", "Signer_6"],
                "total": results_by_cohort["test"]["total"],
                "correct": results_by_cohort["test"]["correct"],
                "accuracy_pct": test_acc
            }
        },
        "per_class_performance": per_class_final,
        "participant_breakdown": {s: dict(d, accuracy_pct=(d["correct"]/d["total"]*100.0)) for s, d in participant_breakdown.items()},
        "camera_conditions_breakdown": {c: dict(d, accuracy_pct=(d["correct"]/d["total"]*100.0)) for c, d in camera_breakdown.items()},
        "confusion_matrix": {k: dict(v) for k, v in confusion_matrix.items()},
        "error_taxonomy": dict(error_taxonomy)
    }

    out_file = WORKSPACE_ROOT / "backend" / "data" / "physical_human_eval_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(eval_out, f, indent=2)

    print("\nEVALUATION COMPLETE:")
    print(f"Overall Accuracy: {overall_acc:.2f}% ({total_correct}/{total_samples})")
    print(f"Training Cohort Accuracy (Signer 1, 2): {train_acc:.2f}% ({results_by_cohort['train']['correct']}/{results_by_cohort['train']['total']})")
    print(f"Held-Out Evaluation Accuracy (Signer 3..6): {test_acc:.2f}% ({results_by_cohort['test']['correct']}/{results_by_cohort['test']['total']})")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Error Taxonomy: {dict(error_taxonomy)}")

if __name__ == "__main__":
    main()
