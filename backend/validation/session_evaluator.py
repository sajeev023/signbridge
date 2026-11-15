"""
SIGNBRIDGE Physical Validation Engine — Session Evaluation & Analysis
Computes empirical metrics across physical-human validation sessions:
- Per-class accuracy, false positive rate, false rejection rate
- Multi-cohort generalization (Training vs Held-Out Evaluation)
- Confusion matrix and error breakdown
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
SESSIONS_DIR = WORKSPACE_ROOT / "backend" / "data" / "physical_validation_sessions"


def evaluate_sessions(sessions_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Evaluates all recorded physical validation sessions.
    """
    s_dir = sessions_dir or SESSIONS_DIR
    if not s_dir.exists():
        return {
            "status": "NO_SESSIONS_RECORDED",
            "total_sessions": 0,
            "total_trials": 0,
            "message": "No physical-human sessions have been conducted yet."
        }

    session_files = list(s_dir.glob("*.json"))
    if not session_files:
        return {
            "status": "NO_SESSIONS_RECORDED",
            "total_sessions": 0,
            "total_trials": 0,
            "message": "No session JSON logs found in directory."
        }

    cohort_stats = {
        "TRAIN_COHORT": {"trials": 0, "correct": 0, "by_class": defaultdict(lambda: {"trials": 0, "correct": 0})},
        "HELDOUT_EVAL_COHORT": {"trials": 0, "correct": 0, "by_class": defaultdict(lambda: {"trials": 0, "correct": 0})}
    }
    confusion_pairs = defaultdict(int)
    error_counts = defaultdict(int)
    total_trials = 0
    total_correct = 0

    for s_file in session_files:
        with open(s_file, "r", encoding="utf-8") as f:
            sess = json.load(f)

        cohort = sess.get("cohort", "TRAIN_COHORT")
        if cohort not in cohort_stats:
            cohort_stats[cohort] = {"trials": 0, "correct": 0, "by_class": defaultdict(lambda: {"trials": 0, "correct": 0})}

        for tr in sess.get("trials", []):
            gt = tr["ground_truth_letter"]
            pred = tr.get("predicted_letter")
            is_corr = tr["is_correct"]
            err_cls = tr.get("error_classification", "NONE")

            total_trials += 1
            if is_corr:
                total_correct += 1

            cohort_stats[cohort]["trials"] += 1
            if is_corr:
                cohort_stats[cohort]["correct"] += 1

            cohort_stats[cohort]["by_class"][gt]["trials"] += 1
            if is_corr:
                cohort_stats[cohort]["by_class"][gt]["correct"] += 1

            if not is_corr:
                confusion_pairs[f"{gt}->{pred}"] += 1
                error_counts[err_cls] += 1

    overall_acc = (total_correct / total_trials * 100.0) if total_trials > 0 else 0.0

    return {
        "status": "EVALUATION_COMPLETE",
        "total_sessions": len(session_files),
        "total_trials": total_trials,
        "total_correct": total_correct,
        "overall_accuracy_pct": overall_acc,
        "cohorts": {
            k: {
                "trials": v["trials"],
                "correct": v["correct"],
                "accuracy_pct": (v["correct"] / v["trials"] * 100.0) if v["trials"] > 0 else 0.0,
                "per_class": {
                    cls: {
                        "trials": d["trials"],
                        "correct": d["correct"],
                        "accuracy_pct": (d["correct"] / d["trials"] * 100.0) if d["trials"] > 0 else 0.0
                    }
                    for cls, d in v["by_class"].items()
                }
            }
            for k, v in cohort_stats.items()
        },
        "top_confusions": dict(sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:10]),
        "error_taxonomy_breakdown": dict(error_counts)
    }
