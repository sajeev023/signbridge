"""
Executes robustness evaluation and saves output to titan_robustness_results.json.
"""

import json
from pathlib import Path
import torch

from backend.titan.architectures import ArchitectureA_SpatialGNN
from backend.titan.robustness import evaluate_robustness_battery
from backend.titan.train import CLASS_TO_IDX

def main():
    base = Path(__file__).resolve().parent.parent
    multi_path = base / "data" / "alphabet_dataset_multisigner.json"
    out_path = base / "saved_models" / "titan_robustness_results.json"

    with open(multi_path, "r", encoding="utf-8") as f:
        samples = json.load(f)["samples"]

    # Filter test samples (50 clean samples for responsive battery)
    test_cohort = [s for s in samples if s.get("split") == "test"]
    if len(test_cohort) == 0:
        test_cohort = samples[-200:]
    test_cohort = test_cohort[:50]

    # Format samples with consistent class index
    clean = []
    for s in test_cohort:
        lbl = s.get("sign_label")
        if lbl in CLASS_TO_IDX:
            clean.append({
                "landmarks": s["landmarks"],
                "label_idx": CLASS_TO_IDX[lbl],
            })

    model_spatial = ArchitectureA_SpatialGNN(num_classes=25, hidden_dim=64, out_dim=64)
    weights_path = base / "saved_models" / "titan_spatial_model.json"
    if weights_path.exists():
        with open(weights_path, "r", encoding="utf-8") as wf:
            w_data = json.load(wf)["weights"]
            state = {k: torch.tensor(v, dtype=torch.float32) for k, v in w_data.items()}
            model_spatial.load_state_dict(state, strict=False)

    res_spatial = evaluate_robustness_battery(model_spatial, clean)

    output = {
        "TITAN_Spatial_GNN": res_spatial,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"[TITAN ROBUSTNESS] Evaluation complete. Saved to {out_path}")

if __name__ == "__main__":
    main()
