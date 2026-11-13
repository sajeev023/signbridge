"""
Trains ArchitectureA_SpatialGNN and saves the trained weights to JSON.
"""

from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader

from backend.titan.architectures import ArchitectureA_SpatialGNN
from backend.titan.train import (
    NUM_CLASSES,
    TitanDataset,
    load_multisigner_dataset,
    run_architecture_training,
    set_seed,
)
from backend.titan.export import export_spatial_gnn_to_json


def main():
    set_seed(42)
    device = torch.device("cpu")
    base = Path(__file__).resolve().parent.parent
    ds_path = base / "data" / "alphabet_dataset_multisigner.json"

    raw = load_multisigner_dataset(ds_path)
    total = len(raw["labels"])
    n_train = int(total * 0.70)
    n_val = int(total * 0.15)

    ds_train = TitanDataset(
        node_features=raw["nodes"][:n_train],
        temporal_sequences=raw["sequences"][:n_train],
        aux_features=raw["aux"][:n_train],
        labels=raw["labels"][:n_train],
        signer_ids=raw["signers"][:n_train],
        provenance=raw["provenance"][:n_train],
    )
    ds_val = TitanDataset(
        node_features=raw["nodes"][n_train:n_train+n_val],
        temporal_sequences=raw["sequences"][n_train:n_train+n_val],
        aux_features=raw["aux"][n_train:n_train+n_val],
        labels=raw["labels"][n_train:n_train+n_val],
        signer_ids=raw["signers"][n_train:n_train+n_val],
        provenance=raw["provenance"][n_train:n_train+n_val],
    )
    ds_test = TitanDataset(
        node_features=raw["nodes"][n_train+n_val:],
        temporal_sequences=raw["sequences"][n_train+n_val:],
        aux_features=raw["aux"][n_train+n_val:],
        labels=raw["labels"][n_train+n_val:],
        signer_ids=raw["signers"][n_train+n_val:],
        provenance=raw["provenance"][n_train+n_val:],
    )

    train_loader = DataLoader(ds_train, batch_size=64, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=64, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=64, shuffle=False)

    model = ArchitectureA_SpatialGNN(num_classes=NUM_CLASSES, hidden_dim=64, out_dim=64)

    print("[TITAN TRAIN] Training Architecture A (Spatial GNN)...")
    res = run_architecture_training(
        "Arch_A_Spatial_GNN",
        model,
        train_loader,
        val_loader,
        test_loader,
        num_epochs=12,
        device=device,
    )
    print(f"[TITAN TRAIN] Accuracy: {res['accuracy']*100:.2f}%, Macro F1: {res['macro_f1']*100:.2f}%, Temp: {res['optimal_temperature']}")

    b_out = base / "saved_models" / "titan_spatial_model.json"
    f_out = base.parent / "frontend" / "src" / "services" / "titan_spatial_model.json"

    export_spatial_gnn_to_json(model, b_out, temperature=res["optimal_temperature"])
    export_spatial_gnn_to_json(model, f_out, temperature=res["optimal_temperature"])


if __name__ == "__main__":
    main()
