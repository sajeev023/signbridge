"""
SIGNBRIDGE TITAN: Systematic Architecture & Fusion Ablation Experiment
Executes controlled ablations across all 5 candidate architectures (A through E),
evaluates fusion mechanisms (Concat vs Gated vs Cross-Attention), and measures
the contribution of learned topological graph convolutions and auxiliary features.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import torch
from torch.utils.data import DataLoader

from .architectures import (
    ArchitectureA_SpatialGNN,
    ArchitectureB_BiLSTMAttention,
    ArchitectureC_STTransformer,
    ArchitectureD_DualStreamGated,
    TitanHybridModel,
)
from .train import (
    NUM_CLASSES,
    TitanDataset,
    load_multisigner_dataset,
    run_architecture_training,
    set_seed,
)


def run_full_ablation_study(
    dataset_path: Path,
    output_path: Path,
    num_epochs: int = 20,
    batch_size: int = 64,
) -> Dict[str, Any]:
    set_seed(42)
    device = torch.device("cpu")

    print("[TITAN ABLATION] Loading multi-signer dataset...")
    raw = load_multisigner_dataset(dataset_path)

    # Participant-based partitioning
    splits = raw["splits"]
    train_idx = [i for i, s in enumerate(splits) if s == "train"]
    val_idx = [i for i, s in enumerate(splits) if s == "val"]
    test_idx = [i for i, s in enumerate(splits) if s == "test"]

    # Fallback to stratified 70/15/15 if splits not pre-populated
    if len(val_idx) == 0 or len(test_idx) == 0:
        total = len(raw["labels"])
        indices = np.random.permutation(total)
        n_train = int(total * 0.70)
        n_val = int(total * 0.15)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

    def build_subset_dataset(indices: List[int]) -> TitanDataset:
        return TitanDataset(
            node_features=raw["nodes"][indices],
            temporal_sequences=raw["sequences"][indices],
            aux_features=raw["aux"][indices],
            labels=raw["labels"][indices],
            signer_ids=[raw["signers"][i] for i in indices],
            provenance=[raw["provenance"][i] for i in indices],
        )

    ds_train = build_subset_dataset(train_idx)
    ds_val = build_subset_dataset(val_idx)
    ds_test = build_subset_dataset(test_idx)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    print(f"[TITAN ABLATION] Partitions: Train={len(ds_train)}, Val={len(ds_val)}, Test={len(ds_test)}")

    results = {}

    # 1. Architecture A: Spatial GNN + MLP (Static Handshape)
    print("\n--- Training Architecture A: Spatial GNN + MLP ---")
    model_a = ArchitectureA_SpatialGNN(num_classes=NUM_CLASSES, hidden_dim=64, out_dim=64)
    results["Arch_A_Spatial_GNN"] = run_architecture_training(
        "Arch_A_Spatial_GNN", model_a, train_loader, val_loader, test_loader, num_epochs=num_epochs, device=device
    )

    # 2. Architecture B: BiLSTM + Attention (Temporal Baseline)
    print("\n--- Training Architecture B: BiLSTM + Attention ---")
    model_b = ArchitectureB_BiLSTMAttention(num_classes=NUM_CLASSES, in_dim=42, hidden_dim=64, out_dim=64)
    results["Arch_B_BiLSTM_Attention"] = run_architecture_training(
        "Arch_B_BiLSTM_Attention", model_b, train_loader, val_loader, test_loader, num_epochs=num_epochs, device=device
    )

    # 3. Architecture C: Spatial-Temporal Transformer (ST-Transformer)
    print("\n--- Training Architecture C: ST-Transformer (Causal) ---")
    model_c = ArchitectureC_STTransformer(num_classes=NUM_CLASSES, in_dim=42, d_model=64, causal=True)
    results["Arch_C_ST_Transformer"] = run_architecture_training(
        "Arch_C_ST_Transformer", model_c, train_loader, val_loader, test_loader, num_epochs=num_epochs, device=device
    )

    # 4. Architecture D: Dual-Stream Spatial GCN + BiLSTM (Gated Fusion)
    print("\n--- Training Architecture D: Dual-Stream Gated ---")
    model_d = ArchitectureD_DualStreamGated(num_classes=NUM_CLASSES, d_model=64)
    results["Arch_D_Dual_Stream_Gated"] = run_architecture_training(
        "Arch_D_Dual_Stream_Gated", model_d, train_loader, val_loader, test_loader, num_epochs=num_epochs, device=device
    )

    # 5. Architecture E (TITAN Full): Hybrid Graph Encoder + Cross-Attention
    print("\n--- Training Architecture E: TITAN Full (Cross-Attention) ---")
    model_e = TitanHybridModel(num_classes=NUM_CLASSES, d_model=64, fusion_type="cross_attention", causal=True)
    results["Arch_E_TITAN_Full_CrossAttention"] = run_architecture_training(
        "Arch_E_TITAN_Full_CrossAttention", model_e, train_loader, val_loader, test_loader, num_epochs=num_epochs, device=device
    )

    # 6. Fusion Ablation: TITAN with Concatenation
    print("\n--- Training Fusion Ablation: TITAN with Concatenation ---")
    model_e_concat = TitanHybridModel(num_classes=NUM_CLASSES, d_model=64, fusion_type="concat", causal=True)
    results["Ablation_TITAN_Concat"] = run_architecture_training(
        "Ablation_TITAN_Concat", model_e_concat, train_loader, val_loader, test_loader, num_epochs=num_epochs, device=device
    )

    # 7. Fusion Ablation: TITAN with Gated Fusion
    print("\n--- Training Fusion Ablation: TITAN with Gated Fusion ---")
    model_e_gated = TitanHybridModel(num_classes=NUM_CLASSES, d_model=64, fusion_type="gated", causal=True)
    results["Ablation_TITAN_Gated"] = run_architecture_training(
        "Ablation_TITAN_Gated", model_e_gated, train_loader, val_loader, test_loader, num_epochs=num_epochs, device=device
    )

    # 8. Feature Ablation: Spatial GNN WITHOUT auxiliary 42-feature input
    print("\n--- Training Feature Ablation: Spatial GNN (Graph Nodes Only, No Aux 42) ---")
    model_a_no_aux = ArchitectureA_SpatialGNN(num_classes=NUM_CLASSES, hidden_dim=64, out_dim=64, use_aux_features=False)
    results["Ablation_Spatial_GNN_NoAux"] = run_architecture_training(
        "Ablation_Spatial_GNN_NoAux", model_a_no_aux, train_loader, val_loader, test_loader, num_epochs=num_epochs, device=device
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[TITAN ABLATION] All 8 experiments completed. Results saved to {output_path}")
    return results


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    ds_path = base_dir / "data" / "alphabet_dataset_multisigner.json"
    out_path = base_dir / "saved_models" / "titan_ablation_results.json"
    run_full_ablation_study(ds_path, out_path, num_epochs=20)
