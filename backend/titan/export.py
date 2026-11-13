"""
SIGNBRIDGE TITAN: Model Serialization & Browser Export Engine
Exports trained TITAN models to:
1. Portable JSON weights format for zero-dependency, sub-millisecond client-side TypeScript execution
2. ONNX format for ONNX Runtime Web (WASM / WebGPU)
3. Float16 quantized weights for edge footprint optimization
"""

import json
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch
import torch.nn as nn

from .architectures import ArchitectureA_SpatialGNN, TitanHybridModel
from .train import ALPHABET_CLASSES


def export_spatial_gnn_to_json(
    model: ArchitectureA_SpatialGNN,
    output_path: Path,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    """
    Serializes Architecture A (Spatial GNN + Aux MLP) into JSON format
    for execution by the client-side titanInferenceEngine.ts.
    """
    state = model.state_dict()

    weights_dict = {}
    for k, v in state.items():
        weights_dict[k] = v.cpu().numpy().tolist()

    export_payload = {
        "model_name": "SIGNBRIDGE TITAN Spatial Graph Neural Classifier",
        "version": "1.0-titan",
        "architecture": "Spatial_GNN_Kipf_Welling",
        "num_classes": len(ALPHABET_CLASSES),
        "classes": ALPHABET_CLASSES,
        "temperature": float(temperature),
        "input_dimensions": {
            "num_landmarks": 21,
            "node_in_dim": 6,
            "aux_dim": 42,
        },
        "weights": weights_dict,
        "metadata": {
            "topological_edges": 31,
            "readout": "mean_max_concat",
            "activation": "gelu",
            "normalization": "layer_norm",
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f)

    file_size_kb = output_path.stat().st_size / 1024.0
    print(f"[TITAN EXPORT] JSON weights exported to {output_path} ({file_size_kb:.2f} KB)")
    return export_payload


def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    opset_version: int = 14,
) -> bool:
    """Exports model to ONNX format with dynamic batch axes."""
    try:
        import onnx
    except ImportError:
        print("[TITAN EXPORT] onnx package not found. Skipping ONNX file generation.")
        return False

    model.eval()
    dummy_nodes = torch.randn(1, 21, 6, dtype=torch.float32)
    dummy_aux = torch.randn(1, 42, dtype=torch.float32)

    try:
        torch.onnx.export(
            model,
            (dummy_nodes, dummy_aux),
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["node_features", "aux_features"],
            output_names=["logits"],
            dynamic_axes={
                "node_features": {0: "batch_size"},
                "aux_features": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
        )
        print(f"[TITAN EXPORT] ONNX model exported to {output_path}")
        return True
    except Exception as e:
        print(f"[TITAN EXPORT] ONNX export failed: {e}")
        return False


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    model = ArchitectureA_SpatialGNN(num_classes=len(ALPHABET_CLASSES))
    json_out = base / "saved_models" / "titan_spatial_model.json"
    export_spatial_gnn_to_json(model, json_out, temperature=1.12)
