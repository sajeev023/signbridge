"""
Export and compute comprehensive evaluation metrics for TITAN Real-World Generalization
"""

import json
import numpy as np
import torch
from backend.titan.asl_architecture import (
    ASLSpatialGNN,
    ASL_CLASS_TO_IDX,
    ASL_VOCABULARY,
    build_adjacency_matrix,
    compute_normalized_laplacian
)
from backend.titan.spatial_encoder import landmarks_to_node_features
from backend.alphabet_feature_extractor import extract_canonical_features
from backend.train_real_world_generalization import run_experiment

# Run training experiment
model, best_state, _ = run_experiment()

# Save backup of current model
curr_model_path = 'backend/saved_models/titan_asl_spatial_model.json'
with open(curr_model_path) as f:
    curr_data = json.load(f)

backup_path = 'backend/saved_models/titan_asl_spatial_model.pre_gen.json'
with open(backup_path, 'w') as f:
    json.dump(curr_data, f, indent=2)
print(f"Backed up pre-generalization model to {backup_path}")

# Update state dict with new trained weights (including learned_adj)
new_state_dict = {}
for k, v in best_state.items():
    new_state_dict[k] = v.cpu().numpy().tolist()

curr_data['state_dict'] = new_state_dict
curr_data['architecture'] = 'Adaptive Kinematic GCN (AK-GCN) + Dual Readout + Aux42 Biomechanics'
curr_data['temperature'] = 1.08
curr_data['min_confidence'] = 0.55
curr_data['entropy_threshold'] = 1.35

with open(curr_model_path, 'w') as f:
    json.dump(curr_data, f, indent=2)
print(f"Saved upgraded AK-GCN weights to {curr_model_path}")

# Also update frontend model copy
frontend_model_path = 'frontend/src/services/titan_asl_spatial_model.json'
with open(frontend_model_path, 'w') as f:
    json.dump(curr_data, f, indent=2)
print(f"Synchronized weights to {frontend_model_path}")

# Evaluate detailed per-class metrics on held-out test cohort
records = json.load(open('backend/titan/datasets/asl_alphabet/metadata/real_human_benchmark.json'))['records']
test_records = [r for r in records if r.get('cohort_split') == 'test']

adj = torch.tensor(compute_normalized_laplacian(build_adjacency_matrix()), dtype=torch.float32)
temp = 1.08

all_preds = []
all_targets = []
all_probs = []

for r in test_records:
    lbl = r.get('alphabet_label') or r.get('target_letter')
    t_idx = ASL_CLASS_TO_IDX[lbl]
    nodes = landmarks_to_node_features(r['landmarks'])
    feats = r.get('features_42') or extract_canonical_features(r['landmarks']).tolist()
    
    with torch.no_grad():
        out = model(torch.tensor(nodes).unsqueeze(0), adj, torch.tensor(feats).unsqueeze(0))
        probs = torch.softmax(out / temp, dim=1)[0].numpy()
        pred = int(np.argmax(probs))
    
    all_preds.append(pred)
    all_targets.append(t_idx)
    all_probs.append(probs)

classes_present = sorted(list(set(all_targets)))
per_class_recall = {}
per_class_precision = {}

for c_idx in classes_present:
    c_name = ASL_VOCABULARY[c_idx]
    tp = sum(1 for p, t in zip(all_preds, all_targets) if p == c_idx and t == c_idx)
    fn = sum(1 for p, t in zip(all_preds, all_targets) if p != c_idx and t == c_idx)
    fp = sum(1 for p, t in zip(all_preds, all_targets) if p == c_idx and t != c_idx)
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    per_class_recall[c_name] = recall
    per_class_precision[c_name] = precision

macro_recall = np.mean(list(per_class_recall.values()))
macro_prec = np.mean(list(per_class_precision.values()))
macro_f1 = 2 * (macro_prec * macro_recall) / (macro_prec + macro_recall + 1e-12)

print("\n=== PER-CLASS METRICS (HELD-OUT REAL-HUMAN COHORT) ===")
print(f"Macro Recall:    {macro_recall*100:.2f}%")
print(f"Macro Precision: {macro_prec*100:.2f}%")
print(f"Macro F1 Score:  {macro_f1:.4f}")

metrics_summary = {
    'accuracy': sum(1 for p, t in zip(all_preds, all_targets) if p == t) / len(all_targets),
    'macro_f1': float(macro_f1),
    'macro_recall': float(macro_recall),
    'macro_precision': float(macro_prec),
    'per_class_recall': per_class_recall,
    'per_class_precision': per_class_precision
}

with open('backend/saved_models/real_world_generalization_results.json', 'w') as f:
    json.dump(metrics_summary, f, indent=2)
print("Saved metrics summary to backend/saved_models/real_world_generalization_results.json")
