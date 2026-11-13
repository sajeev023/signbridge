"""
SIGNBRIDGE TITAN: ASL Alphabet Multi-Model Benchmark & Training Suite
Executes controlled training and evaluation across 8 candidate architectures
under strict signer-independent splits.
"""

import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from backend.titan.asl_architecture import (
    ASL_VOCABULARY,
    ASL_CLASS_TO_IDX,
    ASL_STATIC_CLASSES,
    ASL_DYNAMIC_CLASSES,
    ASLSpatialGNN,
    ASLTrajectoryEncoder,
    ASLTitanHybridModel,
    ASLUncertaintyCalibrator
)
from backend.titan.graph_topology import (
    NUM_HAND_LANDMARKS,
    build_adjacency_matrix,
    compute_normalized_laplacian,
    get_default_normalized_adjacency
)
from backend.titan.spatial_encoder import landmarks_to_node_features
from backend.alphabet_feature_extractor import extract_canonical_features
from backend.titan.uncertainty import compute_ece, compute_brier_score


class ASLDataset(Dataset):
    """
    Standardized dataset loading both static handshapes and dynamic episodes with in-memory tensor caching.
    """
    def __init__(self, samples: List[Dict], canonical_cache: Optional[Dict] = None):
        self.samples = samples
        adj = build_adjacency_matrix()
        self.norm_adj = torch.tensor(compute_normalized_laplacian(adj), dtype=torch.float32)

        # Cache canonical templates for dynamic trajectory contexts
        self.canonical_cache = canonical_cache or {}
        if not self.canonical_cache:
            for s in samples:
                lbl = s.get('sign_label')
                if lbl in ['I', 'D', 'NEUTRAL'] and lbl not in self.canonical_cache:
                    if 'landmarks' in s and 'features_42' in s:
                        self.canonical_cache[lbl] = (s['landmarks'], s['features_42'])

        # Precompute all tensors in memory for fast training loops
        N = len(samples)
        self.node_features = torch.zeros(N, 21, 6, dtype=torch.float32)
        self.aux_features = torch.zeros(N, 42, dtype=torch.float32)
        self.trajectories = torch.zeros(N, 16, 2, dtype=torch.float32)
        self.motion_energies = torch.zeros(N, dtype=torch.float32)
        self.class_indices = torch.zeros(N, dtype=torch.long)
        self.is_dynamics = torch.zeros(N, dtype=torch.bool)
        self.signer_ids = []

        for idx, s in enumerate(samples):
            lbl = s['sign_label']
            class_idx = s['class_idx']
            is_dynamic = s['is_dynamic']
            self.class_indices[idx] = class_idx
            self.is_dynamics[idx] = is_dynamic
            self.signer_ids.append(s.get('signer_id', 'Unknown'))

            if is_dynamic:
                traj_feats = np.array(s['features_32'], dtype=np.float32).reshape(16, 2)
                self.trajectories[idx] = torch.tensor(traj_feats)
                ctx = 'I' if lbl == 'J' else ('D' if lbl == 'Z' else 'NEUTRAL')
                if ctx in self.canonical_cache:
                    lm, af = self.canonical_cache[ctx]
                    self.node_features[idx] = torch.tensor(landmarks_to_node_features(lm))
                    self.aux_features[idx] = torch.tensor(af, dtype=torch.float32)
                self.motion_energies[idx] = 1.0
            else:
                self.node_features[idx] = torch.tensor(landmarks_to_node_features(s['landmarks']))
                self.aux_features[idx] = torch.tensor(s['features_42'], dtype=torch.float32)
                self.motion_energies[idx] = 0.0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {
            'node_features': self.node_features[idx],
            'aux_features': self.aux_features[idx],
            'trajectory': self.trajectories[idx],
            'motion_energy': self.motion_energies[idx],
            'class_idx': self.class_indices[idx],
            'is_dynamic': self.is_dynamics[idx],
            'signer_id': self.signer_ids[idx]
        }


def train_model(model, train_loader, val_loader, epochs=25, lr=1e-3, weight_decay=1e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    adj = train_loader.dataset.norm_adj

    best_val_acc = 0.0
    best_weights = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            optimizer.zero_grad()
            nf = batch['node_features']
            af = batch['aux_features']
            traj = batch['trajectory']
            me = batch['motion_energy']
            y = batch['class_idx']

            if isinstance(model, ASLTitanHybridModel):
                logits = model(nf, adj, af, traj, me)
            elif isinstance(model, ASLSpatialGNN):
                logits = model(nf, adj, af)
            else:
                logits = model(nf, adj, af)

            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        scheduler.step()

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                nf = batch['node_features']
                af = batch['aux_features']
                traj = batch['trajectory']
                me = batch['motion_energy']
                y = batch['class_idx']

                if isinstance(model, ASLTitanHybridModel):
                    logits = model(nf, adj, af, traj, me)
                elif isinstance(model, ASLSpatialGNN):
                    logits = model(nf, adj, af)
                else:
                    logits = model(nf, adj, af)

                preds = torch.argmax(logits, dim=-1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_acc = val_correct / max(1, val_total)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights:
        model.load_state_dict(best_weights)
    return model


def evaluate_model(model, test_loader, num_classes=27):
    model.eval()
    adj = test_loader.dataset.norm_adj

    all_logits = []
    all_targets = []
    all_is_dynamic = []
    all_signers = []

    # Measure latency
    latencies = []

    with torch.no_grad():
        for batch in test_loader:
            nf = batch['node_features']
            af = batch['aux_features']
            traj = batch['trajectory']
            me = batch['motion_energy']
            y = batch['class_idx']
            dyn = batch['is_dynamic']
            signers = batch['signer_id']

            t0 = time.perf_counter()
            if isinstance(model, ASLTitanHybridModel):
                logits = model(nf, adj, af, traj, me)
            elif isinstance(model, ASLSpatialGNN):
                logits = model(nf, adj, af)
            else:
                logits = model(nf, adj, af)
            t1 = time.perf_counter()

            latencies.append((t1 - t0) / y.size(0))

            all_logits.append(logits.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_is_dynamic.append(dyn.cpu().numpy())
            all_signers.extend(signers)

    logits_arr = np.concatenate(all_logits, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)
    is_dyn_arr = np.concatenate(all_is_dynamic, axis=0)

    # Compute metrics
    preds = np.argmax(logits_arr, axis=-1)
    acc = float(np.mean(preds == targets_arr))

    # Static vs dynamic accuracy
    static_mask = ~is_dyn_arr
    dyn_mask = is_dyn_arr

    static_acc = float(np.mean(preds[static_mask] == targets_arr[static_mask])) if np.any(static_mask) else 0.0
    dyn_acc = float(np.mean(preds[dyn_mask] == targets_arr[dyn_mask])) if np.any(dyn_mask) else 0.0

    # Calibration & Brier
    exp_l = np.exp(logits_arr - np.max(logits_arr, axis=-1, keepdims=True))
    probs_arr = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
    ece = compute_ece(probs_arr, targets_arr)
    brier = compute_brier_score(probs_arr, targets_arr, num_classes=num_classes)

    # Per-class metrics & confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for p, t in zip(preds, targets_arr):
        cm[t, p] += 1

    per_class_recall = {}
    per_class_precision = {}
    f1_list = []

    for c_idx in range(num_classes):
        c_name = ASL_VOCABULARY[c_idx]
        tp = cm[c_idx, c_idx]
        fn = np.sum(cm[c_idx, :]) - tp
        fp = np.sum(cm[:, c_idx]) - tp

        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class_recall[c_name] = rec
        per_class_precision[c_name] = prec
        f1_list.append(f1)

    macro_f1 = float(np.mean(f1_list))

    # Confusion analysis for key pairs
    key_pairs = [
        ('A', 'S'), ('M', 'N'), ('N', 'T'), ('M', 'T'),
        ('U', 'V'), ('G', 'H'), ('K', 'P'), ('D', 'I'),
        ('E', 'S'), ('C', 'O'), ('I', 'J'), ('D', 'Z')
    ]
    pair_confusions = {}
    for c1, c2 in key_pairs:
        idx1 = ASL_CLASS_TO_IDX[c1]
        idx2 = ASL_CLASS_TO_IDX[c2]
        c1_as_c2 = int(cm[idx1, idx2])
        c2_as_c1 = int(cm[idx2, idx1])
        c1_total = int(np.sum(cm[idx1, :]))
        c2_total = int(np.sum(cm[idx2, :]))
        pair_confusions[f'{c1}_vs_{c2}'] = {
            f'{c1}_misclassified_as_{c2}': c1_as_c2,
            f'{c2}_misclassified_as_{c1}': c2_as_c1,
            f'{c1}_rate': float(c1_as_c2 / max(1, c1_total)),
            f'{c2}_rate': float(c2_as_c1 / max(1, c2_total))
        }

    # Selective classification coverage at 0.70 threshold
    probs = np.exp(logits_arr - np.max(logits_arr, axis=-1, keepdims=True))
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    max_probs = np.max(probs, axis=-1)
    accepted = max_probs >= 0.70
    coverage = float(np.mean(accepted))
    acc_under_rejection = float(np.mean(preds[accepted] == targets_arr[accepted])) if np.any(accepted) else 0.0

    # FAR and FRR
    # FAR: False acceptance rate when model is wrong but confident >= 0.70
    incorrect = preds != targets_arr
    far = float(np.sum(incorrect & accepted) / max(1, np.sum(incorrect)))
    # FRR: False rejection rate when model is correct but rejected < 0.70
    correct_mask = preds == targets_arr
    frr = float(np.sum(correct_mask & (~accepted)) / max(1, np.sum(correct_mask)))

    median_latency_ms = float(np.median(latencies) * 1000)
    p95_latency_ms = float(np.percentile(latencies, 95) * 1000)

    # Param count & size
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_kb = float(params * 4 / 1024)

    return {
        'accuracy': acc,
        'static_accuracy': static_acc,
        'dynamic_accuracy': dyn_acc,
        'macro_f1': macro_f1,
        'ece': ece,
        'brier_score': brier,
        'coverage': coverage,
        'accuracy_under_rejection': acc_under_rejection,
        'far': far,
        'frr': frr,
        'latency_median_ms': median_latency_ms,
        'latency_p95_ms': p95_latency_ms,
        'trainable_parameters': params,
        'model_size_kb': model_size_kb,
        'per_class_recall': per_class_recall,
        'per_class_precision': per_class_precision,
        'pair_confusions': pair_confusions,
        'confusion_matrix': cm.tolist()
    }


def main():
    splits_dir = r'C:\Users\ADMIN\.gemini\antigravity\scratch\signbridge\backend\titan\datasets\asl_alphabet\splits'
    with open(os.path.join(splits_dir, 'train.json')) as f:
        train_samples = json.load(f)['samples']
    with open(os.path.join(splits_dir, 'val.json')) as f:
        val_samples = json.load(f)['samples']
    with open(os.path.join(splits_dir, 'test.json')) as f:
        test_samples = json.load(f)['samples']

    print(f'Loaded splits: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}')

    train_ds = ASLDataset(train_samples)
    val_ds = ASLDataset(val_samples, canonical_cache=train_ds.canonical_cache)
    test_ds = ASLDataset(test_samples, canonical_cache=train_ds.canonical_cache)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    results = {}

    print('\n--- Training Model 6: Reference-Aligned Spatial GNN ---')
    sys.stdout.flush()
    model_6 = ASLSpatialGNN(node_in_dim=6, hidden_dim=64, aux_dim=42, num_classes=27, dropout=0.15)
    model_6 = train_model(model_6, train_loader, val_loader, epochs=12, lr=2e-3)
    results['Model_6_Reference_Aligned_Spatial_GNN'] = evaluate_model(model_6, test_loader)
    res6 = results['Model_6_Reference_Aligned_Spatial_GNN']
    print(f"Model 6 Acc: {res6['accuracy']:.4f}, Macro F1: {res6['macro_f1']:.4f}")
    sys.stdout.flush()

    print('\n--- Training Model 8: Full Spatial-Temporal TITAN ---')
    sys.stdout.flush()
    model_8 = ASLTitanHybridModel(node_in_dim=6, spatial_hidden_dim=64, aux_dim=42, num_classes=27, dropout=0.15)
    model_8 = train_model(model_8, train_loader, val_loader, epochs=12, lr=2e-3)
    results['Model_8_Full_Spatial_Temporal_TITAN'] = evaluate_model(model_8, test_loader)
    res8 = results['Model_8_Full_Spatial_Temporal_TITAN']
    print(f"Model 8 Acc: {res8['accuracy']:.4f}, Static: {res8['static_accuracy']:.4f}, Dynamic: {res8['dynamic_accuracy']:.4f}")
    sys.stdout.flush()

    # Also evaluate on real-human benchmark
    real_path = r'C:\Users\ADMIN\.gemini\antigravity\scratch\signbridge\backend\titan\datasets\asl_alphabet\metadata\real_human_benchmark.json'
    with open(real_path) as f:
        real_records = json.load(f)['records']

    real_samples = []
    for r in real_records:
        lbl = r.get('alphabet_label') or r.get('target_letter')
        if lbl in ASL_CLASS_TO_IDX:
            feats_42 = r.get('features_42')
            if feats_42 is None:
                feats_42 = extract_canonical_features(r['landmarks']).tolist()
            real_samples.append({
                'sample_id': r.get('capture_id', f'REAL_{len(real_samples)}'),
                'sign_label': lbl,
                'class_idx': ASL_CLASS_TO_IDX[lbl],
                'is_dynamic': False,
                'signer_id': r.get('signer_id', 'Real_Signer'),
                'landmarks': r['landmarks'],
                'features_42': feats_42
            })

    real_ds = ASLDataset(real_samples)
    real_loader = DataLoader(real_ds, batch_size=32, shuffle=False)
    results['Model_6_Real_Human_Benchmark'] = evaluate_model(model_6, real_loader)
    results['Model_8_Real_Human_Benchmark'] = evaluate_model(model_8, real_loader)

    r6_acc = results['Model_6_Real_Human_Benchmark']['accuracy']
    r8_acc = results['Model_8_Real_Human_Benchmark']['accuracy']
    print(f"Real Human Evaluation - Model 6: {r6_acc:.4f}, Model 8: {r8_acc:.4f}")

    # Save results
    out_json = r'C:\Users\ADMIN\.gemini\antigravity\scratch\signbridge\backend\saved_models\titan_asl_experiment_results.json'
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved complete ASL experimental suite results to {out_json}')

    # Export Model 6 weights to JSON for browser execution
    model_6_weights = {}
    for k, v in model_6.state_dict().items():
        model_6_weights[k] = v.cpu().numpy().tolist()

    export_json = {
        'model_name': 'SIGNBRIDGE TITAN ASL Spatial GNN',
        'architecture': '2-Layer Residual GCN + Dual Readout + Aux42 Biomechanics',
        'classes': ASL_VOCABULARY,
        'temperature': 1.08,
        'entropy_threshold': 1.35,
        'min_confidence': 0.55,
        'state_dict': model_6_weights
    }

    out_model = r'C:\Users\ADMIN\.gemini\antigravity\scratch\signbridge\backend\saved_models\titan_asl_spatial_model.json'
    frontend_model = r'C:\Users\ADMIN\.gemini\antigravity\scratch\signbridge\frontend\src\services\titan_asl_spatial_model.json'
    with open(out_model, 'w') as f:
        json.dump(export_json, f)
    with open(frontend_model, 'w') as f:
        json.dump(export_json, f)
    print(f'Exported JSON weights to {out_model} ({os.path.getsize(out_model)/1024:.2f} KB)')


if __name__ == '__main__':
    main()
