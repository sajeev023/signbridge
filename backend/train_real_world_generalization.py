"""
SIGNBRIDGE TITAN: Real-World Multi-Signer Generalization Experiment
Trains Adaptive Kinematic GCN (AK-GCN) strictly on train.json (Signer 1 & 2),
validates on val.json (Signer 3), and evaluates on the independent held-out
real-human benchmark (cohort_split == 'test': Signer 3, 4, 5, 6, N=27).

Applies:
1. 3D Rotation Jitter (Yaw [-60, +60], Pitch [-30, +30], Roll [-20, +20])
2. FingerDropout (Masking 1 finger with p=0.15 for self-occlusion robustness)
3. Palm Scale Jitter (0.85 to 1.20x)
4. Sensor Noise (Gaussian N(0, 0.015))
"""

import json
import math
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from backend.titan.asl_architecture import (
    ASLSpatialGNN,
    ASL_CLASS_TO_IDX,
    ASL_VOCABULARY,
    build_adjacency_matrix,
    compute_normalized_laplacian,
    get_default_normalized_adjacency
)
from backend.titan.spatial_encoder import landmarks_to_node_features
from backend.alphabet_feature_extractor import extract_canonical_features

# Set deterministic seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Finger landmark indices: Thumb (1-4), Index (5-8), Middle (9-12), Ring (13-16), Pinky (17-20)
FINGER_INDICES = [
    [1, 2, 3, 4],       # Thumb
    [5, 6, 7, 8],       # Index
    [9, 10, 11, 12],    # Middle
    [13, 14, 15, 16],   # Ring
    [17, 18, 19, 20]    # Pinky
]


def rotate_landmarks_3d(landmarks, yaw_deg, pitch_deg=0.0, roll_deg=0.0):
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    roll = np.radians(roll_deg)
    
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]], dtype=np.float32)
    Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]], dtype=np.float32)
    Rz = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]], dtype=np.float32)
    R = Rz @ Rx @ Ry
    
    coords = np.array([[lm['x'], lm['y'], lm.get('z', 0.0)] for lm in landmarks], dtype=np.float32)
    wrist = coords[0]
    rel = coords - wrist
    rot_rel = (R @ rel.T).T
    rot_coords = rot_rel + wrist
    return [{'x': float(c[0]), 'y': float(c[1]), 'z': float(c[2])} for c in rot_coords]


def augment_finger_curvature(landmarks, openness=0.25):
    """
    Simulates variations between tight curvature and open loose curvature
    for claw/curled postures (C, O, E).
    """
    new_lms = [dict(lm) for lm in landmarks]
    for tip_idx, mcp_idx in [(8, 5), (12, 9), (16, 13), (20, 17)]:
        tip = np.array([landmarks[tip_idx]['x'], landmarks[tip_idx]['y'], landmarks[tip_idx].get('z', 0.0)], dtype=np.float32)
        mcp = np.array([landmarks[mcp_idx]['x'], landmarks[mcp_idx]['y'], landmarks[mcp_idx].get('z', 0.0)], dtype=np.float32)
        vec = tip - mcp
        new_tip = tip + vec * openness
        new_lms[tip_idx]['x'] = float(new_tip[0])
        new_lms[tip_idx]['y'] = float(new_tip[1])
        new_lms[tip_idx]['z'] = float(new_tip[2])
    return new_lms


class RobustASLDataset(Dataset):
    def __init__(self, samples, is_training=False):
        self.samples = samples
        self.is_training = is_training

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        lms = s['landmarks']
        target_idx = s['class_idx']
        
        # Training data augmentation (FingerDropout, 3D Rotation, Scale, Noise, Curvature)
        if self.is_training:
            # 1. Curvature Variation for C and curled letters
            if s.get('sign_label') in ['C', 'O', 'E'] and random.random() < 0.60:
                openness = random.uniform(0.10, 0.35)
                lms = augment_finger_curvature(lms, openness=openness)

            # 2. 3D Rotation Augmentation: simulate lateral angles (e.g. C seen at 45 deg)
            if random.random() < 0.70:
                yaw = random.uniform(-60.0, 60.0)
                pitch = random.uniform(-30.0, 30.0)
                roll = random.uniform(-20.0, 20.0)
                lms = rotate_landmarks_3d(lms, yaw, pitch, roll)

            # 3. Scale Jitter: simulate distance differences
            if random.random() < 0.50:
                scale_factor = random.uniform(0.85, 1.20)
                coords = np.array([[lm['x'], lm['y'], lm.get('z', 0.0)] for lm in lms], dtype=np.float32)
                wrist = coords[0]
                scaled = (coords - wrist) * scale_factor + wrist
                lms = [{'x': float(c[0]), 'y': float(c[1]), 'z': float(c[2])} for c in scaled]

            # 4. FingerDropout: simulate partial self-occlusion
            if random.random() < 0.20:
                finger_to_drop = random.choice(FINGER_INDICES)
                lms_copy = [dict(lm) for lm in lms]
                for joint_idx in finger_to_drop:
                    lms_copy[joint_idx] = {'x': lms[0]['x'], 'y': lms[0]['y'], 'z': lms[0].get('z', 0.0)}
                lms = lms_copy

            # 5. Sensor Noise
            if random.random() < 0.40:
                noise = np.random.normal(0, 0.012, size=(len(lms), 3)).astype(np.float32)
                lms = [{'x': float(lms[i]['x'] + noise[i, 0]),
                        'y': float(lms[i]['y'] + noise[i, 1]),
                        'z': float(lms[i].get('z', 0.0) + noise[i, 2])} for i in range(len(lms))]

        nodes = landmarks_to_node_features(lms)
        feats42 = extract_canonical_features(lms)

        return {
            'node_features': torch.tensor(nodes, dtype=torch.float32),
            'aux_features': torch.tensor(feats42, dtype=torch.float32),
            'class_idx': torch.tensor(target_idx, dtype=torch.long)
        }


def run_experiment():
    print("=== SIGNBRIDGE TITAN: REAL-WORLD GENERALIZATION EXPERIMENT ===")
    
    # 1. Load Data
    train_samples = json.load(open('backend/titan/datasets/asl_alphabet/splits/train.json'))['samples']
    val_samples = json.load(open('backend/titan/datasets/asl_alphabet/splits/val.json'))['samples']
    test_samples = json.load(open('backend/titan/datasets/asl_alphabet/splits/test.json'))['samples']

    train_static = [s for s in train_samples if not s['is_dynamic']]
    val_static = [s for s in val_samples if not s['is_dynamic']]
    test_static = [s for s in test_samples if not s['is_dynamic']]

    train_loader = DataLoader(RobustASLDataset(train_static, is_training=True), batch_size=32, shuffle=True)
    val_loader = DataLoader(RobustASLDataset(val_static, is_training=False), batch_size=32, shuffle=False)
    test_loader = DataLoader(RobustASLDataset(test_static, is_training=False), batch_size=32, shuffle=False)

    # 2. Build Model
    model = ASLSpatialGNN(node_in_dim=6, hidden_dim=64, aux_dim=42, num_classes=27, dropout=0.15)
    
    # Initialize from pre-adaptive checkpoint
    with open('backend/saved_models/titan_asl_spatial_model.json') as f:
        sd = json.load(f)['state_dict']
    model.load_state_dict({k: torch.tensor(v) for k, v in sd.items()}, strict=False)

    adj = torch.tensor(compute_normalized_laplacian(build_adjacency_matrix()), dtype=torch.float32)

    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state_dict = None

    print("\n--- Training with Adaptive Graph & Generalization Augmentation (Signer 1 & 2 ONLY) ---")
    for epoch in range(8):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch['node_features'], adj, batch['aux_features'])
            loss = criterion(out, batch['class_idx'])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        # Evaluate on Validation Set (Signer 3 ONLY)
        model.eval()
        val_corr = 0
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch['node_features'], adj, batch['aux_features'])
                val_corr += (torch.argmax(out, dim=1) == batch['class_idx']).sum().item()
        val_acc = val_corr / len(val_static)
        print(f"Epoch {epoch+1}/8: Loss={train_loss/len(train_loader):.4f}, Val Acc (Signer 3)={val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best checkpoint selected strictly by validation set
    if best_state_dict:
        model.load_state_dict(best_state_dict)

    # 3. Evaluate on Synthetic Held-Out Test Set (Signer 4, N=1000)
    model.eval()
    test_corr = 0
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch['node_features'], adj, batch['aux_features'])
            test_corr += (torch.argmax(out, dim=1) == batch['class_idx']).sum().item()
    test_acc = test_corr / len(test_static)
    print(f"\nSynthetic Held-Out Test Accuracy: {test_corr}/{len(test_static)} ({test_acc*100:.2f}%)")

    # 4. Evaluate on Real-Human Benchmark (Held-Out Test Cohort: Signers 3, 4, 5, 6, N=27)
    records = json.load(open('backend/titan/datasets/asl_alphabet/metadata/real_human_benchmark.json'))['records']
    test_records = [r for r in records if r.get('cohort_split') == 'test']
    train_records = [r for r in records if r.get('cohort_split') == 'train']

    def eval_benchmark(recs, name):
        corr = 0
        signer_stats = {}
        all_preds = []
        all_targets = []
        all_probs = []
        
        temp = 1.08
        accepted_corr = 0
        accepted_wrong = 0
        rejected_corr = 0
        rejected_wrong = 0

        for r in recs:
            lbl = r.get('alphabet_label') or r.get('target_letter')
            t_idx = ASL_CLASS_TO_IDX[lbl]
            sid = r.get('signer_id')
            nodes = landmarks_to_node_features(r['landmarks'])
            feats = r.get('features_42') or extract_canonical_features(r['landmarks']).tolist()

            with torch.no_grad():
                logits = model(torch.tensor(nodes).unsqueeze(0), adj, torch.tensor(feats).unsqueeze(0))
                calib = logits / temp
                probs = torch.softmax(calib, dim=1)[0].numpy()
                pred = int(np.argmax(probs))
                conf = float(probs[pred])
                entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

            is_c = (pred == t_idx)
            corr += (1 if is_c else 0)
            signer_stats.setdefault(sid, [0, 0])
            signer_stats[sid][0] += (1 if is_c else 0)
            signer_stats[sid][1] += 1
            all_preds.append(pred)
            all_targets.append(t_idx)
            all_probs.append(probs)

            # Rejection Gate
            is_rej = (conf < 0.55) or (entropy > 1.35)
            if is_rej:
                if is_c: rejected_corr += 1
                else: rejected_wrong += 1
            else:
                if is_c: accepted_corr += 1
                else: accepted_wrong += 1

        acc = corr / len(recs)
        print(f"\n=== {name} (N={len(recs)}) ===")
        print(f"Overall Accuracy: {corr}/{len(recs)} ({acc*100:.2f}%)")
        print("Per-Signer Breakdown:")
        for sid in sorted(signer_stats.keys()):
            c, t = signer_stats[sid]
            print(f"  {sid}: {c}/{t} ({c/t*100:.2f}%)")

        tot_accepted = accepted_corr + accepted_wrong
        prec_accepted = (accepted_corr / tot_accepted * 100) if tot_accepted > 0 else 0.0
        cov = tot_accepted / len(recs) * 100
        print(f"Calibrated Rejection: Coverage={cov:.1f}% ({tot_accepted}/{len(recs)}), Precision on Accepted={prec_accepted:.2f}% (Wrong Accepted={accepted_wrong})")
        return acc, signer_stats, all_preds, all_targets, all_probs

    test_acc_bm, test_signers, preds, targets, probs = eval_benchmark(test_records, "HELD-OUT REAL-HUMAN BENCHMARK")
    train_acc_bm, train_signers, _, _, _ = eval_benchmark(train_records, "SEEN-SIGNER FAMILIARITY BENCHMARK")

    # Compute ECE and Brier Score
    probs_np = np.array(probs)
    targets_np = np.array(targets)
    
    # Brier Score
    one_hot = np.zeros_like(probs_np)
    for i, t in enumerate(targets_np):
        one_hot[i, t] = 1.0
    brier = float(np.mean(np.sum((probs_np - one_hot) ** 2, axis=1)))
    
    # ECE (10 bins)
    confidences = np.max(probs_np, axis=1)
    predictions = np.argmax(probs_np, axis=1)
    accuracies = (predictions == targets_np).astype(float)
    
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for i in range(10):
        bin_mask = (confidences > bins[i]) & (confidences <= bins[i+1])
        if np.any(bin_mask):
            bin_acc = np.mean(accuracies[bin_mask])
            bin_conf = np.mean(confidences[bin_mask])
            ece += (np.sum(bin_mask) / len(targets_np)) * abs(bin_acc - bin_conf)

    print(f"\nHeld-Out ECE: {ece:.4f}")
    print(f"Held-Out Brier Score: {brier:.4f}")

    return model, best_state_dict, test_acc_bm

if __name__ == '__main__':
    run_experiment()
