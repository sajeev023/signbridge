"""
SIGNBRIDGE — Isolated ASL Letter A Recognition & Verification Specialist
Phase 1: Isolated ASL Alphabet Recognition Training (Letter A Only)

Constructs, trains, and validates:
1. An isolated Letter A specialist detector with hard-negative rejection (E, S, M, N, T, NEUTRAL).
2. Biomechanical geometric gate for canonical ASL Letter A configuration.
3. Multi-signer cohort evaluation (train: Signer_1, Signer_2; val: Signer_3; test: Signer_4 + Real Human Benchmarks).
4. Strict metrics: Precision, Recall, F1, FPR, FNR, ECE (Expected Calibration Error), Confusion Matrix.
"""

import json
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# Compute standard classification metrics without external sklearn dependency
def calc_metrics(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'precision': prec, 'recall': rec, 'f1': f1,
        'fpr': fpr, 'fnr': fnr,
        'accuracy': (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    }

try:
    from backend.alphabet_feature_extractor import (
        extract_canonical_features,
        FEATURE_NAMES_42,
        NUM_CANONICAL_FEATURES
    )
except ImportError:
    from alphabet_feature_extractor import (
        extract_canonical_features,
        FEATURE_NAMES_42,
        NUM_CANONICAL_FEATURES
    )

# Set deterministic random seeds
torch.manual_seed(42)
np.random.seed(42)

# --- 1. Model Architecture ---
class LetterASpecialist(nn.Module):
    """
    Lightweight calibrated MLP specialist for ASL Letter A recognition.
    Input: 42 canonical biomechanical features
    Output: 2 logits (0: Non-A / Rejection / Hard Negative, 1: Letter A)
    """
    def __init__(self, input_dim: int = 42, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)
        )
        self.temperature = nn.Parameter(torch.ones(1) * 1.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return logits / torch.clamp(self.temperature, min=0.1, max=10.0)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=-1)


# --- 2. Biomechanical Geometric Rule Gate ---
def verify_letter_a_geometry(feats_42: np.ndarray) -> dict:
    """
    Evaluates kinematic biomechanical criteria for canonical ASL Letter A.
    Returns boolean pass and fine-grained violation flags.
    Calibrated against empirical human and multi-signer kinematics:
    - 4 non-thumb fingers curled into fist (ext < 0.35)
    - Thumb upright on radial margin: thumb_y >= 0.55, thumb_x >= 0.12
    - Thumb not crossing knuckles: cos_spread < 0.15 (rules out S)
    - Thumb not tucked under fingers: dist_thumb_mid >= 0.38 (rules out M/N/T)
    """
    ext_thumb = feats_42[0]
    ext_idx = feats_42[1]
    ext_mid = feats_42[2]
    ext_ring = feats_42[3]
    ext_pinky = feats_42[4]
    cos_spread = feats_42[15]
    dist_thumb_mid = feats_42[20]
    thumb_x = feats_42[24]
    thumb_y = feats_42[25]
    thumb_z = feats_42[26]
    
    # 1. Four fingers tightly curled
    fist_curled = bool(
        ext_idx < 0.35 and
        ext_mid < 0.35 and
        ext_ring < 0.35 and
        ext_pinky < 0.35
    )
    
    # 2. Thumb upright and extended along radial edge (not folded flat like E)
    thumb_elevated = bool(thumb_y >= 0.55)
    
    # 3. Thumb on lateral radial side (x >= 0.12; in S/E thumb_x <= 0.0)
    thumb_lateral = bool(thumb_x >= 0.12)
    
    # 4. Thumb bone vector spread vs index (in S, thumb crosses over index with cos_spread > 0.40)
    thumb_not_crossing = bool(cos_spread < 0.15)
    
    # 5. Thumb tip not tucked under middle/ring (in M/N dist_thumb_mid < 0.25)
    thumb_not_tucked = bool(dist_thumb_mid >= 0.38)
    
    is_valid_a = bool(fist_curled and thumb_elevated and thumb_lateral and thumb_not_crossing and thumb_not_tucked)
    
    return {
        'is_valid_letter_a': is_valid_a,
        'fist_curled': fist_curled,
        'thumb_elevated': thumb_elevated,
        'thumb_lateral': thumb_lateral,
        'thumb_not_crossing': thumb_not_crossing,
        'thumb_not_tucked': thumb_not_tucked,
        'details': {
            'ext_thumb': float(ext_thumb),
            'max_ext_fingers': float(max(ext_idx, ext_mid, ext_ring, ext_pinky)),
            'thumb_x': float(thumb_x),
            'thumb_y': float(thumb_y),
            'cos_spread': float(cos_spread),
            'dist_thumb_mid': float(dist_thumb_mid)
        }
    }


# --- 3. Dataset Preparation ---
class AlphabetDataset(Dataset):
    def __init__(self, samples: list, binary_target_class: str = 'A', augment: bool = False):
        self.features = []
        self.labels = []
        self.raw_labels = []
        self.signers = []
        self.augment = augment
        
        for s in samples:
            label = s.get('sign_label') or s.get('alphabet_label') or s.get('label')
            feat = s.get('features_42')
            signer = s.get('participant_id') or s.get('signer_id') or 'Unknown'
            
            if feat is not None and len(feat) == 42:
                self.features.append(feat)
                self.raw_labels.append(label)
                is_target = 1 if label == binary_target_class else 0
                self.labels.append(is_target)
                self.signers.append(signer)
                
                # If training with augmentation for target class A, add natural resting thumb variations
                if augment and is_target == 1:
                    for _ in range(3):
                        aug_feat = np.array(feat, dtype=np.float32)
                        # Natural resting thumb variation:
                        # Vary canonical_thumb_y (index 25) between 0.60 and 1.12
                        aug_feat[25] = float(np.random.uniform(0.60, 1.12))
                        # Vary canonical_thumb_x (index 24) between 0.16 and 0.40
                        aug_feat[24] = float(np.random.uniform(0.16, 0.40))
                        # Small noise on other dimensions
                        noise = np.random.normal(0, 0.015, size=42).astype(np.float32)
                        aug_feat += noise
                        self.features.append(aug_feat.tolist())
                        self.raw_labels.append(label)
                        self.labels.append(1)
                        self.signers.append(signer)
                
        self.features = np.array(self.features, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])


def load_datasets():
    with open('backend/data/alphabet_dataset_multisigner.json') as f:
        data = json.load(f)
    samples = data['samples']
    
    # Split by participant ID:
    # Train: Signer_1, Signer_2 (2000 samples)
    # Val: Signer_3 (1000 samples)
    # Test: Signer_4 (1000 samples)
    train_samples = [s for s in samples if s['participant_id'] in ['Signer_1', 'Signer_2']]
    val_samples = [s for s in samples if s['participant_id'] == 'Signer_3']
    test_samples = [s for s in samples if s['participant_id'] == 'Signer_4']
    
    # Also load empirical real-human benchmarks
    real_bm_samples = []
    if os.path.exists('backend/titan/datasets/asl_alphabet/metadata/real_human_benchmark.json'):
        with open('backend/titan/datasets/asl_alphabet/metadata/real_human_benchmark.json') as f:
            bm = json.load(f)
        real_bm_samples = bm.get('records', [])
    
    return train_samples, val_samples, test_samples, real_bm_samples


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(probs)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (probs >= bin_lower) & (probs < bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin] == (probs[in_bin] >= 0.5))
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return float(ece)


# --- 4. Main Training & Evaluation Routine ---
def train_and_evaluate():
    train_samples, val_samples, test_samples, real_bm_samples = load_datasets()
    print(f"Dataset split: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}, RealBM={len(real_bm_samples)}")
    
    train_ds = AlphabetDataset(train_samples, augment=True)
    val_ds = AlphabetDataset(val_samples)
    test_ds = AlphabetDataset(test_samples)
    real_bm_ds = AlphabetDataset(real_bm_samples)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    
    # Calculate class weights to handle imbalance (1:24 ratio for A vs rest)
    pos_count = int(np.sum(train_ds.labels))
    neg_count = len(train_ds.labels) - pos_count
    pos_weight = torch.tensor([1.0, float(neg_count) / float(max(1, pos_count))])
    print(f"Train counts: Pos(A)={pos_count}, Neg={neg_count}, Weight={pos_weight[1]:.2f}")
    
    model = LetterASpecialist(input_dim=42, hidden_dim=32)
    criterion = nn.CrossEntropyLoss(weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    
    best_val_f1 = 0.0
    best_state = None
    
    model.train()
    for epoch in range(40):
        total_loss = 0.0
        for x_b, y_b in train_loader:
            optimizer.zero_grad()
            out = model(x_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        
        # Validation pass
        model.eval()
        val_preds, val_targets, val_confs = [], [], []
        with torch.no_grad():
            for x_b, y_b in val_loader:
                probs = torch.softmax(model(x_b), dim=-1)[:, 1].numpy()
                val_confs.extend(probs)
                val_preds.extend((probs >= 0.5).astype(int))
                val_targets.extend(y_b.numpy())
        
        val_metrics = calc_metrics(np.array(val_targets), np.array(val_preds))
        f1 = val_metrics['f1']
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        model.train()

    model.load_state_dict(best_state)
    model.eval()
    print(f"Training Complete. Best Validation F1: {best_val_f1:.4f}")

    # --- 5. Comprehensive Test Split Evaluation ---
    test_x = torch.tensor(test_ds.features)
    test_y = test_ds.labels
    raw_labels = test_ds.raw_labels
    
    with torch.no_grad():
        test_probs = torch.softmax(model(test_x), dim=-1)[:, 1].numpy()
        test_preds = (test_probs >= 0.5).astype(int)
        
    m = calc_metrics(test_y, test_preds)
    prec, rec, f1 = m['precision'], m['recall'], m['f1']
    tp, fp, tn, fn = m['tp'], m['fp'], m['tn'], m['fn']
    fpr, fnr = m['fpr'], m['fnr']
    ece = compute_ece(test_probs, test_y)
    
    print("\n--- Test Set (Held-Out Signer_4) Performance ---")
    print(f"Accuracy:  {(tp + tn) / len(test_y) * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1 Score:  {f1 * 100:.2f}%")
    print(f"FPR:       {fpr * 100:.3f}% (FP={fp}, TN={tn})")
    print(f"FNR:       {fnr * 100:.3f}% (FN={fn}, TP={tp})")
    print(f"ECE:       {ece:.4f}")
    
    # Detailed confusion breakdown by true class
    class_confusion = {}
    for i, raw_lbl in enumerate(raw_labels):
        if raw_lbl not in class_confusion:
            class_confusion[raw_lbl] = {'total': 0, 'classified_as_A': 0}
        class_confusion[raw_lbl]['total'] += 1
        if test_preds[i] == 1:
            class_confusion[raw_lbl]['classified_as_A'] += 1
            
    print("\nPer-Class Classification as A on Test Set:")
    for lbl, stats in sorted(class_confusion.items()):
        rate = stats['classified_as_A'] / stats['total'] * 100
        print(f"  Class {lbl:7s}: {stats['classified_as_A']:2d}/{stats['total']:2d} ({rate:5.1f}%)")

    # --- 6. Empirical Real Human Benchmark Evaluation ---
    print("\n--- Empirical Real Human Benchmark Performance ---")
    real_x = torch.tensor(real_bm_ds.features)
    real_y = real_bm_ds.labels
    real_raw = real_bm_ds.raw_labels
    real_signers = real_bm_ds.signers
    
    with torch.no_grad():
        real_probs = torch.softmax(model(real_x), dim=-1)[:, 1].numpy()
        real_preds = (real_probs >= 0.5).astype(int)

    for i in range(len(real_bm_ds)):
        lbl = real_raw[i]
        sgn = real_signers[i]
        prob = real_probs[i]
        pred_a = real_preds[i] == 1
        geom = verify_letter_a_geometry(real_bm_ds.features[i])
        print(f"Sample {i+1:2d} | Label: {lbl:2s} | Signer: {sgn:9s} | P(A): {prob:.4f} | Pred: {'A' if pred_a else 'Other'} | GeomPass: {geom['is_valid_letter_a']}")

    # --- 7. Save Specialist Artifacts & Model Weights ---
    os.makedirs('backend/saved_models', exist_ok=True)
    
    # Export weights for frontend TypeScript engine
    weights_json = {
        'model_name': 'SIGNBRIDGE ASL Letter A Specialist Discriminator',
        'version': '1.0.0-letter-a-mastery',
        'target_letter': 'A',
        'threshold': 0.5,
        'temperature': float(model.temperature.item()),
        'input_dim': 42,
        'hidden_dim': 32,
        'weights': {
            'w1': model.net[0].weight.detach().numpy().tolist(),
            'b1': model.net[0].bias.detach().numpy().tolist(),
            'bn1_mean': model.net[1].running_mean.numpy().tolist(),
            'bn1_var': model.net[1].running_var.numpy().tolist(),
            'bn1_weight': model.net[1].weight.detach().numpy().tolist(),
            'bn1_bias': model.net[1].bias.detach().numpy().tolist(),
            'w2': model.net[4].weight.detach().numpy().tolist(),
            'b2': model.net[4].bias.detach().numpy().tolist(),
            'w3': model.net[6].weight.detach().numpy().tolist(),
            'b3': model.net[6].bias.detach().numpy().tolist()
        },
        'metrics': {
            'test_accuracy': float((tp + tn) / len(test_y)),
            'test_precision': float(prec),
            'test_recall': float(rec),
            'test_f1': float(f1),
            'test_fpr': float(fpr),
            'test_fnr': float(fnr),
            'test_ece': float(ece),
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        },
        'class_confusion': class_confusion
    }
    
    with open('backend/saved_models/letter_a_specialist_model.json', 'w') as f:
        json.dump(weights_json, f, indent=2)
    print("\nSaved backend/saved_models/letter_a_specialist_model.json")
    
    return weights_json


if __name__ == '__main__':
    train_and_evaluate()
