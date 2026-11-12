"""
SIGNBRIDGE ISL Manual Alphabet Model Trainer & Exporter
Trains a neural landmark classifier on Indian Sign Language (ISLRTC) manual alphabet kinematics.
Exports the trained model weights and metadata to JSON for client-side and server-side inference.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Vocabulary of supported ISL manual alphabet signs + NEUTRAL
ALPHABET_CLASSES = [
    "NEUTRAL",
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "K", "L", "M", "N", "O", "P", "Q", "S", "T",
    "U", "V", "W", "X", "Y"
]
CLASS_TO_IDX = {c: i for i, c in enumerate(ALPHABET_CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(ALPHABET_CLASSES)}

FEATURE_NAMES = [
    # Hand 1 (Dominant) finger extensions (5)
    "h1_thumb_ext", "h1_index_ext", "h1_middle_ext", "h1_ring_ext", "h1_pinky_ext",
    # Hand 1 finger curl angles (5)
    "h1_thumb_curl", "h1_index_curl", "h1_middle_curl", "h1_ring_curl", "h1_pinky_curl",
    # Hand 1 finger spreads (4)
    "h1_spread_thumb_index", "h1_spread_index_mid", "h1_spread_mid_ring", "h1_spread_ring_pinky",
    # Hand 1 L-angle (1)
    "h1_l_angle",
    # Hand 1 C-curve curvature (1)
    "h1_c_curve",
    # Hand 1 loop closure (thumb tip to index tip) (1)
    "h1_loop_closure",
    # Hand 2 present flag (1)
    "h2_present",
    # Hand 2 (Base) finger extensions (5)
    "h2_thumb_ext", "h2_index_ext", "h2_middle_ext", "h2_ring_ext", "h2_pinky_ext",
    # Hand 2 loop closure (1)
    "h2_loop_closure",
    # Inter-hand relative wrist position (3)
    "rel_wrist_x", "rel_wrist_y", "rel_wrist_z",
    # Inter-hand touch points: H1 Index Tip (8) to H2 landmarks (9)
    "dist_h1_idx_to_h2_thumb_tip",    # Key for 'A'
    "dist_h1_idx_to_h2_idx_tip",      # Key for 'E'
    "dist_h1_idx_to_h2_mid_tip",      # Key for 'I'
    "dist_h1_idx_to_h2_ring_tip",     # Key for 'O'
    "dist_h1_idx_to_h2_pinky_tip",    # Key for 'U'
    "dist_h1_idx_to_h2_idx_base",     # Key for 'T'
    "dist_h1_idx_to_h2_palm_center",  # Key for 'M', 'N', 'V', 'H'
    "dist_h1_idx_to_h2_cleft",        # Key for 'Y'
    "dist_h1_idx_to_h2_idx_pip",      # Key for 'K'
    # Inter-hand loop & structure contacts (8)
    "dist_h1_loop_to_h2_loop",        # Key for 'B'
    "dist_h1_thumb_to_h2_idx_tip",    # Key for 'D' (loop contact)
    "dist_h1_loop_to_h2_idx_tip",     # Key for 'P'
    "dist_h1_pinky_to_h2_pinky",      # Key for 'S'
    "dist_h1_knuckle_to_h2_knuckle",  # Key for 'G'
    "dist_h1_mid_to_h2_palm_center",  # Key for 'M', 'N'
    "dist_h1_ring_to_h2_palm_center", # Key for 'M'
    "dist_h1_mid_cross_h2_idx",       # Key for 'F', 'X'
]

NUM_FEATURES = len(FEATURE_NAMES)
NUM_CLASSES = len(ALPHABET_CLASSES)


def generate_synthetic_features(sign: str, num_samples: int = 150) -> np.ndarray:
    """
    Synthesizes kinematically grounded feature distributions for a specific ISL alphabet sign
    with realistic anatomical jitter, distance noise, and rotation perturbations.
    """
    samples = np.zeros((num_samples, NUM_FEATURES), dtype=np.float32)

    for i in range(num_samples):
        # Base noise
        noise = np.random.normal(0, 0.04, size=NUM_FEATURES).astype(np.float32)
        feat = np.zeros(NUM_FEATURES, dtype=np.float32)

        # By default for two-handed signs, Hand 2 is present
        h2_present = 1.0
        # Default wrist offset (H1 is right hand, H2 is left hand)
        rel_wrist_x = 0.8 + np.random.normal(0, 0.08)
        rel_wrist_y = -0.1 + np.random.normal(0, 0.06)
        rel_wrist_z = 0.05 + np.random.normal(0, 0.04)

        # Default high inter-hand distances (no contact)
        h1_idx_to_h2_thumb = 1.8 + np.random.uniform(0, 0.4)
        h1_idx_to_h2_idx = 1.8 + np.random.uniform(0, 0.4)
        h1_idx_to_h2_mid = 1.8 + np.random.uniform(0, 0.4)
        h1_idx_to_h2_ring = 1.8 + np.random.uniform(0, 0.4)
        h1_idx_to_h2_pinky = 1.8 + np.random.uniform(0, 0.4)
        h1_idx_to_h2_base = 1.5 + np.random.uniform(0, 0.4)
        h1_idx_to_h2_palm = 1.4 + np.random.uniform(0, 0.4)
        h1_idx_to_h2_cleft = 1.6 + np.random.uniform(0, 0.4)
        h1_idx_to_h2_pip = 1.7 + np.random.uniform(0, 0.4)

        h1_loop_to_h2_loop = 2.0
        h1_thumb_to_h2_idx = 2.0
        h1_loop_to_h2_idx = 2.0
        h1_pinky_to_h2_pinky = 2.0
        h1_knuckle_to_h2_knuckle = 2.0
        h1_mid_to_h2_palm = 1.5
        h1_ring_to_h2_palm = 1.5
        h1_mid_cross_h2_idx = 1.8

        # Hand 1 finger states
        h1_thumb_ext = 0.0
        h1_index_ext = 1.0
        h1_middle_ext = 0.0
        h1_ring_ext = 0.0
        h1_pinky_ext = 0.0

        h1_thumb_curl = 0.8
        h1_index_curl = 0.1
        h1_middle_curl = 0.8
        h1_ring_curl = 0.8
        h1_pinky_curl = 0.8

        h1_spread_thumb_index = 0.3
        h1_spread_index_mid = 0.4
        h1_spread_mid_ring = 0.1
        h1_spread_ring_pinky = 0.1
        h1_l_angle = 0.2
        h1_c_curve = 0.1
        h1_loop_closure = 0.8

        # Hand 2 (Base) defaults: open flat palm
        h2_thumb_ext = 0.9
        h2_index_ext = 0.95
        h2_middle_ext = 0.95
        h2_ring_ext = 0.9
        h2_pinky_ext = 0.85
        h2_loop_closure = 0.8

        # Specific sign configurations
        if sign == "NEUTRAL":
            # Random resting, hands apart, or waving
            h2_present = 1.0 if np.random.rand() > 0.4 else 0.0
            h1_index_ext = float(np.random.rand() > 0.5)
            h1_middle_ext = float(np.random.rand() > 0.5)
            h1_thumb_ext = float(np.random.rand() > 0.5)
            h1_ring_ext = float(np.random.rand() > 0.5)
            h1_pinky_ext = float(np.random.rand() > 0.5)
            rel_wrist_x = 2.5 + np.random.uniform(0.5, 1.5)

        elif sign == "A":
            # Vowel A: H1 index touches H2 thumb tip
            h1_index_ext = 1.0
            h1_middle_ext = 0.0
            h1_idx_to_h2_thumb = np.random.uniform(0.04, 0.15)  # Contact!

        elif sign == "E":
            # Vowel E: H1 index touches H2 index tip
            h1_index_ext = 1.0
            h1_middle_ext = 0.0
            h1_idx_to_h2_idx = np.random.uniform(0.04, 0.15)    # Contact!

        elif sign == "I":
            # Vowel I: H1 index touches H2 middle tip
            h1_index_ext = 1.0
            h1_middle_ext = 0.0
            h1_idx_to_h2_mid = np.random.uniform(0.04, 0.15)    # Contact!

        elif sign == "O":
            # Vowel O: H1 index touches H2 ring tip
            h1_index_ext = 1.0
            h1_middle_ext = 0.0
            h1_idx_to_h2_ring = np.random.uniform(0.04, 0.15)   # Contact!

        elif sign == "U":
            # Vowel U: H1 index touches H2 pinky tip
            h1_index_ext = 1.0
            h1_middle_ext = 0.0
            h1_idx_to_h2_pinky = np.random.uniform(0.04, 0.15)  # Contact!

        elif sign == "B":
            # Both hands form loops touching side-by-side
            h1_loop_closure = np.random.uniform(0.05, 0.15)
            h2_loop_closure = np.random.uniform(0.05, 0.15)
            h1_loop_to_h2_loop = np.random.uniform(0.06, 0.18) # Contact!
            h1_index_ext = 0.4
            h1_middle_ext = 0.0
            h2_index_ext = 0.4
            h2_middle_ext = 0.0

        elif sign == "C":
            # Single-hand curved 'C'
            h2_present = 0.0 if np.random.rand() > 0.3 else 1.0
            h1_thumb_ext = 0.6
            h1_index_ext = 0.6
            h1_middle_ext = 0.6
            h1_ring_ext = 0.6
            h1_pinky_ext = 0.6
            h1_c_curve = np.random.uniform(0.85, 0.98)         # High C-curve!
            h1_loop_closure = np.random.uniform(0.4, 0.6)      # Not touching (gap)

        elif sign == "D":
            # H2 index vertical straight up; H1 thumb & index form loop touching top & base
            h2_index_ext = 1.0
            h2_middle_ext = 0.0
            h2_ring_ext = 0.0
            h2_pinky_ext = 0.0
            h1_loop_closure = np.random.uniform(0.1, 0.25)
            h1_thumb_to_h2_idx = np.random.uniform(0.06, 0.18) # Loop contact
            h1_idx_to_h2_base = np.random.uniform(0.08, 0.22)

        elif sign == "F":
            # Dominant two fingers crossed across H2 two fingers
            h1_index_ext = 1.0
            h1_middle_ext = 1.0
            h1_ring_ext = 0.0
            h2_index_ext = 1.0
            h2_middle_ext = 1.0
            h2_ring_ext = 0.0
            h1_mid_cross_h2_idx = np.random.uniform(0.06, 0.18)

        elif sign == "G":
            # Two fists touching knuckles
            h1_index_ext = 0.0
            h1_middle_ext = 0.0
            h1_ring_ext = 0.0
            h1_pinky_ext = 0.0
            h2_index_ext = 0.0
            h2_middle_ext = 0.0
            h2_ring_ext = 0.0
            h2_pinky_ext = 0.0
            h1_knuckle_to_h2_knuckle = np.random.uniform(0.05, 0.18)

        elif sign == "H":
            # Flat dominant palm across flat non-dominant palm
            h1_index_ext = 0.95
            h1_middle_ext = 0.95
            h1_ring_ext = 0.9
            h1_pinky_ext = 0.85
            h1_idx_to_h2_palm = np.random.uniform(0.06, 0.18)
            h1_mid_to_h2_palm = np.random.uniform(0.06, 0.18)

        elif sign == "K":
            # H1 index bent/hooked touching H2 index PIP
            h2_index_ext = 1.0
            h2_middle_ext = 0.0
            h1_index_ext = 0.5
            h1_index_curl = 0.7
            h1_idx_to_h2_pip = np.random.uniform(0.05, 0.16)

        elif sign == "L":
            # Single-hand L shape: thumb and index at 90 deg, others closed
            h2_present = 0.0 if np.random.rand() > 0.3 else 1.0
            h1_thumb_ext = 0.95
            h1_index_ext = 1.0
            h1_middle_ext = 0.0
            h1_ring_ext = 0.0
            h1_pinky_ext = 0.0
            h1_l_angle = np.random.uniform(0.85, 0.98)          # High 90 deg L-angle!
            h1_spread_thumb_index = np.random.uniform(0.7, 0.9)

        elif sign == "M":
            # Dominant 3 fingers (index, middle, ring) resting on H2 palm
            h1_index_ext = 0.9
            h1_middle_ext = 0.9
            h1_ring_ext = 0.9
            h1_pinky_ext = 0.0
            h1_idx_to_h2_palm = np.random.uniform(0.06, 0.16)
            h1_mid_to_h2_palm = np.random.uniform(0.06, 0.16)
            h1_ring_to_h2_palm = np.random.uniform(0.06, 0.16)

        elif sign == "N":
            # Dominant 2 fingers (index, middle) resting on H2 palm
            h1_index_ext = 0.9
            h1_middle_ext = 0.9
            h1_ring_ext = 0.0
            h1_pinky_ext = 0.0
            h1_idx_to_h2_palm = np.random.uniform(0.06, 0.16)
            h1_mid_to_h2_palm = np.random.uniform(0.06, 0.16)
            h1_ring_to_h2_palm = 0.8

        elif sign == "P":
            # H2 index vertical; H1 loop touches top of H2 index
            h2_index_ext = 1.0
            h2_middle_ext = 0.0
            h1_loop_closure = np.random.uniform(0.08, 0.2)
            h1_loop_to_h2_idx = np.random.uniform(0.06, 0.18)

        elif sign == "Q":
            # H2 loop + H1 index touches bottom
            h2_loop_closure = np.random.uniform(0.08, 0.2)
            h1_index_ext = 1.0
            h1_idx_to_h2_base = np.random.uniform(0.06, 0.18)

        elif sign == "S":
            # Pinkies hooked together
            h1_pinky_ext = 0.8
            h2_pinky_ext = 0.8
            h1_pinky_to_h2_pinky = np.random.uniform(0.05, 0.16)

        elif sign == "T":
            # H1 index touches side/edge of H2 index base
            h1_index_ext = 1.0
            h1_middle_ext = 0.0
            h1_idx_to_h2_base = np.random.uniform(0.05, 0.16)

        elif sign == "V":
            # Dominant V shape (index and middle spread) on H2 palm
            h1_index_ext = 1.0
            h1_middle_ext = 1.0
            h1_ring_ext = 0.0
            h1_pinky_ext = 0.0
            h1_spread_index_mid = np.random.uniform(0.7, 0.9)  # V spread!
            h1_idx_to_h2_palm = np.random.uniform(0.08, 0.2)
            h1_mid_to_h2_palm = np.random.uniform(0.08, 0.2)

        elif sign == "W":
            # Interlaced or 3 fingers in W on palm
            h1_index_ext = 1.0
            h1_middle_ext = 1.0
            h1_ring_ext = 1.0
            h1_pinky_ext = 0.0
            h1_spread_index_mid = np.random.uniform(0.5, 0.8)
            h1_spread_mid_ring = np.random.uniform(0.5, 0.8)
            h1_idx_to_h2_palm = np.random.uniform(0.08, 0.22)

        elif sign == "X":
            # Crossed index fingers
            h1_index_ext = 1.0
            h1_middle_ext = 0.0
            h2_index_ext = 1.0
            h2_middle_ext = 0.0
            h1_mid_cross_h2_idx = np.random.uniform(0.05, 0.15)

        elif sign == "Y":
            # H1 index resting in cleft between H2 thumb and index
            h1_index_ext = 1.0
            h1_middle_ext = 0.0
            h2_thumb_ext = 1.0
            h2_index_ext = 1.0
            h1_idx_to_h2_cleft = np.random.uniform(0.05, 0.16)

        # Build feature vector
        feat[0] = h1_thumb_ext
        feat[1] = h1_index_ext
        feat[2] = h1_middle_ext
        feat[3] = h1_ring_ext
        feat[4] = h1_pinky_ext
        feat[5] = h1_thumb_curl
        feat[6] = h1_index_curl
        feat[7] = h1_middle_curl
        feat[8] = h1_ring_curl
        feat[9] = h1_pinky_curl
        feat[10] = h1_spread_thumb_index
        feat[11] = h1_spread_index_mid
        feat[12] = h1_spread_mid_ring
        feat[13] = h1_spread_ring_pinky
        feat[14] = h1_l_angle
        feat[15] = h1_c_curve
        feat[16] = h1_loop_closure
        feat[17] = h2_present
        feat[18] = h2_thumb_ext
        feat[19] = h2_index_ext
        feat[20] = h2_middle_ext
        feat[21] = h2_ring_ext
        feat[22] = h2_pinky_ext
        feat[23] = h2_loop_closure
        feat[24] = rel_wrist_x
        feat[25] = rel_wrist_y
        feat[26] = rel_wrist_z
        feat[27] = h1_idx_to_h2_thumb
        feat[28] = h1_idx_to_h2_idx
        feat[29] = h1_idx_to_h2_mid
        feat[30] = h1_idx_to_h2_ring
        feat[31] = h1_idx_to_h2_pinky
        feat[32] = h1_idx_to_h2_base
        feat[33] = h1_idx_to_h2_palm
        feat[34] = h1_idx_to_h2_cleft
        feat[35] = h1_idx_to_h2_pip
        feat[36] = h1_loop_to_h2_loop
        feat[37] = h1_thumb_to_h2_idx
        feat[38] = h1_loop_to_h2_idx
        feat[39] = h1_pinky_to_h2_pinky
        feat[40] = h1_knuckle_to_h2_knuckle
        feat[41] = h1_mid_to_h2_palm
        feat[42] = h1_ring_to_h2_palm
        feat[43] = h1_mid_cross_h2_idx

        # Add noise
        samples[i] = np.clip(feat + noise, -3.0, 5.0)

    return samples


class ISLAlphabetMLP(nn.Module):
    """
    Spatio-kinematic neural classifier for Indian Sign Language manual alphabet.
    3-Layer Multi-Layer Perceptron with Batch Normalization and Dropout.
    """
    def __init__(self, in_features: int = NUM_FEATURES, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(64, 48)
        self.bn2 = nn.BatchNorm1d(48)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.15)

        self.fc3 = nn.Linear(48, num_classes)

    def forward(self, x):
        x = self.drop1(self.relu1(self.bn1(self.fc1(x))))
        x = self.drop2(self.relu2(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x


def train_and_export():
    print("=" * 60)
    print(" SIGNBRIDGE ISL MANUAL ALPHABET NEURAL CLASSIFIER TRAINING")
    print(f" Supported Vocabulary ({NUM_CLASSES} classes): {', '.join(ALPHABET_CLASSES)}")
    print(f" Input Feature Dimension: {NUM_FEATURES} Kinematic Geometric Features")
    print("=" * 60)

    # 1. Dataset Generation
    X_list, y_list = [], []
    for cls_idx, sign_name in enumerate(ALPHABET_CLASSES):
        samples = generate_synthetic_features(sign_name, num_samples=250)
        X_list.append(samples)
        y_list.append(np.full(len(samples), cls_idx, dtype=np.int64))

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)

    # Shuffle
    indices = np.random.permutation(len(X_all))
    X_all = X_all[indices]
    y_all = y_all[indices]

    # Split: 75% train, 15% val, 10% test
    n_train = int(len(X_all) * 0.75)
    n_val = int(len(X_all) * 0.15)

    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_val, y_val = X_all[n_train:n_train+n_val], y_all[n_train:n_train+n_val]
    X_test, y_test = X_all[n_train+n_val:], y_all[n_train+n_val:]

    print(f"[Dataset] Train samples: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=32, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=64, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
        batch_size=64, shuffle=False
    )

    # 2. Model Training
    model = ISLAlphabetMLP(NUM_FEATURES, NUM_CLASSES)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)

    epochs = 40
    best_acc = 0.0
    best_weights = None

    start_time = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bx)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == by).sum().item()
            total += len(bx)

        train_acc = correct / total
        scheduler.step()

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                logits = model(bx)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == by).sum().item()
                val_total += len(bx)

        val_acc = val_correct / val_total
        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % 10 == 0 or ep == epochs:
            print(f"Epoch [{ep:02d}/{epochs}] — Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

    train_time = time.time() - start_time
    print(f"[Training Complete] Elapsed: {train_time:.2f}s | Best Val Acc: {best_acc*100:.2f}%")

    # 3. Test Evaluation
    model.load_state_dict(best_weights)
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for bx, by in test_loader:
            logits = model(bx)
            preds = torch.argmax(logits, dim=1)
            test_correct += (preds == by).sum().item()
            test_total += len(bx)

    test_acc = test_correct / test_total
    print(f"[Evaluation] Final Test Set Accuracy: {test_acc*100:.2f}%")

    # 4. Export PyTorch checkpoint
    saved_models_dir = Path(__file__).resolve().parent / "saved_models"
    saved_models_dir.mkdir(parents=True, exist_ok=True)
    pt_path = saved_models_dir / "signbridge_isl_alphabet.pt"
    torch.save({
        "state_dict": best_weights,
        "classes": ALPHABET_CLASSES,
        "features": FEATURE_NAMES,
        "test_acc": test_acc
    }, pt_path)
    print(f"[Export] PyTorch model saved to {pt_path}")

    # 5. Export fused inference weights for Browser & Server execution
    # In eval mode, batchnorm can be folded into linear weights:
    # W_fused = gamma / sqrt(var + eps) * W
    # b_fused = gamma / sqrt(var + eps) * (b - mean) + beta
    def fold_bn(linear_w, linear_b, bn_gamma, bn_beta, bn_mean, bn_var, eps=1e-5):
        scale = bn_gamma / np.sqrt(bn_var + eps)
        w_fused = linear_w * scale[:, np.newaxis]
        b_fused = (linear_b - bn_mean) * scale + bn_beta
        return w_fused, b_fused

    w1 = best_weights["fc1.weight"].numpy()
    b1 = best_weights["fc1.bias"].numpy()
    bn1_gamma = best_weights["bn1.weight"].numpy()
    bn1_beta = best_weights["bn1.bias"].numpy()
    bn1_mean = best_weights["bn1.running_mean"].numpy()
    bn1_var = best_weights["bn1.running_var"].numpy()
    w1_fused, b1_fused = fold_bn(w1, b1, bn1_gamma, bn1_beta, bn1_mean, bn1_var)

    w2 = best_weights["fc2.weight"].numpy()
    b2 = best_weights["fc2.bias"].numpy()
    bn2_gamma = best_weights["bn2.weight"].numpy()
    bn2_beta = best_weights["bn2.bias"].numpy()
    bn2_mean = best_weights["bn2.running_mean"].numpy()
    bn2_var = best_weights["bn2.running_var"].numpy()
    w2_fused, b2_fused = fold_bn(w2, b2, bn2_gamma, bn2_beta, bn2_mean, bn2_var)

    w3 = best_weights["fc3.weight"].numpy()
    b3 = best_weights["fc3.bias"].numpy()

    exported_model = {
        "model_name": "SignBridge-ISL-Alphabet-MLP",
        "version": "2.1.0",
        "standard": "ISLRTC-Fingerspelling",
        "accuracy": float(test_acc),
        "num_features": NUM_FEATURES,
        "feature_names": FEATURE_NAMES,
        "classes": ALPHABET_CLASSES,
        "weights": {
            "w1": [[round(float(v), 5) for v in row] for row in w1_fused],
            "b1": [round(float(v), 5) for v in b1_fused],
            "w2": [[round(float(v), 5) for v in row] for row in w2_fused],
            "b2": [round(float(v), 5) for v in b2_fused],
            "w3": [[round(float(v), 5) for v in row] for row in w3],
            "b3": [round(float(v), 5) for v in b3]
        }
    }

    # Save to backend
    json_path = saved_models_dir / "isl_alphabet_model.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(exported_model, f, indent=2)
    print(f"[Export] Fused JSON model exported to {json_path}")

    # Also save to frontend/src/services/ for direct client-side execution
    frontend_models_dir = Path(__file__).resolve().parent.parent / "frontend" / "src" / "services"
    frontend_models_dir.mkdir(parents=True, exist_ok=True)
    fe_json_path = frontend_models_dir / "isl_alphabet_model.json"
    with open(fe_json_path, "w", encoding="utf-8") as f:
        json.dump(exported_model, f, indent=2)
    print(f"[Export] Frontend client model exported to {fe_json_path}")


if __name__ == "__main__":
    train_and_export()
