"""
SIGNBRIDGE Single-Handed Manual Alphabet Model Trainer & Exporter
Trains a neural landmark classifier on the 1-handed manual alphabet
(matching the visual reference chart) with scale and translation invariance.
"""

import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

SINGLE_HAND_CLASSES = [
    "NEUTRAL",
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "K", "L", "O", "R", "U", "V", "W", "X", "Y"
]
CLASS_TO_IDX = {c: i for i, c in enumerate(SINGLE_HAND_CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(SINGLE_HAND_CLASSES)}

SINGLE_HAND_FEATURES = [
    # 5 Finger Extension Ratios
    "thumb_ext", "index_ext", "middle_ext", "ring_ext", "pinky_ext",
    # 5 Tip to Wrist Distances
    "dist_thumb_wrist", "dist_idx_wrist", "dist_mid_wrist", "dist_ring_wrist", "dist_pinky_wrist",
    # 5 Finger Curl Angles
    "thumb_curl", "index_curl", "middle_curl", "ring_curl", "pinky_curl",
    # 4 Inter-finger Spreads
    "spread_thumb_idx", "spread_idx_mid", "spread_mid_ring", "spread_ring_pinky",
    # Specific Handshape Invariants
    "loop_thumb_idx",       # Key for 'F', 'D', 'O'
    "loop_thumb_mid",       # Key for 'D'
    "l_angle",              # Key for 'L' (~90 deg)
    "y_span",               # Key for 'Y' (thumb to pinky wide spread)
    "crossed_idx_mid",      # Key for 'R'
    "fist_compactness",     # Key for 'A', 'E', 'S'
    "horizontal_alignment", # Key for 'G', 'H'
    "c_curve_metric",       # Key for 'C'
    "u_parallel_metric"     # Key for 'U' (index & middle touching upright)
]

NUM_FEATURES = len(SINGLE_HAND_FEATURES)
NUM_CLASSES = len(SINGLE_HAND_CLASSES)


def generate_single_hand_features(sign: str, num_samples: int = 250) -> np.ndarray:
    samples = np.zeros((num_samples, NUM_FEATURES), dtype=np.float32)

    for i in range(num_samples):
        feat = np.zeros(NUM_FEATURES, dtype=np.float32)
        noise = np.random.normal(0, 0.03, size=NUM_FEATURES).astype(np.float32)

        # Baseline defaults
        thumb_ext = 0.2
        index_ext = 0.0
        middle_ext = 0.0
        ring_ext = 0.0
        pinky_ext = 0.0

        dist_thumb_wrist = 0.9
        dist_idx_wrist = 0.8
        dist_mid_wrist = 0.8
        dist_ring_wrist = 0.75
        dist_pinky_wrist = 0.7

        thumb_curl = 0.5
        index_curl = 0.8
        middle_curl = 0.8
        ring_curl = 0.8
        pinky_curl = 0.8

        spread_thumb_idx = 0.4
        spread_idx_mid = 0.15
        spread_mid_ring = 0.15
        spread_ring_pinky = 0.15

        loop_thumb_idx = 0.6
        loop_thumb_mid = 0.7
        l_angle = 0.1
        y_span = 0.8
        crossed_idx_mid = 0.0
        fist_compactness = 0.6
        horizontal_alignment = 0.1
        c_curve_metric = 0.1
        u_parallel_metric = 0.1

        if sign == "NEUTRAL":
            # Relaxed, resting or waving hand
            thumb_ext = np.random.uniform(-0.2, 0.8)
            index_ext = np.random.uniform(-0.2, 0.8)
            middle_ext = np.random.uniform(-0.2, 0.8)
            ring_ext = np.random.uniform(-0.2, 0.8)
            pinky_ext = np.random.uniform(-0.2, 0.8)
            dist_idx_wrist = 1.0 + np.random.uniform(-0.3, 0.3)

        elif sign == "A":
            # Fist with thumb resting alongside index
            thumb_ext = 0.4
            index_ext = -0.3
            middle_ext = -0.3
            ring_ext = -0.3
            pinky_ext = -0.3
            fist_compactness = np.random.uniform(0.85, 0.98)
            dist_thumb_wrist = 0.95

        elif sign == "B":
            # 4 fingers upright together, thumb tucked across palm
            thumb_ext = -0.4
            index_ext = 1.2
            middle_ext = 1.2
            ring_ext = 1.15
            pinky_ext = 1.1
            spread_idx_mid = np.random.uniform(0.05, 0.12)  # Together!
            spread_mid_ring = np.random.uniform(0.05, 0.12) # Together!
            spread_ring_pinky = np.random.uniform(0.05, 0.12)
            dist_idx_wrist = 1.7
            dist_mid_wrist = 1.75
            dist_ring_wrist = 1.7
            dist_pinky_wrist = 1.6

        elif sign == "C":
            # Curved C-shape with all fingers curved facing thumb
            thumb_ext = 0.5
            index_ext = 0.5
            middle_ext = 0.5
            ring_ext = 0.5
            pinky_ext = 0.5
            c_curve_metric = np.random.uniform(0.88, 0.99)
            loop_thumb_idx = np.random.uniform(0.4, 0.6) # C gap

        elif sign == "D":
            # Index upright; thumb touching middle/ring/pinky tips
            index_ext = 1.25
            middle_ext = -0.2
            ring_ext = -0.2
            pinky_ext = -0.2
            dist_idx_wrist = 1.7
            loop_thumb_mid = np.random.uniform(0.05, 0.15) # Touching

        elif sign == "E":
            # All fingers curled tightly with thumb tucked under
            thumb_ext = -0.4
            index_ext = -0.5
            middle_ext = -0.5
            ring_ext = -0.5
            pinky_ext = -0.5
            fist_compactness = np.random.uniform(0.9, 0.99)
            dist_idx_wrist = 0.6

        elif sign == "F":
            # Index and thumb touching in circle, middle/ring/pinky upright
            loop_thumb_idx = np.random.uniform(0.04, 0.14) # Circle!
            index_ext = 0.2
            middle_ext = 1.2
            ring_ext = 1.15
            pinky_ext = 1.1
            dist_mid_wrist = 1.75

        elif sign == "G":
            # Index pointing horizontal, thumb parallel forward
            index_ext = 1.1
            middle_ext = -0.3
            ring_ext = -0.3
            pinky_ext = -0.3
            horizontal_alignment = np.random.uniform(0.85, 0.98)
            spread_thumb_idx = np.random.uniform(0.15, 0.3)

        elif sign == "H":
            # Index and middle pointing horizontal together
            index_ext = 1.1
            middle_ext = 1.1
            ring_ext = -0.3
            pinky_ext = -0.3
            horizontal_alignment = np.random.uniform(0.88, 0.99)
            spread_idx_mid = np.random.uniform(0.05, 0.12)

        elif sign == "I":
            # Pinky upright, all other fingers closed into palm
            thumb_ext = -0.2
            index_ext = -0.4
            middle_ext = -0.4
            ring_ext = -0.4
            pinky_ext = 1.15
            dist_pinky_wrist = 1.55

        elif sign == "K":
            # Index upright, middle pointing forward, thumb at base of middle
            index_ext = 1.2
            middle_ext = 0.7
            ring_ext = -0.3
            pinky_ext = -0.3
            spread_idx_mid = np.random.uniform(0.3, 0.5)

        elif sign == "L":
            # L shape: thumb and index at 90 deg, others closed
            thumb_ext = 1.05
            index_ext = 1.25
            middle_ext = -0.4
            ring_ext = -0.4
            pinky_ext = -0.4
            l_angle = np.random.uniform(0.88, 0.99) # 90 degree L!
            spread_thumb_idx = np.random.uniform(0.7, 0.95)

        elif sign == "O":
            # All 5 fingers touching tips forming O
            loop_thumb_idx = np.random.uniform(0.05, 0.15)
            loop_thumb_mid = np.random.uniform(0.05, 0.15)
            index_ext = 0.2
            middle_ext = 0.2
            ring_ext = 0.2
            pinky_ext = 0.2
            c_curve_metric = np.random.uniform(0.75, 0.9)

        elif sign == "R":
            # Index and middle crossed
            index_ext = 1.15
            middle_ext = 1.15
            ring_ext = -0.3
            pinky_ext = -0.3
            crossed_idx_mid = np.random.uniform(0.85, 0.98) # Crossed!

        elif sign == "U":
            # Index and middle upright together (parallel)
            index_ext = 1.2
            middle_ext = 1.2
            ring_ext = -0.4
            pinky_ext = -0.4
            u_parallel_metric = np.random.uniform(0.88, 0.99)
            spread_idx_mid = np.random.uniform(0.04, 0.11)

        elif sign == "V":
            # Index and middle upright spread in V
            index_ext = 1.2
            middle_ext = 1.2
            ring_ext = -0.4
            pinky_ext = -0.4
            spread_idx_mid = np.random.uniform(0.65, 0.9) # V-spread!
            u_parallel_metric = 0.1

        elif sign == "W":
            # Index, middle, ring upright spread in W
            index_ext = 1.15
            middle_ext = 1.2
            ring_ext = 1.15
            pinky_ext = -0.4
            spread_idx_mid = np.random.uniform(0.4, 0.7)
            spread_mid_ring = np.random.uniform(0.4, 0.7)

        elif sign == "X":
            # Hooked index finger
            index_ext = 0.3
            index_curl = np.random.uniform(0.75, 0.95) # Bent hook!
            middle_ext = -0.3
            ring_ext = -0.3
            pinky_ext = -0.3

        elif sign == "Y":
            # Thumb and pinky extended, middle 3 closed (shaka)
            thumb_ext = 1.05
            index_ext = -0.4
            middle_ext = -0.4
            ring_ext = -0.4
            pinky_ext = 1.1
            y_span = np.random.uniform(1.3, 1.8) # Wide span!

        feat[0] = thumb_ext
        feat[1] = index_ext
        feat[2] = middle_ext
        feat[3] = ring_ext
        feat[4] = pinky_ext
        feat[5] = dist_thumb_wrist
        feat[6] = dist_idx_wrist
        feat[7] = dist_mid_wrist
        feat[8] = dist_ring_wrist
        feat[9] = dist_pinky_wrist
        feat[10] = thumb_curl
        feat[11] = index_curl
        feat[12] = middle_curl
        feat[13] = ring_curl
        feat[14] = pinky_curl
        feat[15] = spread_thumb_idx
        feat[16] = spread_idx_mid
        feat[17] = spread_mid_ring
        feat[18] = spread_ring_pinky
        feat[19] = loop_thumb_idx
        feat[20] = loop_thumb_mid
        feat[21] = l_angle
        feat[22] = y_span
        feat[23] = crossed_idx_mid
        feat[24] = fist_compactness
        feat[25] = horizontal_alignment
        feat[26] = c_curve_metric
        feat[27] = u_parallel_metric

        samples[i] = np.clip(feat + noise, -3.0, 5.0)

    return samples


class SingleHandMLP(nn.Module):
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


def train_single_hand():
    print("=" * 60)
    print(" TRAINING SINGLE-HAND MANUAL ALPHABET CLASSIFIER")
    print(f" Classes ({NUM_CLASSES}): {', '.join(SINGLE_HAND_CLASSES)}")
    print(f" Features: {NUM_FEATURES}")
    print("=" * 60)

    X_list, y_list = [], []
    for cls_idx, sign_name in enumerate(SINGLE_HAND_CLASSES):
        samples = generate_single_hand_features(sign_name, num_samples=300)
        X_list.append(samples)
        y_list.append(np.full(len(samples), cls_idx, dtype=np.int64))

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)

    idx = np.random.permutation(len(X_all))
    X_all, y_all = X_all[idx], y_all[idx]

    n_train = int(len(X_all) * 0.75)
    n_val = int(len(X_all) * 0.15)

    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_val, y_val = X_all[n_train:n_train+n_val], y_all[n_train:n_train+n_val]
    X_test, y_test = X_all[n_train+n_val:], y_all[n_train+n_val:]

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

    model = SingleHandMLP(NUM_FEATURES, NUM_CLASSES)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)

    epochs = 40
    best_acc = 0.0
    best_weights = None

    for ep in range(1, epochs + 1):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        val_corr, val_tot = 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                preds = torch.argmax(model(bx), dim=1)
                val_corr += (preds == by).sum().item()
                val_tot += len(bx)

        val_acc = val_corr / val_tot
        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % 10 == 0 or ep == epochs:
            print(f"Epoch [{ep:02d}/{epochs}] — Val Acc: {val_acc*100:.2f}%")

    # Evaluate test
    model.load_state_dict(best_weights)
    model.eval()
    test_corr, test_tot = 0, 0
    with torch.no_grad():
        for bx, by in test_loader:
            preds = torch.argmax(model(bx), dim=1)
            test_corr += (preds == by).sum().item()
            test_tot += len(bx)
    test_acc = test_corr / test_tot
    print(f"[Single Hand Model] Test Accuracy: {test_acc*100:.2f}%")

    # Fold BatchNorm
    def fold_bn(linear_w, linear_b, bn_gamma, bn_beta, bn_mean, bn_var, eps=1e-5):
        scale = bn_gamma / np.sqrt(bn_var + eps)
        w_fused = linear_w * scale[:, np.newaxis]
        b_fused = (linear_b - bn_mean) * scale + bn_beta
        return w_fused, b_fused

    w1_fused, b1_fused = fold_bn(
        best_weights["fc1.weight"].numpy(), best_weights["fc1.bias"].numpy(),
        best_weights["bn1.weight"].numpy(), best_weights["bn1.bias"].numpy(),
        best_weights["bn1.running_mean"].numpy(), best_weights["bn1.running_var"].numpy()
    )
    w2_fused, b2_fused = fold_bn(
        best_weights["fc2.weight"].numpy(), best_weights["fc2.bias"].numpy(),
        best_weights["bn2.weight"].numpy(), best_weights["bn2.bias"].numpy(),
        best_weights["bn2.running_mean"].numpy(), best_weights["bn2.running_var"].numpy()
    )
    w3 = best_weights["fc3.weight"].numpy()
    b3 = best_weights["fc3.bias"].numpy()

    exported_model = {
        "model_name": "SignBridge-SingleHand-Alphabet-MLP",
        "version": "2.1.0",
        "reference": "Single-Hand-Manual-Alphabet-Chart",
        "accuracy": float(test_acc),
        "num_features": NUM_FEATURES,
        "feature_names": SINGLE_HAND_FEATURES,
        "classes": SINGLE_HAND_CLASSES,
        "weights": {
            "w1": [[round(float(v), 5) for v in row] for row in w1_fused],
            "b1": [round(float(v), 5) for v in b1_fused],
            "w2": [[round(float(v), 5) for v in row] for row in w2_fused],
            "b2": [round(float(v), 5) for v in b2_fused],
            "w3": [[round(float(v), 5) for v in row] for row in w3],
            "b3": [round(float(v), 5) for v in b3]
        }
    }

    # Save to frontend
    fe_json = Path(__file__).resolve().parent.parent / "frontend" / "src" / "services" / "single_hand_alphabet_model.json"
    with open(fe_json, "w", encoding="utf-8") as f:
        json.dump(exported_model, f, indent=2)
    print(f"[Export] Saved to {fe_json}")


if __name__ == "__main__":
    train_single_hand()
