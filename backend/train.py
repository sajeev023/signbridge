import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import (
    CLASS_NAMES, SEQUENCE_LENGTH, NUM_FEATURES, NUM_CLASSES,
    MODEL_CHECKPOINT_PATH, MODEL_METADATA_PATH, DATASET_FILE
)
from app.models.architecture import SignBridgeBiLSTM
from app.services.dataset_generator import ISLKinematicSynthesizer

def train_model(epochs: int = 35, batch_size: int = 32, lr: float = 1e-3):
    print("=" * 60)
    print(" SIGNBRIDGE AI RECOGNITION ENGINE — MODEL TRAINING")
    print(" Architecture: 1D-Conv + Bidirectional LSTM + Temporal Attention")
    print(f" Target Vocabulary: {len(CLASS_NAMES)} Indian Sign Language Classes")
    print("=" * 60)

    # 1. Dataset verification or generation
    if not DATASET_FILE.exists():
        print("[Training] Dataset not found on disk. Initializing synthesis pipeline...")
        synthesizer = ISLKinematicSynthesizer(seq_len=SEQUENCE_LENGTH)
        dataset = synthesizer.build_dataset(samples_per_class=120, save_path=DATASET_FILE)
    else:
        print(f"[Training] Loading verified ISL sequence dataset from {DATASET_FILE}...")
        dataset = np.load(DATASET_FILE, allow_pickle=True)

    X_train = torch.tensor(dataset["X_train"], dtype=torch.float32)
    y_train = torch.tensor(dataset["y_train"], dtype=torch.long)
    X_val = torch.tensor(dataset["X_val"], dtype=torch.float32)
    y_val = torch.tensor(dataset["y_val"], dtype=torch.long)
    X_test = torch.tensor(dataset["X_test"], dtype=torch.float32)
    y_test = torch.tensor(dataset["y_test"], dtype=torch.long)

    print(f"[Dataset] Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    print(f"[Dataset] Tensor Shape: (Batch, Seq_Len={X_train.shape[1]}, Features={X_train.shape[2]})")

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    # 2. Hardware setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Hardware] Training device: {device}")

    # 3. Model instantiation
    model = SignBridgeBiLSTM(
        input_dim=NUM_FEATURES,
        hidden_dim=128,
        num_layers=2,
        num_classes=NUM_CLASSES,
        dropout=0.3
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Trainable Parameters: {total_params:,}")

    # 4. Optimizer, Loss & Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_weights = None
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    start_time = time.time()

    # 5. Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits, _ = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == batch_y).sum().item()
            total_train += batch_x.size(0)

        scheduler.step()

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits, _ = model(batch_x)
                loss = criterion(logits, batch_y)
                val_running_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(logits, dim=1)
                correct_val += (preds == batch_y).sum().item()
                total_val += batch_x.size(0)

        epoch_val_loss = val_running_loss / total_val
        epoch_val_acc = correct_val / total_val

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)

        if epoch_val_acc > best_val_acc or (epoch_val_acc == best_val_acc and epoch_val_loss < best_val_loss):
            best_val_acc = epoch_val_acc
            best_val_loss = epoch_val_loss
            best_weights = model.state_dict().copy()
            saved_indicator = "*"
        else:
            saved_indicator = ""

        if epoch % 5 == 0 or epoch == epochs or saved_indicator == "*":
            print(f"Epoch [{epoch:02d}/{epochs:02d}] "
                  f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc*100:.2f}% | "
                  f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc*100:.2f}% {saved_indicator}")

    elapsed = time.time() - start_time
    print(f"\n[Training Complete] Elapsed: {elapsed:.2f}s | Best Val Accuracy: {best_val_acc*100:.2f}%")

    # 6. Save checkpoint
    MODEL_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_weights, MODEL_CHECKPOINT_PATH)
    print(f"[Model Saved] Weights serialized to {MODEL_CHECKPOINT_PATH}")

    # 7. Evaluate on held-out Test Set
    model.load_state_dict(best_weights)
    model.eval()
    correct_test = 0
    total_test = 0
    test_preds = []
    test_targets = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits, _ = model(batch_x)
            preds = torch.argmax(logits, dim=1)
            correct_test += (preds == batch_y).sum().item()
            total_test += batch_x.size(0)
            test_preds.extend(preds.cpu().numpy().tolist())
            test_targets.extend(batch_y.cpu().numpy().tolist())

    test_acc = correct_test / total_test
    print(f"[Test Set Evaluation] Unseen Test Accuracy: {test_acc*100:.2f}% ({correct_test}/{total_test})")

    # 8. Save Metadata
    metadata = {
        "model_name": "SignBridge-BiLSTM-ISL",
        "version": "1.0.0",
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "architecture": {
            "type": "1D-CNN + Bidirectional LSTM + Temporal Attention",
            "input_dim": NUM_FEATURES,
            "hidden_dim": 128,
            "num_layers": 2,
            "num_classes": NUM_CLASSES,
            "total_parameters": total_params,
            "sequence_length": SEQUENCE_LENGTH
        },
        "dataset_split": {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "total_samples": len(X_train) + len(X_val) + len(X_test)
        },
        "performance": {
            "best_val_accuracy": float(best_val_acc),
            "test_accuracy": float(test_acc),
            "epochs_trained": epochs,
            "training_time_seconds": round(elapsed, 2)
        },
        "vocabulary": CLASS_NAMES,
        "history": history
    }

    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[Metadata Saved] Model specifications saved to {MODEL_METADATA_PATH}")
    return metadata

if __name__ == "__main__":
    train_model()
