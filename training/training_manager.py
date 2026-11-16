"""
training_manager.py
--------------------
Trains the Manager CNN that classifies audio segments by noise type and
routes them to the appropriate denoising Expert in the MoE pipeline.

Usage
-----
Run from the project root or the training/ directory:

    python training/training_manager.py

Inputs  (must exist — run extract_features.py first)
------
training/X_features.npy  — shape (N, N_MFCC, TIME_STEPS), float32
training/y_labels.npy    — shape (N,), int64

Outputs
-------
training/manager_model.pth   — saved model weights (state_dict)
training/class_map.txt       — class-index → name mapping (read-only here)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = "training"
X_PATH     = os.path.join(DATA_DIR, "X_features.npy")
Y_PATH     = os.path.join(DATA_DIR, "y_labels.npy")
MODEL_PATH = os.path.join(DATA_DIR, "manager_model.pth")
CLASS_MAP_PATH = os.path.join(DATA_DIR, "class_map.txt")

# ─── Hyperparameters ──────────────────────────────────────────────────────────
BATCH_SIZE   = 32
EPOCHS       = 20
LEARNING_RATE = 1e-3
TRAIN_SPLIT  = 0.80   # 80% train, 20% test
DROPOUT_RATE = 0.4    # Regularization — tune if overfitting persists


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD AND PREPARE DATA
# ─────────────────────────────────────────────────────────────────────────────
print("Loading extracted features...")

if not os.path.isfile(X_PATH) or not os.path.isfile(Y_PATH):
    raise FileNotFoundError(
        f"Feature files not found at {X_PATH} / {Y_PATH}.\n"
        "Run extract_features.py first."
    )

X = np.load(X_PATH)   # (N, N_MFCC, TIME_STEPS)
y = np.load(Y_PATH)   # (N,)

num_classes = int(y.max()) + 1
print(f"Dataset: {len(X)} samples | {num_classes} classes")
print(f"Class distribution: { {int(c): int((y == c).sum()) for c in np.unique(y)} }")

# Load human-readable class names (optional — for the final report print)
class_names = {}
if os.path.isfile(CLASS_MAP_PATH):
    with open(CLASS_MAP_PATH) as f:
        for line in f:
            idx, name = line.strip().split("\t")
            class_names[int(idx)] = name

# PyTorch CNNs expect (Batch, Channels, Height, Width) — add a grayscale channel dim
X = np.expand_dims(X, axis=1)                      # (N, 1, N_MFCC, TIME_STEPS)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

dataset    = TensorDataset(X_tensor, y_tensor)
train_size = int(TRAIN_SPLIT * len(dataset))
test_size  = len(dataset) - train_size
train_dataset, test_dataset = random_split(
    dataset, [train_size, test_size],
    generator=torch.Generator().manual_seed(42),   # Reproducible split
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Train samples: {train_size} | Test samples: {test_size}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. DEFINE THE CNN "MANAGER" ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
class ManagerCNN(nn.Module):
    """
    Lightweight CNN that classifies 1-second MFCC windows into noise categories.

    Architecture overview
    ─────────────────────
    Input  : (B, 1, N_MFCC, TIME_STEPS)  — grayscale spectrogram
    Conv1  : 16 filters, 3×3 → BatchNorm → ReLU → MaxPool /2
    Conv2  : 32 filters, 3×3 → BatchNorm → ReLU → MaxPool /2
    Conv3  : 64 filters, 3×3 → BatchNorm → ReLU → GlobalAvgPool
    FC1    : 64 → 128 → Dropout → ReLU
    FC2    : 128 → num_classes  (logits for CrossEntropyLoss)

    BatchNorm after each conv stabilises training on small datasets.
    GlobalAvgPool makes the classifier resolution-agnostic, so it works
    regardless of exact TIME_STEPS (which varies slightly with hop_length).
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.4):
        super().__init__()

        # ── Convolutional feature extractor ──────────────────────────────────
        # Block 1: basic edge / contour detection
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Block 2: mid-level texture / harmonic pattern detection
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Block 3: high-level class-specific pattern detection
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Global Average Pooling: (B, 64, H, W) → (B, 64)
        # Removes the spatial dimensions so the FC head is resolution-agnostic.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ── Classification head ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)   # Flatten (B, 64, 1, 1) → (B, 64)
        x = self.classifier(x)
        return x                     # Raw logits; loss fn applies softmax internally


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAIN THE MODEL
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

model     = ManagerCNN(num_classes=num_classes, dropout=DROPOUT_RATE).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# Decay LR by ×0.5 if validation loss doesn't improve for 5 epochs
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

print(f"\nStarting training for {EPOCHS} epochs...")
print("-" * 50)

best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    # ── Training pass ─────────────────────────────────────────────────────
    model.train()
    total_train_loss = 0.0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss    = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # ── Validation pass (no gradient) ─────────────────────────────────────
    model.eval()
    total_val_loss = 0.0
    correct = 0
    total   = 0

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            total_val_loss += criterion(outputs, batch_y).item()
            _, predicted = torch.max(outputs, 1)
            total   += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    avg_val_loss = total_val_loss / len(test_loader)
    val_acc      = 100.0 * correct / total

    scheduler.step(avg_val_loss)

    print(
        f"Epoch {epoch:>3}/{EPOCHS} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Val Acc: {val_acc:.1f}%"
    )

    # Save checkpoint whenever validation loss improves
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  ✓ New best model saved (val_loss={best_val_loss:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# 4. FINAL EVALUATION WITH PER-CLASS BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("Final evaluation on test set")
print("=" * 50)

# Reload best checkpoint for final evaluation
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

all_preds  = []
all_labels = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        outputs  = model(batch_X)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_y.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

overall_acc = 100.0 * (all_preds == all_labels).mean()
print(f"Overall Accuracy: {overall_acc:.2f}%\n")

# Confusion matrix — rows = actual, cols = predicted
print("Confusion Matrix (rows=actual, cols=predicted):")
header = "       " + "  ".join(
    f"{class_names.get(i, str(i)):>8}" for i in range(num_classes)
)
print(header)
for actual in range(num_classes):
    row = "  ".join(
        f"{int((all_preds[all_labels == actual] == pred).sum()):>8}"
        for pred in range(num_classes)
    )
    label_name = class_names.get(actual, f"class_{actual}")
    print(f"{label_name:>7}: {row}")

# Per-class precision / recall
print("\nPer-class metrics:")
print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
print("-" * 60)
for cls in range(num_classes):
    tp = int(((all_preds == cls) & (all_labels == cls)).sum())
    fp = int(((all_preds == cls) & (all_labels != cls)).sum())
    fn = int(((all_preds != cls) & (all_labels == cls)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    support   = int((all_labels == cls).sum())

    name = class_names.get(cls, f"class_{cls}")
    print(f"{name:<20} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f} {support:>10}")

print(f"\nBest model weights saved to '{MODEL_PATH}'")
print("Done! Use this model in cleaner.py to route frames to the right Expert.")
