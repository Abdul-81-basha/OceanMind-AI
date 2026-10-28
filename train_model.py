
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_loader import MARIDADataset
from unet_model import UNet

# Fast training settings
EPOCHS = 3
BATCH_SIZE = 2
LEARNING_RATE = 0.0005
NUM_CLASSES = 15
IGNORE_INDEX = 255

ROOT = Path("data/raw")
MODEL_PATH = Path("outputs/marine_debris_segmentation.pth")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, flush=True)

print("Loading datasets...", flush=True)

train_data = MARIDADataset(root=str(ROOT), split="train")
val_data = MARIDADataset(root=str(ROOT), split="val")

train_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_data, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0
)

print("Training patches:", len(train_data), flush=True)
print("Validation patches:", len(val_data), flush=True)

# Pixel counts from the training split, MARIDA classes 1-15
counts = np.array([
    1943, 870, 1091, 723, 3289,
    65295, 86877, 154335, 469, 86820,
    13852, 2975, 5675, 4974, 224
], dtype=np.float64)

# Moderate class weighting to help rare classes
weights = np.sqrt(counts.sum() / (NUM_CLASSES * counts))
weights /= weights.mean()
weights = np.clip(weights, 0.5, 3.0)

class_weights = torch.tensor(
    weights, dtype=torch.float32, device=device
)

print("Class weights:", np.round(weights, 2).tolist(), flush=True)

model = UNet(in_channels=11, num_classes=NUM_CLASSES).to(device)

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    ignore_index=IGNORE_INDEX
)

optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE
)

best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    train_batches = 0

    print(f"\nEpoch {epoch + 1}/{EPOCHS} started", flush=True)

    for step, (images, masks) in enumerate(train_loader, start=1):
        images = images.to(device)
        masks = masks.to(device)

        if not (masks != IGNORE_INDEX).any():
            continue

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)

        if not torch.isfinite(loss):
            continue

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_batches += 1

        if step % 50 == 0:
            print(
                f"  Batch {step}/{len(train_loader)}",
                flush=True
            )

    model.eval()
    val_loss = 0.0
    val_batches = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            if not (masks != IGNORE_INDEX).any():
                continue

            outputs = model(images)
            loss = criterion(outputs, masks)

            if torch.isfinite(loss):
                val_loss += loss.item()
                val_batches += 1

    avg_train = train_loss / max(train_batches, 1)
    avg_val = (
        val_loss / val_batches
        if val_batches else float("inf")
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train loss: {avg_train:.4f} | "
        f"Val loss: {avg_val:.4f}",
        flush=True
    )

    if val_batches and avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(), MODEL_PATH)
        print("Saved best model:", MODEL_PATH, flush=True)

print("\nTraining finished.", flush=True)
print("Best validation loss:", best_val_loss, flush=True)
