
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from dataset_loader import MARIDADataset
from unet_model import UNet

ROOT = Path("data/raw")
MODEL_PATH = Path("outputs/marine_debris_segmentation.pth")
NUM_CLASSES = 15
IGNORE_INDEX = 255
BATCH_SIZE = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

dataset = MARIDADataset(root=str(ROOT), split="test")
loader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

model = UNet(in_channels=11, num_classes=NUM_CLASSES).to(device)

# Accept either a plain state dictionary or a wrapped checkpoint.
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model.eval()

cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

with torch.no_grad():
    for batch_no, (images, masks) in enumerate(loader, start=1):
        images = images.to(device)
        masks = masks.to(device)

        predictions = model(images).argmax(dim=1)

        valid = masks != IGNORE_INDEX
        y_true = masks[valid].cpu().numpy()
        y_pred = predictions[valid].cpu().numpy()

        if len(y_true):
            cm += confusion_matrix(
                y_true,
                y_pred,
                labels=list(range(NUM_CLASSES))
            )

        if batch_no % 50 == 0:
            print(f"Evaluated {batch_no}/{len(loader)} batches")

support = cm.sum(axis=1)
correct = np.diag(cm)
total = cm.sum()

accuracy = correct.sum() / total if total else 0.0

print("\nTest pixels evaluated:", int(total))
print(f"Pixel accuracy: {accuracy:.4f}")
print("\nClass ID | Support | Precision | Recall | Dice")

dice_scores = []

for i in range(NUM_CLASSES):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0

    if support[i] > 0:
        dice_scores.append(dice)

    print(
        f"{i + 1:8} | {support[i]:7} | "
        f"{precision:.4f}    | {recall:.4f} | {dice:.4f}"
    )

mean_dice = np.mean(dice_scores) if dice_scores else 0.0
print(f"\nMean Dice (classes with test support): {mean_dice:.4f}")

