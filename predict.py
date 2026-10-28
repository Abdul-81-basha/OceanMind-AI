
from pathlib import Path

import numpy as np
import rasterio
import torch

from dataset_loader import MARIDADataset
from unet_model import UNet

ROOT = Path("data/raw")
MODEL_PATH = Path("outputs/marine_debris_segmentation.pth")
OUTPUT_DIR = Path("outputs/predictions")
NUM_CLASSES = 15
IGNORE_INDEX = 255


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    dataset = MARIDADataset(root=str(ROOT), split="test")
    image, target = dataset[0]
    sample_id = dataset.ids[0]

    model = UNet(in_channels=11, num_classes=NUM_CLASSES).to(device)

    checkpoint = torch.load(
        MODEL_PATH, map_location=device, weights_only=True
    )
    state_dict = (
        checkpoint["model_state_dict"]
        if "model_state_dict" in checkpoint
        else checkpoint
    )

    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device))
        prediction = logits.argmax(dim=1).squeeze(0).cpu().numpy()

    # Convert model IDs 0-14 back to MARIDA IDs 1-15.
    prediction = prediction.astype(np.uint8) + 1

    # Preserve unlabeled areas as 0 for visualization.
    valid = target.numpy() != IGNORE_INDEX
    prediction[~valid] = 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{sample_id}_prediction.tif"

    with rasterio.open(
        next((ROOT / "patches").rglob(f"S2_{sample_id}.tif"))
    ) as src:
        profile = src.profile.copy()

    profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        height=prediction.shape[0],
        width=prediction.shape[1],
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(prediction, 1)

    print("Test sample:", sample_id)
    print("Prediction shape:", prediction.shape)
    print("Predicted MARIDA class IDs:", np.unique(prediction).tolist())
    print("Saved prediction:", output_path)


if __name__ == "__main__":
    main()

