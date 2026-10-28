
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


class MARIDADataset(Dataset):
    def __init__(self, root="data/raw", split="train", normalize=True):
        self.root = Path(root)
        self.patch_dir = self.root / "patches"
        self.normalize = normalize

        split_file = self.root / "splits" / f"{split}_X.txt"

        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        self.ids = [
            line.strip()
            for line in split_file.read_text().splitlines()
            if line.strip()
        ]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_name = f"S2_{self.ids[index]}.tif"
        matches = list(self.patch_dir.rglob(image_name))

        if not matches:
            raise FileNotFoundError(f"Image not found: {image_name}")

        image_path = matches[0]
        mask_path = image_path.with_name(
            image_path.stem + "_cl.tif"
        )
        conf_path = image_path.with_name(
            image_path.stem + "_conf.tif"
        )

        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        if not conf_path.exists():
            raise FileNotFoundError(f"Confidence map not found: {conf_path}")

        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32)

        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int64)

        with rasterio.open(conf_path) as src:
            confidence = src.read(1)

        if image.shape != (11, 256, 256):
            raise ValueError(
                f"Unexpected image shape {image.shape}: {image_path}"
            )

        if mask.shape != confidence.shape:
            raise ValueError("Mask and confidence-map shapes differ")

        # Preliminary per-band normalization.
        if self.normalize:
            lo = image.min(axis=(1, 2), keepdims=True)
            hi = image.max(axis=(1, 2), keepdims=True)
            image = (image - lo) / np.maximum(hi - lo, 1e-6)

        # Keep only pixels with a positive confidence value
        # and a named MARIDA class ID from 1 through 15.
        valid = (confidence > 0) & (mask >= 1) & (mask <= 15)

        # 255 is ignored by CrossEntropyLoss.
        target = np.full(mask.shape, 255, dtype=np.int64)
        target[valid] = mask[valid] - 1

        return (
            torch.from_numpy(image.copy()).float(),
            torch.from_numpy(target.copy()).long(),
        )


if __name__ == "__main__":
    dataset = MARIDADataset("data/raw", "train")
    image, mask = dataset[0]

    valid = mask != 255

    print("Training samples:", len(dataset))
    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)
    print("Valid pixels:", valid.sum().item())
    print("Ignored pixels:", (~valid).sum().item())
    print("Target class IDs:", torch.unique(mask[valid]).tolist())
    print("Image finite:", torch.isfinite(image).all().item())

