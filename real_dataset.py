import rasterio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class MARIDABinaryDataset(Dataset):

    def __init__(self, image_paths, mask_paths):

        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        # -----------------------------
        # Read satellite image
        # -----------------------------

        with rasterio.open(self.image_paths[index]) as src:
            image = src.read().astype(np.float32)

        # -----------------------------
        # Normalize each band
        # -----------------------------

        for band in range(image.shape[0]):

            minimum = image[band].min()
            maximum = image[band].max()

            if maximum > minimum:

                image[band] = (
                    (image[band] - minimum)
                    / (maximum - minimum)
                )

        # -----------------------------
        # Select RGB
        # -----------------------------

        image = image[[2, 1, 0]]

        # -----------------------------
        # Convert NumPy → PyTorch
        # -----------------------------

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        # -----------------------------
        # Resize to ViT input
        # -----------------------------

        image = image.unsqueeze(0)

        image = F.interpolate(
            image,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        image = image.squeeze(0)

        # -----------------------------
        # Read classification mask
        # -----------------------------

        with rasterio.open(self.mask_paths[index]) as src:
            mask = src.read(1)

        # -----------------------------
        # Class 1 = Marine Debris
        # -----------------------------

        if np.any(mask == 1):
            label = 1
        else:
            label = 0

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label


# =====================================
# Our currently available MARIDA data
# =====================================

image_paths = [
    "dataset/raw/S2_1-12-19_48MYU_0.tif",
    "dataset/raw/S2_1-12-19_48MYU_2.tif"
]

mask_paths = [
    "dataset/raw/S2_1-12-19_48MYU_0_cl.tif",
    "dataset/raw/S2_1-12-19_48MYU_2_cl.tif"
]


# Create dataset
dataset = MARIDABinaryDataset(
    image_paths,
    mask_paths
)


# =====================================
# Test dataset
# =====================================

print("Number of samples:", len(dataset))

for i in range(len(dataset)):

    image, label = dataset[i]

    print()
    print("Sample:", i)
    print("Image shape:", image.shape)
    print("Label:", label.item())

    