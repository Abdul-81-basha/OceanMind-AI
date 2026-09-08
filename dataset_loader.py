import rasterio
import numpy as np
import torch
from torch.utils.data import Dataset


class MARIDADataset(Dataset):

    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]

        # Read satellite image
        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32)

        # Normalize each band independently
        for band in range(image.shape[0]):

            minimum = image[band].min()
            maximum = image[band].max()

            if maximum > minimum:
                image[band] = (
                    (image[band] - minimum)
                    / (maximum - minimum)
                )

        # Convert NumPy → PyTorch
        image = torch.tensor(image, dtype=torch.float32)

        return image


# ---------------------------------
# Our two currently available images
# ---------------------------------

image_paths = [
    "dataset/raw/S2_1-12-19_48MYU_0.tif",
    "dataset/raw/S2_1-12-19_48MYU_2.tif"
]


# Create dataset
dataset = MARIDADataset(image_paths)

print("Number of images:", len(dataset))

# Load first image
image = dataset[0]

print("Tensor shape:", image.shape)
print("Tensor type:", image.dtype)
print("Minimum value:", image.min().item())
print("Maximum value:", image.max().item())