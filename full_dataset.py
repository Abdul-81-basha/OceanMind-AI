import rasterio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class FullMARIDADataset(Dataset):

    def __init__(
        self,
        split_file,
        max_samples=None
    ):

        self.root = "patches"

        # Read the MARIDA split file
        with open(split_file, "r") as f:

            self.names = [
                line.strip()
                for line in f
                if line.strip()
            ]

        # Limit samples while testing
        if max_samples is not None:
            self.names = self.names[:max_samples]

        print(
            "Samples loaded:",
            len(self.names)
        )

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):

        # Name from train_X.txt
        name = self.names[index]

        # --------------------------------
        # Add S2_ prefix
        # --------------------------------

        actual_name = "S2_" + name

        # --------------------------------
        # Find the scene folder
        # --------------------------------

        parts = name.split("_")

        scene = "S2_" + "_".join(parts[:-1])

        # --------------------------------
        # Build image path
        # --------------------------------

        image_path = (
            f"{self.root}/{scene}/{actual_name}.tif"
        )

        # --------------------------------
        # Build mask path
        # --------------------------------

        mask_path = (
            f"{self.root}/{scene}/{actual_name}_cl.tif"
        )

        print()
        print("Loading:", actual_name)

        # --------------------------------
        # Read Sentinel-2 image
        # --------------------------------

        with rasterio.open(image_path) as src:

            image = src.read().astype(
                np.float32
            )

        # --------------------------------
        # Normalize each band
        # --------------------------------

        for band in range(image.shape[0]):

            minimum = image[band].min()
            maximum = image[band].max()

            if maximum > minimum:

                image[band] = (
                    (image[band] - minimum)
                    / (maximum - minimum)
                )

        # --------------------------------
        # Select RGB bands
        # --------------------------------

        image = image[[2, 1, 0]]

        # --------------------------------
        # NumPy → PyTorch
        # --------------------------------

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        # --------------------------------
        # Resize to ViT input
        # --------------------------------

        image = image.unsqueeze(0)

        image = F.interpolate(
            image,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        image = image.squeeze(0)

        # --------------------------------
        # Read classification mask
        # --------------------------------

        with rasterio.open(mask_path) as src:

            mask = src.read(1)

        # --------------------------------
        # Determine Marine Debris label
        # --------------------------------

        if np.any(mask == 1):

            label = 1

        else:

            label = 0

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label


# ======================================
# TEST THE DATASET
# ======================================

dataset = FullMARIDADataset(
    "splits/train_X.txt",
    max_samples=100
)


print()
print("Dataset length:", len(dataset))


for i in range(
    min(3, len(dataset))
):

    image, label = dataset[i]

    print()
    print("Sample:", i)
    print("Image shape:", image.shape)
    print("Label:", label.item())

