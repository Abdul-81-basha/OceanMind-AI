import json
import os

import rasterio
import numpy as np
import torch

from torch.utils.data import Dataset
import torch.nn.functional as F


class MARIDAResNetDataset(Dataset):

    def __init__(
        self,
        split_file,
        max_samples=None
    ):

        self.root = "patches"

        # ======================================
        # Load official MARIDA labels
        # ======================================

        with open(
            "labels_mapping.txt",
            "r"
        ) as f:

            self.labels = json.load(f)


        # ======================================
        # Load split
        # ======================================

        with open(
            split_file,
            "r"
        ) as f:

            names = [
                line.strip()
                for line in f
                if line.strip()
            ]


        # ======================================
        # Match split names with labels
        # ======================================

        self.names = []

        for name in names:

            # Split files don't contain S2_
            filename = "S2_" + name + ".tif"

            if filename in self.labels:

                self.names.append(filename)


        # ======================================
        # Limit samples for testing
        # ======================================

        if max_samples is not None:

            self.names = self.names[
                :max_samples
            ]


        print()
        print(
            "Dataset:",
            split_file
        )

        print(
            "Samples:",
            len(self.names)
        )


    def __len__(self):

        return len(self.names)


    def __getitem__(self, index):

        # ======================================
        # Filename
        # ======================================

        filename = self.names[index]


        # ======================================
        # Get scene folder
        # ======================================

        name_without_extension = (
            filename[:-4]
        )

        parts = name_without_extension.split("_")

        scene = "_".join(parts[:-1])


        # ======================================
        # Image path
        # ======================================

        image_path = os.path.join(
            self.root,
            scene,
            filename
        )


        # ======================================
        # Read Sentinel-2 image
        # ======================================

        with rasterio.open(
            image_path
        ) as src:

            image = src.read().astype(
                np.float32
            )


        # ======================================
        # Normalize each band
        # ======================================

        for band in range(
            image.shape[0]
        ):

            minimum = image[band].min()
            maximum = image[band].max()

            if maximum > minimum:

                image[band] = (
                    image[band] - minimum
                ) / (
                    maximum - minimum
                )


        # ======================================
        # Convert Sentinel-2 to RGB
        # ======================================

        image = image[
            [2, 1, 0]
        ]


        # ======================================
        # NumPy → PyTorch
        # ======================================

        image = torch.tensor(
            image,
            dtype=torch.float32
        )


        # ======================================
        # Resize to 224 × 224
        # ======================================

        image = image.unsqueeze(0)

        image = F.interpolate(
            image,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        image = image.squeeze(0)


        # ======================================
        # Official MARIDA label
        # ======================================

        classes = self.labels[
            filename
        ]


        # First class = Marine Debris
        #
        # 0 = Non-Marine-Debris
        # 1 = Marine-Debris

        if classes[0] == 1:

            label = 1

        else:

            label = 0


        label = torch.tensor(
            label,
            dtype=torch.long
        )


        return image, label


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    dataset = MARIDAResNetDataset(
        "splits/train_X.txt",
        max_samples=10
    )


    print()
    print(
        "Dataset length:",
        len(dataset)
    )


    # --------------------------------------
    # Check first few samples
    # --------------------------------------

    for i in range(
        min(3, len(dataset))
    ):

        image, label = dataset[i]

        print()
        print(
            "Sample:",
            i
        )

        print(
            "Image:",
            image.shape
        )

        print(
            "Label:",
            label.item()
        )