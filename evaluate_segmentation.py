import os
import glob

import numpy as np
import rasterio

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "outputs/marine_debris_segmentation.pth"

PATCHES_DIR = "patches"

VAL_SPLIT = "splits/val_X.txt"

IMAGE_SIZE = 128

THRESHOLD = 0.5

MARINE_DEBRIS_CLASS = 1


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print()
print("========================================")
print(" OCEANMIND AI MODEL EVALUATION")
print("========================================")

print()
print("Device:", device)


# ============================================================
# NORMALIZE FILE NAME
# ============================================================

def normalize_name(name):

    name = name.strip()

    name = name.replace(
        "\\",
        "/"
    )

    name = os.path.basename(
        name
    )

    if name.lower().endswith(".tif"):
        name = name[:-4]

    if name.lower().endswith("_cl"):
        name = name[:-3]

    if name.lower().endswith("_conf"):
        name = name[:-5]

    if not name.lower().startswith("s2_"):
        name = "s2_" + name

    return name.lower()


# ============================================================
# INDEX IMAGES
# ============================================================

print()
print("Indexing validation images...")

image_index = {}

all_tifs = glob.glob(
    os.path.join(
        PATCHES_DIR,
        "**",
        "*.tif"
    ),
    recursive=True
)


for path in all_tifs:

    filename = os.path.basename(
        path
    )

    if filename.lower().endswith(
        "_cl.tif"
    ):
        continue

    if filename.lower().endswith(
        "_conf.tif"
    ):
        continue

    key = normalize_name(
        filename
    )

    image_index[key] = path


# ============================================================
# READ VALIDATION SPLIT
# ============================================================

val_images = []

missing = []


with open(
    VAL_SPLIT,
    "r",
    encoding="utf-8"
) as file:

    for line in file:

        name = line.strip()

        if not name:
            continue

        key = normalize_name(
            name
        )

        if key in image_index:

            val_images.append(
                image_index[key]
            )

        else:

            missing.append(name)


print()
print(
    "Validation images:",
    len(val_images)
)

print(
    "Missing:",
    len(missing)
)


if len(val_images) == 0:

    print()
    print(
        "ERROR: No validation images found."
    )

    raise SystemExit


# ============================================================
# MASK PATH
# ============================================================

def get_mask_path(image_path):

    directory = os.path.dirname(
        image_path
    )

    filename = os.path.basename(
        image_path
    )

    mask_filename = (
        filename[:-4] +
        "_cl.tif"
    )

    return os.path.join(
        directory,
        mask_filename
    )


# ============================================================
# U-NET
# ============================================================

class DoubleConv(nn.Module):

    def __init__(
        self,
        input_channels,
        output_channels
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                input_channels,
                output_channels,
                3,
                padding=1
            ),

            nn.BatchNorm2d(
                output_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                output_channels,
                output_channels,
                3,
                padding=1
            ),

            nn.BatchNorm2d(
                output_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )


    def forward(self, x):

        return self.block(x)


class UNet(nn.Module):

    def __init__(
        self,
        input_channels=11
    ):

        super().__init__()

        self.enc1 = DoubleConv(
            input_channels,
            32
        )

        self.enc2 = DoubleConv(
            32,
            64
        )

        self.enc3 = DoubleConv(
            64,
            128
        )

        self.pool = nn.MaxPool2d(
            2
        )

        self.bottleneck = DoubleConv(
            128,
            256
        )

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            2,
            stride=2
        )

        self.dec3 = DoubleConv(
            256,
            128
        )

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            2,
            stride=2
        )

        self.dec2 = DoubleConv(
            128,
            64
        )

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            2,
            stride=2
        )

        self.dec1 = DoubleConv(
            64,
            32
        )

        self.final = nn.Conv2d(
            32,
            1,
            1
        )


    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        b = self.bottleneck(
            self.pool(e3)
        )

        d3 = self.up3(b)

        d3 = torch.cat(
            [
                d3,
                e3
            ],
            dim=1
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        d2 = torch.cat(
            [
                d2,
                e2
            ],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [
                d1,
                e1
            ],
            dim=1
        )

        d1 = self.dec1(d1)

        return self.final(d1)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading trained model...")

model = UNet(
    input_channels=11
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("Model loaded successfully.")


# ============================================================
# METRIC VARIABLES
# ============================================================

true_positive = 0

true_negative = 0

false_positive = 0

false_negative = 0


# ============================================================
# EVALUATION
# ============================================================

print()
print("Evaluating validation dataset...")
print()


with torch.no_grad():

    for number, image_path in enumerate(
        val_images,
        start=1
    ):

        mask_path = get_mask_path(
            image_path
        )

        if not os.path.exists(
            mask_path
        ):

            print(
                "Skipping missing mask:",
                mask_path
            )

            continue


        # ====================================================
        # LOAD IMAGE
        # ====================================================

        with rasterio.open(
            image_path
        ) as src:

            image = src.read().astype(
                np.float32
            )


        # ====================================================
        # CHECK BANDS
        # ====================================================

        if image.shape[0] != 11:

            print(
                "Skipping image with",
                image.shape[0],
                "bands"
            )

            continue


        # ====================================================
        # CLEAN IMAGE
        # ====================================================

        image = np.nan_to_num(
            image,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )


        # ====================================================
        # NORMALIZE
        # ====================================================

        for band in range(
            image.shape[0]
        ):

            band_data = image[band]

            minimum = np.min(
                band_data
            )

            maximum = np.max(
                band_data
            )

            if (
                np.isfinite(minimum)
                and
                np.isfinite(maximum)
                and
                maximum > minimum
            ):

                image[band] = (
                    band_data - minimum
                ) / (
                    maximum - minimum
                )

            else:

                image[band] = 0.0


        # ====================================================
        # LOAD GROUND TRUTH MASK
        # ====================================================

        with rasterio.open(
            mask_path
        ) as src:

            mask = src.read(1)


        mask = (
            mask ==
            MARINE_DEBRIS_CLASS
        ).astype(
            np.float32
        )


        # ====================================================
        # CONVERT IMAGE TO TENSOR
        # ====================================================

        image_tensor = torch.tensor(
            image,
            dtype=torch.float32
        )


        # ====================================================
        # CONVERT MASK TO TENSOR
        # ====================================================

        mask_tensor = torch.tensor(
            mask,
            dtype=torch.float32
        )


        # ====================================================
        # RESIZE
        # ====================================================

        image_tensor = F.interpolate(
            image_tensor.unsqueeze(0),
            size=(
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            mode="bilinear",
            align_corners=False
        )


        mask_tensor = F.interpolate(
            mask_tensor.unsqueeze(0).unsqueeze(0),
            size=(
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            mode="nearest"
        )


        image_tensor = image_tensor.to(
            device
        )

        mask_tensor = mask_tensor.to(
            device
        )


        # ====================================================
        # PREDICTION
        # ====================================================

        output = model(
            image_tensor
        )

        probability = torch.sigmoid(
            output
        )


        prediction = (
            probability >= THRESHOLD
        ).float()


        # ====================================================
        # CONFUSION MATRIX
        # ====================================================

        prediction = prediction.reshape(
            -1
        )

        target = mask_tensor.reshape(
            -1
        )


        tp = (
            (prediction == 1)
            &
            (target == 1)
        ).sum().item()


        tn = (
            (prediction == 0)
            &
            (target == 0)
        ).sum().item()


        fp = (
            (prediction == 1)
            &
            (target == 0)
        ).sum().item()


        fn = (
            (prediction == 0)
            &
            (target == 1)
        ).sum().item()


        true_positive += tp

        true_negative += tn

        false_positive += fp

        false_negative += fn


        if (
            number % 25 == 0
            or
            number == len(val_images)
        ):

            print(
                f"Processed "
                f"{number}/"
                f"{len(val_images)}"
            )


# ============================================================
# CALCULATE METRICS
# ============================================================

epsilon = 1e-8


iou = (
    true_positive
    /
    (
        true_positive
        +
        false_positive
        +
        false_negative
        +
        epsilon
    )
)


dice = (
    2 * true_positive
    /
    (
        2 * true_positive
        +
        false_positive
        +
        false_negative
        +
        epsilon
    )
)


precision = (
    true_positive
    /
    (
        true_positive
        +
        false_positive
        +
        epsilon
    )
)


recall = (
    true_positive
    /
    (
        true_positive
        +
        false_negative
        +
        epsilon
    )
)


accuracy = (
    (
        true_positive
        +
        true_negative
    )
    /
    (
        true_positive
        +
        true_negative
        +
        false_positive
        +
        false_negative
        +
        epsilon
    )
)


# ============================================================
# RESULTS
# ============================================================

print()
print("========================================")
print(" OCEANMIND AI EVALUATION RESULTS")
print("========================================")

print()

print(
    "True Positives:",
    true_positive
)

print(
    "True Negatives:",
    true_negative
)

print(
    "False Positives:",
    false_positive
)

print(
    "False Negatives:",
    false_negative
)

print()

print(
    f"Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall   : {recall * 100:.2f}%"
)

print(
    f"Dice     : {dice * 100:.2f}%"
)

print(
    f"IoU      : {iou * 100:.2f}%"
)

print()

print("========================================")
print(" EVALUATION COMPLETE")
print("========================================")
