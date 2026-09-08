import os
import glob

import numpy as np
import rasterio
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "outputs/marine_debris_segmentation.pth"

PATCHES_DIR = "patches"

IMAGE_SIZE = 128

THRESHOLD = 0.5

OUTPUT_MASK = "outputs/predicted_debris_mask.png"

OUTPUT_OVERLAY = "outputs/debris_overlay.png"


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print()
print("========================================")
print(" OCEANMIND AI MARINE DEBRIS DETECTION")
print("========================================")

print()
print("Device:", device)


# ============================================================
# FIND INPUT IMAGE
# ============================================================

all_images = glob.glob(
    os.path.join(
        PATCHES_DIR,
        "**",
        "*.tif"
    ),
    recursive=True
)


input_images = []

for path in all_images:

    filename = os.path.basename(path).lower()

    if filename.endswith("_cl.tif"):
        continue

    if filename.endswith("_conf.tif"):
        continue

    input_images.append(path)


if len(input_images) == 0:

    print()
    print("ERROR: No satellite images found.")

    raise SystemExit


IMAGE_PATH = input_images[0]


print()
print("Input image:")
print(IMAGE_PATH)


# ============================================================
# U-NET COMPONENT
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


# ============================================================
# U-NET
# ============================================================

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
# CREATE MODEL
# ============================================================

print()
print("Loading trained U-Net...")

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
# LOAD SATELLITE IMAGE
# ============================================================

print()
print("Reading satellite image...")

with rasterio.open(
    IMAGE_PATH
) as src:

    image = src.read().astype(
        np.float32
    )


print(
    "Number of bands:",
    image.shape[0]
)

print(
    "Original size:",
    image.shape[1],
    "x",
    image.shape[2]
)


# ============================================================
# CHECK BANDS
# ============================================================

if image.shape[0] != 11:

    print()
    print(
        "ERROR: Expected 11 Sentinel-2 bands."
    )

    print(
        "Found:",
        image.shape[0]
    )

    raise SystemExit


# ============================================================
# FIX NaN / INFINITY
# ============================================================

image = np.nan_to_num(
    image,
    nan=0.0,
    posinf=1.0,
    neginf=0.0
)


# ============================================================
# NORMALIZE BANDS
# ============================================================

print()
print("Normalizing image...")

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


# ============================================================
# CONVERT TO TENSOR
# ============================================================

image_tensor = torch.tensor(
    image,
    dtype=torch.float32
)


# ============================================================
# RESIZE IMAGE
# ============================================================

image_tensor = F.interpolate(
    image_tensor.unsqueeze(0),
    size=(
        IMAGE_SIZE,
        IMAGE_SIZE
    ),
    mode="bilinear",
    align_corners=False
)


image_tensor = image_tensor.to(
    device
)


# ============================================================
# AI PREDICTION
# ============================================================

print()
print("Running OceanMind AI...")

with torch.no_grad():

    output = model(
        image_tensor
    )

    probability = torch.sigmoid(
        output
    )


# ============================================================
# CREATE MASK
# ============================================================

mask = (
    probability >= THRESHOLD
).float()


mask = mask.squeeze().cpu().numpy()


# ============================================================
# CALCULATE DETECTION
# ============================================================

total_pixels = mask.size

debris_pixels = np.sum(
    mask > 0
)

debris_percentage = (
    debris_pixels /
    total_pixels
) * 100


# ============================================================
# SAVE MASK
# ============================================================

os.makedirs(
    "outputs",
    exist_ok=True
)


mask_image = (
    mask * 255
).astype(
    np.uint8
)


Image.fromarray(
    mask_image
).save(
    OUTPUT_MASK
)


# ============================================================
# CREATE TRUE-COLOR RGB IMAGE
# ============================================================

# Sentinel-2 band order in the 11-band input:
#
# B1 = index 0
# B2 = index 1  -> Blue
# B3 = index 2  -> Green
# B4 = index 3  -> Red
#
# True color = B4, B3, B2

rgb = image[
    [3, 2, 1]
]


rgb = np.transpose(
    rgb,
    (1, 2, 0)
)


rgb = np.clip(
    rgb,
    0,
    1
)


rgb = (
    rgb * 255
).astype(
    np.uint8
)


rgb_image = Image.fromarray(
    rgb
).resize(
    (
        IMAGE_SIZE,
        IMAGE_SIZE
    )
)


# ============================================================
# CREATE OVERLAY
# ============================================================

rgb_array = np.array(
    rgb_image
)


overlay = rgb_array.copy()


# Highlight detected marine debris

detected = mask > 0


overlay[detected] = [
    255,
    0,
    0
]


# Blend original image and detection

overlay = (
    0.6 * rgb_array +
    0.4 * overlay
).astype(
    np.uint8
)


Image.fromarray(
    overlay
).save(
    OUTPUT_OVERLAY
)


# ============================================================
# RESULTS
# ============================================================

print()
print(
    "========================================"
)

print(
    " OCEANMIND AI RESULT"
)

print(
    "========================================"
)

print()

print(
    "Marine debris pixels:",
    int(debris_pixels)
)

print(
    "Total pixels:",
    int(total_pixels)
)

print(
    f"Detected debris area: "
    f"{debris_percentage:.2f}%"
)

print()


if debris_percentage > 1:

    print(
        "RESULT: Marine debris detected."
    )

else:

    print(
        "RESULT: Low/no marine debris detected."
    )


print()

print(
    "Segmentation mask saved:"
)

print(
    OUTPUT_MASK
)

print()

print(
    "Overlay image saved:"
)

print(
    OUTPUT_OVERLAY
)

print()

print(
    "========================================"
)

print(
    " PREDICTION COMPLETE"
)

print(
    "========================================"
)