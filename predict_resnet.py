import os

import torch
import torch.nn as nn

import rasterio
import numpy as np

from torchvision.models import resnet18

import torch.nn.functional as F


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = (
    "outputs/marine_debris_resnet18_500.pth"
)

# Best threshold found from our testing
THRESHOLD = 0.55


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CREATE MODEL
# ============================================================

model = resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()


# ============================================================
# FUNCTION: LOAD SATELLITE IMAGE
# ============================================================

def load_satellite_image(
    image_path
):

    print()
    print(
        "Loading:",
        image_path
    )


    # ----------------------------------------
    # Open TIFF
    # ----------------------------------------

    with rasterio.open(
        image_path
    ) as src:

        image = src.read().astype(
            np.float32
        )


    print(
        "Original shape:",
        image.shape
    )


    # ----------------------------------------
    # Check bands
    # ----------------------------------------

    if image.shape[0] < 3:

        raise ValueError(
            "Image must contain at least 3 bands."
        )


    # ----------------------------------------
    # Normalize bands
    # ----------------------------------------

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


    # ----------------------------------------
    # Sentinel-2 → RGB
    #
    # Band indexes:
    #
    # 2 = Blue
    # 1 = Green
    # 0 = Red
    #
    # ----------------------------------------

    image = image[
        [2, 1, 0]
    ]


    # ----------------------------------------
    # NumPy → Tensor
    # ----------------------------------------

    image = torch.tensor(
        image,
        dtype=torch.float32
    )


    # ----------------------------------------
    # Add batch dimension
    # ----------------------------------------

    image = image.unsqueeze(0)


    # ----------------------------------------
    # Resize
    # ----------------------------------------

    image = F.interpolate(
        image,
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    )


    # ----------------------------------------
    # Move to device
    # ----------------------------------------

    image = image.to(device)


    print(
        "Processed shape:",
        image.shape
    )


    return image


# ============================================================
# FUNCTION: PREDICT
# ============================================================

def predict(
    image_path
):

    image = load_satellite_image(
        image_path
    )


    # ----------------------------------------
    # Prediction
    # ----------------------------------------

    with torch.no_grad():

        output = model(image)


        probabilities = torch.softmax(
            output,
            dim=1
        )


        debris_probability = (
            probabilities[0][1].item()
        )


        non_debris_probability = (
            probabilities[0][0].item()
        )


    # ----------------------------------------
    # Decision
    # ----------------------------------------

    if debris_probability >= THRESHOLD:

        result = (
            "MARINE DEBRIS DETECTED"
        )

    else:

        result = (
            "NO MARINE DEBRIS DETECTED"
        )


    # ----------------------------------------
    # Display
    # ----------------------------------------

    print()
    print(
        "========================================"
    )

    print(
        "          OCEANMIND AI"
    )

    print(
        "     MARINE DEBRIS DETECTOR"
    )

    print(
        "========================================"
    )

    print()

    print(
        f"Non-Debris Probability: "
        f"{non_debris_probability:.2%}"
    )

    print(
        f"Marine Debris Probability: "
        f"{debris_probability:.2%}"
    )

    print()

    print(
        f"Decision Threshold: "
        f"{THRESHOLD:.0%}"
    )

    print()

    print(
        "RESULT:"
    )

    print(
        result
    )

    print()

    print(
        f"Confidence: "
        f"{max(
            debris_probability,
            non_debris_probability
        ):.2%}"
    )

    print(
        "========================================"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Change this to any MARIDA TIFF you want to test
    # --------------------------------------------------------

    image_path = (
        "patches/"
        "S2_1-12-19_48MYU/"
        "S2_1-12-19_48MYU_2.tif"
    )


    if not os.path.exists(
        image_path
    ):

        print()
        print(
            "ERROR: Image not found."
        )

        print()
        print(
            "Expected:"
        )

        print(
            image_path
        )

    else:

        predict(
            image_path
        )
        