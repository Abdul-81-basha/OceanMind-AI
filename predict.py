import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio

from torchvision.models import vit_b_16


# -----------------------------
# Load satellite image
# -----------------------------

def load_image(path):

    with rasterio.open(path) as src:
        image = src.read().astype("float32")

    # Normalize each band
    for band in range(image.shape[0]):

        minimum = image[band].min()
        maximum = image[band].max()

        if maximum > minimum:
            image[band] = (
                (image[band] - minimum)
                / (maximum - minimum)
            )

    image = torch.tensor(image)

    # RGB
    image = image[[2, 1, 0]]

    # Add batch dimension
    image = image.unsqueeze(0)

    # Resize
    image = F.interpolate(
        image,
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    )

    return image


# -----------------------------
# Create model
# -----------------------------

model = vit_b_16(weights=None)

model.heads.head = nn.Linear(
    model.heads.head.in_features,
    2
)


# -----------------------------
# Load saved model
# -----------------------------

model.load_state_dict(
    torch.load(
        "outputs/marine_debris_vit.pth",
        map_location="cpu"
    )
)

model.eval()


# -----------------------------
# Select image to test
# -----------------------------

image_path = (
    "dataset/raw/S2_1-12-19_48MYU_2.tif"
)

image = load_image(image_path)


# -----------------------------
# Prediction
# -----------------------------

with torch.no_grad():

    output = model(image)

    probabilities = torch.softmax(
        output,
        dim=1
    )

    prediction = probabilities.argmax(
        dim=1
    ).item()


# -----------------------------
# Display result
# -----------------------------

print()
print("Prediction")
print("-----------------------")

print(
    "Non-Marine-Debris:",
    f"{probabilities[0][0].item() * 100:.2f}%"
)

print(
    "Marine-Debris:",
    f"{probabilities[0][1].item() * 100:.2f}%"
)

print()

if prediction == 1:
    print("Result: MARINE DEBRIS")
else:
    print("Result: NON-MARINE-DEBRIS")

    