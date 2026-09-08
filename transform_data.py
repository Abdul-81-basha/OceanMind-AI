import torch
import torch.nn.functional as F

from dataset_loader import MARIDADataset


# Our two satellite images
image_paths = [
    "dataset/raw/S2_1-12-19_48MYU_0.tif",
    "dataset/raw/S2_1-12-19_48MYU_2.tif"
]


# Load dataset
dataset = MARIDADataset(image_paths)

# Get first image
image = dataset[0]

print("Original shape:", image.shape)


# --------------------------------
# Select RGB bands
# --------------------------------

rgb = image[[2, 1, 0]]

print("RGB shape:", rgb.shape)


# --------------------------------
# Add batch dimension
# --------------------------------

rgb = rgb.unsqueeze(0)

print("Before resize:", rgb.shape)


# --------------------------------
# Resize to 224 × 224
# --------------------------------

rgb = F.interpolate(
    rgb,
    size=(224, 224),
    mode="bilinear",
    align_corners=False
)


print("After resize:", rgb.shape)


# Remove batch dimension
rgb = rgb.squeeze(0)

print("Final shape:", rgb.shape)
print("Minimum:", rgb.min().item())
print("Maximum:", rgb.max().item())
