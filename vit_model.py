import torch
import torch.nn as nn
from torchvision.models import vit_b_16


# Load Vision Transformer
model = vit_b_16(weights=None)


# Change the final classification layer
# We currently have 2 classes:
# 0 = Non-Marine-Debris
# 1 = Marine-Debris

model.heads.head = nn.Linear(
    model.heads.head.in_features,
    2
)


# Create a sample input
sample = torch.randn(
    1, 3, 224, 224
)


# Run the image through the model
output = model(sample)


print("Input shape:", sample.shape)
print("Output shape:", output.shape)
print("Output:", output)
