import rasterio
import numpy as np
import matplotlib.pyplot as plt

image_path = "dataset/raw/S2_1-12-19_48MYU_2.tif"
mask_path = "dataset/raw/S2_1-12-19_48MYU_2_cl.tif"

# Read satellite image
with rasterio.open(image_path) as src:
    image = src.read().astype(np.float32)

# Read mask
with rasterio.open(mask_path) as src:
    mask = src.read(1)

# Create RGB image
red = image[2]
green = image[1]
blue = image[0]

rgb = np.stack([red, green, blue], axis=-1)

# Contrast stretch for visualization
for i in range(3):
    low = np.percentile(rgb[:, :, i], 2)
    high = np.percentile(rgb[:, :, i], 98)

    rgb[:, :, i] = np.clip(
        (rgb[:, :, i] - low) / (high - low),
        0,
        1
    )

# Marine Debris = class 1
debris = mask == 1

plt.figure(figsize=(8, 8))
plt.imshow(rgb)

# Highlight debris locations
y, x = np.where(debris)

plt.scatter(x, y, s=40, facecolors="none", edgecolors="red")

plt.title("Marine Debris Locations")
plt.axis("off")
plt.show()

print("Marine Debris pixels:", debris.sum())