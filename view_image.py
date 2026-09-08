import rasterio
import numpy as np
import matplotlib.pyplot as plt

image_path = "dataset/raw/S2_1-12-19_48MYU_0.tif"

with rasterio.open(image_path) as src:
    image = src.read().astype(np.float32)

print("Image shape:", image.shape)
print("Number of bands:", image.shape[0])

# MARIDA contains 11 Sentinel-2 bands.
# For true color we use:
# B04 = Red
# B03 = Green
# B02 = Blue

red = image[2]
green = image[1]
blue = image[0]

# Stack into RGB
rgb = np.stack([red, green, blue], axis=-1)

# Contrast stretch each channel
for i in range(3):
    low = np.percentile(rgb[:, :, i], 2)
    high = np.percentile(rgb[:, :, i], 98)

    rgb[:, :, i] = np.clip(
        (rgb[:, :, i] - low) / (high - low),
        0,
        1
    )

plt.figure(figsize=(8, 8))
plt.imshow(rgb)
plt.title("MARIDA Sentinel-2 True Color")
plt.axis("off")
plt.show()