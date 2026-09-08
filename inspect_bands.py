import rasterio
import numpy as np

image_path = "dataset/raw/S2_1-12-19_48MYU_0.tif"

with rasterio.open(image_path) as src:
    image = src.read().astype(np.float32)

print("Image shape:", image.shape)
print()

for band in range(image.shape[0]):
    band_data = image[band]

    print(
        f"Band {band + 1}: "
        f"Min={band_data.min():.2f}, "
        f"Max={band_data.max():.2f}, "
        f"Mean={band_data.mean():.2f}"
    )
import numpy as np

image_path = "dataset/raw/S2_1-12-19_48MYU_0.tif"

with rasterio.open(image_path) as src:
    image = src.read().astype(np.float32)

print("Image shape:", image.shape)
print()

for band in range(image.shape[0]):
    band_data = image[band]

    print(
        f"Band {band + 1}: "
        f"Min={band_data.min():.2f}, "
        f"Max={band_data.max():.2f}, "
        f"Mean={band_data.mean():.2f}"
    )