
from pathlib import Path

import numpy as np
import rasterio

root = Path("data/raw/patches")
images = list(root.rglob("S2_9-10-17_16PEC_20.tif"))

if not images:
    print("Sample image not found.")
    raise SystemExit

image_path = images[0]
mask_path = image_path.with_name(image_path.stem + "_cl.tif")
conf_path = image_path.with_name(image_path.stem + "_conf.tif")

with rasterio.open(image_path) as src:
    image = src.read().astype(np.float32)

with rasterio.open(mask_path) as src:
    mask = src.read(1)

with rasterio.open(conf_path) as src:
    confidence = src.read(1)

print("Image shape:", image.shape)
print("Image finite:", np.isfinite(image).all())
print("Image min/max:", np.nanmin(image), np.nanmax(image))

for band in range(image.shape[0]):
    values = image[band]
    values = values[np.isfinite(values)]
    print(
        f"Band {band + 1}: min={values.min():.6f}, "
        f"max={values.max():.6f}, mean={values.mean():.6f}"
    )

print("Mask classes:", np.unique(mask, return_counts=True))
print("Confidence values:", np.unique(confidence, return_counts=True))
print(
    "Nonzero class pixels:",
    int(((mask >= 1) & (mask <= 15)).sum())
)
print(
    "Labeled pixels with confidence > 0:",
    int(((mask >= 1) & (mask <= 15) & (confidence > 0)).sum())
)

