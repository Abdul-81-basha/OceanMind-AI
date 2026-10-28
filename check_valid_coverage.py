
from pathlib import Path
import numpy as np
import rasterio

root = Path("data/raw/patches")
total_pixels = 0
valid_pixels = 0
samples = 0

for mask_path in root.rglob("*_cl.tif"):
    conf_path = mask_path.with_name(
        mask_path.stem.replace("_cl", "_conf") + ".tif"
    )

    if not conf_path.exists():
        continue

    with rasterio.open(mask_path) as src:
        mask = src.read(1)

    with rasterio.open(conf_path) as src:
        conf = src.read(1)

    total_pixels += mask.size
    valid_pixels += int(np.sum(conf == 1))
    samples += 1

print("Samples checked:", samples)
print("Total pixels:", total_pixels)
print("Pixels with confidence 1:", valid_pixels)
print("Valid pixel percentage:", round(100 * valid_pixels / total_pixels, 4))
