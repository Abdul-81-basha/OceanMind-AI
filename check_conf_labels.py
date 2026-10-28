
import numpy as np
import rasterio
from pathlib import Path

root = Path("data/raw/patches")
image = next(root.rglob("S2_9-10-17_16PEC_20.tif"))

mask_path = image.with_name(image.stem + "_cl.tif")
conf_path = image.with_name(image.stem + "_conf.tif")

with rasterio.open(mask_path) as src:
    mask = src.read(1).astype(int)

with rasterio.open(conf_path) as src:
    conf = src.read(1)

print("Class ID | Total pixels | Conf=0 | Conf=1")

for class_id in np.unique(mask):
    pixels = mask == class_id
    print(
        f"{class_id:8} | {pixels.sum():12} | "
        f"{np.sum(pixels & (conf == 0)):6} | "
        f"{np.sum(pixels & (conf == 1)):6}"
    )

