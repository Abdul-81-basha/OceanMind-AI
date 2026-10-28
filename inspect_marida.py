
from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt

root = Path("data/raw/patches/S2_9-10-17_16PEC")

image_path = root / "S2_9-10-17_16PEC_4.tif"
mask_path = root / "S2_9-10-17_16PEC_4_cl.tif"

with rasterio.open(image_path) as src:
    image = src.read(1)
    print("Image shape:", image.shape)

with rasterio.open(mask_path) as src:
    mask = src.read(1)
    print("Mask shape:", mask.shape)
    print("Class IDs:", np.unique(mask, return_counts=True))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].imshow(image, cmap="gray")
ax[0].set_title("Satellite image band 4")
ax[0].axis("off")

ax[1].imshow(mask, cmap="tab20")
ax[1].set_title("Classification mask")
ax[1].axis("off")

plt.tight_layout()
plt.show()
