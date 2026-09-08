import rasterio
import numpy as np

mask_path = "dataset/raw/S2_1-12-19_48MYU_2_cl.tif"

with rasterio.open(mask_path) as src:
    mask = src.read(1)

print("Mask shape:", mask.shape)
print("Data type:", mask.dtype)

values, counts = np.unique(mask, return_counts=True)

print("\nClasses found in this patch:")

for value, count in zip(values, counts):
    print(f"Class {value}: {count} pixels")