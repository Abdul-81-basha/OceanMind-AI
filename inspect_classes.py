
import rasterio
import numpy as np
from pathlib import Path
from collections import Counter

root = Path("data/raw/patches")
counts = Counter()
mask_count = 0

for file in root.rglob("*_cl.tif"):
    with rasterio.open(file) as src:
        mask = src.read(1)
        values, frequencies = np.unique(mask, return_counts=True)

        for value, frequency in zip(values, frequencies):
            counts[int(value)] += int(frequency)

    mask_count += 1

print("Masks inspected:", mask_count)
print("Class ID | Pixel count")

for class_id, count in sorted(counts.items()):
    print(f"{class_id:8} | {count}")

print("All class IDs:", sorted(counts.keys()))

