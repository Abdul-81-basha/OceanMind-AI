
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path("data/raw")
PATCHES = ROOT / "patches"
SPLIT = ROOT / "splits" / "train_X.txt"

counts = Counter()
total_labeled = 0

ids = [
    line.strip()
    for line in SPLIT.read_text().splitlines()
    if line.strip()
]

for item in ids:
    image_name = f"S2_{item}_cl.tif"
    matches = list(PATCHES.rglob(image_name))

    if not matches:
        print("Missing mask:", image_name)
        continue

    with rasterio.open(matches[0]) as src:
        mask = src.read(1)

    # Count only named classes with a positive confidence value.
    conf_path = matches[0].with_name(
        matches[0].stem.replace("_cl", "_conf") + ".tif"
    )

    with rasterio.open(conf_path) as src:
        confidence = src.read(1)

    valid = (confidence > 0) & (mask >= 1) & (mask <= 15)
    values, frequencies = np.unique(mask[valid], return_counts=True)

    for value, frequency in zip(values, frequencies):
        counts[int(value)] += int(frequency)
        total_labeled += int(frequency)

print("Training patches:", len(ids))
print("Labeled pixels:", total_labeled)
print("\nClass ID | Class name | Pixel count")

names = [
    "Marine Debris",
    "Dense Sargassum",
    "Sparse Sargassum",
    "Natural Organic Material",
    "Ship",
    "Clouds",
    "Marine Water",
    "Sediment-Laden Water",
    "Foam",
    "Turbid Water",
    "Shallow Water",
    "Waves",
    "Cloud Shadows",
    "Wakes",
    "Mixed Water",
]

for class_id, name in enumerate(names, start=1):
    print(f"{class_id:8} | {name:24} | {counts[class_id]}")

print("\nTotal class pixels:", sum(counts.values()))

