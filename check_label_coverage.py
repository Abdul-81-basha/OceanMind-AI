
from pathlib import Path
import numpy as np
import rasterio

root = Path("data/raw/patches")
split = Path("data/raw/splits/train_X.txt")

ids = [
    line.strip()
    for line in split.read_text().splitlines()
    if line.strip()
]

covered = 0
empty = 0
class_patches = {i: 0 for i in range(1, 16)}

for item in ids:
    matches = list(root.rglob(f"S2_{item}_cl.tif"))
    if not matches:
        continue

    with rasterio.open(matches[0]) as src:
        mask = src.read(1)

    conf_path = matches[0].with_name(
        matches[0].stem.replace("_cl", "_conf") + ".tif"
    )

    with rasterio.open(conf_path) as src:
        conf = src.read(1)

    valid = (conf > 0) & (mask >= 1) & (mask <= 15)

    if valid.any():
        covered += 1
    else:
        empty += 1

    for c in range(1, 16):
        if np.any(valid & (mask == c)):
            class_patches[c] += 1

print("Training patches:", len(ids))
print("Patches with labeled pixels:", covered)
print("Patches without labeled pixels:", empty)
print("\nClass ID | Patches containing class")
for c, count in class_patches.items():
    print(f"{c:8} | {count}")

