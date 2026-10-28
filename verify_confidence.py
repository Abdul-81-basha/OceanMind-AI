
import numpy as np
import rasterio
from pathlib import Path
from collections import Counter

root = Path("data/raw/patches")
counts = Counter()

for mask_path in root.rglob("*_cl.tif"):
    conf_path = mask_path.with_name(
        mask_path.stem.replace("_cl", "_conf") + ".tif"
    )

    if not conf_path.exists():
        continue

    with rasterio.open(mask_path) as src:
        mask = src.read(1).astype(int)

    with rasterio.open(conf_path) as src:
        conf = src.read(1)

    counts["total_pixels"] += mask.size
    counts["class0_conf0"] += int(np.sum((mask == 0) & (conf == 0)))
    counts["class0_conf1"] += int(np.sum((mask == 0) & (conf == 1)))
    counts["nonzero_conf0"] += int(np.sum((mask != 0) & (conf == 0)))
    counts["nonzero_conf1"] += int(np.sum((mask != 0) & (conf == 1)))

for key, value in counts.items():
    print(f"{key}: {value}")
