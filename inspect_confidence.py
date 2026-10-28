
import numpy as np
import rasterio
from pathlib import Path

root = Path("data/raw/patches")
files = list(root.rglob("S2_9-10-17_16PEC_20_conf.tif"))

if not files:
    print("Confidence map not found")
else:
    with rasterio.open(files[0]) as src:
        conf = src.read(1)

    print("Shape:", conf.shape)
    print("Data type:", conf.dtype)
    print("Unique values:", np.unique(conf, return_counts=True))

