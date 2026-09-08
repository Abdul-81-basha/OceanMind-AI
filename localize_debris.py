import os

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ============================================================
# SETTINGS
# ============================================================

IMAGE_PATH = (
    "patches/"
    "S2_1-12-19_48MYU/"
    "S2_1-12-19_48MYU_2.tif"
)

MASK_PATH = (
    "patches/"
    "S2_1-12-19_48MYU/"
    "S2_1-12-19_48MYU_2_cl.tif"
)

OUTPUT_PATH = (
    "outputs/marine_debris_localization_box.png"
)

# MARIDA class 1 = Marine Debris
MARINE_DEBRIS_CLASS = 1


# ============================================================
# LOAD SATELLITE IMAGE
# ============================================================

print()
print("Loading satellite image...")

with rasterio.open(IMAGE_PATH) as src:
    image = src.read().astype(np.float32)

print(
    "Image shape:",
    image.shape
)


# ============================================================
# CREATE RGB IMAGE
# ============================================================

# Using the first three bands:
# Band 1 -> Red
# Band 2 -> Green
# Band 3 -> Blue

rgb = image[[0, 1, 2]]


# ============================================================
# NORMALIZE RGB
# ============================================================

for band in range(3):

    minimum = rgb[band].min()
    maximum = rgb[band].max()

    if maximum > minimum:

        rgb[band] = (
            rgb[band] - minimum
        ) / (
            maximum - minimum
        )


# Convert from:
# (3, height, width)
#
# to:
# (height, width, 3)

rgb = np.transpose(
    rgb,
    (1, 2, 0)
)


# ============================================================
# LOAD MARIDA MASK
# ============================================================

print()
print("Loading MARIDA mask...")

with rasterio.open(MASK_PATH) as src:
    mask = src.read(1)

print(
    "Mask shape:",
    mask.shape
)


# ============================================================
# FIND MARINE DEBRIS
# ============================================================

debris_mask = (
    mask == MARINE_DEBRIS_CLASS
)


debris_pixels = int(
    np.sum(debris_mask)
)


total_pixels = int(
    debris_mask.size
)


debris_percentage = (
    debris_pixels /
    total_pixels
) * 100


print()
print("========================================")
print("      MARINE DEBRIS LOCALIZATION")
print("========================================")

print()
print(
    "Marine debris pixels:",
    debris_pixels
)

print(
    "Total pixels:",
    total_pixels
)

print(
    f"Marine debris area: "
    f"{debris_percentage:.2f}%"
)


# ============================================================
# FIND BOUNDING BOX
# ============================================================

rows, cols = np.where(
    debris_mask
)


if len(rows) == 0:

    print()
    print(
        "No Marine Debris pixels found."
    )

    # Still save the image

    plt.figure(
        figsize=(10, 8)
    )

    plt.imshow(rgb)

    plt.title(
        "OceanMind AI - No Marine Debris Found"
    )

    plt.axis("off")

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    plt.savefig(
        OUTPUT_PATH,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print(
        "Image saved:",
        OUTPUT_PATH
    )

    exit()


# Bounding-box coordinates

min_row = int(rows.min())
max_row = int(rows.max())

min_col = int(cols.min())
max_col = int(cols.max())


box_width = (
    max_col - min_col + 1
)

box_height = (
    max_row - min_row + 1
)


print()
print("Debris bounding box:")
print(
    f"Top:    {min_row}"
)

print(
    f"Bottom: {max_row}"
)

print(
    f"Left:   {min_col}"
)

print(
    f"Right:  {max_col}"
)

print()
print(
    "Bounding box width:",
    box_width
)

print(
    "Bounding box height:",
    box_height
)


# ============================================================
# CREATE VISUALIZATION
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 8)
)


# Display satellite image

ax.imshow(rgb)


# ============================================================
# DRAW DEBRIS PIXELS
# ============================================================

overlay = np.zeros(
    (
        mask.shape[0],
        mask.shape[1],
        4
    )
)


# Red transparent overlay

overlay[
    debris_mask
] = [
    1.0,
    0.0,
    0.0,
    0.8
]


ax.imshow(
    overlay
)


# ============================================================
# DRAW BOUNDING BOX
# ============================================================

rectangle = Rectangle(
    (
        min_col,
        min_row
    ),
    box_width,
    box_height,

    linewidth=3,

    edgecolor="red",

    facecolor="none"
)


ax.add_patch(
    rectangle
)


# ============================================================
# ADD LABEL
# ============================================================

ax.text(
    min_col,
    max(
        0,
        min_row - 8
    ),
    "MARINE DEBRIS",

    fontsize=12,

    fontweight="bold",

    bbox=dict(
        facecolor="white",
        alpha=0.8,
        edgecolor="red"
    )
)


# ============================================================
# TITLE
# ============================================================

ax.set_title(
    "OceanMind AI - Marine Debris Localization",
    fontsize=16,
    fontweight="bold"
)


ax.axis("off")


# ============================================================
# INFORMATION BOX
# ============================================================

info = (
    f"Debris pixels: {debris_pixels}\n"
    f"Area: {debris_percentage:.2f}%\n"
    f"Box: {box_width} × {box_height} px\n"
    f"Coordinates: "
    f"({min_col}, {min_row}) → "
    f"({max_col}, {max_row})"
)


ax.text(
    0.02,
    0.02,
    info,

    transform=ax.transAxes,

    fontsize=10,

    verticalalignment="bottom",

    bbox=dict(
        facecolor="white",
        alpha=0.85,
        edgecolor="red"
    )
)


# ============================================================
# SAVE RESULT
# ============================================================

os.makedirs(
    "outputs",
    exist_ok=True
)


plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# FINISHED
# ============================================================

print()
print("========================================")
print("LOCALIZATION COMPLETE")
print("========================================")

print()
print(
    "Saved to:"
)

print(
    OUTPUT_PATH
)
