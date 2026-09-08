import os
import glob

import numpy as np
import rasterio
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

PATCHES_DIR = "patches"

MARINE_DEBRIS_CLASS = 1

OUTPUT_DIR = "outputs/data_exploration"


# ============================================================
# START
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print()
print("========================================")
print(" OCEANMIND AI DATA EXPLORATION")
print("========================================")


# ============================================================
# FIND TIFF FILES
# ============================================================

print()
print("Searching MARIDA patches...")

all_tifs = glob.glob(
    os.path.join(
        PATCHES_DIR,
        "**",
        "*.tif"
    ),
    recursive=True
)

images = []
masks = []

for path in all_tifs:

    filename = os.path.basename(
        path
    ).lower()

    if filename.endswith("_cl.tif"):

        masks.append(path)

    elif filename.endswith("_conf.tif"):

        continue

    else:

        images.append(path)


print()
print("Satellite images:", len(images))
print("Label masks:", len(masks))


if len(images) == 0:

    print()
    print("ERROR: No satellite images found.")

    raise SystemExit


# ============================================================
# DATASET INFORMATION
# ============================================================

print()
print("========================================")
print(" DATASET INFORMATION")
print("========================================")


sample_path = images[0]

with rasterio.open(
    sample_path
) as src:

    sample = src.read()

    band_count = src.count

    height = src.height

    width = src.width

    dtype = src.dtypes[0]


print()
print(
    "Sample image:",
    os.path.basename(sample_path)
)

print(
    "Number of bands:",
    band_count
)

print(
    "Image size:",
    height,
    "x",
    width
)

print(
    "Data type:",
    dtype
)


# ============================================================
# BAND STATISTICS
# ============================================================

print()
print("========================================")
print(" BAND STATISTICS")
print("========================================")


band_min = np.full(
    band_count,
    np.inf
)

band_max = np.full(
    band_count,
    -np.inf
)

band_mean_sum = np.zeros(
    band_count,
    dtype=np.float64
)


for number, image_path in enumerate(
    images,
    start=1
):

    with rasterio.open(
        image_path
    ) as src:

        data = src.read().astype(
            np.float32
        )


    data = np.nan_to_num(
        data,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


    for band in range(
        band_count
    ):

        band_data = data[band]

        band_min[band] = min(
            band_min[band],
            np.min(band_data)
        )

        band_max[band] = max(
            band_max[band],
            np.max(band_data)
        )

        band_mean_sum[band] += np.mean(
            band_data
        )


    if (
        number % 100 == 0
        or
        number == len(images)
    ):

        print(
            f"Processed images: "
            f"{number}/{len(images)}"
        )


print()

print(
    f"{'Band':<10}"
    f"{'Minimum':<15}"
    f"{'Maximum':<15}"
    f"{'Mean':<15}"
)


for band in range(
    band_count
):

    mean_value = (
        band_mean_sum[band]
        /
        len(images)
    )


    print(
        f"{band + 1:<10}"
        f"{band_min[band]:<15.4f}"
        f"{band_max[band]:<15.4f}"
        f"{mean_value:<15.4f}"
    )


# ============================================================
# MARINE DEBRIS DISTRIBUTION
# ============================================================

print()
print("========================================")
print(" MARINE DEBRIS DISTRIBUTION")
print("========================================")


total_pixels = 0

debris_pixels = 0

non_debris_pixels = 0

images_with_debris = 0

images_without_debris = 0

valid_masks = 0


for number, image_path in enumerate(
    images,
    start=1
):

    filename = os.path.basename(
        image_path
    )

    mask_path = os.path.join(
        os.path.dirname(image_path),
        filename[:-4] + "_cl.tif"
    )


    if not os.path.exists(
        mask_path
    ):

        continue


    valid_masks += 1


    with rasterio.open(
        mask_path
    ) as src:

        mask = src.read(1)


    total = mask.size

    debris = np.sum(
        mask == MARINE_DEBRIS_CLASS
    )

    non_debris = (
        total -
        debris
    )


    total_pixels += total

    debris_pixels += debris

    non_debris_pixels += non_debris


    if debris > 0:

        images_with_debris += 1

    else:

        images_without_debris += 1


    if (
        number % 100 == 0
        or
        number == len(images)
    ):

        print(
            f"Processed masks: "
            f"{number}/{len(images)}"
        )


debris_percentage = (
    debris_pixels /
    max(
        1,
        total_pixels
    )
) * 100


non_debris_percentage = (
    non_debris_pixels /
    max(
        1,
        total_pixels
    )
) * 100


print()
print(
    "Valid masks:",
    valid_masks
)

print(
    "Total pixels:",
    total_pixels
)

print(
    "Marine debris pixels:",
    debris_pixels
)

print(
    "Non-debris pixels:",
    non_debris_pixels
)

print()

print(
    f"Marine debris percentage: "
    f"{debris_percentage:.4f}%"
)

print(
    f"Non-debris percentage: "
    f"{non_debris_percentage:.4f}%"
)

print()

print(
    "Images containing debris:",
    images_with_debris
)

print(
    "Images without debris:",
    images_without_debris
)


# ============================================================
# CLASS DISTRIBUTION GRAPH
# ============================================================

print()
print("Creating class distribution graph...")


plt.figure(
    figsize=(8, 5)
)

plt.bar(
    [
        "Marine Debris",
        "Non-Debris"
    ],
    [
        debris_pixels,
        non_debris_pixels
    ]
)

plt.title(
    "MARIDA Pixel Class Distribution"
)

plt.ylabel(
    "Number of Pixels"
)

plt.tight_layout()


class_graph = os.path.join(
    OUTPUT_DIR,
    "class_distribution.png"
)


plt.savefig(
    class_graph,
    dpi=200
)

plt.close()


print(
    "Saved:",
    class_graph
)


# ============================================================
# CREATE TRUE COLOR IMAGE
# ============================================================

print()
print("Creating sample visualization...")


with rasterio.open(
    sample_path
) as src:

    image = src.read().astype(
        np.float32
    )


image = np.nan_to_num(
    image,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)


# Sentinel-2:
# B2 = Blue
# B3 = Green
# B4 = Red
#
# Input order:
# index 0 = B1
# index 1 = B2
# index 2 = B3
# index 3 = B4

if image.shape[0] >= 4:

    rgb = image[
        [3, 2, 1]
    ]

else:

    rgb = image[
        [0, 1, 2]
    ]


rgb = np.transpose(
    rgb,
    (1, 2, 0)
)


# ============================================================
# CONTRAST NORMALIZATION
# ============================================================

for channel in range(3):

    channel_data = rgb[:, :, channel]


    minimum = np.percentile(
        channel_data,
        2
    )

    maximum = np.percentile(
        channel_data,
        98
    )


    if maximum > minimum:

        rgb[:, :, channel] = (
            channel_data - minimum
        ) / (
            maximum - minimum
        )

    else:

        rgb[:, :, channel] = 0


rgb = np.clip(
    rgb,
    0,
    1
)


# ============================================================
# SAVE SAMPLE IMAGE
# ============================================================

plt.figure(
    figsize=(7, 7)
)

plt.imshow(
    rgb
)

plt.title(
    "MARIDA Sample Satellite Image"
)

plt.axis(
    "off"
)

plt.tight_layout()


sample_output = os.path.join(
    OUTPUT_DIR,
    "sample_satellite_image.png"
)


plt.savefig(
    sample_output,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved:",
    sample_output
)


# ============================================================
# LOAD GROUND TRUTH MASK
# ============================================================

sample_filename = os.path.basename(
    sample_path
)

sample_mask_path = os.path.join(
    os.path.dirname(sample_path),
    sample_filename[:-4] + "_cl.tif"
)


if os.path.exists(
    sample_mask_path
):

    with rasterio.open(
        sample_mask_path
    ) as src:

        sample_mask = src.read(1)


    debris_mask = (
        sample_mask ==
        MARINE_DEBRIS_CLASS
    )


else:

    debris_mask = np.zeros(
        (
            image.shape[1],
            image.shape[2]
        ),
        dtype=bool
    )


# ============================================================
# GROUND TRUTH MASK
# ============================================================

plt.figure(
    figsize=(7, 7)
)

plt.imshow(
    debris_mask
)

plt.title(
    "Marine Debris Ground Truth Mask"
)

plt.axis(
    "off"
)

plt.tight_layout()


mask_output = os.path.join(
    OUTPUT_DIR,
    "ground_truth_mask.png"
)


plt.savefig(
    mask_output,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved:",
    mask_output
)


# ============================================================
# GROUND TRUTH OVERLAY
# ============================================================

overlay = rgb.copy()


overlay[
    debris_mask
] = [
    1.0,
    0.0,
    0.0
]


plt.figure(
    figsize=(7, 7)
)

plt.imshow(
    overlay
)

plt.title(
    "Marine Debris Ground Truth Overlay"
)

plt.axis(
    "off"
)

plt.tight_layout()


overlay_output = os.path.join(
    OUTPUT_DIR,
    "ground_truth_overlay.png"
)


plt.savefig(
    overlay_output,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved:",
    overlay_output
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("========================================")
print(" OCEANMIND AI DATASET SUMMARY")
print("========================================")

print()

print(
    "Satellite patches:",
    len(images)
)

print(
    "Label masks:",
    len(masks)
)

print(
    "Spectral bands:",
    band_count
)

print(
    "Patch dimensions:",
    f"{height} x {width}"
)

print(
    f"Marine debris: "
    f"{debris_percentage:.4f}%"
)

print(
    f"Non-debris: "
    f"{non_debris_percentage:.4f}%"
)

print(
    "Images with debris:",
    images_with_debris
)

print(
    "Images without debris:",
    images_without_debris
)

print()

print(
    "EDA files saved to:"
)

print(
    OUTPUT_DIR
)

print()

print("========================================")
print(" DATA EXPLORATION COMPLETE")
print("========================================")

