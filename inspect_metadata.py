import rasterio

image_path = "dataset/raw/S2_1-12-19_48MYU_0.tif"

with rasterio.open(image_path) as src:

    print("Number of bands:", src.count)
    print("Image width:", src.width)
    print("Image height:", src.height)
    print("Data type:", src.dtypes)
    print()

    for i in range(1, src.count + 1):
        print(
            f"Band {i}: "
            f"description={src.descriptions[i-1]}"
        )