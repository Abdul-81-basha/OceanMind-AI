
from pathlib import Path

root = Path("data/raw")
patches = root / "patches"
splits = root / "splits"

for split in ["train_X.txt", "val_X.txt", "test_X.txt"]:
    ids = (splits / split).read_text().splitlines()
    found_images = 0
    found_masks = 0
    missing = []

    for item in ids:
        item = item.strip()
        if not item:
            continue

        # Split IDs omit the S2_ prefix.
        image_name = f"S2_{item}.tif"
        matches = list(patches.rglob(image_name))

        if matches:
            found_images += 1
            mask = matches[0].with_name(
                matches[0].stem + "_cl.tif"
            )
            if mask.exists():
                found_masks += 1
            else:
                missing.append(str(mask))
        else:
            missing.append(image_name)

    print(f"\n{split}")
    print("Listed entries:", len(ids))
    print("Images found:", found_images)
    print("Masks found:", found_masks)
    print("Missing files:", len(missing))
    if missing:
        print("Examples:", missing[:3])

