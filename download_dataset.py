import json
import urllib.request
from pathlib import Path

BASE_URL = "https://data.source.coop/ntua/marida/"

LABEL_FILE = Path("dataset/labels_mapping.txt")
TRAIN_FILE = Path("dataset/train_X.txt")

DEBRIS_DIR = Path("dataset/raw/debris")
NON_DEBRIS_DIR = Path("dataset/raw/non_debris")

NUMBER_EACH = 20

DEBRIS_DIR.mkdir(parents=True, exist_ok=True)
NON_DEBRIS_DIR.mkdir(parents=True, exist_ok=True)


# Read training split
with open(TRAIN_FILE, "r") as f:
    train_names = [line.strip() for line in f if line.strip()]


# Read labels
with open(LABEL_FILE, "r") as f:
    labels = json.load(f)


debris = []
non_debris = []

for name in train_names:

    filename = "S2_" + name + ".tif"

    if filename not in labels:
        continue

    classes = labels[filename]

    if classes[0] == 1:
        debris.append(filename)
    else:
        non_debris.append(filename)

    if len(debris) >= NUMBER_EACH and len(non_debris) >= NUMBER_EACH:
        break


print("Marine Debris samples:", len(debris))
print("Non-Debris samples:", len(non_debris))


def download_file(url, destination):

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    with urllib.request.urlopen(request) as response:
        data = response.read()

    with open(destination, "wb") as f:
        f.write(data)


def download_patch(filename, destination):

    scene = filename.rsplit("_", 1)[0]

    image_url = (
        BASE_URL
        + "patches/"
        + scene
        + "/"
        + filename
    )

    mask_filename = filename.replace(
        ".tif",
        "_cl.tif"
    )

    mask_url = (
        BASE_URL
        + "patches/"
        + scene
        + "/"
        + mask_filename
    )

    image_path = destination / filename
    mask_path = destination / mask_filename

    print("\nImage:", filename)

    try:

        if not image_path.exists():
            download_file(
                image_url,
                image_path
            )

            print("Image downloaded.")

        else:
            print("Image already exists.")

        if not mask_path.exists():
            download_file(
                mask_url,
                mask_path
            )

            print("Mask downloaded.")

        else:
            print("Mask already exists.")

    except Exception as e:

        print("FAILED:", e)


print("\n--- MARINE DEBRIS ---")

for filename in debris:
    download_patch(
        filename,
        DEBRIS_DIR
    )


print("\n--- NON-MARINE-DEBRIS ---")

for filename in non_debris:
    download_patch(
        filename,
        NON_DEBRIS_DIR
    )


print("\nFinished.")