import os
import glob

import numpy as np
import rasterio

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


# ============================================================
# SETTINGS
# ============================================================

PATCHES_DIR = "patches"

TRAIN_SPLIT = "splits/train_X.txt"
VAL_SPLIT = "splits/val_X.txt"

MODEL_PATH = "outputs/marine_debris_segmentation.pth"

IMAGE_SIZE = 128

BATCH_SIZE = 4

EPOCHS = 10

LEARNING_RATE = 0.0001

MARINE_DEBRIS_CLASS = 1

# Maximum positive-class weight.
# Prevents extremely large weights from destabilizing training.
MAX_POS_WEIGHT = 50.0


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print()
print("========================================")
print(" OCEANMIND AI IMPROVED TRAINING")
print("========================================")

print()
print("Device:", device)


# ============================================================
# NORMALIZE FILE NAME
# ============================================================

def normalize_name(name):

    name = name.strip()

    name = name.replace(
        "\\",
        "/"
    )

    name = os.path.basename(
        name
    )

    if name.lower().endswith(".tif"):
        name = name[:-4]

    if name.lower().endswith("_cl"):
        name = name[:-3]

    if name.lower().endswith("_conf"):
        name = name[:-5]

    if not name.lower().startswith("s2_"):
        name = "s2_" + name

    return name.lower()


# ============================================================
# INDEX IMAGES
# ============================================================

print()
print("Indexing MARIDA patches...")

image_index = {}

all_tifs = glob.glob(
    os.path.join(
        PATCHES_DIR,
        "**",
        "*.tif"
    ),
    recursive=True
)


for path in all_tifs:

    filename = os.path.basename(
        path
    )

    if filename.lower().endswith(
        "_cl.tif"
    ):
        continue

    if filename.lower().endswith(
        "_conf.tif"
    ):
        continue

    key = normalize_name(
        filename
    )

    image_index[key] = path


print(
    "Images indexed:",
    len(image_index)
)


# ============================================================
# READ SPLIT
# ============================================================

def read_split(split_file):

    samples = []

    missing = []

    with open(
        split_file,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            original_name = line.strip()

            if not original_name:
                continue

            key = normalize_name(
                original_name
            )

            if key in image_index:

                samples.append(
                    image_index[key]
                )

            else:

                missing.append(
                    original_name
                )


    print()
    print(
        "Split:",
        split_file
    )

    print(
        "Matched:",
        len(samples)
    )

    print(
        "Missing:",
        len(missing)
    )

    return samples


# ============================================================
# LOAD DATA
# ============================================================

train_images = read_split(
    TRAIN_SPLIT
)

val_images = read_split(
    VAL_SPLIT
)


print()
print("========================================")

print(
    "Training images:",
    len(train_images)
)

print(
    "Validation images:",
    len(val_images)
)

print("========================================")


if len(train_images) == 0:

    print()
    print(
        "ERROR: No training images."
    )

    raise SystemExit


if len(val_images) == 0:

    print()
    print(
        "ERROR: No validation images."
    )

    raise SystemExit


# ============================================================
# MASK PATH
# ============================================================

def get_mask_path(image_path):

    directory = os.path.dirname(
        image_path
    )

    filename = os.path.basename(
        image_path
    )

    mask_filename = (
        filename[:-4] +
        "_cl.tif"
    )

    return os.path.join(
        directory,
        mask_filename
    )


# ============================================================
# DATASET
# ============================================================

class MARIDASegmentationDataset(
    Dataset
):

    def __init__(
        self,
        image_paths,
        augment=False
    ):

        self.image_paths = image_paths

        self.augment = augment


    def __len__(self):

        return len(
            self.image_paths
        )


    def __getitem__(
        self,
        index
    ):

        image_path = (
            self.image_paths[index]
        )

        mask_path = get_mask_path(
            image_path
        )


        if not os.path.exists(
            mask_path
        ):

            raise FileNotFoundError(
                f"Mask not found:\n{mask_path}"
            )


        # ====================================================
        # LOAD IMAGE
        # ====================================================

        with rasterio.open(
            image_path
        ) as src:

            image = src.read().astype(
                np.float32
            )


        # ====================================================
        # CHECK BANDS
        # ====================================================

        if image.shape[0] != 11:

            raise ValueError(
                f"Expected 11 bands but found "
                f"{image.shape[0]} in {image_path}"
            )


        # ====================================================
        # FIX NaN / INFINITY
        # ====================================================

        image = np.nan_to_num(
            image,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )


        # ====================================================
        # NORMALIZE EACH BAND
        # ====================================================

        for band in range(
            image.shape[0]
        ):

            band_data = image[band]

            minimum = np.min(
                band_data
            )

            maximum = np.max(
                band_data
            )


            if (
                np.isfinite(minimum)
                and
                np.isfinite(maximum)
                and
                maximum > minimum
            ):

                image[band] = (
                    band_data - minimum
                ) / (
                    maximum - minimum
                )

            else:

                image[band] = 0.0


        # ====================================================
        # LOAD MASK
        # ====================================================

        with rasterio.open(
            mask_path
        ) as src:

            mask = src.read(1)


        # ====================================================
        # MARINE DEBRIS = CLASS 1
        # ====================================================

        mask = (
            mask ==
            MARINE_DEBRIS_CLASS
        ).astype(
            np.float32
        )


        # ====================================================
        # TENSORS
        # ====================================================

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        mask = torch.tensor(
            mask,
            dtype=torch.float32
        )


        # ====================================================
        # RESIZE IMAGE
        # ====================================================

        image = F.interpolate(
            image.unsqueeze(0),
            size=(
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)


        # ====================================================
        # RESIZE MASK
        # ====================================================

        mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0),
            size=(
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            mode="nearest"
        ).squeeze(0)


        # ====================================================
        # DATA AUGMENTATION
        # ====================================================

        if self.augment:

            # Horizontal flip

            if torch.rand(1).item() < 0.5:

                image = torch.flip(
                    image,
                    dims=[2]
                )

                mask = torch.flip(
                    mask,
                    dims=[2]
                )


            # Vertical flip

            if torch.rand(1).item() < 0.5:

                image = torch.flip(
                    image,
                    dims=[1]
                )

                mask = torch.flip(
                    mask,
                    dims=[1]
                )


            # Random 90-degree rotation

            rotation = torch.randint(
                0,
                4,
                (1,)
            ).item()


            if rotation > 0:

                image = torch.rot90(
                    image,
                    rotation,
                    dims=[1, 2]
                )

                mask = torch.rot90(
                    mask,
                    rotation,
                    dims=[1, 2]
                )


        # ====================================================
        # FINAL SAFETY
        # ====================================================

        image = torch.nan_to_num(
            image,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )

        mask = torch.nan_to_num(
            mask,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )


        return image, mask


# ============================================================
# DATASETS
# ============================================================

train_dataset = (
    MARIDASegmentationDataset(
        train_images,
        augment=True
    )
)


val_dataset = (
    MARIDASegmentationDataset(
        val_images,
        augment=False
    )
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


print()
print(
    "Train batches:",
    len(train_loader)
)

print(
    "Validation batches:",
    len(val_loader)
)


# ============================================================
# U-NET
# ============================================================

class DoubleConv(
    nn.Module
):

    def __init__(
        self,
        input_channels,
        output_channels
    ):

        super().__init__()


        self.block = nn.Sequential(

            nn.Conv2d(
                input_channels,
                output_channels,
                3,
                padding=1
            ),

            nn.BatchNorm2d(
                output_channels
            ),

            nn.ReLU(
                inplace=True
            ),


            nn.Conv2d(
                output_channels,
                output_channels,
                3,
                padding=1
            ),

            nn.BatchNorm2d(
                output_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )


    def forward(self, x):

        return self.block(x)


# ============================================================
# U-NET
# ============================================================

class UNet(
    nn.Module
):

    def __init__(
        self,
        input_channels=11
    ):

        super().__init__()


        self.enc1 = DoubleConv(
            input_channels,
            32
        )

        self.enc2 = DoubleConv(
            32,
            64
        )

        self.enc3 = DoubleConv(
            64,
            128
        )


        self.pool = nn.MaxPool2d(
            2
        )


        self.bottleneck = DoubleConv(
            128,
            256
        )


        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            2,
            stride=2
        )

        self.dec3 = DoubleConv(
            256,
            128
        )


        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            2,
            stride=2
        )

        self.dec2 = DoubleConv(
            128,
            64
        )


        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            2,
            stride=2
        )

        self.dec1 = DoubleConv(
            64,
            32
        )


        self.final = nn.Conv2d(
            32,
            1,
            1
        )


    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )


        b = self.bottleneck(
            self.pool(e3)
        )


        d3 = self.up3(b)

        d3 = torch.cat(
            [
                d3,
                e3
            ],
            dim=1
        )

        d3 = self.dec3(d3)


        d2 = self.up2(d3)

        d2 = torch.cat(
            [
                d2,
                e2
            ],
            dim=1
        )

        d2 = self.dec2(d2)


        d1 = self.up1(d2)

        d1 = torch.cat(
            [
                d1,
                e1
            ],
            dim=1
        )

        d1 = self.dec1(d1)


        return self.final(d1)


# ============================================================
# CALCULATE CLASS WEIGHT
# ============================================================

print()
print("Calculating marine-debris class weight...")

positive_pixels = 0

negative_pixels = 0


for number, image_path in enumerate(
    train_images,
    start=1
):

    mask_path = get_mask_path(
        image_path
    )


    if not os.path.exists(
        mask_path
    ):

        continue


    with rasterio.open(
        mask_path
    ) as src:

        mask = src.read(1)


    positive_pixels += np.sum(
        mask == MARINE_DEBRIS_CLASS
    )

    negative_pixels += np.sum(
        mask != MARINE_DEBRIS_CLASS
    )


    if (
        number % 100 == 0
        or
        number == len(train_images)
    ):

        print(
            f"Processed masks: "
            f"{number}/"
            f"{len(train_images)}"
        )


if positive_pixels == 0:

    print()
    print(
        "ERROR: No marine-debris pixels found."
    )

    raise SystemExit


pos_weight_value = (
    negative_pixels /
    positive_pixels
)


pos_weight_value = min(
    pos_weight_value,
    MAX_POS_WEIGHT
)


print()
print(
    "Positive pixels:",
    int(positive_pixels)
)

print(
    "Negative pixels:",
    int(negative_pixels)
)

print(
    f"Calculated positive weight: "
    f"{pos_weight_value:.2f}"
)


pos_weight = torch.tensor(
    [pos_weight_value],
    dtype=torch.float32,
    device=device
)


# ============================================================
# MODEL
# ============================================================

print()
print("Creating improved U-Net...")


model = UNet(
    input_channels=11
)


model = model.to(
    device
)


print(
    "Model ready."
)


# ============================================================
# LOSS FUNCTIONS
# ============================================================

bce_loss = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)


def dice_loss(
    predictions,
    targets
):

    predictions = torch.sigmoid(
        predictions
    )


    predictions = predictions.reshape(
        -1
    )

    targets = targets.reshape(
        -1
    )


    intersection = (
        predictions *
        targets
    ).sum()


    dice = (
        2.0 * intersection
        +
        1.0
    ) / (
        predictions.sum()
        +
        targets.sum()
        +
        1.0
    )


    return 1.0 - dice


def focal_loss(
    predictions,
    targets,
    alpha=0.75,
    gamma=2.0
):

    bce = F.binary_cross_entropy_with_logits(
        predictions,
        targets,
        reduction="none"
    )


    probabilities = torch.sigmoid(
        predictions
    )


    pt = torch.where(
        targets == 1,
        probabilities,
        1 - probabilities
    )


    alpha_factor = torch.where(
        targets == 1,
        alpha,
        1 - alpha
    )


    focal = (
        alpha_factor
        *
        (1 - pt) ** gamma
        *
        bce
    )


    return focal.mean()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# ============================================================
# TRAINING
# ============================================================

best_val_loss = float(
    "inf"
)


for epoch in range(
    EPOCHS
):


    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    train_loss = 0.0

    valid_train_batches = 0


    for batch_number, (
        images,
        masks
    ) in enumerate(
        train_loader,
        start=1
    ):


        images = images.to(
            device
        )

        masks = masks.to(
            device
        )


        optimizer.zero_grad()


        outputs = model(
            images
        )


        if not torch.isfinite(
            outputs
        ).all():

            print()
            print(
                "WARNING: Non-finite output."
            )

            continue


        loss_bce = bce_loss(
            outputs,
            masks
        )


        loss_dice = dice_loss(
            outputs,
            masks
        )


        loss_focal = focal_loss(
            outputs,
            masks
        )


        # Combined loss

        loss = (
            0.40 * loss_bce
            +
            0.40 * loss_dice
            +
            0.20 * loss_focal
        )


        if not torch.isfinite(
            loss
        ):

            print()
            print(
                "WARNING: Non-finite loss."
            )

            continue


        loss.backward()


        # Gradient clipping

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


        optimizer.step()


        train_loss += (
            loss.item()
        )

        valid_train_batches += 1


        if (
            batch_number % 25 == 0
            or
            batch_number ==
            len(train_loader)
        ):

            print(
                f"Epoch "
                f"{epoch + 1}/{EPOCHS} "
                f"| Batch "
                f"{batch_number}/"
                f"{len(train_loader)}",
                end="\r"
            )


    average_train_loss = (
        train_loss /
        max(
            1,
            valid_train_batches
        )
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss = 0.0

    valid_val_batches = 0


    with torch.no_grad():

        for images, masks in val_loader:


            images = images.to(
                device
            )

            masks = masks.to(
                device
            )


            outputs = model(
                images
            )


            if not torch.isfinite(
                outputs
            ).all():

                continue


            loss_bce = bce_loss(
                outputs,
                masks
            )


            loss_dice = dice_loss(
                outputs,
                masks
            )


            loss_focal = focal_loss(
                outputs,
                masks
            )


            loss = (
                0.40 * loss_bce
                +
                0.40 * loss_dice
                +
                0.20 * loss_focal
            )


            if torch.isfinite(
                loss
            ):

                val_loss += (
                    loss.item()
                )

                valid_val_batches += 1


    average_val_loss = (
        val_loss /
        max(
            1,
            valid_val_batches
        )
    )


    # ========================================================
    # LEARNING RATE UPDATE
    # ========================================================

    scheduler.step(
        average_val_loss
    )


    current_lr = optimizer.param_groups[0]["lr"]


    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print()

    print(
        "========================================"
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print(
        "========================================"
    )

    print(
        f"Train Loss: "
        f"{average_train_loss:.4f}"
    )

    print(
        f"Validation Loss: "
        f"{average_val_loss:.4f}"
    )

    print(
        f"Learning Rate: "
        f"{current_lr:.7f}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if (
        np.isfinite(
            average_val_loss
        )
        and
        average_val_loss <
        best_val_loss
    ):

        best_val_loss = (
            average_val_loss
        )


        os.makedirs(
            "outputs",
            exist_ok=True
        )


        torch.save(
            model.state_dict(),
            MODEL_PATH
        )


        print()
        print(
            "BEST SEGMENTATION MODEL SAVED!"
        )


# ============================================================
# COMPLETE
# ============================================================

print()
print(
    "========================================"
)

print(
    " IMPROVED SEGMENTATION TRAINING COMPLETE"
)

print(
    "========================================"
)

print()

print(
    "Training images:",
    len(train_images)
)

print(
    "Validation images:",
    len(val_images)
)

print(
    "Best validation loss:",
    best_val_loss
)

print()

if os.path.exists(
    MODEL_PATH
):

    print(
        "Model saved successfully:"
    )

    print(
        MODEL_PATH
    )

else:

    print(
        "WARNING: No valid model was saved."
    )

print()

print(
    "Next step: Run model evaluation."
)

print()
