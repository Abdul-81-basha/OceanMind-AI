import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)

from torchvision import transforms

from resnet_dataset import MARIDAResNetDataset


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 8

EPOCHS = 8

LEARNING_RATE = 0.00005

MODEL_PATH = (
    "outputs/marine_debris_resnet18_full.pth"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print()
print("========================================")
print("FULL MARIDA RESNET18 TRAINING")
print("========================================")

print()
print("Device:", device)


# ============================================================
# AUGMENTATION
# ============================================================

train_transform = transforms.Compose([

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomVerticalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=15
    )
])


# ============================================================
# DATASET WRAPPER
# ============================================================

class TransformDataset(Dataset):

    def __init__(
        self,
        dataset,
        transform=None
    ):

        self.dataset = dataset

        self.transform = transform


    def __len__(self):

        return len(self.dataset)


    def __getitem__(self, index):

        image, label = self.dataset[index]

        if self.transform is not None:

            image = self.transform(image)

        return image, label


# ============================================================
# LOAD OFFICIAL TRAINING DATA
# ============================================================

train_base = MARIDAResNetDataset(
    "splits/train_X.txt"
)


# ============================================================
# LOAD OFFICIAL VALIDATION DATA
# ============================================================

val_dataset = MARIDAResNetDataset(
    "splits/val_X.txt"
)


print()
print(
    "Full training samples:",
    len(train_base)
)

print(
    "Validation samples:",
    len(val_dataset)
)


# ============================================================
# TRAINING DATA WITH AUGMENTATION
# ============================================================

train_dataset = TransformDataset(
    train_base,
    transform=train_transform
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


# ============================================================
# COUNT CLASSES
# ============================================================

print()
print("Counting training classes...")


class_0 = 0
class_1 = 0


for i in range(len(train_base)):

    _, label = train_base[i]

    if label.item() == 0:

        class_0 += 1

    else:

        class_1 += 1


print()
print(
    "Non-Marine-Debris:",
    class_0
)

print(
    "Marine-Debris:",
    class_1
)


# ============================================================
# CALCULATE CLASS WEIGHTS
# ============================================================

total = class_0 + class_1

weight_0 = total / (
    2 * class_0
)

weight_1 = total / (
    2 * class_1
)


class_weights = torch.tensor(
    [
        weight_0,
        weight_1
    ],
    dtype=torch.float32
).to(device)


print()
print(
    "Class weight 0:",
    round(weight_0, 4)
)

print(
    "Class weight 1:",
    round(weight_1, 4)
)


# ============================================================
# CREATE PRETRAINED RESNET18
# ============================================================

print()
print("Loading pretrained ResNet18...")


model = resnet18(
    weights=ResNet18_Weights.DEFAULT
)


# ============================================================
# CHANGE FINAL LAYER
# ============================================================

model.fc = nn.Linear(
    model.fc.in_features,
    2
)


model = model.to(device)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.5
)


# ============================================================
# TRAINING LOOP
# ============================================================

best_val_accuracy = 0.0


for epoch in range(EPOCHS):

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


    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    train_loss = 0.0

    train_correct = 0

    train_total = 0


    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)


        optimizer.zero_grad()


        outputs = model(images)


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()


        optimizer.step()


        train_loss += loss.item()


        predictions = outputs.argmax(
            dim=1
        )


        train_correct += (
            predictions == labels
        ).sum().item()


        train_total += labels.size(0)


    train_accuracy = (
        train_correct /
        train_total
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss = 0.0

    val_correct = 0

    val_total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)

            labels = labels.to(device)


            outputs = model(images)


            loss = criterion(
                outputs,
                labels
            )


            val_loss += loss.item()


            predictions = outputs.argmax(
                dim=1
            )


            val_correct += (
                predictions == labels
            ).sum().item()


            val_total += labels.size(0)


    val_accuracy = (
        val_correct /
        val_total
    )


    # ========================================================
    # AVERAGE LOSS
    # ========================================================

    average_train_loss = (
        train_loss /
        len(train_loader)
    )

    average_val_loss = (
        val_loss /
        len(val_loader)
    )


    # ========================================================
    # PRINT
    # ========================================================

    print(
        f"Train Loss: "
        f"{average_train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy:.2%}"
    )

    print(
        f"Validation Loss: "
        f"{average_val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{val_accuracy:.2%}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print()
        print(
            "BEST MODEL SAVED!"
        )


    scheduler.step()


# ============================================================
# FINISHED
# ============================================================

print()
print(
    "========================================"
)

print(
    "FULL TRAINING COMPLETE!"
)

print(
    "========================================"
)

print()
print(
    "Best validation accuracy:",
    f"{best_val_accuracy:.2%}"
)

print()
print(
    "Model saved to:"
)

print(
    MODEL_PATH
)

