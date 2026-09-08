import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)

from resnet_dataset import MARIDAResNetDataset


# ==========================================
# 1. Settings
# ==========================================

MAX_SAMPLES = 500

BATCH_SIZE = 8

EPOCHS = 5

LEARNING_RATE = 0.0001


# ==========================================
# 2. Device
# ==========================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print()
print("Device:", device)


# ==========================================
# 3. Load MARIDA training dataset
# ==========================================

dataset = MARIDAResNetDataset(
    "splits/train_X.txt",
    max_samples=MAX_SAMPLES
)

print()
print("Total samples:", len(dataset))


# ==========================================
# 4. Split training / validation
# ==========================================

train_size = int(
    0.8 * len(dataset)
)

val_size = (
    len(dataset) - train_size
)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

print(
    "Training samples:",
    len(train_dataset)
)

print(
    "Validation samples:",
    len(val_dataset)
)


# ==========================================
# 5. DataLoaders
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ==========================================
# 6. Create pretrained ResNet18
# ==========================================

print()
print("Loading pretrained ResNet18...")


model = resnet18(
    weights=ResNet18_Weights.DEFAULT
)


# ==========================================
# 7. Replace final classification layer
# ==========================================

model.fc = nn.Linear(
    model.fc.in_features,
    2
)


model = model.to(device)


# ==========================================
# 8. Class-weighted loss
# ==========================================

# Class 0 = Non-Marine-Debris
# Class 1 = Marine-Debris

class_weights = torch.tensor(
    [1.0, 2.7],
    dtype=torch.float32
).to(device)


criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ==========================================
# 9. Optimizer
# ==========================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# 10. Training
# ==========================================

for epoch in range(EPOCHS):

    # --------------------------------------
    # Training
    # --------------------------------------

    model.train()

    train_loss = 0.0

    train_correct = 0

    train_total = 0


    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)


        # Clear gradients
        optimizer.zero_grad()


        # Forward pass
        outputs = model(images)


        # Loss
        loss = criterion(
            outputs,
            labels
        )


        # Backpropagation
        loss.backward()


        # Update model
        optimizer.step()


        # Track loss
        train_loss += loss.item()


        # Predictions
        predictions = outputs.argmax(
            dim=1
        )


        train_correct += (
            predictions == labels
        ).sum().item()


        train_total += labels.size(0)


    train_accuracy = (
        train_correct / train_total
    )


    # --------------------------------------
    # Validation
    # --------------------------------------

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
        val_correct / val_total
    )


    # --------------------------------------
    # Print results
    # --------------------------------------

    print()

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Train Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy:.2f}"
    )

    print(
        f"Validation Loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{val_accuracy:.2f}"
    )


# ==========================================
# 11. Save model
# ==========================================

torch.save(
    model.state_dict(),
    "outputs/marine_debris_resnet18_500.pth"
)


print()
print("================================")
print("RESNET18 MODEL SAVED!")
print("================================")

print()
print(
    "Saved to:"
)

print(
    "outputs/marine_debris_resnet18_500.pth"
)

