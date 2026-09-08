import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split

from torchvision.models import vit_b_16

from full_dataset import FullMARIDADataset


# ==========================================
# 1. Load MARIDA dataset
# ==========================================

dataset = FullMARIDADataset(
    "splits/train_X.txt",
    max_samples=100
)

print()
print("Total samples:", len(dataset))


# ==========================================
# 2. Split into training and validation
# ==========================================

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


# ==========================================
# 3. Create DataLoaders
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False
)


# ==========================================
# 4. Create Vision Transformer
# ==========================================

model = vit_b_16(weights=None)

# 2 classes:
# 0 = Non-Marine-Debris
# 1 = Marine-Debris

model.heads.head = nn.Linear(
    model.heads.head.in_features,
    2
)


# ==========================================
# 5. Class-weighted loss
# ==========================================

# Marine Debris gets a higher penalty.
#
# Class 0 = Non-Debris
# Class 1 = Marine Debris

class_weights = torch.tensor(
    [1.0, 2.5],
    dtype=torch.float32
)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ==========================================
# 6. Optimizer
# ==========================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)


# ==========================================
# 7. Training
# ==========================================

epochs = 5

for epoch in range(epochs):

    # --------------------------------------
    # Training mode
    # --------------------------------------

    model.train()

    train_correct = 0
    train_total = 0
    train_loss = 0.0


    for images, labels in train_loader:

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate weighted loss
        loss = criterion(
            outputs,
            labels
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()


        # Track loss
        train_loss += loss.item()


        # Predictions
        predictions = outputs.argmax(
            dim=1
        )


        # Accuracy
        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)


    train_accuracy = (
        train_correct / train_total
    )


    # --------------------------------------
    # Validation mode
    # --------------------------------------

    model.eval()

    val_correct = 0
    val_total = 0
    val_loss = 0.0


    with torch.no_grad():

        for images, labels in val_loader:

            # Forward pass
            outputs = model(images)

            # Validation loss
            loss = criterion(
                outputs,
                labels
            )

            val_loss += loss.item()


            # Predictions
            predictions = outputs.argmax(
                dim=1
            )


            # Accuracy
            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)


    val_accuracy = (
        val_correct / val_total
    )


    # --------------------------------------
    # Display results
    # --------------------------------------

    print()
    print(
        f"Epoch {epoch + 1}/{epochs}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
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
# 8. Save improved model
# ==========================================

torch.save(
    model.state_dict(),
    "outputs/marine_debris_vit_weighted.pth"
)


print()
print("================================")
print("WEIGHTED MODEL SAVED!")
print("================================")
print()
print(
    "Saved to:"
)
print(
    "outputs/marine_debris_vit_weighted.pth"
)
