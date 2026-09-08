import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision.models import resnet18

from resnet_dataset import MARIDAResNetDataset


# ==========================================
# 1. Settings
# ==========================================

MODEL_PATH = (
    "outputs/marine_debris_resnet18_500.pth"
)

TEST_SPLIT = "splits/test_X.txt"

BATCH_SIZE = 8

MAX_SAMPLES = 100


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
# 3. Load test dataset
# ==========================================

test_dataset = MARIDAResNetDataset(
    TEST_SPLIT,
    max_samples=MAX_SAMPLES
)

print()
print(
    "Test samples:",
    len(test_dataset)
)


# ==========================================
# 4. DataLoader
# ==========================================

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ==========================================
# 5. Create ResNet18
# ==========================================

model = resnet18(
    weights=None
)


# Two classes:
#
# 0 = Non-Marine-Debris
# 1 = Marine-Debris

model.fc = nn.Linear(
    model.fc.in_features,
    2
)


# ==========================================
# 6. Load trained model
# ==========================================

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()


print()
print("Model loaded successfully.")


# ==========================================
# 7. Evaluation variables
# ==========================================

correct = 0
total = 0

true_positive = 0
true_negative = 0

false_positive = 0
false_negative = 0


# ==========================================
# 8. Run test
# ==========================================

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        labels = labels.to(device)


        # Model prediction
        outputs = model(images)

        predictions = outputs.argmax(
            dim=1
        )


        # Accuracy
        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)


        # Confusion matrix
        for prediction, label in zip(
            predictions,
            labels
        ):

            prediction = prediction.item()

            label = label.item()


            if prediction == 1 and label == 1:

                true_positive += 1


            elif prediction == 0 and label == 0:

                true_negative += 1


            elif prediction == 1 and label == 0:

                false_positive += 1


            elif prediction == 0 and label == 1:

                false_negative += 1


# ==========================================
# 9. Calculate metrics
# ==========================================

accuracy = correct / total


# Precision

if (
    true_positive + false_positive
) > 0:

    precision = (
        true_positive
        /
        (
            true_positive
            +
            false_positive
        )
    )

else:

    precision = 0.0


# Recall

if (
    true_positive + false_negative
) > 0:

    recall = (
        true_positive
        /
        (
            true_positive
            +
            false_negative
        )
    )

else:

    recall = 0.0


# F1

if (
    precision + recall
) > 0:

    f1 = (
        2
        *
        precision
        *
        recall
        /
        (
            precision
            +
            recall
        )
    )

else:

    f1 = 0.0


# ==========================================
# 10. Display results
# ==========================================

print()
print("========================================")
print("     RESNET18 MARIDA TEST RESULTS")
print("========================================")

print()

print(
    f"Test Accuracy:       {accuracy:.2%}"
)

print(
    f"Precision:            {precision:.2%}"
)

print(
    f"Recall:               {recall:.2%}"
)

print(
    f"F1 Score:             {f1:.2%}"
)


# ==========================================
# 11. Confusion matrix
# ==========================================

print()
print("Confusion Matrix")
print("-------------------------")

print(
    f"True Negative:        {true_negative}"
)

print(
    f"False Positive:       {false_positive}"
)

print(
    f"False Negative:       {false_negative}"
)

print(
    f"True Positive:        {true_positive}"
)


# ==========================================
# 12. Marine debris detection
# ==========================================

actual_debris = (
    true_positive
    +
    false_negative
)

detected_debris = true_positive


print()
print("Marine Debris Detection")
print("-------------------------")

print(
    f"Actual debris:        {actual_debris}"
)

print(
    f"Detected debris:      {detected_debris}"
)

if actual_debris > 0:

    detection_rate = (
        detected_debris
        /
        actual_debris
    )

    print(
        f"Detection rate:       "
        f"{detection_rate:.2%}"
    )
    