import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision.models import resnet18

from resnet_dataset import MARIDAResNetDataset


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = (
    "outputs/marine_debris_resnet18_full.pth"
)

TEST_SPLIT = "splits/test_X.txt"

BATCH_SIZE = 8

MAX_SAMPLES = 100


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print()
print("Device:", device)


# ============================================================
# TEST DATASET
# ============================================================

test_dataset = MARIDAResNetDataset(
    TEST_SPLIT,
    max_samples=MAX_SAMPLES
)

print()
print(
    "Test samples:",
    len(test_dataset)
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

model = resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)


# ============================================================
# LOAD FULL MODEL
# ============================================================

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()


print()
print("Full ResNet18 model loaded successfully.")


# ============================================================
# CONFUSION MATRIX
# ============================================================

true_negative = 0
false_positive = 0
false_negative = 0
true_positive = 0

correct = 0
total = 0


# ============================================================
# TEST
# ============================================================

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = outputs.argmax(
            dim=1
        )


        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)


        for prediction, label in zip(
            predictions,
            labels
        ):

            prediction = prediction.item()
            label = label.item()


            if prediction == 0 and label == 0:

                true_negative += 1


            elif prediction == 1 and label == 0:

                false_positive += 1


            elif prediction == 0 and label == 1:

                false_negative += 1


            elif prediction == 1 and label == 1:

                true_positive += 1


# ============================================================
# METRICS
# ============================================================

accuracy = correct / total


if true_positive + false_positive > 0:

    precision = (
        true_positive /
        (
            true_positive +
            false_positive
        )
    )

else:

    precision = 0.0


if true_positive + false_negative > 0:

    recall = (
        true_positive /
        (
            true_positive +
            false_negative
        )
    )

else:

    recall = 0.0


if precision + recall > 0:

    f1 = (
        2 * precision * recall
        /
        (precision + recall)
    )

else:

    f1 = 0.0


# ============================================================
# RESULTS
# ============================================================

print()
print("========================================")
print(" FULL RESNET18 MARIDA TEST RESULTS")
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


# ============================================================
# CONFUSION MATRIX
# ============================================================

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


# ============================================================
# MARINE DEBRIS DETECTION
# ============================================================

actual_debris = (
    true_positive +
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
        detected_debris /
        actual_debris
    )

    print(
        f"Detection rate:       "
        f"{detection_rate:.2%}"
    )

    