import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision.models import resnet18

from resnet_dataset import MARIDAResNetDataset


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = (
    "outputs/marine_debris_resnet18_500.pth"
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
# LOAD TEST DATA
# ============================================================

test_dataset = MARIDAResNetDataset(
    TEST_SPLIT,
    max_samples=MAX_SAMPLES
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


print()
print(
    "Test samples:",
    len(test_dataset)
)


# ============================================================
# CREATE MODEL
# ============================================================

model = resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()


print(
    "Model loaded successfully."
)


# ============================================================
# COLLECT PROBABILITIES
# ============================================================

all_probabilities = []

all_labels = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)


        # Convert logits to probabilities

        probabilities = torch.softmax(
            outputs,
            dim=1
        )


        # Probability of Marine Debris

        debris_probability = (
            probabilities[:, 1]
        )


        all_probabilities.extend(
            debris_probability.cpu().tolist()
        )


        all_labels.extend(
            labels.tolist()
        )


# ============================================================
# TEST DIFFERENT THRESHOLDS
# ============================================================

print()
print("==============================================")
print("      THRESHOLD ANALYSIS")
print("==============================================")

print()

print(
    "Threshold | Precision | Recall | F1"
)

print(
    "----------------------------------------------"
)


best_threshold = 0.50

best_f1 = 0.0


for threshold in [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80
]:

    true_positive = 0

    false_positive = 0

    false_negative = 0


    for probability, label in zip(
        all_probabilities,
        all_labels
    ):

        prediction = (
            1
            if probability >= threshold
            else 0
        )


        if prediction == 1 and label == 1:

            true_positive += 1


        elif prediction == 1 and label == 0:

            false_positive += 1


        elif prediction == 0 and label == 1:

            false_negative += 1


    # Precision

    if (
        true_positive +
        false_positive
    ) > 0:

        precision = (
            true_positive /
            (
                true_positive +
                false_positive
            )
        )

    else:

        precision = 0.0


    # Recall

    if (
        true_positive +
        false_negative
    ) > 0:

        recall = (
            true_positive /
            (
                true_positive +
                false_negative
            )
        )

    else:

        recall = 0.0


    # F1

    if precision + recall > 0:

        f1 = (
            2 *
            precision *
            recall
            /
            (
                precision +
                recall
            )
        )

    else:

        f1 = 0.0


    print(
        f"{threshold:8.2f} | "
        f"{precision:9.2%} | "
        f"{recall:6.2%} | "
        f"{f1:6.2%}"
    )


    # Best F1

    if f1 > best_f1:

        best_f1 = f1

        best_threshold = threshold


# ============================================================
# BEST THRESHOLD
# ============================================================

print()
print("==============================================")

print(
    f"BEST THRESHOLD: {best_threshold:.2f}"
)

print(
    f"BEST F1:        {best_f1:.2%}"
)

print("==============================================")

