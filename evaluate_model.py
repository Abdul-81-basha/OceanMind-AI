import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision.models import vit_b_16

from full_dataset import FullMARIDADataset


# ==========================================
# 1. Load TEST dataset
# ==========================================

test_dataset = FullMARIDADataset(
    "splits/test_X.txt",
    max_samples=100
)

print()
print("Test samples:", len(test_dataset))


# ==========================================
# 2. DataLoader
# ==========================================

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False
)


# ==========================================
# 3. Create Vision Transformer
# ==========================================

model = vit_b_16(weights=None)

model.heads.head = nn.Linear(
    model.heads.head.in_features,
    2
)


# ==========================================
# 4. Load WEIGHTED model
# ==========================================

model.load_state_dict(
    torch.load(
        "outputs/marine_debris_vit_weighted.pth",
        map_location="cpu"
    )
)

model.eval()


# ==========================================
# 5. Evaluation
# ==========================================

correct = 0
total = 0

true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0


with torch.no_grad():

    for images, labels in test_loader:

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
# 6. Calculate metrics
# ==========================================

accuracy = correct / total


if true_positive + false_positive > 0:

    precision = (
        true_positive
        /
        (true_positive + false_positive)
    )

else:

    precision = 0


if true_positive + false_negative > 0:

    recall = (
        true_positive
        /
        (true_positive + false_negative)
    )

else:

    recall = 0


if precision + recall > 0:

    f1 = (
        2 * precision * recall
        /
        (precision + recall)
    )

else:

    f1 = 0


# ==========================================
# 7. Results
# ==========================================

print()
print("================================")
print("WEIGHTED MARIDA TEST RESULTS")
print("================================")

print(
    f"Test Accuracy:  {accuracy:.2f}"
)

print(
    f"Precision:      {precision:.2f}"
)

print(
    f"Recall:         {recall:.2f}"
)

print(
    f"F1 Score:       {f1:.2f}"
)


print()
print("Confusion Matrix")
print("----------------")

print(
    f"True Negative:  {true_negative}"
)

print(
    f"False Positive: {false_positive}"
)

print(
    f"False Negative: {false_negative}"
)

print(
    f"True Positive:  {true_positive}"
)


# ==========================================
# 8. Marine Debris Detection Rate
# ==========================================

actual_debris = (
    true_positive + false_negative
)

detected_debris = true_positive


print()
print("Marine Debris Detection")
print("------------------------")

print(
    f"Actual debris samples: "
    f"{actual_debris}"
)

print(
    f"Detected debris samples: "
    f"{detected_debris}"
)