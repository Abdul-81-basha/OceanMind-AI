import json
from collections import Counter


# ==========================================
# Load MARIDA label mapping
# ==========================================

with open(
    "labels_mapping.txt",
    "r"
) as f:

    labels = json.load(f)


print()
print("================================")
print("MARIDA LABEL DISTRIBUTION")
print("================================")

print(
    "Total labeled patches:",
    len(labels)
)


# ==========================================
# Count Marine Debris
# ==========================================

debris = 0
non_debris = 0


for filename, classes in labels.items():

    # First position = Marine Debris
    if classes[0] == 1:

        debris += 1

    else:

        non_debris += 1


print()
print(
    "Marine Debris:",
    debris
)

print(
    "Non-Marine-Debris:",
    non_debris
)


# ==========================================
# Percentages
# ==========================================

total = debris + non_debris

if total > 0:

    print()

    print(
        f"Marine Debris %: "
        f"{(debris / total) * 100:.2f}%"
    )

    print(
        f"Non-Debris %: "
        f"{(non_debris / total) * 100:.2f}%"
    )


# ==========================================
# Show examples
# ==========================================

print()
print("Examples:")
print("--------------------------------")

count = 0

for filename, classes in labels.items():

    print(
        filename,
        "MarineDebris =",
        classes[0]
    )

    count += 1

    if count >= 10:
        break 