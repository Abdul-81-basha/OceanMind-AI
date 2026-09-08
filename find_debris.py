import json

labels_path = "dataset/labels_mapping.txt"

with open(labels_path, "r") as f:
    labels = json.load(f)

count = 0

print("Patches containing Marine Debris:\n")

for filename, classes in labels.items():

    # Marine Debris = first class in the mapping
    if classes[0] == 1:
        print(filename)
        count += 1

        if count == 10:
            break

print("\nFound", count, "Marine Debris patches.")