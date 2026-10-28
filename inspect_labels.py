import json
from pathlib import Path

p = Path("data/raw/labels_mapping.txt")
data = json.loads(p.read_text(encoding="utf-8-sig"))

print("Total image entries:", len(data))

for filename, labels in list(data.items())[:5]:
    print(filename, "->", labels)

print("Label vector lengths:", sorted(set(len(v) for v in data.values())))