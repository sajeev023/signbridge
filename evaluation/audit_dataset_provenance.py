import json
from pathlib import Path

data_path = Path("evaluation/real_world_landmarks_dataset.json")
with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total samples: {len(data)}")
participants = {}
conventions = {}
categories = {}

for s in data:
    p = s.get("participantId", "Unknown")
    participants[p] = participants.get(p, 0) + 1
    c = s.get("convention", "Unknown")
    conventions[c] = conventions.get(c, 0) + 1
    cat = s.get("category", "Unknown")
    categories[cat] = categories.get(cat, 0) + 1

print("\n--- Participants Breakdown ---")
for p, count in sorted(participants.items()):
    print(f"{p}: {count} samples")

print("\n--- Conventions Breakdown ---")
for c, count in sorted(conventions.items()):
    print(f"{c}: {count} samples")

print("\n--- Category Breakdown ---")
for cat, count in sorted(categories.items()):
    print(f"{cat}: {count} samples")

print("\n--- Detailed Per-Sample Inventory ---")
for idx, s in enumerate(data):
    sid = s.get("sampleId")
    p = s.get("participantId")
    lbl = s.get("signLabel")
    conv = s.get("convention")
    cat = s.get("category")
    print(f"{idx+1:02d}. ID: {sid:18s} | Signer: {p:10s} | Label: {lbl:2s} | Conv: {conv:20s} | Category: {cat}")
