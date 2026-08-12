"""
setup_dataset.py
────────────────────────────────────────────
Run this ONCE to create the full folder structure
for your medicine defect dataset.

Usage:
    python setup_dataset.py
"""

import os

CLASSES = [
    "normal",
    "pill",
    "capsule",
    "cracked",
    "broken",
    "chipped",
    "discolored",
    "contaminated",
    "coating_damage",
    "capped",
    "black_spot",
    "rough_surface",
    "dented",
]

SPLITS = ["train", "validation", "test"]

for split in SPLITS:
    for cls in CLASSES:
        path = os.path.join("dataset", split, cls)
        os.makedirs(path, exist_ok=True)
        # Add a placeholder so git tracks the (initially empty) folder
        placeholder = os.path.join(path, ".gitkeep")
        with open(placeholder, "w") as f:
            f.write("")

print("Dataset folder structure created successfully!\n")
print("Folder layout:")
for split in SPLITS:
    print(f"  dataset/{split}/")
    for cls in CLASSES:
        print(f"    |-- {cls}/   <- put your {cls} images here")
print("\nMinimum recommended images per class per split:")
print("  train:      >= 100 images per class")
print("  validation: >=  20 images per class")
print("  test:       >=  20 images per class")
print("\nIMPORTANT: dataset/train, dataset/validation, and dataset/test")
print("must all contain the exact same set of class subfolders, or")
print("training will stop with a clear error explaining the mismatch.")
