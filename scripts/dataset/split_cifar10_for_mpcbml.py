import os
import json
from tqdm import tqdm

# ============== Config ==============
SRC_ROOT = "data/cifar10"                     # your existing folder

# Output files (same style as CUB)
DST_ROOT = r"data\cifar10"
TRAIN_FILE = os.path.join(DST_ROOT, "train.txt")
TEST_FILE  = os.path.join(DST_ROOT, "test.txt")
CLASS_COUNTS_FILE = os.path.join(DST_ROOT, "class_counts.json")
# ====================================

os.makedirs(DST_ROOT, exist_ok=True)


def process_split(split_name):
    src_split = os.path.join(SRC_ROOT, split_name)
    data_list = []
    class_counts = {}

    print(f"Processing {split_name} set...")

    for class_id in tqdm(sorted(os.listdir(src_split), key=lambda x: int(x))):
        class_path = os.path.join(src_split, class_id)
        if not os.path.isdir(class_path):
            continue

        label = int(class_id)

        for img_name in os.listdir(class_path):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            # Relative path that will be written in the txt file
            # Example: train/0/xxxx.png
            rel_path = os.path.join(split_name, class_id, img_name)

            data_list.append((rel_path, label))

            if split_name == "train":
                class_counts[label] = class_counts.get(label, 0) + 1

    return data_list, class_counts


def main():
    train_list, class_counts = process_split("train")
    test_list, _ = process_split("test")

    # Write train.txt and test.txt
    for filepath, data in [(TRAIN_FILE, train_list), (TEST_FILE, test_list)]:
        with open(filepath, "w") as f:
            for rel_path, label in data:
                # Format similar to your CUB script
                print(f"{rel_path},{label}", file=f)

    # Write class_counts.json
    with open(CLASS_COUNTS_FILE, "w") as f:
        json.dump(class_counts, f, indent=2)

    print("\n✅ Done!")
    print(f"train.txt         → {TRAIN_FILE}")
    print(f"test.txt          → {TEST_FILE}")
    print(f"class_counts.json → {CLASS_COUNTS_FILE}")


if __name__ == "__main__":
    main()