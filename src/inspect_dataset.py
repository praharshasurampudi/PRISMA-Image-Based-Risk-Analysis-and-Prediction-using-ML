import os

# 🔴 IMPORTANT: replace this with the path printed by kagglehub
DATASET_PATH = r"C:\Users\hp\.cache\kagglehub\datasets\amanullahasraf\covid19-pneumonia-normal-chest-xray-pa-dataset\versions\1"

def inspect_directory(path, max_items=5):
    print(f"\nInspecting: {path}")
    items = os.listdir(path)
    print(f"Total items: {len(items)}")
    print("Sample items:", items[:max_items])

if __name__ == "__main__":
    inspect_directory(DATASET_PATH)

    for item in os.listdir(DATASET_PATH):
        sub_path = os.path.join(DATASET_PATH, item)
        if os.path.isdir(sub_path):
            inspect_directory(sub_path)
