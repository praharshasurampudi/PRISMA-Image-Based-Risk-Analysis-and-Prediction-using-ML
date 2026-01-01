import os
import pandas as pd
import numpy as np

# ✅ Correct dataset root (confirmed)
DATASET_PATH = r"C:\Users\hp\.cache\kagglehub\datasets\amanullahasraf\covid19-pneumonia-normal-chest-xray-pa-dataset\versions\1"

RISK_MAPPING = {
    "normal": 0,       # low risk
    "covid": 1,        # high risk
    "pneumonia": 1     # high risk
}

np.random.seed(42)

records = []

for class_name, risk in RISK_MAPPING.items():
    class_dir = os.path.join(DATASET_PATH, class_name)

    if not os.path.exists(class_dir):
        raise FileNotFoundError(f"Expected folder not found: {class_dir}")

    for img_name in os.listdir(class_dir):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(class_dir, img_name)

        record = {
            "image_path": img_path,
            "age": np.random.randint(18, 90),
            "bmi": round(np.random.uniform(18.0, 35.0), 1),
            "prior_conditions": np.random.randint(0, 5),
            "risk_label": risk
        }

        records.append(record)

df = pd.DataFrame(records)

os.makedirs("data", exist_ok=True)
df.to_csv("data/structured_data.csv", index=False)

print("Structured dataset created successfully")
print(df.head())
print("\nClass distribution:")
print(df["risk_label"].value_counts())
