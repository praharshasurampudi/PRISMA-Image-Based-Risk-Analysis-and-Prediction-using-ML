import pandas as pd
import numpy as np

# Load structured data
structured_df = pd.read_csv("data/structured_data.csv")

# Load image embeddings
embeddings_df = pd.read_csv("embeddings/image_embeddings.csv")

# Select structured features
structured_features = structured_df[[
    "age",
    "bmi",
    "prior_conditions"
]]

# Combine structured + image features
X = pd.concat([structured_features, embeddings_df], axis=1)

# Target variable
y = structured_df["risk_label"]

# Sanity checks
assert len(X) == len(y), "Feature-label length mismatch"

# Save final datasets
X.to_csv("data/X_features.csv", index=False)
y.to_csv("data/y_labels.csv", index=False)

print("Feature engineering completed")
print("Feature matrix shape:", X.shape)
print("Label distribution:")
print(y.value_counts())
