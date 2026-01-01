import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score

# Load data
X = pd.read_csv("data/X_features.csv")
y = pd.read_csv("data/y_labels.csv").squeeze()

# Same test split
_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Choose best-performing model (Gradient Boosting here)
model = joblib.load("models/gradient_boosting.pkl")

# Predicted probabilities
y_prob = model.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.1, 0.91, 0.05)
results = []

for t in thresholds:
    y_pred = (y_prob >= t).astype(int)
    recall = recall_score(y_test, y_pred)

    results.append({
        "threshold": round(t, 2),
        "recall": recall
    })

results_df = pd.DataFrame(results)
best = results_df.loc[results_df["recall"].idxmax()]

print("\nThreshold vs Recall")
print(results_df)
print("\nBest threshold for recall:")
print(best)
