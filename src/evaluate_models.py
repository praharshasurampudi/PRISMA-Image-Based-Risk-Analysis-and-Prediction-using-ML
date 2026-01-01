import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score

# Load data
X = pd.read_csv("data/X_features.csv")
y = pd.read_csv("data/y_labels.csv").squeeze()

# Same split as training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

models = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Random Forest": "models/random_forest.pkl",
    "Gradient Boosting": "models/gradient_boosting.pkl",
    "SVM": "models/svm.pkl"
}

print("\nModel Evaluation Results\n" + "-" * 40)

for name, path in models.items():
    model = joblib.load(path)

    # Probability scores for ROC-AUC
    y_prob = model.predict_proba(X_test)[:, 1]

    # Binary predictions for Recall
    y_pred = model.predict(X_test)

    roc_auc = roc_auc_score(y_test, y_prob)
    recall = recall_score(y_test, y_pred)

    print(f"{name}")
    print(f"  ROC-AUC : {roc_auc:.4f}")
    print(f"  Recall  : {recall:.4f}")
    print("-" * 40)
