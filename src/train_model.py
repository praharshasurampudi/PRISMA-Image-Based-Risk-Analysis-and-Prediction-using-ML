import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Load features and labels
X = pd.read_csv("data/X_features.csv")
y = pd.read_csv("data/y_labels.csv").squeeze()

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# -------------------------
# 1. Logistic Regression
# -------------------------
log_reg = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    n_jobs=-1
)
log_reg.fit(X_train, y_train)
joblib.dump(log_reg, "models/logistic_regression.pkl")

# -------------------------
# 2. Random Forest
# -------------------------
rf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
joblib.dump(rf, "models/random_forest.pkl")

# -------------------------
# 3. Gradient Boosting
# -------------------------
gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)
gb.fit(X_train, y_train)
joblib.dump(gb, "models/gradient_boosting.pkl")

# -------------------------
# 4. Support Vector Machine
# -------------------------
svm = SVC(
    kernel="linear",
    probability=True,
    class_weight="balanced",
    random_state=42
)
svm.fit(X_train, y_train)
joblib.dump(svm, "models/svm.pkl")

print("All 4 models trained and saved successfully")
print("Train size:", X_train.shape)
print("Test size:", X_test.shape)
