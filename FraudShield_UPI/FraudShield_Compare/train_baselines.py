import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("model/fraudshield_upi.csv")

# Features & target
X = df.drop("class", axis=1)
y = df["class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
dt = DecisionTreeClassifier(random_state=42)
svm = SVC(probability=True, random_state=42)
rf = RandomForestClassifier(n_estimators=200, random_state=42)

# Train
dt.fit(X_train, y_train)
svm.fit(X_train_scaled, y_train)
rf.fit(X_train, y_train)

# Predict
dt_pred = dt.predict(X_test)
svm_pred = svm.predict(X_test_scaled)
rf_pred = rf.predict(X_test)

# Metrics
def evaluate(name, y_true, y_pred):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
        "precision": round(precision_score(y_true, y_pred), 3),
        "recall": round(recall_score(y_true, y_pred), 3),
        "f1": round(f1_score(y_true, y_pred), 3),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }

results = {
    "Decision Tree": evaluate("DT", y_test, dt_pred),
    "SVM": evaluate("SVM", y_test, svm_pred),
    "Random Forest": evaluate("RF", y_test, rf_pred),
}

# Save models + scaler
joblib.dump(dt, "model/dt_model.pkl")
joblib.dump(svm, "model/svm_model.pkl")
joblib.dump(rf, "model/rf_model.pkl")
joblib.dump(scaler, "model/svm_scaler.pkl")

print("\nBaseline Model Results:")
for k, v in results.items():
    print(f"\n{k}:")
    for m, val in v.items():
        print(f"  {m}: {val}")

print("\nModels saved successfully!")
