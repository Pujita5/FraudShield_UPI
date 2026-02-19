import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from imblearn.over_sampling import SMOTE

# =========================
# 1. LOAD DATASET
# =========================
data = pd.read_csv("../dataset/fraudshield_upi.csv")

print("Original Dataset Shape:", data.shape)
print("\nOriginal Class Distribution:")
print(data['class'].value_counts())

# =========================
# 2. SEPARATE FEATURES & LABEL
# =========================
X = data.drop('class', axis=1)
y = data['class']

# =========================
# 3. FEATURE SCALING
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 4. APPLY SMOTE
# =========================
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_scaled, y)

print("\nAfter SMOTE Shape:", X_smote.shape)
print("\nAfter SMOTE Class Distribution:")
print(pd.Series(y_smote).value_counts())

# =========================
# 5. APPLY PCA
# =========================
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_smote)

print("\nShape After PCA:", X_pca.shape)

# =========================
# 6. TRAIN-TEST SPLIT (80-20)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y_smote, test_size=0.2, random_state=42
)

print("\nTraining Set Size:", X_train.shape)
print("Testing Set Size:", X_test.shape)
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# =========================
# 7. XGBOOST MODEL TRAINING
# =========================
xgb_model = XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)

xgb_model.fit(X_train, y_train)

# =========================
# 8. MODEL PREDICTION
# =========================
y_pred = xgb_model.predict(X_test)

# =========================
# 9. MODEL EVALUATION
# =========================
accuracy = accuracy_score(y_test, y_pred)

print("\nXGBoost Model Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =========================
# 10. SAVE MODEL & PCA
# =========================
joblib.dump(xgb_model, "fraudshield_xgboost.pkl")
joblib.dump(pca, "pca_transform.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel and preprocessing files saved successfully!")
